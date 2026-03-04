#!/usr/bin/env python3
"""
Real-Time i-ppDRE Monitor for RTX 2000 Ada GPU

This script demonstrates the incremental drift detection running in parallel
with the Student agent. It monitors activations in real-time and triggers
the safety gate when drift exceeds the calibrated threshold.

Hardware Target: RTX 2000 Ada (16GB, Tensor Cores)
Latency Target: <5ms per step for drift score computation
Memory Overhead: ~50MB for RFF weights + running statistics
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Add project root to path for detector imports
project_root = Path(__file__).parent.parent.parent
if str(project_root / "02_SRC") not in sys.path:
    sys.path.insert(0, str(project_root / "02_SRC"))

from acc_core.detector.ipp_dre import IncrementalDriftTracker
from acc_core.control.conformal import ConformalSafetyGate

print("LFG UAI 2026: LOADING UPDATED GPU MONITOR FROM DISK")

class GPUDriftMonitor:
    """
    High-Performance Drift Monitor using i-ppDRE.
    Optimized for RTX 2000 Ada Tensor Cores.
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        input_dim: int = 128,
        rff_dim: int = 512,
        device: str = "cuda",
        alpha_lambda: float = 0.99,
        epsilon: float = 0.05,
    ):
        """
        Args:
            hidden_size: Hidden dimension of the student model (e.g., 4096, 2048)
            input_dim: Dimensionality of projected activations (128D)
            rff_dim: Number of random Fourier features (512 recommended)
            device: cuda for RTX 2000 Ada
            alpha_lambda: Forgetting factor (0.99 = ~100 step memory)
            epsilon: Conformal error rate (0.05 = 95% coverage guarantee)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        print(f"Initializing GPUDriftMonitor on {self.device}")
        
        self.rff_dim = rff_dim
        self.alpha_lambda = alpha_lambda
        
        # Initialize i-ppDRE detector
        self.detector = IncrementalDriftTracker(
            input_dim=input_dim,
            rff_dim=rff_dim,
            alpha_lambda=alpha_lambda,
            device=self.device,
        )
        
        # Initialize conformal safety gate
        self.safety_gate = ConformalSafetyGate(epsilon=epsilon)
        
        # Performance metrics
        self.latencies = []
        self.drift_scores = []
        self.handoff_count = 0
        
        # Random projection matrix for hidden_size -> 128D compression
        print(f"Creating projection matrix: {hidden_size}D -> {input_dim}D")
        self.projection_matrix = torch.randn(
            hidden_size, input_dim, device=self.device, dtype=torch.float32
        ) / np.sqrt(hidden_size)
        self.projection_matrix.requires_grad = False
        
        print(f"i-ppDRE initialized: {input_dim}D input, {rff_dim} RFF features")
        print(f"Safety gate: ε={epsilon} (95% coverage guarantee)")

    def calibrate(self, teacher_activations_path: Path):
        """
        Calibrate the safety gate threshold using teacher's baseline activations.
        [UAI 2026] This is Phase 1 of the high-integrity control flow.
        """
        print(f"Calibrating monitor with baseline: {teacher_activations_path.name}")
        
        if not teacher_activations_path.exists():
            print("Warning: Calibration file not found. Using conservative defaults.")
            self.safety_gate.lambda_star = 0.002  # Default conservative threshold
            return

        # Load teacher activations (standard format: [N, hidden_size])
        teacher_data = np.load(teacher_activations_path)
        
        # Standardize hidden size for calibration
        if teacher_data.shape[1] == self.hidden_size:
            print(f"Projecting baseline {self.hidden_size}D -> 128D...")
            teacher_tensor = torch.as_tensor(
                teacher_data, device=self.device, dtype=torch.float32
            )
            teacher_projected = teacher_tensor @ self.projection_matrix
            teacher_data_np = teacher_projected.cpu().numpy()
        elif teacher_data.shape[1] == self.input_dim:
            teacher_data_np = teacher_data
        else:
            raise ValueError(f"Teacher data shape {teacher_data.shape} mismatch with monitor hidden_size {self.hidden_size}")

        # Update i-ppDRE baseline weights
        self.detector.update_teacher_baseline(torch.from_numpy(teacher_data_np))
        
        # Perform conformal calibration (find lambda_star for error rate epsilon)
        # In a real environment, we'd use multiple sequences. For the benchmark,
        # we compute the mean drift of the teacher on its own data (residual noise floor).
        residual_scores = []
        for i in range(min(100, len(teacher_data_np))):
            z_i = teacher_data_np[i]
            score = self.detector.score(z_i)
            residual_scores.append(score)
            
        # Set threshold to (1-epsilon) quantile of noise floor
        self.safety_gate.calibrate(residual_scores)
        print(f"Calibration complete. lambda* threshold: {self.safety_gate.lambda_star:.6f}")

    def monitor_step(self, hidden_state: torch.Tensor) -> Tuple[float, bool]:
        """
        Perform a single step of drift detection.
        This is called inside the Student's generation loop after each token.
        It must be extremely fast to avoid stalling the pipeline.
        
        [UAI 2026] This implements the 'Predict' and 'Audit' steps.
        If audit fails (drift too high), it signals the 'Correct' step 
        and check if handoff to teacher is required.
        
        Args:
            hidden_state: Student's last hidden state [hidden_size] on GPU
            
        Returns:
            (drift_score, should_handoff): Drift score and handoff decision
        """
        start = time.perf_counter()
        
        # Ensure hidden_state has batch dim removed if passed as [1, D]
        # and ensure it's on the correct device for multiplication
        hidden_state = hidden_state.to(self.device, dtype=torch.float32)
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.squeeze()
            
        # 1. Project to 128D (Tensor Core accelerated)
        if hidden_state.shape[-1] == self.hidden_size:
            z_t = (hidden_state @ self.projection_matrix).detach()
        elif hidden_state.shape[-1] == self.input_dim:
            z_t = hidden_state.detach()
        else:
            # Emergency: if it's still wrong, project it anyway if possible or fail gracefully
            raise ValueError(f"Unexpected hidden_state shape: {hidden_state.shape}. Expected last dim {self.hidden_size} or {self.input_dim}")
            
        # Convert to numpy for detector (ipp_dre expects 1D or 2D handled internally)
        z_t_np = z_t.cpu().numpy()
        
        # 2. Compute drift score with i-ppDRE
        w_raw = self.detector.score(z_t_np)
        # [UAI 2026] Normalization Shield: 
        # Scale score relative to feature dimension to keep it in calibrated range
        w_x = w_raw / np.sqrt(self.detector.feature_dim)
        
        # [AUDIT] Persistent Sparse Log (Every 50 steps) to ensure user we are not stuck
        if len(self.drift_scores) % 50 == 0:
             avg_lat = np.mean(self.latencies[-10:]) if self.latencies else 0
             print(f" [Drift Audit] Total Step {len(self.drift_scores)}: w={w_x:.6f} | Lat={avg_lat:.2f}ms", flush=True)
        
        # 3. Update weights for next step
        self.detector.update(z_t_np)
        
        # 4. Check safety gate
        should_handoff = self.safety_gate.check(w_x)
        
        # [UAI 2026 DEBUG]
        if w_x < 1e-6:
             z_norm = float(np.linalg.norm(z_t_np))
             if self.handoff_count < 5: # Limit spam
                 print(f" [DEBUG] Step w_x={w_x:.6f} z_norm={z_norm:.4f}")
        
        if should_handoff:
            self.handoff_count += 1
            
        latency = (time.perf_counter() - start) * 1000
        self.latencies.append(latency)
        self.drift_scores.append(w_x)
        
        return w_x, should_handoff

    def reset(self):
        """Reset the monitor state for a fresh task, preventing history leaks."""
        # Re-initialize the tracker to clear incremental weights 
        # (while preserving projection and config)
        self.detector = IncrementalDriftTracker(
            input_dim=self.input_dim,
            rff_dim=self.rff_dim,
            alpha_lambda=self.alpha_lambda,
            device=self.device,
        )
        self.latencies = []
        self.drift_scores = []
        self.handoff_count = 0
        # Restore teacher baseline if it was set
        # (For this campaign, we use direct sweet spots, so we don't strictly 
        # need the incremental baseline, but this is cleaner)

    def get_stats(self) -> dict:
        """Return performance and safety statistics."""
        avg_latency = np.mean(self.latencies) if self.latencies else 0
        p95_latency = np.percentile(self.latencies, 95) if self.latencies else 0
        avg_drift = np.mean(self.drift_scores) if self.drift_scores else 0
        
        return {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "avg_drift": avg_drift,
            "handoff_count": self.handoff_count,
            "total_steps": len(self.drift_scores)
        }
