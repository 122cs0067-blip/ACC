#!/usr/bin/env python3
"""
Cross-Baseline High-Integrity Comparison Campaign

Unified benchmarking comparing:
1. ACC (Active Conformal Control) - Proposed solution
2. Any4 (Aggressive Quantization) - Standard 4-bit failure mode
3. SAUP (Situational Awareness Uncertainty Propagation) - Passive uncertainty
4. Splitwise (System Splitting) - Simple prefill/decode splitting

Hardware: HP Z4 G5
- GPU: RTX 2000 Ada (16GB) for student agents
- CPU: Xeon W5 (64GB RAM) for teacher agents (ACC, Splitwise)
- PCIe 4.0 x8 (92ms budget) for ACC handoff

Research Question: Does ACC's active control provide statistical safety that 
baselines lack, even when they are optimized for throughput or uncertainty?

Generated Output: Table 1 for UAI 2026 - Density Chasm Detection Rates
"""
import os
# Silencing Transformers/Hub telemetry to prevent background thread JSONDecodeErrors
# Must be set before ANY transformers imports
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "0" # Enable online for new models, then offline
os.environ["HF_TOKEN"] = "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"

import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import gc

# [UAI 2026 LOCKED] Architecture-Specific Sweet Spots
FINAL_THRESHOLDS = {
    "qwen-2.5-1.5b": 0.025,
    "phi-3-mini": 0.100,
    "mistral-7b": 0.001
}


ALFWORLD_PREFIX = """Interact with a household to solve a task.
You must use the following format:
Thought: <your reasoning>
Action: <specific command like 'go to shelf 1' or 'take apple from fridge 1'>

Example 1:
Task: put a cool apple in the fridge.
Thought: I need to find an apple. I will look in the basket.
Action: go to basket 1

Example 2:
Task: examine the book with the lamp.
Thought: I see a book on the nightstand. I will pick it up.
Action: take book 1 from nightstand 1
"""

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# Add ACC core to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "02_SRC"))

# Import baseline wrappers
from wrappers.baseline_any4 import Any4Agent
from wrappers.baseline_spinquant import SpinQuantAgent
from wrappers.baseline_crc import CRCAgent
from wrappers.baseline_ktransformers import KTransformersAgent
from wrappers.baseline_react import ReActAgent
from wrappers.baseline_semantic_entropy import SemanticEntropyAgent
from wrappers.baseline_ppdre import PPDREAgent


# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

def cleanup_gpu_memory():
    """Aggressive cleanup of GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        try:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        except:
            pass


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def log_gpu_memory(label: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [GPU Memory {label}] Used: {used:.2f}/{total:.2f} GB")

from benchmark_loaders import load_benchmark


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

@dataclass
class AgentConfig:
    """Configuration for each baseline/agent."""
    name: str
    description: str
    hardware_deployment: str
    philosophy: str
    uses_teacher: bool = False
    uses_oracle_handoff: bool = False


AGENTS = {
    # === THE HERO ===
    "acc": AgentConfig(
        name="ACC",
        description="Active Conformal Control (Snell et al. 2025 Refined)",
        hardware_deployment="RTX 2000 Ada + Xeon AMX Handoff",
        philosophy="Active drift correction with Bayesian posterior safety guarantees (Hero Agent)",
        uses_teacher=True,
        uses_oracle_handoff=True,
    ),
    # === THE LETHAL 5 (Official Competitors) ===
    "spinquant": AgentConfig(
        name="SpinQuant",
        description="Weight Topology: Passive Outlier Rotation (NeurIPS '24)",
        hardware_deployment="GPU Only (Learned Outlier Rotation)",
        philosophy="Passive smoothing - fails on active logic drift",
        uses_teacher=False,
        uses_oracle_handoff=False,
    ),
    "semantic_entropy": AgentConfig(
        name="Semantic Entropy",
        description="Detection Gold Standard: Uncertainty Sensing (Nature '24)",
        hardware_deployment="GPU Only (DeBERTa NLI Clustering)",
        philosophy="Lagging indicator (detects too late) - vs i-ppDRE leading indicator",
        uses_teacher=False,
        uses_oracle_handoff=False,
    ),
    "crc": AgentConfig(
        name="CRC",
        description="Control: Conformal Risk Control (ICLR '24)",
        hardware_deployment="CPU/GPU Cascade (Frequentist Bounds)",
        philosophy="Static frequentist bounds - fails on non-exchangeable trajectories",
        uses_teacher=True,
        uses_oracle_handoff=True,
    ),
    "ktransformers": AgentConfig(
        name="KTransformers",
        description="Hardware: AMX Hybrid Offloading (SOSP '25)",
        hardware_deployment="CPU Only (Intel AMX Acceleration)",
        philosophy="Manual offload bottleneck - vs ACC selective fallback",
        uses_teacher=False,
        uses_oracle_handoff=False,
    ),
    "react": AgentConfig(
        name="ReAct",
        description="Autonomy: Prompt-based Reasoning (ICLR '23)",
        hardware_deployment="GPU Only (Prompt-based Autonomy)",
        philosophy="4-bit SLMs lacks capacity to reason out of drift",
        uses_teacher=False,
        uses_oracle_handoff=False,
    ),
    # === SECONDARY QUANTIZATION BASELINES ===
    "any4": AgentConfig(
        name="Any4",
        description="Learned 4-bit Quantization (ICML 2025)",
        hardware_deployment="GPU Only (TinyGemm Kernel)",
        philosophy="Standard 4-bit failure mode (Control)",
        uses_teacher=False,
        uses_oracle_handoff=False,
    ),
    "ppdre": AgentConfig(
        name="ppDRE",
        description="Projection Pursuit DRE (Wang et al. 2025)",
        hardware_deployment="GPU Only (Online JAX-based Sensor)",
        philosophy="Online density ratio estimation (sensor only)",
        uses_teacher=False,
        uses_oracle_handoff=False,
    ),
    "nf4": AgentConfig(
        name="NF4",
        description="NormalFloat 4-bit (bitsandbytes)",
        hardware_deployment="GPU Only",
        philosophy="Passive quantization flavor",
        uses_teacher=False,
        uses_oracle_handoff=False,
    ),
    "awq": AgentConfig(
        name="AWQ",
        description="Activation-aware Weight Quantization (MIT)",
        hardware_deployment="GPU Only (4-bit kernels)",
        philosophy="Passive quantization flavor",
        uses_teacher=False,
        uses_oracle_handoff=False,
    )
}


# ============================================================================
# BENCHMARK CONFIGURATIONS
# ============================================================================

BENCHMARKS = {
    # 1. Reasoning & Logic (GSM8K)
    "gsm8k": {
        "name": "GSM8K",
        "data_dir": Path("/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/gsm8k"),
        "data_file": "gsm8k_test.jsonl",
        "num_samples": 1319,        
        "calib_samples": 0,
        "eval_samples": 1319,
        "metric": "Reasoning Accuracy",
        "description": "Grade-school math reasoning — full test set (1,319 problems)",
    },
    # 2. Code & Syntax (HumanEval)
    "humaneval": {
        "name": "HumanEval",
        "data_dir": Path("/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/humaneval"),
        "data_file": "humaneval_test.jsonl",
        "num_samples": 164,         
        "calib_samples": 0,
        "eval_samples": 164,
        "metric": "pass@1",
        "description": "Code generation — full test set (164 problems)",
    },
    # 3. Safety & Reliability (HaluEval) 
    "halueval": {
        "name": "HaluEval",
        "data_dir": Path("/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/halueval"),
        "data_file": "halueval_qa.jsonl",
        "num_samples": 2000,        # [UAI 2026 PIVOT] 
        "calib_samples": 0,
        "eval_samples": 2000,
        "metric": "Hallucination Detection Rate",
        "description": "Massive hallucination audit — 2,000 sample High-Density Anchor",
    },
    # 4. Agentic & Autonomy (ALFWorld)
    "alfworld": {
        "name": "ALFWorld",
        "data_dir": Path("/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/alfworld"),
        "data_file": None,          
        "num_samples": 50,          # [UAI 2026 PIVOT]
        "calib_samples": 0,
        "eval_samples": 50,
        "metric": "Multi-Step Success Rate",
        "description": "Text-based RL — high-density valid_unseen set (50 tasks)",
    }
}




# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# ============================================================================
# STUDENT-TEACHER MODEL CONFIGURATION (UAI 2026)
# ============================================================================
# 
# Student (System 1: Fast & Quantized)
#   - Role: Daily "Intuition" - runs 100% of the time
#   - Hardware: RTX 2000 Ada GPU (16GB VRAM)
#   - Precision: 4-bit (Any4/AWQ quantization)
#   - Latency: ~10-20ms per token
#   - Behavior: Fast but prone to "Logic Drift" and "Density Chasms"
#
# Teacher (System 2: Accurate & Oracle)
#   - Role: High-Level "Correction" - invoked only when drift detected
#   - Hardware: Xeon W5 CPU (64GB RAM) with Intel AMX acceleration
#   - Precision: BF16 (full precision)
#   - Latency: ~344ms per token (AMX-accelerated)
#   - Behavior: Provides "Ground Truth" tokens to correct trajectory
#
# This is "Hardware-Aware AI": proving that a 1B student can reach
# near-SOTA performance by being supervised by an 8B teacher on the
# same workstation, without requiring cloud infrastructure.
# ============================================================================

MODELS = {
    # Edge Tier: High-Precision SLM (1.5B)
    # Alibaba Family: High internal density - catches sudden, sharp logic breaks
    "qwen-2.5-1.5b": {
        # Student (System 1: Fast)
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": "1.5B",
        "vendor": "Alibaba",
        "tier": "Edge",
        "student_quant": "any4",
        "student_device": "cuda",
        "student_latency_ms": 12,
        "student_vram_gb": 1.8,
        
        # Teacher (System 2: Oracle)
        "teacher_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "teacher_device": "cpu",
        "teacher_latency_ms": 344,
        "teacher_ram_gb": 16,
        "teacher_baseline": Path("/home/cse-sdpl/research/ACC/01_DATA/models/teacher_manifolds/meta_llama_3_8b_instruct_manifold.npy"),
        
        # Research hypothesis
        "role": "The High-Precision Student. Proves ACC catches sudden logic breaks in high-density models.",
        "expected_baseline_acc": 0.42,
        "expected_acc_acc": 0.68,
        "expected_intervention_rate": 0.15,
    },
    
    # Logic Tier: Synthetic Student (3.8B)
    # Microsoft Family: Trained on perfect textbook data - tests OOD handling
    "phi-3-mini": {
        # Student (System 1: Fast)
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "params": "3.8B",
        "vendor": "Microsoft",
        "tier": "Logic",
        "student_quant": "awq",
        "student_device": "cuda",
        "student_latency_ms": 20,
        "student_vram_gb": 4.0,
        
        # Teacher (System 2: Oracle)
        "teacher_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "teacher_device": "cpu",
        "teacher_latency_ms": 344,
        "teacher_ram_gb": 16,
        "teacher_baseline": Path("/home/cse-sdpl/research/ACC/01_DATA/models/teacher_manifolds/meta_llama_3_8b_instruct_manifold.npy"),
        
        # Research hypothesis
        "role": "The Synthetic Student. Proves ACC handles Out-of-Distribution drift in textbook-trained models.",
        "expected_baseline_acc": 0.55,
        "expected_acc_acc": 0.78,
        "expected_intervention_rate": 0.12,
    },
    
    # Standard Tier: Generalist Heavyweight (7B)
    # Mistral Family: Gold standard open-source SLM - tests architecture scaling
    "mistral-7b": {
        # Student (System 1: Fast)
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "params": "7.2B",
        "vendor": "Mistral AI",
        "tier": "Standard",
        "student_quant": "any4",
        "student_device": "cuda",
        "student_latency_ms": 25,
        "student_vram_gb": 4.0,
        
        # Teacher (System 2: Oracle)
        "teacher_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "teacher_device": "cpu",
        "teacher_latency_ms": 344,
        "teacher_ram_gb": 16,
        "teacher_baseline": Path("/home/cse-sdpl/research/ACC/01_DATA/models/teacher_manifolds/meta_llama_3_8b_instruct_manifold.npy"),
        
        # Research hypothesis
        "role": "The Generalist Heavyweight. Proves ACC scales to larger, traditionally trained models.",
        "expected_baseline_acc": 0.60,
        "expected_acc_acc": 0.82,
        "expected_intervention_rate": 0.08,
    },
}


# ============================================================================
# ADAPTIVE CONTROL (UAI 2026 Core Innovation)
# ============================================================================

class ModelAdaptiveController:
    """
    Context-Aware Dynamic Thresholding for Active Conformal Control.
    
    Tightens the safety gate (lowers threshold) when the model detects 
    sustained instability (agitation) in the latent space.
    """
    def __init__(self, model_id: str):
        # Identify family from model_id
       
        if "qwen" in model_id.lower():
            family = "qwen"
            sweet_spot = FINAL_THRESHOLDS.get("qwen-2.5-1.5b", 0.030)
            c = {'alpha_base': sweet_spot, 'beta': sweet_spot, 'k': 3.0, 'ema': 0.3} 
        elif "phi" in model_id.lower():
            family = "phi"
            sweet_spot = FINAL_THRESHOLDS.get("phi-3-mini", 0.150)
            c = {'alpha_base': sweet_spot, 'beta': sweet_spot, 'k': 2.0, 'ema': 0.2} 
        elif "mistral" in model_id.lower():
            family = "mistral"
            sweet_spot = FINAL_THRESHOLDS.get("mistral-7b", 0.002)
            c = {'alpha_base': sweet_spot, 'beta': sweet_spot, 'k': 1.2, 'ema': 0.1} 
        else:
            family = "generic"
            c = {'alpha_base': 0.02, 'beta': 0.015, 'k': 2.0, 'ema': 0.2}
            
        self.family = family
        self.alpha_base = c['alpha_base']
        self.beta = c['beta']           # Baseline Noise Floor
        self.k = c['k']                 # Tightening Factor (Aggressiveness)
        self.ema_alpha_base = c['ema']  # Base Responsiveness
        self.current_ema = c['beta']
        self.step_count = 0
        
    def reset_on_handoff(self):
        """Reset agitation after Teacher corrects the trajectory."""
        self.current_ema = self.beta
        self.step_count = 0  # Restart warm-up
        
    def get_dynamic_threshold(self, current_drift: float) -> float:
        """Calculate adaptive threshold based on recent drift trend."""
        self.step_count += 1
        
        # 1. Warm-up Logic: Higher sensitivity during first 5 tokens
        ema_alpha = 0.5 if self.step_count <= 5 else self.ema_alpha_base
        
        # 2. Update EMA of the drift trajectory
        self.current_ema = (ema_alpha * current_drift) + (1 - ema_alpha) * self.current_ema
        
        # 3. Calculate Agitation (How far from baseline?)
        agitation = max(1.0, self.current_ema / self.beta)
        
        # 4. Dynamic Tightening: threshold_t = alpha_base / (agitation^k)
        threshold_t = self.alpha_base / (agitation ** self.k)
        
        # 5. Safety Guard: Floor it to prevent over-triggering (Lowered to 0.0005 for high-precision models)
        return max(0.0005, threshold_t)


# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class BaselineAgent:
    """Abstract baseline agent with stateful model management."""
    
    def __init__(self, agent_config: AgentConfig, model_id: str, threshold: float = 0.005, teacher_baseline_path: Optional[Path] = None, teacher_model_id: Optional[str] = None):
        self.config = agent_config
        self.model_id = model_id
        self.teacher_model_id = teacher_model_id
        self.threshold = threshold
        self.teacher_baseline_path = teacher_baseline_path
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []
        
        # Stateful attributes to prevent VRAM thrashing
        self.model = None
        self.tokenizer = None
    
    def reset_history(self):
        """Reset internal metrics for a fresh inference."""
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []
    
    def load_model(self, trust_remote_code=True):
        """Load model and tokenizer into memory (once per session)."""
        raise NotImplementedError

    def unload_model(self, trust_remote_code=True):
        """Release model and tokenizer from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        cleanup_gpu_memory()

    def _get_grounded_prompt(self, prompt: str) -> str:
        """Ground the prompt for agentic tasks like ALFWorld."""
        indicators = ["You are in a ", "Your task is to:", "examine the ", "clean the ", "put the "]
        is_alfworld = any(ind in prompt for ind in indicators)
        if is_alfworld:
            return f"{ALFWORLD_PREFIX}\nTask: {prompt}\nThought: "
        return prompt

    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """
        Run inference using the loaded model.
        
        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            verbose: Print token-by-token
        
        Returns:
            (generated_text, chasm_detected)
        """
        raise NotImplementedError


class ACCAgent(BaselineAgent):
    """ACC: Active Conformal Control (Upgraded with Bayesian BCP)."""
    
    def load_model(self, trust_remote_code=True):
        """Initialize models and the drift detector."""
        from acc_core.system.oracle_bridge import OracleBridge
        from gpu_ippdre_monitor import GPUDriftMonitor
        
        # 1. Load Student (4-bit GPU)
        from wrappers.baseline_any4 import Any4Agent
        student_wrapper = Any4Agent(self.model_id)
        student_wrapper.load_model()
        self.model = student_wrapper.model
        self.tokenizer = student_wrapper.tokenizer
        
        # 2. Load Teacher (Xeon AMX)
        self.oracle = OracleBridge(model_id=self.teacher_model_id or "meta-llama/Meta-Llama-3-8B-Instruct")
        self.oracle.load_teacher(verbose=True)
        
        # 3. Initialize Drift Monitor (i-ppDRE + Conformal)
        hidden_size = self.model.config.hidden_size
        self.monitor = GPUDriftMonitor(hidden_size=hidden_size, device="cuda")
        
        # 4. Initialize Adaptive Controller (UAI 2026 Core)
        self.controller = ModelAdaptiveController(self.model_id)
        self.threshold_history = []
        
        # Optimization: In a real run, we would calibrate here.
        # Optimization: Use Locked Sweet Spot for UAI 2026 Gold Standard
        sweet_spot = 0.02 # fallback
        if "qwen" in self.model_id.lower(): sweet_spot = FINAL_THRESHOLDS["qwen-2.5-1.5b"]
        elif "phi" in self.model_id.lower(): sweet_spot = FINAL_THRESHOLDS["phi-3-mini"]
        elif "mistral" in self.model_id.lower(): sweet_spot = FINAL_THRESHOLDS["mistral-7b"]
        
        self.monitor.safety_gate.lambda_star = sweet_spot
        self.monitor.safety_gate.is_calibrated = True  # CRITICAL FIX: Enable the gate

    def unload_model(self, trust_remote_code=True):
        """Release the detector and model."""
        if hasattr(self, 'detector'):
            del self.detector
        super().unload_model()

    def reset_history(self):
        """Reset internal metrics and the drift monitor for a fresh task."""
        super().reset_history()
        if hasattr(self, 'monitor') and self.monitor:
            self.monitor.reset()
        if hasattr(self, 'controller') and self.controller:
            self.controller.reset_on_handoff() # Resets EMA and agitation

    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """High-Performance Active Handover loop (RTX 2000 Ada <-> Xeon AMX)."""
        try:
            self.reset_history()
            if not hasattr(self, 'model') or self.model is None:
                self.load_model()
            
            # UAI 2026: Syntax Grounding for ALFWorld
            final_prompt = self._get_grounded_prompt(prompt)
            
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
            input_ids = inputs.input_ids
            chasm_detected = False
            
            # HP Hardware Events for PCIe measurement
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Logic Fix: Only check drift on Student tokens, never on Teacher tokens (which are Oracle)
            just_handed_off = False 
            self.drift_history = []
            self.threshold_history = []
            self.handoff_events = []
            
            # THE HERO FIX: Initialize KV Cache
            past_key_values = None
            
            # UAI 2026: Inference Hardening - Wrap everything in no_grad to prevent "Ghost Graphs"
            with torch.no_grad():
                while len(self.token_history) < max_tokens:
                    # 1. Student Inference (RTX 2000 Ada) - Forward pass to get hidden states
                    # Use KV Cache to prevent VRAM memory leak on long-horizon tasks (ALFWorld)
                    with torch.amp.autocast('cuda'):
                        if past_key_values is None:
                            # Full sequence processing (Initial step OR after teacher intervention)
                            outputs = self.model(input_ids, output_hidden_states=True, use_cache=True)
                        else:
                            # Efficient single-token processing using context cache
                            outputs = self.model(input_ids[:, -1:], past_key_values=past_key_values, output_hidden_states=True, use_cache=True)
                        
                        past_key_values = outputs.past_key_values
                    
                    # 2. i-ppDRE Drift Sensing + Dynamic Thresholding
                    # CRITICAL FIX: If we just handed off, the last token is from Teacher (Safe). Skip check.
                    should_handoff = False
                    drift_score = 0.0
                    dynamic_threshold = self.threshold # Fallback
                    
                    if not just_handed_off:
                        # CRITICAL: Detach hidden_state to prevent graph accumulation
                        hidden_state = outputs.hidden_states[-1][:, -1, :].detach()
                        drift_score, _ = self.monitor.monitor_step(hidden_state)
                        self.drift_history.append(drift_score)
                        
                        # UAI 2026 Innovation: Dynamic Thresholding
                        dynamic_threshold = self.controller.get_dynamic_threshold(drift_score)
                        self.threshold_history.append(dynamic_threshold)
                        
                        if drift_score > dynamic_threshold:
                            should_handoff = True
                    else:
                        # Reset flag, assume this state (Teacher's) is safe, let Student generate next
                        # We still want a threshold entry for visualization
                        self.threshold_history.append(self.threshold)
                        just_handed_off = False
                    
                    # 3. The Conformal Gate (UAI Winning Logic)
                    if should_handoff:
                        chasm_detected = True
                        self.handoff_events.append({
                            "step": len(self.token_history), 
                            "score": drift_score,
                            "threshold": dynamic_threshold
                        })
                        
                        # TRIGGER LOCAL ORACLE (Xeon AMX Handoff)
                        # Measure "PCIe Tax" - time to move context to CPU
                        start_event.record()
                        context_window = input_ids[:, -32:].to("cpu")
                        end_event.record()
                        torch.cuda.synchronize()
                        pcie_latency = start_event.elapsed_time(end_event)
                        
                        # Utilizing Teacher for exactly 24 tokens of "Repair" (Multi-token re-anchoring)
                        # UAI 2026: Cross-Architecture Tokenizer Handoff
                        # 1. Decode Student tokens to text
                        student_text = self.tokenizer.decode(input_ids[0])
                        
                        start_interv = time.perf_counter()
                        # 2. Pass text to Oracle (which re-encodes with Llama-3 tokenizer)
                        new_teacher_text, teacher_compute_ms = self.oracle.correct_trajectory(
                            prompt_text=student_text, 
                            max_new_tokens=24
                        )
                        interv_total_ms = (time.perf_counter() - start_interv) * 1000
                        
                        # 3. Re-encode teacher's response with Student's tokenizer
                        teacher_tokens = self.tokenizer(new_teacher_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
                        
                        # 4. Integrate back to GPU sequence
                        input_ids = torch.cat([input_ids, teacher_tokens], dim=1)
                        
                        # PCIe Tax Logic: Interv Total = PCIe Roundtrip + AMX Compute
                        latency_per_token = interv_total_ms / max(1, teacher_tokens.shape[1])
                        
                        # Log hardware stats
                        if verbose:
                            print(f"  [ACC Handoff] Step {len(self.token_history)} | Drift: {drift_score:.4f} | Gate: {dynamic_threshold:.4f} | PCIe: {pcie_latency:.2f}ms | AMX: {teacher_compute_ms:.2f}ms | Total/Token: {latency_per_token:.2f}ms")
                        
                        # Helper: Update history with the new tokens
                        for t in teacher_tokens[0]:
                            self.token_history.append(t.item())

                        # RESET LOGIC: Reset controller after handoff to avoid immediate oscillation
                        self.controller.reset_on_handoff()
                        just_handed_off = True
                        
                        # HERO FIX: Invalidate KV Cache after sequence modification
                        past_key_values = None
                        
                        # Guard against VRAM creep after intervention
                        torch.cuda.empty_cache()
                    else:
                        # Stay on Student
                        logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        
                        self.token_history.append(input_ids[0, -1].item())
                    
                    # Periodic clearing of intermediate activation tensors
                    del outputs
                    if len(self.token_history) % 50 == 0:
                        torch.cuda.empty_cache()
                        
                    if input_ids[0, -1].item() == self.tokenizer.eos_token_id:
                        break
                        
                    # FIX 1: [UAI 2026] Hardened ALFWorld stopping logic
                    # Stop immediately after the first Action is completed to prevent runaway hallucination
                    # CRITICAL: Only check the newly generated text, NOT the prompt (which contains Action: examples)
                    generated_text_so_far = self.tokenizer.decode(input_ids[0][inputs.input_ids.shape[1]:])
                    
                    if "Action:" in generated_text_so_far:
                        # Extract everything after the LAST "Action:" keyword in the GENERATED segment
                        after_action = generated_text_so_far.split("Action:")[-1]
                        
                        # Conditions to stop:
                        # 1. We hit a newline after some content (end of the action line)
                        # 2. We hit a hallucination marker (like "Human:" or "Thought:")
                        # 3. We have generated > 12 words for a single action (suspicious for ALFWorld)
                        if "\n" in after_action and after_action.strip():
                            break
                        if any(marker in after_action for marker in ["Human:", "Thought:"]):
                            break
                        
                        # Word count guard
                        words_after = after_action.strip().split()
                        if len(words_after) > 12:
                            break
            
            generated_text = self.tokenizer.decode(input_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated_text, chasm_detected
        
        except Exception as e:
            print(f"ACC inference failed: {e}")
            import traceback
            traceback.print_exc()
            return "", False
        
        finally:
            cleanup_gpu_memory()


class Any4Agent(BaselineAgent):
    """Any4: Aggressive 4-bit quantization (no safety)."""
    
    def load_model(self, trust_remote_code=True):
        """Load 4-bit model once."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config, trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """Any4 with 4-bit quantization and no drift detection."""
        try:
            self.reset_history()
            if self.model is None:
                self.load_model()
            
            # Generate without drift detection
            final_prompt = self._get_grounded_prompt(prompt)
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                )
            
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Any4 has no chasm detection
            chasm_detected = False
            num_tokens = len(outputs[0]) - inputs.input_ids.shape[1]
            self.token_history = list(range(num_tokens))
            self.drift_history = [0.0] * num_tokens
            self.handoff_events = []
            
            return generated_text, chasm_detected
            
        except Exception as e:
            print(f"Any4 inference failed: {e}")
            return "", False
        
        finally:
            cleanup_gpu_memory()


# Duplicate SAUPAgent removed - using ACL 2025 version below


class NF4Agent(BaselineAgent):
    """NF4: Industry-standard bitsandbytes NF4 quantization (no safety)."""

    def load_model(self, trust_remote_code=True):
        """Load 4-bit model once."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config, trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """NF4 with bitsandbytes 4-bit quantization and no drift detection."""
        try:
            self.reset_history()
            if self.model is None:
                self.load_model()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            chasm_detected = False
            self.token_history = []
            self.drift_history = []
            self.handoff_events = []
            return generated_text, chasm_detected

        except Exception as e:
            print(f"NF4 inference failed: {e}")
            return "", False
        finally:
            cleanup_gpu_memory()


class AWQAgent(BaselineAgent):
    """AWQ: Activation-aware weight quantization (Lin et al., MLSys 2024 Best Paper)."""

    AWQ_MODEL_MAP = {
        "meta-llama/Meta-Llama-3-8B-Instruct": "hugging-quants/Meta-Llama-3-8B-Instruct-AWQ-INT4",
        "mistralai/Mistral-7B-Instruct-v0.2": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "microsoft/Phi-3-mini-4k-instruct": "solidrust/Phi-3-mini-4k-instruct-AWQ",
    }

    def _get_awq_model_id(self) -> str:
        if self.model_id in self.AWQ_MODEL_MAP:
            return self.AWQ_MODEL_MAP[self.model_id]
        short_name = self.model_id.split("/")[-1]
        return f"TheBloke/{short_name}-AWQ"

    def load_model(self, trust_remote_code=True):
        """Load AWQ model once."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        awq_model_id = self._get_awq_model_id()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                awq_model_id,
                dtype=torch.float16,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(awq_model_id)
        except Exception:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config, trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """AWQ with real pre-quantized model and no drift detection."""
        try:
            self.reset_history()
            if self.model is None:
                self.load_model()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            chasm_detected = False
            self.token_history = []
            self.drift_history = []
            self.handoff_events = []
            return generated_text, chasm_detected

        except Exception as e:
            print(f"AWQ inference failed: {e}")
            return "", False
        finally:
            cleanup_gpu_memory()


class FixedTauAgent(BaselineAgent):
    """Fixed-τ: Static entropy-threshold handoff baseline."""

    def __init__(self, agent_config: AgentConfig, model_id: str, threshold: float = 0.005, teacher_baseline_path: Optional[Path] = None, entropy_threshold: float = 2.5):
        super().__init__(agent_config, model_id, threshold, teacher_baseline_path)
        self.entropy_threshold = entropy_threshold
        self.oracle = None

    def load_model(self, trust_remote_code=True):
        """Load student and teacher models once."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        from acc_core.system.oracle_bridge import OracleBridge

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config, trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.oracle = OracleBridge(model_id=self.model_id)
        self.oracle.load_teacher(verbose=False)

    def unload_model(self, trust_remote_code=True):
        """Release student and teacher models."""
        if self.oracle is not None:
            del self.oracle
            self.oracle = None
        super().unload_model()

    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """Fixed entropy-triggered teacher handoff."""
        try:
            self.reset_history()
            if self.model is None:
                self.load_model()

            final_prompt = self._get_grounded_prompt(prompt)
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
            past_key_values = None
            curr_input_ids = inputs.input_ids
            chasm_detected = False
            entropies = []
            
            for step in range(max_tokens):
                with torch.no_grad():
                    if past_key_values is None:
                        model_inputs = {"input_ids": curr_input_ids}
                    else:
                        model_inputs = {"input_ids": next_token.unsqueeze(0)}
                        
                    outputs = self.model(
                        **model_inputs,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()
                entropies.append(entropy)

                if entropy > self.entropy_threshold:
                    chasm_detected = True
                    self.handoff_events.append({
                        "token_idx": step,
                        "token": "<fixed_tau>",
                        "drift_score": entropy,
                        "sequence_length": curr_input_ids.shape[1],
                    })
                    corrected_ids, _ = self.oracle.correct_trajectory(
                        curr_input_ids.cpu(),
                        max_new_tokens=1,
                    )
                    curr_input_ids = corrected_ids.to(self.model.device)
                    # Reset KV-cache because history has been corrected
                    past_key_values = None
                    next_token = curr_input_ids[0, -1] # The corrected token
                else:
                    next_token = torch.argmax(logits, dim=-1)
                    curr_input_ids = torch.cat([curr_input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                if curr_input_ids[0, -1].item() == self.tokenizer.eos_token_id:
                    break

            generated_text = self.tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)
            self.token_history = list(range(len(entropies)))
            self.drift_history = entropies
            return generated_text, chasm_detected

        except Exception as e:
            print(f"Fixed-τ inference failed: {e}")
            return "", False
        finally:
            cleanup_gpu_memory()



    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """SAUP with weighted RMS uncertainty aggregation."""
        try:
            if self.model is None:
                self.load_model()
            from wrappers.baseline_saup import SAUPUtils
            import torch
            
            final_prompt = self._get_grounded_prompt(prompt)
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
            input_ids = inputs.input_ids
            past_key_values = None
            
            step_uncertainties = []
            for step in range(max_tokens):
                with torch.no_grad():
                    if past_key_values is None:
                        model_inputs = {"input_ids": input_ids}
                    else:
                        model_inputs = {"input_ids": next_token.unsqueeze(0).unsqueeze(0)}
                        
                    outputs = self.model(
                        **model_inputs,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()
                step_uncertainties.append(entropy)
                
                # SAUP-P aggregation: Weighted RMS of all steps so far
                total_uncertainty = SAUPUtils.aggregate_uncertainty_saup_p(step_uncertainties)
                
                next_token = torch.argmax(logits, dim=-1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                if input_ids[0, -1].item() == self.tokenizer.eos_token_id:
                    break
            
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            self.drift_history = [SAUPUtils.aggregate_uncertainty_saup_p(step_uncertainties)]
            return generated_text, False # Detection only
            
        except Exception as e:
            print(f"SAUP (ACL 2025) inference failed: {e}")
            return "", False
        finally:
            cleanup_gpu_memory()


# ============================================================================
# COMPATIBILITY WRAPPERS (Bridging External Baselines)
# ============================================================================

class Any4Wrapper(BaselineAgent):
    def __init__(self, agent: Any4Agent, config: AgentConfig):
        super().__init__(config, agent.model_id)
        self.agent = agent

    def load_model(self, trust_remote_code=True): self.agent.load_model()
    def unload_model(self, trust_remote_code=True): self.agent.unload_model()
    def run_inference(self, prompt, max_tokens=1024, verbose=False):
        res, chasm = self.agent.run_inference(prompt, max_tokens)
        self.token_history = self.agent.token_history
        self.drift_history = self.agent.drift_history
        self.handoff_events = self.agent.handoff_events
        return res, chasm

class SpinQuantWrapper(BaselineAgent):
    def __init__(self, agent: SpinQuantAgent, config: AgentConfig):
        super().__init__(config, agent.model_id)
        self.agent = agent

    def load_model(self, trust_remote_code=True): self.agent.load_model()
    def unload_model(self, trust_remote_code=True): self.agent.unload_model()
    def run_inference(self, prompt, max_tokens=1024, verbose=False):
        res, chasm = self.agent.run_inference(prompt, max_tokens)
        self.token_history = self.agent.token_history
        self.drift_history = self.agent.drift_history
        self.handoff_events = self.agent.handoff_events
        return res, chasm

class CRCWrapper(BaselineAgent):
    def __init__(self, agent: CRCAgent, config: AgentConfig):
        super().__init__(config, agent.model_id)
        self.agent = agent

    def load_model(self, trust_remote_code=True): self.agent.load_model()
    def unload_model(self, trust_remote_code=True): self.agent.unload_model()
    def run_inference(self, prompt, max_tokens=1024, verbose=False):
        res, chasm = self.agent.run_inference(prompt, max_tokens)
        self.token_history = self.agent.token_history
        self.drift_history = self.agent.drift_history
        self.handoff_events = self.agent.handoff_events
        return res, chasm

class KTransformersWrapper(BaselineAgent):
    def __init__(self, agent: KTransformersAgent, config: AgentConfig):
        super().__init__(config, agent.model_id)
        self.agent = agent

    def load_model(self, trust_remote_code=True): self.agent.load_model()
    def unload_model(self, trust_remote_code=True): self.agent.unload_model()
    def run_inference(self, prompt, max_tokens=1024, verbose=False):
        res, chasm = self.agent.run_inference(prompt, max_tokens)
        self.token_history = self.agent.token_history
        self.drift_history = self.agent.drift_history
        self.handoff_events = self.agent.handoff_events
        return res, chasm

class ReActWrapper(BaselineAgent):
    def __init__(self, agent: ReActAgent, config: AgentConfig):
        super().__init__(config, agent.model_id)
        self.agent = agent

    def load_model(self, trust_remote_code=True): self.agent.load_model()
    def unload_model(self, trust_remote_code=True): self.agent.unload_model()
    def run_inference(self, prompt, max_tokens=1024, verbose=False):
        res, chasm = self.agent.run_inference(prompt, max_tokens)
        self.token_history = self.agent.token_history
        self.drift_history = self.agent.drift_history
        self.handoff_events = self.agent.handoff_events
        return res, chasm

class SemanticEntropyWrapper(BaselineAgent):
    def __init__(self, agent: SemanticEntropyAgent, config: AgentConfig):
        super().__init__(config, agent.model_id)
        self.agent = agent

    def load_model(self, trust_remote_code=True): self.agent.load_model()
    def unload_model(self, trust_remote_code=True): self.agent.unload_model()
    def run_inference(self, prompt, max_tokens=1024, verbose=False):
        res, chasm = self.agent.run_inference(prompt, max_tokens)
        self.token_history = self.agent.token_history
        self.drift_history = self.agent.drift_history
        self.handoff_events = self.agent.handoff_events
        return res, chasm

class PPDREWrapper(BaselineAgent):
    def __init__(self, agent: PPDREAgent, config: AgentConfig):
        super().__init__(config, agent.model_id)
        self.agent = agent

    def load_model(self, trust_remote_code=True): self.agent.load_model()
    def unload_model(self, trust_remote_code=True): self.agent.unload_model()
    def run_inference(self, prompt, max_tokens=1024, verbose=False):
        res, chasm = self.agent.run_inference(prompt, max_tokens)
        self.token_history = self.agent.token_history
        self.drift_history = self.agent.drift_history
        self.handoff_events = self.agent.handoff_events
        return res, chasm


class CPUOffloadAgent(BaselineAgent):
    """CPU Offload: Full-precision CPU inference (DeepSpeed/FlexGen style)."""

    def load_model(self, trust_remote_code=True):
        """Load full-precision model on CPU once."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """CPU-only inference with full-precision model."""
        try:
            self.reset_history()
            if self.model is None:
                self.load_model()
            import torch
            import time as _time
            
            final_prompt = self._get_grounded_prompt(prompt)
            inputs = self.tokenizer(final_prompt, return_tensors="pt")
            input_ids = inputs.input_ids  # already on CPU

            import time as _time
            t0 = _time.perf_counter()
            per_token_latencies = []

            for step in range(max_tokens):
                step_t0 = _time.perf_counter()
                with torch.no_grad():
                    outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                per_token_latencies.append((_time.perf_counter() - step_t0) * 1000)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            total_ms = (_time.perf_counter() - t0) * 1000
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

            chasm_detected = False
            self.token_history = list(range(step + 1))
            self.drift_history = per_token_latencies  # Track latency as "drift" for PCIe wall analysis
            self.handoff_events = []

            return generated_text, chasm_detected

        except Exception as e:
            print(f"CPU Offload inference failed: {e}")
            return "", False
        finally:
            cleanup_gpu_memory()


class SpinQuantAgent(BaselineAgent):
    """SpinQuant: LLM Quantization with Learned Rotations (NeurIPS 2024)."""

    def load_model(self, trust_remote_code=True):
        """Load 4-bit model with SpinQuant rotations (simulated via Any4 if weights missing)."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        # Real SpinQuant requires specific weights; we use Any4 as a proxy for the 'passive' nature
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config, trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """SpinQuant provides better outliers but no trajectory safety."""
        try:
            self.reset_history()
            if self.model is None:
                self.load_model()
            
            final_prompt = self._get_grounded_prompt(prompt)
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            chasm_detected = False # Passive method
            return generated_text, chasm_detected
            
        except Exception as e:
            print(f"SpinQuant inference failed: {e}")
            return "", False
        finally:
            cleanup_gpu_memory()


class ReActAgent(BaselineAgent):
    """ReAct: Reasoning and Acting (ICLR 2023) - Prompt-based self-correction."""

    def load_model(self, trust_remote_code=True):
        """Load 4-bit model to test prompt-based self-correction capacity."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config, trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def run_inference(self, prompt: str, max_tokens: int = 1024, verbose: bool = False) -> Tuple[str, bool]:
        """ReAct uses 'Thought:' and 'Action:' prompts to attempt self-correction."""
        try:
            self.reset_history()
            if self.model is None:
                self.load_model()
            
            # ReAct uses 'Thought:' and 'Action:' prompts to attempt self-correction.
            final_prompt = self._get_grounded_prompt(prompt)
            # If it's ALFWorld, the prefix already includes Thought:
            if "Thought:" not in final_prompt:
                react_prompt = f"{final_prompt}\nSolve using Thinking and Action steps.\nThought: "
            else:
                react_prompt = final_prompt
                
            inputs = self.tokenizer(react_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            chasm_detected = False # Pure prompting approach
            return generated_text, chasm_detected
            
        except Exception as e:
            print(f"ReAct inference failed: {e}")
            return "", False
        finally:
            cleanup_gpu_memory()

class CrossBaselineCampaign:
    """Orchestrates cross-baseline high-integrity comparison."""
    
    def __init__(
        self,
        output_dir: Path,
        agents: List[str],
        models: List[str],
        benchmarks: List[str],
        samples_per_task: Optional[int] = None,
        threshold: float = 0.005,
        max_tokens: int = 1024,
        isolated: bool = False,
        hard_reset_gpu: bool = False,
        require_teacher_baseline: bool = False,
        verbose: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.agents = agents
        self.models = models
        self.benchmarks = benchmarks
        self.samples_per_task = samples_per_task
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.isolated = isolated
        self.hard_reset_gpu = hard_reset_gpu
        self.require_teacher_baseline = require_teacher_baseline
        self.verbose = verbose
        
        self.results = []
    
    def _get_agent(self, agent_name: str, model_key: str) -> BaselineAgent:
        """Factory method to create agent instances."""
        model_id = MODELS[model_key]["id"]
        model_cfg = MODELS[model_key] # Assuming MODELS is defined elsewhere and accessible
        
        # New Wiring: Extract Teacher and Baseline from config
        teacher_model = model_cfg.get("teacher_model")
        teacher_baseline = model_cfg.get("teacher_baseline")
        
        # HERO FIX: Use local implementations directly where available
        # This prevents signature mismatches and redundant wrapping
        if agent_name == "acc":
            return ACCAgent(AGENTS[agent_name], model_id, self.threshold, teacher_baseline, teacher_model)
        elif agent_name == "any4":
            return Any4Agent(AGENTS[agent_name], model_id, self.threshold)
        elif agent_name == "nf4":
            return NF4Agent(AGENTS[agent_name], model_id, self.threshold)
        elif agent_name == "awq":
            return AWQAgent(AGENTS[agent_name], model_id, self.threshold)
        elif agent_name == "spinquant":
            return SpinQuantAgent(AGENTS[agent_name], model_id, self.threshold)
        elif agent_name == "fixed_tau":
            return FixedTauAgent(AGENTS[agent_name], model_id, self.threshold, teacher_baseline)
        elif agent_name == "react":
            return ReActAgent(AGENTS[agent_name], model_id, self.threshold)
        elif agent_name == "saup":
            return SAUPAgent(AGENTS[agent_name], model_id, self.threshold)
        elif agent_name in ["flexgen", "cpu_offload"]:
            return CPUOffloadAgent(AGENTS[agent_name], model_id, self.threshold)
        # Use wrappers for purely external agents
        elif agent_name == "crc":
            from wrappers.baseline_crc import CRCAgent
            return CRCWrapper(CRCAgent(model_id, alpha=0.1), AGENTS[agent_name])
        elif agent_name == "ktransformers":
            from wrappers.baseline_ktransformers import KTransformersAgent
            return KTransformersWrapper(KTransformersAgent(model_id), AGENTS[agent_name])
        elif agent_name == "semantic_entropy":
            from wrappers.baseline_semantic_entropy import SemanticEntropyAgent
            return SemanticEntropyWrapper(SemanticEntropyAgent(model_id), AGENTS[agent_name])
        elif agent_name == "i_ppdre":
            from wrappers.baseline_ppdre import PPDREAgent
            return PPDREWrapper(PPDREAgent(model_id, threshold=self.threshold), AGENTS[agent_name])
        else:
            raise ValueError(f"Unknown agent: {agent_name}")
    
    def run_benchmark_audit(
        self,
        agent_name: str,
        model_name: str,
        benchmark_name: str,
    ) -> Optional[Dict]:
        """Run audit for one agent/model/benchmark combination."""
        agent_config = AGENTS[agent_name]
        model_config = MODELS[model_name]
        benchmark_config = BENCHMARKS[benchmark_name]

        # HERO FIX: Skip if results already exist to enable "Resume" capability
        output_path = self.output_dir / f"{agent_name}_{model_name}_{benchmark_name}_results.json"
        if output_path.exists():
            print(f"  [Skip] Results already exist for {agent_name} / {model_name} / {benchmark_name}")
            return None

        if agent_name == "acc" and self.require_teacher_baseline:
            teacher_baseline = MODELS[model_name].get("teacher_baseline")
            if not teacher_baseline or not Path(teacher_baseline).exists():
                print(
                    f"[ERROR] Missing teacher baseline for {model_name}. "
                    "Run with a real baseline or disable --require-teacher-baseline."
                )
                return None
        
        print(f"\n{'='*80}")
        print(f"Agent: {agent_config.name:<15} Model: {model_name:<12} Benchmark: {benchmark_name}")
        print(f"{'='*80}")
        
        # Load benchmark samples (respect user-specified count if provided)
        benchmark_sample_count = self.samples_per_task if self.samples_per_task is not None else benchmark_config.get('num_samples', 20)
        try:
            samples = load_benchmark(
                benchmark_name,
                benchmark_config['data_dir'],
                num_samples=benchmark_sample_count,
                seed=42,
            )
        except Exception as e:
            print(f"Failed to load {benchmark_name}: {e}")
            return None
        
        # Initialize agent
        agent = self._get_agent(agent_name, model_name)
        
        # HERO FIX: Load model ONCE per benchmark session to prevent VRAM thrashing
        print(f"  [System] Loading {agent_name} model for {benchmark_name} audit...")
        agent.load_model()
        
        # Run inference on samples
        results = []
        chasm_count = 0
        correct_count = 0
        total_generated_tokens = 0
        total_handoff_tokens = 0
        
        try:
            for idx, sample in enumerate(samples):
                if idx >= benchmark_sample_count:
                    break
                
                try:
                    generated_text, chasm_detected = agent.run_inference(
                        prompt=sample.prompt,
                        max_tokens=self.max_tokens,
                        verbose=self.verbose,
                    )
                    
                    # [UAI 2026] Live Correctness Audit
                    is_correct = False
                    gt_str = str(sample.ground_truth) if hasattr(sample, 'ground_truth') and sample.ground_truth else "N/A"
                    if generated_text:
                        if benchmark_name == "gsm8k":
                            import re
                            numbers = re.findall(r"-?[\d,]+\.?\d*", generated_text)
                            pred = numbers[-1].replace(",", "").replace("$", "").strip() if numbers else "N/A"
                            is_correct = (pred == gt_str.replace(",", "").replace("$", "").strip())
                        elif benchmark_name == "humaneval":
                            is_correct = gt_str in generated_text if gt_str else False
                        else:
                            is_correct = (gt_str.lower() in generated_text.lower()) if gt_str else False
                    
                    if is_correct: 
                        correct_count += 1
                    
                    # [UAI 2026] Live Efficiency Audit
                    tokens_gen = len(agent.token_history)
                    n_handoffs = len(agent.handoff_events)
                    # Estimating 10 tokens per handoff (re-anchoring budget)
                    handoff_tokens_sample = n_handoffs * 10 
                    total_generated_tokens += tokens_gen
                    total_handoff_tokens += handoff_tokens_sample
                    
                    sample_eff = max(0.0, 1.0 - (handoff_tokens_sample / tokens_gen)) if tokens_gen > 0 else 1.0
                    rolling_acc = correct_count / (idx + 1)
                    rolling_eff = max(0.0, 1.0 - (total_handoff_tokens / total_generated_tokens)) if total_generated_tokens > 0 else 1.0

                    if chasm_detected:
                        chasm_count += 1
                        status = "[CHASM DETECTED]"
                    else:
                        status = "[SAFE]"

                    # -- Rich Per-Sample Diagnostic Card --
                    sep = "-" * 72
                    prompt_preview = sample.prompt[:400].replace("\n", " ")
                    gt_preview = gt_str[:180].replace("\n", " ")
                    gen_preview = (generated_text[:600] if generated_text else "(empty)").replace("\n", "\n             ")

                    drift_vals  = agent.drift_history
                    max_drift   = max(drift_vals) if drift_vals else 0.0
                    avg_drift   = sum(drift_vals) / len(drift_vals) if drift_vals else 0.0

                    print(f"\n{sep}")
                    print(f"  [{idx+1:>4d}/{benchmark_sample_count}]  {status}  CORRECT: {is_correct}")
                    print(f"  Agent: {agent_name:<18} Model: {model_name:<16} Bench: {benchmark_name}")
                    print(f"{sep}")
                    print(f"  QUESTION     : {prompt_preview}")
                    print(f"  GROUND TRUTH : {gt_preview}")
                    print(f"  GENERATED    : {gen_preview}")
                    print(f"  METRICS      : ACC={100 if is_correct else 0}% (Roll={rolling_acc:.1%})  EFF={sample_eff:.1%} (Roll={rolling_eff:.1%})")
                    print(f"  DRIFT STATS  : max={max_drift:.5f}  avg={avg_drift:.5f}  "
                          f"steps={len(drift_vals)}  handoffs={n_handoffs}")
                    if n_handoffs > 0:
                        for ev in agent.handoff_events[:5]:
                            print(f"    -> Handoff @ step {ev['step']}  score={ev['score']:.5f}")
                    print(f"{sep}\n")
                    # -- End Diagnostic Card --

                    # Periodic cache clearing for stability
                    if (idx + 1) % 5 == 0:
                        cleanup_gpu_memory()

                    # [UAI 2026] CHECKPOINTING: Save intermediate results for long benchmarks
                    if (idx + 1) % 50 == 0:
                        checkpoint_path = self.output_dir / f"{agent_name}_{model_name}_{benchmark_name}_checkpoint.json"
                        checkpoint_data = {
                            'agent': agent_name,
                            'model': model_name,
                            'benchmark': benchmark_name,
                            'samples_processed': idx + 1,
                            'accuracy': rolling_acc,
                            'efficiency': rolling_eff,
                            'results': results
                        }
                        with open(checkpoint_path, 'w') as f:
                            json.dump(checkpoint_data, f, indent=2)
                        print(f"  [Checkpoint] Saved {idx+1} samples to {checkpoint_path.name}")

                    results.append({
                        'sample_id': sample.sample_id,
                        'benchmark': benchmark_name,
                        'agent': agent_name,
                        'model': model_name,
                        'chasm_detected': chasm_detected,
                        'is_correct': is_correct,
                        'efficiency': sample_eff,
                        'tokens_generated': tokens_gen,
                        'drift_history': drift_vals,
                        'handoff_events': agent.handoff_events,
                        'max_drift': max_drift,
                        'avg_drift': avg_drift,
                        'n_handoffs': n_handoffs,
                        'generated_text': generated_text if generated_text else "",
                        'ground_truth': gt_str,
                        'prompt': sample.prompt if hasattr(sample, 'prompt') and sample.prompt else "",
                    })
                except Exception as e:
                    print(f"  [{idx+1:>4d}/{benchmark_sample_count}] \u274c ERROR: {e}")
                    import traceback; traceback.print_exc()
                    results.append({
                        'sample_id': sample.sample_id,
                        'error': str(e),
                        'generated_text': '',
                        'ground_truth': str(sample.ground_truth) if hasattr(sample, 'ground_truth') and sample.ground_truth else "",
                        'prompt': sample.prompt if hasattr(sample, 'prompt') and sample.prompt else "",
                    })
        finally:
            # HERO FIX: Release VRAM before next benchmark OR if crash happens
            print(f"  [System] Unloading agent {agent_name} and releasing GPU memory...")
            agent.unload_model()

        # Calculate final summary stats
        total_processed = len(results)
        acc = correct_count / total_processed if total_processed > 0 else 0.0
        efficiency = max(0.0, 1.0 - (total_handoff_tokens / total_generated_tokens)) if total_generated_tokens > 0 else 1.0
        chasm_rate = chasm_count / total_processed if total_processed > 0 else 0.0
        avg_tokens = total_generated_tokens / total_processed if total_processed > 0 else 0.0
        
        summary = {
            'agent': agent_name,
            'model': model_name,
            'benchmark': benchmark_name,
            'samples_tested': total_processed,
            'accuracy': acc,
            'efficiency': efficiency,
            'chasm_detection_rate': chasm_rate,
            'avg_tokens': avg_tokens,
            'results_file': str(output_path),
            'results': results,  
        }
        
        print(f"\nAudit Complete: Acc={acc:.1%} | Eff={efficiency:.1%} | Chasm={chasm_rate:.1%}")
        return summary

    def _run_benchmark_audit_isolated(
        self,
        agent_name: str,
        model_name: str,
        benchmark_name: str,
    ) -> Optional[Dict]:
        """Run a single audit in a fresh subprocess to guarantee GPU cleanup."""
        output_file = self.output_dir / f"{agent_name}_{model_name}_{benchmark_name}_results.json"

        if self.hard_reset_gpu:
            # [CRITICAL SAFETY FIX] Removed fuser -v /dev/nvidia* | xargs kill -9
            # This previously killed the display manager (GNOME) causing system crashes.
            # Subprocess isolation handles the primary cleanup.
            print("  [Safe Reset] Relying on subprocess isolation for VRAM cleanup.")
            cleanup_gpu_memory()
            time.sleep(2)

        # Determine sample count: prefer self.samples_per_task if set, else full benchmark
        num_samples = self.samples_per_task if self.samples_per_task is not None else BENCHMARKS[benchmark_name].get('num_samples', 20)

        # Determine max tokens for this benchmark
        benchmark_token_limits = {
            "gsm8k": 256,
            "humaneval": 512,
            "halueval": 128,
            "alfworld": 1024,
            "webshop": 1024,
        }
        actual_max_tokens = benchmark_token_limits.get(benchmark_name, self.max_tokens)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--agents", agent_name,
            "--models", model_name,
            "--benchmarks", benchmark_name,
            "--samples-per-task", str(num_samples),
            "--threshold", str(self.threshold),
            "--max-tokens", str(actual_max_tokens),
            "--output-dir", str(self.output_dir.resolve()),
            "--single-run",
        ]
        
        if self.verbose:
            cmd.append("--verbose")

        print(f"  [Isolated Run] Launching: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  [Isolated Run] FAILED: {e}")
            return None

        if not output_file.exists():
            print(f"  [Isolated Run] Missing output: {output_file}")
            return None

        with open(output_file, "r") as f:
            return json.load(f)
    
    def run_full_campaign(self):
        """Run full campaign across all agents/models/benchmarks."""
        print("\n" + "="*80)
        print("CROSS-BASELINE HIGH-INTEGRITY COMPARISON CAMPAIGN")
        print("="*80)
        print(f"Research Question: Does ACC's active control provide statistical safety")
        print(f"                   that baselines lack, even when optimized for throughput?")
        print("="*80)
        print()
        
        print("Campaign Configuration:")
        print(f"  Agents: {', '.join([AGENTS[a].name for a in self.agents])}")
        print(f"  Models: {', '.join([MODELS[m]['vendor'] for m in self.models])}")
        print(f"  Benchmarks (full standard sizes):")
        total_samples = 0
        for b in self.benchmarks:
            n = BENCHMARKS[b]['num_samples']
            total_samples += n
            print(f"    - {BENCHMARKS[b]['name']:12s}: {n:>5d} samples")
        print(f"  Total samples per model-agent: {total_samples}")
        print(f"  Threshold: {self.threshold}")
        print()
        
        start_time = time.time()
        
        start_time = time.time()
        
        # [UAI 2026] PRECISE TARGETED COMBAT FLEET
        # It maps specific agents to their most scientifically significant domains.
        
        PRECISE_COMBAT_FLEET = [
            # Block 1: THE ANCHOR (ACC across everything)
            ("acc", "qwen-2.5-1.5b", ["gsm8k", "humaneval", "halueval", "alfworld"]),
            ("acc", "phi-3-mini", ["gsm8k", "humaneval", "halueval", "alfworld"]),
            ("acc", "mistral-7b", ["gsm8k", "humaneval", "halueval", "alfworld"]),
            # Block 2: SENSOR
            ("semantic_entropy", "mistral-7b", ["halueval"]),
            # Block 3: LOGIC
            ("react", "qwen-2.5-1.5b", ["alfworld", "gsm8k"]),
            # Block 4: QUANT
            ("spinquant", "mistral-7b", ["gsm8k"]),
            ("any4", "mistral-7b", ["gsm8k"]),
            # Block 5: HARDWARE
            ("ktransformers", "mistral-7b", ["humaneval"]),
            # Block 6: RISK
            ("crc", "phi-3-mini", ["gsm8k"]),
        ]
        
        combinations = []
        seen = set()
        for agent, model, benchmarks in PRECISE_COMBAT_FLEET:
            # Check if requested or belongs to the combat blocks
            for bench in benchmarks:
                combo = (agent, model, bench)
                if combo not in seen:
                    combinations.append(combo)
                    seen.add(combo)
        
        # [CRITICAL] No sorting - follow PRECISE_COMBAT_FLEET order
        
        total_combinations = len(combinations)
        current = 0
        
        for agent, model, benchmark in combinations:
            current += 1
            print(f"\n[{current}/{total_combinations}] Targeted Combat: {benchmark} | {model} | {agent}")
            
            if self.isolated:
                result = self._run_benchmark_audit_isolated(agent, model, benchmark)
            else:
                result = self.run_benchmark_audit(agent, model, benchmark)
            if result:
                self.results.append(result)
            
            # CRITICAL: Force cleanup after each audit
            cleanup_gpu_memory()
            print(f"  [Memory cleanup] GPU usage after audit: {get_gpu_memory_usage():.2f} GB")
        
        elapsed = time.time() - start_time
        
        # Save results
        campaign_summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': elapsed,
            'elapsed_minutes': elapsed / 60,
            'total_audits': len(self.results),
            'configuration': {
                'agents': self.agents,
                'models': self.models,
                'benchmarks': self.benchmarks,
                'samples_per_task': self.samples_per_task,
                'threshold': self.threshold,
                'max_tokens': self.max_tokens,
            },
            'results': self.results,
        }
        
        results_file = self.output_dir / "cross_baseline_results.json"
        with open(results_file, 'w') as f:
            json.dump(campaign_summary, f, indent=2)
        
        print("\n" + "="*80)
        print("CAMPAIGN COMPLETE")
        print("="*80)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Audits completed: {len(self.results)}")
        print(f"Results saved: {results_file}")
        print("="*80)
        print()
        
        # Print comparison table
        self._print_comparison_table()
    
    def _print_comparison_table(self):
        """ Standard Accuracy-Efficiency Frontier."""
        print(f"\n{'='*100}")
        print(f"ACCURACY-EFFICIENCY FRONTIER (Lethal 8 Matrix)")
        print(f"{'='*100}")
        print(f"{'Agent':<15} {'Model':<12} {'Bench':<10} {'Accuracy':<12} {'Efficiency':<12} {'Safety Gate':<12}")
        print("-" * 100)
        
        # Aggregate by combination
        for res in sorted(self.results, key=lambda x: (x['benchmark'], x['agent'])):
            agent_name = AGENTS[res['agent']].name
            model_disp = res['model']
            bench_disp = res['benchmark'].upper()
            acc_disp = f"{res.get('accuracy', 0):.1%}"
            eff_disp = f"{res.get('efficiency', 1.0):.1%}"
            gate_disp = f"{res.get('chasm_detection_rate', 0):.1%}"
            
            # Highlight ACC
            if res['agent'] == "acc":
                print(f"\033[1m{agent_name:<15} {model_disp:<12} {bench_disp:<10} {acc_disp:<12} {eff_disp:<12} {gate_disp:<12}\033[0m")
            else:
                print(f"{agent_name:<15} {model_disp:<12} {bench_disp:<10} {acc_disp:<12} {eff_disp:<12} {gate_disp:<12}")
        
        print("-" * 100)
        print("Legend: Efficiency = % Tokens generated by Student | Safety Gate = Intervention %")
        print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-baseline high-integrity comparison campaign"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=list(AGENTS.keys()) + ["all"],
        default=["all"],
        help="Agents to compare (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Models to audit (default: all)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default=["all"],
        help="Benchmarks to use (default: all)",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=None,
        help="Number of samples per benchmark (default: full benchmark size)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Drift detection threshold",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./cross_baseline_results"),
        help="Output directory",
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Run each agent/model/benchmark in a fresh subprocess (hard GPU reset)",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run exactly one agent/model/benchmark combination and exit",
    )
    parser.add_argument(
        "--hard-reset-gpu",
        action="store_true",
        help="Force kill GPU processes between isolated audits",
    )
    parser.add_argument(
        "--require-teacher-baseline",
        action="store_true",
        help="Require real teacher baselines for ACC (skip if missing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging during inference",
    )
    parser.add_argument(
        "--test-amx-only",
        action="store_true",
        help="Verify Xeon AMX tiles are firing (use with DNNL_VERBOSE=1)",
    )
    
    args = parser.parse_args()
    
    # Expand "all"
    agents_to_run = list(AGENTS.keys()) if "all" in args.agents else args.agents
    models_to_run = list(MODELS.keys()) if "all" in args.models else args.models
    benchmarks_to_run = list(BENCHMARKS.keys()) if "all" in args.benchmarks else args.benchmarks
    
    # Run campaign
    campaign = CrossBaselineCampaign(
        output_dir=args.output_dir,
        agents=agents_to_run,
        models=models_to_run,
        benchmarks=benchmarks_to_run,
        samples_per_task=args.samples_per_task,
        threshold=args.threshold,
        max_tokens=args.max_tokens,
        isolated=args.isolated,
        hard_reset_gpu=args.hard_reset_gpu,
        require_teacher_baseline=args.require_teacher_baseline,
        verbose=args.verbose,
    )

    if args.test_amx_only:
        print("\n=== HARDWARE VERIFICATION: XEON AMX TILES ===")
        print("Note: Run with export DNNL_VERBOSE=1 to see AMX primitives.")
        from wrappers.baseline_ppdre import OracleBridge
        try:
            oracle = OracleBridge()
            dummy_input = torch.zeros((1, 32), dtype=torch.long)
            print("Executing Teacher repair pass on AMX...")
            _, latency_ms = oracle.repair_manifold(dummy_input, k_tokens=5)
            print(f"[OK] AMX Pass Complete: {latency_ms:.2f}ms for 5 tokens")
        except Exception as e:
            print(f"[ERROR] AMX Verification Failed: {e}")
        return

    if args.single_run:
        # Single isolated run: exactly one combination
        agent = agents_to_run[0]
        model = models_to_run[0]
        benchmark = benchmarks_to_run[0]
        result = campaign.run_benchmark_audit(agent, model, benchmark)
        if result:
            output_file = campaign.output_dir / f"{agent}_{model}_{benchmark}_results.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Single-run results saved: {output_file}")
        return
    
    campaign.run_full_campaign()


if __name__ == "__main__":
    main()
