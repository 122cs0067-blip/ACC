#!/usr/bin/env python3
import os
import sys
import torch
from pathlib import Path

# Add project root to sys.path
ROOT = Path("/home/cse-sdpl/research/ACC")
sys.path.append(str(ROOT / "02_SRC"))
sys.path.append(str(ROOT / "05_EXPERIMENTS" / "phase_4_cross_arch_validation"))

from wrappers.wrapper_qwen25vl_3b import QwenVLStudentAgent
from benchmark_loaders_vlm import load_vlm_benchmark

def test_logging():
    print("Verifying Granular Performance Logging Architecture...")
    
    # 1. Load 1 sample from POPE
    samples = load_vlm_benchmark("pope", Path("/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/pope"), num_samples=1)
    sample = samples[0]

    # 2. Initialize Agent with low threshold to force handoff
    agent = QwenVLStudentAgent(use_teacher=True, lambda_star=0.001)
    
    # 3. Run Inference
    print(f"Running inference on Sample {sample.sample_id}...")
    res = agent.run_text(sample.question, image=sample.image, max_new_tokens=20, task_id=sample.sample_id)
    
    # 4. Verify results
    print("\n--- Granular Metrics Captured ---")
    print(f"Handoff Index:   {res.get('handoff_idx')}")
    print(f"VLM Latency:     {res.get('vlm_ms')} ms")
    print(f"ACC Overhead:    {res.get('acc_ms')} ms")
    print(f"Quant State:     {res.get('quant_state')}")
    print(f"Efficiency:      {res.get('efficiency')}")
    
    # 5. Verify Logger Files
    log_dir = agent.logger.log_dir
    print(f"\nVerifying Log Directory: {log_dir}")
    
    traj_file = log_dir / "trajectory_data.csv"
    handoff_file = log_dir / "handoffs.log"
    
    if traj_file.exists():
        print(f"  [OK] {traj_file.name} created.")
        with open(traj_file, 'r') as f:
            lines = f.readlines()
            print(f"  [OK] Trajectory Header: {lines[0].strip()}")
            if len(lines) > 1:
                print(f"  [OK] First Data Line:   {lines[1].strip()}")
    else:
        print(f"  [ERROR] {traj_file.name} MISSING.")

    if handoff_file.exists():
        print(f"  [OK] {handoff_file.name} created.")
        with open(handoff_file, 'r') as f:
            content = f.read()
            print(f"  [OK] Handoff Log Content:\n{content}")
    else:
        print(f"  [ERROR] {handoff_file.name} MISSING.")

if __name__ == "__main__":
    test_logging()
