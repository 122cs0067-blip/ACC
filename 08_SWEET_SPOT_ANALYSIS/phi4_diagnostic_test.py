#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
import random

# Add Project Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "02_SRC"))
sys.path.insert(0, str(ROOT_DIR / "05_EXPERIMENTS" / "phase_4_cross_arch_validation"))

from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent
from benchmark_loaders_vlm import load_vlm_benchmark

def main():
    print("--- PHI-4 MULTIMODAL DIAGNOSTIC TEST (N=5) ---")
    
    # Configuration
    BENCHMARK_PATH = ROOT_DIR / "01_DATA" / "Benchmarks" / "mathvista"
    
    print("Loading 5 mathvista samples...")
    samples = load_vlm_benchmark("mathvista", BENCHMARK_PATH, num_samples=5)
    
    print("Initializing Student Agent...")
    agent = Phi4MultimodalStudentAgent(use_teacher=True) # Forces teacher load
    
    print("\nStarting Inference Loop...")
    for i, s in enumerate(samples):
        print(f"\n[Sample {i+1}/5] ID: {s.sample_id}")
        try:
            # Student Run
            res = agent.run_text(s.question, image=s.image, max_new_tokens=30)
            print(f"  Student output: {res['generated_text'][:50]}...")
            print(f"  Drift scores: {res['drift_scores'][:5]}")
            
            # Direct Teacher call (via agent.oracle)
            if agent.oracle:
                t_text = agent.oracle.generate(s.question, image=s.image, max_new_tokens=30)
                print(f"  Teacher output: {t_text[:50]}...")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Diagnostic Test Complete ---")

if __name__ == "__main__":
    main()
