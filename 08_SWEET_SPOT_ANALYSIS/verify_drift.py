#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import torch

# Standard ACC Environment
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "02_SRC"))
sys.path.insert(0, str(ROOT_DIR / "05_EXPERIMENTS" / "phase_4_cross_arch_validation"))

from benchmark_loaders import load_benchmark
from cross_baseline_campaign import ACCAgent, AGENTS, MODELS, BENCHMARKS

def verify_phi3():
    model_key = "phi-3-mini"
    print(f"--- Verifying normalized drift for {model_key} ---")
    
    # Load 5 samples from GSM8K
    cfg = BENCHMARKS["gsm8k"]
    samples = load_benchmark("gsm8k", cfg['data_dir'], num_samples=5, seed=42)
    
    # Initialize ACC Agent with corrected monitor
    agent = ACCAgent(AGENTS["acc"], MODELS[model_key]["id"], threshold=1.0)
    agent.load_model()
    
    # Force high threshold to see natural drift
    agent.controller.beta = 1.0
    agent.controller.k = 0.0
    
    for i, s in enumerate(samples):
        print(f"\nSample {i+1}: {s.sample_id}")
        # run_inference will print "Drift Audit" scores due to my recent edit
        text, _ = agent.run_inference(s.prompt, max_tokens=20, verbose=False)
        print(f"  Drift counts: {len(agent.drift_history)}")
        print(f"  Avg Drift: {sum(agent.drift_history)/len(agent.drift_history) if agent.drift_history else 0:.6f}")

    agent.unload_model()

if __name__ == "__main__":
    verify_phi3()
