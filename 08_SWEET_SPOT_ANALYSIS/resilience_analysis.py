#!/usr/bin/env python3
import json
import numpy as np
import os
from pathlib import Path

def analyze_resilience():
    """
    campaign: Multimodal Manifold Resilience Analysis.
    Compares the latent drift profile of AWQ-quantized Phi-4
    vs NF4-quantized LLaVA-v1.6-7B.
    """
    phi4_raw = "08_SWEET_SPOT_ANALYSIS/raw_capture_phi4.json"
    llava_raw = "08_SWEET_SPOT_ANALYSIS/raw_capture_llava.json"
    
    results = {}
    
    for name, path in [("Phi-4 (AWQ)", phi4_raw), ("LLaVA (NF4)", llava_raw)]:
        if not os.path.exists(path):
            print(f"Waiting for {path}...")
            continue
            
        with open(path, "r") as f:
            data = json.load(f)
            
        all_drifts = []
        for sample in data:
            if "drifts" in sample:
                all_drifts.extend(sample["drifts"])
        
        if not all_drifts:
            continue
            
        mu = np.mean(all_drifts)
        std = np.std(all_drifts)
        p95 = np.percentile(all_drifts, 95)
        
        results[name] = {
            "mean_drift": mu,
            "std_drift": std,
            "p95_drift": p95,
            "resilience": 1.0 / mu if mu > 0 else 0
        }
    
    if len(results) == 2:
        names = list(results.keys())
        r0 = results[names[0]]
        r1 = results[names[1]]
        
        ratio = r0["resilience"] / r1["resilience"]
        print(f"\n" + "="*60)
        print(f"campaign: MANIFOLD RESILIENCE ANALYSIS")
        print("="*60)
        print(f"{names[0]}: Mean Drift={r0['mean_drift']:.6f} | Resilience={r0['resilience']:.2f}")
        print(f"{names[1]}: Mean Drift={r1['mean_drift']:.6f} | Resilience={r1['resilience']:.2f}")
        print("-" * 60)
        print(f"Theoretical Finding: {names[0]} is {ratio:.2f}x more manifold-resilient.")
        print("="*60 + "\n")
    else:
        print("Waiting for both captures to complete (Phi-4 and LLaVA).")

if __name__ == "__main__":
    analyze_resilience()
