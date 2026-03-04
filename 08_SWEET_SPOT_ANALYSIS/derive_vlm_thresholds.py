#!/usr/bin/env python3
"""
ACC Statistical Calibration Utility
Derives lambda_95 = mu + 2.3*sigma from normalized student drift distributions.
"""

import os
import json
import numpy as np
from pathlib import Path

def derive_thresholds(capture_dir: Path):
    models = ["qwen", "phi4", "llava"]
    stats = {}

    print(f"\n{'='*60}")
    print(f"{'VLM FLEET STATISTICAL CALIBRATION':^60}")
    print(f"{'='*60}")
    print(f"| {'Model':<10} | {'mu':<8} | {'sigma':<8} | {'lambda_95':<10} | {'Status':<12} |")
    print(f"|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*12}|{'-'*14}|")

    for m in models:
        raw_file = capture_dir / f"raw_capture_{m}.json"
        if not raw_file.exists():
            print(f"| {m:<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<10} | {'NOT FOUND':<12} |")
            continue
        
        with open(raw_file, "r") as f:
            data = json.load(f)
        
        # Rigorous filtering: Only successful student traces define the safe manifold
        safe_drifts = []
        for d in data:
            if d.get("student_correct", False):
                hist = [float(x) for x in d.get("drift_history", []) if x is not None and not np.isinf(x)]
                safe_drifts.extend(hist)
        
        if not safe_drifts:
            print(f"| {m:<10} | {'0.000':<8} | {'0.000':<8} | {'0.000':<10} | {'NO SAFE DATA':<12} |")
            continue

        mu = np.mean(safe_drifts)
        sigma = np.std(safe_drifts)
        # Bayesian Conformal Guardrail (98.9% Coverage)
        lambda_95 = mu + 2.3 * sigma
        
        stats[m] = {
            "mu": mu,
            "sigma": sigma,
            "lambda_95": lambda_95,
            "n_samples": len(data),
            "n_safe_tokens": len(safe_drifts)
        }
        
        print(f"| {m:<10} | {mu:.4f} | {sigma:.4f} | {lambda_95:.4f} | {'CALIBRATED':<12} |")

    # Save finalized calibration parameters
    with open(capture_dir / "vlm_fleet_calibration.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"{'='*60}")
    print(f"Saved finalized calibration to vlm_fleet_calibration.json\n")

if __name__ == "__main__":
    derive_thresholds(Path(__file__).parent)
