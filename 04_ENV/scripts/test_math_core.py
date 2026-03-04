import sys
import os
# Ensure we can import from SRC
sys.path.append(os.path.abspath('/home/cse-sdpl/research/ACC/02_SRC'))

import torch
import numpy as np
from acc_core.detector.ipp_dre import IncrementalDriftTracker

def run_synthetic_test():
    print(" Starting Synthetic Drift Test (i-ppDRE)...")
    
    # 1. Initialize Tracker
    # input_dim=128 (matches our compressed Llama vector)
    tracker = IncrementalDriftTracker(input_dim=128, rff_dim=256)
    
    # 2. Simulate Teacher (Standard Normal Distribution)
    print("   ...Calibrating on 'Teacher' Data (Normal Distribution)")
    teacher_data = torch.randn(100, 128) # Mean=0, Std=1
    tracker.update_teacher_baseline(teacher_data)
    
    scores = []
    
    # 3. Run Simulation (Steps 0-200)
    print("   ...Running Simulation")
    for t in range(200):
        # Phase A: Normal Operation (Steps 0-100) -> Student matches Teacher
        if t < 100:
            z_t = torch.randn(128) 
        # Phase B: DRIFT! (Steps 100-200) -> Student Mean Shifts by +1.5
        else:
            z_t = torch.randn(128) + 1.5 
            
        # A. Calculate Score BEFORE update (Test)
        w_x = tracker.score(z_t)
        
        # B. Update Tracker (Train)
        tracker.update(z_t) 
        
        scores.append(w_x)
        
        if t % 20 == 0:
            status = "Normal" if t < 100 else "DRIFTING"
            print(f"   Step {t:03d} [{status}]: Drift Score w(x) = {w_x:.4f}")

    # 4. Analysis
    avg_normal = np.mean(scores[:100])
    avg_drift = np.mean(scores[120:]) # Give it 20 steps to adapt
    
    print("-" * 50)
    print(f" Normal Baseline Avg: {avg_normal:.4f}")
    print(f"  Drift Regime Avg:   {avg_drift:.4f}")
    print(f"   Ratio (Drift/Normal): {avg_drift / avg_normal:.2f}x")
    
    # 5. Success Criteria
    # The score should inherently rise because the "Student" distribution 
    # is deviating from the "Teacher" baseline.
    if avg_drift > avg_normal * 1.2:
        print(" SUCCESS: Drift detected significantly! Math engine is valid.")
        return True
    else:
        print(" FAILURE: Sensitivity too low. Check Learning Rate or RFF dim.")
        return False

if __name__ == "__main__":
    success = run_synthetic_test()
    sys.exit(0 if success else 1)
