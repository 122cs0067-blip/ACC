#!/usr/bin/env python3
"""
ACC Campaign Progress Tracker
Monitors the results directory and provides real-time completion statistics.
"""

import os
import json
import time
from pathlib import Path

# Load config to get target sample counts
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.append(str(PROJECT_ROOT / "02_SRC"))
from acc_core.system.deployment_config import DATASET_SPLITS, ACC_VLM_FLEET

RESULTS_DIR = PROJECT_ROOT / "06_RESULTS/final_campaign/phase_1"

def track_progress():
    models = list(ACC_VLM_FLEET.keys())
    benchmarks = list(DATASET_SPLITS.keys())
    
    total_target = len(models) * sum(DATASET_SPLITS[b]["num_samples"] for b in benchmarks)
    
    print("\n" + "="*60)
    print("ACC FINAL CAMPAIGN: PHASE 1 PROGRESS TRACKER")
    print("="*60)
    
    while True:
        completed_total = 0
        status_table = []
        
        for m_name in models:
            m_comp = 0
            m_target = sum(DATASET_SPLITS[b]["num_samples"] for b in benchmarks)
            
            for b_name in benchmarks:
                res_file = RESULTS_DIR / f"acc_results_{m_name}_{b_name}.jsonl"
                count = 0
                if res_file.exists():
                    with open(res_file, "r") as f:
                        count = sum(1 for _ in f)
                
                m_comp += count
                completed_total += count
            
            status_table.append(f"  - {m_name.ljust(20)}: {m_comp}/{m_target} ({ (m_comp/m_target)*100:.1f}%)")

        os.system('clear')
        print("="*60)
        print(f"ACC CAMPAIGN PROGRESS | {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print(f"Overall Completion: {completed_total}/{total_target} ({(completed_total/total_target)*100:.2f}%)")
        print("-" * 60)
        for line in status_table:
            print(line)
        print("-" * 60)
        print("Press Ctrl+C to stop tracking.")
        
        if completed_total >= total_target:
            print("\n[OK] CAMPAIGN PHASE 1 COMPLETE!")
            break
            
        time.sleep(10)

if __name__ == "__main__":
    try:
        track_progress()
    except KeyboardInterrupt:
        print("\nTracker stopped.")
