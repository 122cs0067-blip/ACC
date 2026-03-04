import os
import json
import numpy as np

results_dir = "/home/cse-sdpl/research/ACC/05_EXPERIMENTS/phase_4_cross_arch_validation/campaign_results_20260218_164327"
all_drift_scores = []

for filename in os.listdir(results_dir):
    if filename.endswith(".json") and "results" in filename:
        with open(os.path.join(results_dir, filename), "r") as f:
            data = json.load(f)
            for res in data.get("results", []):
                for event in res.get("handoff_events", []):
                    all_drift_scores.append(event["drift_score"])

if not all_drift_scores:
    print("No drift scores found.")
else:
    all_drift_scores = np.sort(all_drift_scores)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"Total Handoff Events: {len(all_drift_scores)}")
    print(f"Min Drift: {all_drift_scores[0]:.6f}")
    for p in percentiles:
        print(f"{p}th Percentile: {np.percentile(all_drift_scores, p):.6f}")
    print(f"Max Drift: {all_drift_scores[-1]:.6f}")
