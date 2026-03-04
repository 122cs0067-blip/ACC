import json
import glob
import os
import numpy as np

def analyze_jsonl(filepath):
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass
    
    if not results:
        return None
    
    count = len(results)
    handoffs = [1 if r.get('acc_active', False) or r.get('interventions', 0) > 0 else 0 for r in results]
    handoff_rate = np.mean(handoffs) * 100
    
    latencies = [r.get('latency_sec', 0) for r in results]
    avg_latency = np.mean(latencies)
    
    # Simple accuracy check
    correct = 0
    valid_count = 0
    for r in results:
        gt = str(r.get('ground_truth', '')).strip().lower()
        pred = str(r.get('prediction', '')).strip().lower()
        if gt:
            valid_count += 1
            if gt in pred or pred in gt: # Simple check for accuracy
                correct += 1
    
    accuracy = (correct / valid_count * 100) if valid_count > 0 else 0
    
    return {
        "benchmark": os.path.basename(filepath),
        "samples": count,
        "handoff_rate": f"{handoff_rate:.2f}%",
        "avg_latency": f"{avg_latency:.2f}s",
        "accuracy": f"{accuracy:.2f}%"
    }

def main():
    files = glob.glob("/home/cse-sdpl/research/ACC/06_RESULTS/final_campaign_v3/phase_1/*.jsonl")
    summary = []
    for f in sorted(files):
        analysis = analyze_jsonl(f)
        if analysis:
            summary.append(analysis)
    
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
