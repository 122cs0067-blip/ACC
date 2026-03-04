import json
from pathlib import Path

def run_simulation(captured_data):
    thresholds = [0.001, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    frontier = []
    for tau in thresholds:
        correct_count = 0
        student_tokens = 0
        total_tokens = 0
        for d in captured_data:
            drift = d["drift_history"]
            handoff_idx = -1
            for idx, val in enumerate(drift):
                if val > tau:
                    handoff_idx = idx
                    break
            if handoff_idx == -1:
                correct_count += 1 if d["student_correct"] else 0
                student_tokens += d["tokens"]
            else:
                correct_count += 1 if d["teacher_correct"] else 0
                student_tokens += handoff_idx
            total_tokens += d["tokens"]
            
        acc = correct_count / len(captured_data) if captured_data else 0
        eff = student_tokens / total_tokens if total_tokens > 0 else 0
        
        frontier.append({
            "threshold": tau,
            "accuracy": acc,
            "efficiency": eff,
            "handoff_rate": sum(1 for d in captured_data if any(v > tau for v in d["drift_history"])) / len(captured_data) if captured_data else 0
        })
    return frontier

data_path = "/home/cse-sdpl/research/ACC/08_SWEET_SPOT_ANALYSIS/phi4_rerun_capture.json"
with open(data_path, "r") as f:
    data = json.load(f)

frontier = run_simulation(data)

print(f"\n[RESULTS] Phi-4 Rerun | Sweet Spot Candidates:")
print("| Tau   | Acc   | Eff   | H/O % |")
for f in frontier:
    print(f"| {f['threshold']:.3f} | {f['accuracy']:.3f} | {f['efficiency']:.3f} | {f['handoff_rate']:.1%} |")
