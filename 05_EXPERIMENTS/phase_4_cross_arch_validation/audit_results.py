#!/usr/bin/env python3
"""Audit all campaign result JSON files and print a summary table."""
import json
import glob
import os
from pathlib import Path

results_dir = Path('/home/cse-sdpl/research/ACC/05_EXPERIMENTS/phase_4_cross_arch_validation/campaign_results_20260218_222001')

rows = []
for f in sorted(results_dir.glob('*.json')):
    name = f.stem  # e.g. acc_llama-3.2-1b_gsm8k_results
    # Strip trailing _results
    name = name.replace('_results', '')
    # Split on first two underscores to get agent, model, benchmark
    # But model names have hyphens, not underscores, so split on _ gives:
    # ['acc', 'llama-3.2-1b', 'gsm8k']
    parts = name.split('_', 2)
    if len(parts) < 3:
        continue
    agent = parts[0]
    model = parts[1]
    bench = parts[2]

    try:
        with open(f) as fh:
            d = json.load(fh)
        samples = d.get('samples_tested', 0)
        detect_count = d.get('chasm_detection_count', 0)
        detect_rate = d.get('chasm_detection_rate', 0.0)
        total_handoffs = sum(len(r.get('handoff_events', [])) for r in d.get('results', []))
        rows.append((agent, model, bench, samples, detect_count, round(detect_rate * 100, 1), total_handoffs))
    except Exception as e:
        rows.append((agent, model, bench, 'ERR', 'ERR', str(e), 0))

header = ('Agent', 'Model', 'Benchmark', 'Samples', 'Detected', 'Det %', 'Handoffs')
print(f"{header[0]:<8} {header[1]:<15} {header[2]:<14} {header[3]:>8} {header[4]:>9} {header[5]:>7} {header[6]:>9}")
print('-' * 75)
prev_agent = ''
for r in rows:
    agent_str = r[0] if r[0] != prev_agent else ''
    prev_agent = r[0]
    print(f"{agent_str:<8} {r[1]:<15} {r[2]:<14} {str(r[3]):>8} {str(r[4]):>9} {str(r[5]):>7} {str(r[6]):>9}")

print()
print(f"Total result files: {len(rows)}")

# Summary by agent
print()
print("=== SUMMARY BY AGENT ===")
by_agent = {}
for r in rows:
    agent = r[0]
    if agent not in by_agent:
        by_agent[agent] = {'samples': 0, 'detected': 0, 'handoffs': 0, 'runs': 0}
    if isinstance(r[3], int):
        by_agent[agent]['samples'] += r[3]
        by_agent[agent]['detected'] += r[4] if isinstance(r[4], int) else 0
        by_agent[agent]['handoffs'] += r[6]
        by_agent[agent]['runs'] += 1

print(f"{'Agent':<10} {'Runs':>5} {'Total Samples':>14} {'Total Detected':>15} {'Avg Det %':>10} {'Total Handoffs':>15}")
print('-' * 70)
for agent, stats in sorted(by_agent.items()):
    avg_det = (stats['detected'] / stats['samples'] * 100) if stats['samples'] > 0 else 0
    print(f"{agent:<10} {stats['runs']:>5} {stats['samples']:>14} {stats['detected']:>15} {avg_det:>9.1f}% {stats['handoffs']:>15}")
