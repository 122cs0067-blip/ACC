#!/usr/bin/env python3
"""
Generate all 3 scientific figures for ACC UAI 2026 paper.
Run from: /home/cse-sdpl/research/ACC/06_PAPER/
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# RESULTS_DIR = '../05_EXPERIMENTS/results'
OUT_DIR = './'

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

ACC_BLUE   = '#1f6feb'
SAFE_GREEN = '#2ea043'
CHASM_RED  = '#cf222e'
GRAY       = '#6e7781'
ORANGE     = '#f0883e'
PURPLE     = '#8250df'
TEAL       = '#1abc9c'


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Density Chasm: drift score trace across token steps
# ══════════════════════════════════════════════════════════════════════════════
def figure1_density_chasm():
    print("Generating Figure 1: Density Chasm...")

    # Load Mistral GSM8K results — best model to show clear drift events
    path = os.path.join(RESULTS_DIR, 'acc_mistral-7b_gsm8k_results.json')
    with open(path) as f:
        data = json.load(f)
    results = data['results']

    # Pick 3 interesting traces:
    # 1. A chasm-detected WRONG answer (false-negative style)
    # 2. A chasm-detected CORRECT answer (successful recovery)
    # 3. A SAFE (no chasm) correct answer (baseline)
    chasm_wrong  = [r for r in results if r['chasm_detected'] and not r['is_correct'] and len(r.get('drift_history', [])) > 15]
    chasm_right  = [r for r in results if r['chasm_detected'] and r['is_correct']     and len(r.get('drift_history', [])) > 15]
    safe_right   = [r for r in results if not r['chasm_detected'] and r['is_correct'] and len(r.get('drift_history', [])) > 15]

    if not chasm_wrong or not chasm_right or not safe_right:
        print("  Warning: insufficient traces, using available data")

    samples = {
        'Chasm → Wrong\n(no recovery)':  (chasm_wrong[0]  if chasm_wrong  else results[0],  CHASM_RED,  '--'),
        'Chasm → Correct\n(recovered)':  (chasm_right[3]  if len(chasm_right)>3 else results[1], SAFE_GREEN, '-'),
        'Safe → Correct\n(no drift)':    (safe_right[0]   if safe_right   else results[2],  GRAY,       ':'),
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for label, (sample, color, ls) in samples.items():
        drift = sample.get('drift_history', [])
        if not drift:
            continue
        xs = range(len(drift))
        ax.plot(xs, drift, color=color, linestyle=ls, linewidth=1.6, label=label, alpha=0.85)

        # Mark handoff events
        for ev in sample.get('handoff_events', []):
            step = ev.get('step', ev.get('token_idx', 0))
            if step < len(drift):
                ax.axvline(step, color=color, alpha=0.35, linewidth=0.8)

    # Threshold line — Mistral lambda* = 0.002
    lambda_star = 0.002
    ax.axhline(lambda_star, color=ORANGE, linewidth=1.2, linestyle='-.', alpha=0.9,
               label=f'$\\lambda^*$ = {lambda_star} (Mistral)')
    ax.fill_between(range(80), lambda_star, ax.get_ylim()[1] if ax.get_ylim()[1] > lambda_star else lambda_star*5,
                    alpha=0.05, color=CHASM_RED)

    ax.set_xlabel('Token Generation Step')
    ax.set_ylabel('Drift Score $w(\\mathbf{x}_t)$')
    # ax.set_title('Figure 1: Density Chasm — Token-Level Drift Traces\n'
    #              '(Mistral-7B × GSM8K, selected samples)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0, 60)
    ymax = max(lambda_star * 6, 0.015)
    ax.set_ylim(-0.0002, ymax)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig1_density_chasm.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Efficiency Frontier: Accuracy vs Intervention Rate
# ══════════════════════════════════════════════════════════════════════════════
def figure2_efficiency_frontier():
    print("Generating Figure 2: Efficiency Frontier...")

    # [agent, model, benchmark, acc%, eff%, dcdr%, ir%, marker, color, size]
    # Intervention Rate = 1 - EFF%  (fraction of tokens involving teacher)
    points = [
        # ACC family
        ('ACC\nQwen GSM8K',        24.5, 78.0, 'o', ACC_BLUE,   100),
        ('ACC\nPhi GSM8K',         31.2, 84.8, 'o', ACC_BLUE,   100),
        ('ACC\nMistral GSM8K',     25.0, 76.7, 'o', ACC_BLUE,   100),
        ('ACC\nMistral HaluEval',  39.9, 83.1, 'o', ACC_BLUE,   100),
        ('ACC\nPhi HaluEval',      30.0, 99.6, 'o', ACC_BLUE,   100),
        ('ACC\nQwen HaluEval',     26.4, 93.6, 'o', ACC_BLUE,   100),
        # Baselines
        ('CRC\nPhi GSM8K',         35.5, 100.0, 's', CHASM_RED,  80),
        ('Any4\nMistral GSM8K',    34.0, 100.0, 's', ORANGE,     80),
        ('SpinQuant\nMistral GSM8K',33.7, 100.0, 'D', PURPLE,    80),
        ('Sem. Entropy\nMistral Halu',37.2, 100.0,'^', TEAL,     80),
        ('ReAct\nQwen GSM8K',       8.9, 100.0, 'v', GRAY,       80),
        ('KTransformers\nMistral HE',1.2, 100.0, 'P', '#e67e22', 80),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    acc_plotted = False
    base_plotted = False
    for label, accuracy, efficiency, marker, color, size in points:
        ir = 100.0 - efficiency  # intervention rate
        is_acc = (color == ACC_BLUE)
        lbl = ('ACC (Ours)' if is_acc and not acc_plotted
               else ('Baselines' if not is_acc and not base_plotted
               else None))
        if is_acc: acc_plotted = True
        if not is_acc: base_plotted = True

        ax.scatter(ir, accuracy, c=color, marker=marker, s=size,
                   zorder=5, alpha=0.88, label=lbl,
                   edgecolors='white', linewidths=0.5)

        # Annotate each point
        short = label.split('\n')[0]
        ax.annotate(short, (ir, accuracy),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7, color=color, alpha=0.9)

    # Ideal quadrant annotation
    # Shaded region for Ideal Zone
    ideal_rect = mpatches.Rectangle((0, 25), 10, 21, color='#d4edda', alpha=0.3, zorder=1)
    ax.add_patch(ideal_rect)
    
    ax.annotate('Ideal Zone\n(High Acc, Low IR)',
                xy=(2, 38), fontsize=8, color='#155724', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4edda', alpha=0.6))
    ax.axhline(25, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.axvline(10, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)

    ax.set_xlabel('Intervention Rate (%) [= 100 − EFF%]', labelpad=6)
    ax.set_ylabel('Task Accuracy (%)', labelpad=6)
    # ax.set_title('Figure 2: Efficiency Frontier\n'
    #              'Accuracy vs. Teacher Intervention Rate')
    ax.set_xlim(-2, 30)
    ax.set_ylim(-2, 46)

    legend_els = [
        mpatches.Patch(facecolor=ACC_BLUE,   label='ACC (Ours)'),
        mpatches.Patch(facecolor=CHASM_RED,  label='CRC'),
        mpatches.Patch(facecolor=ORANGE,     label='Any4'),
        mpatches.Patch(facecolor=PURPLE,     label='SpinQuant'),
        mpatches.Patch(facecolor=TEAL,       label='Sem. Entropy'),
        mpatches.Patch(facecolor=GRAY,       label='ReAct'),
        mpatches.Patch(facecolor='#e67e22',  label='KTransformers'),
    ]
    ax.legend(handles=legend_els, loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig2_efficiency_frontier.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Latency Breakdown (Hardware Cascade)
# ══════════════════════════════════════════════════════════════════════════════
def figure3_latency_breakdown():
    print("Generating Figure 3: Latency Breakdown...")

    # From measured hardware data (Table 4 in paper + oracle_bridge timing)
    phases = ['ppDRE\nSensor', 'PCIe\nTransfer', 'AMX Teacher\n(per token)', 'Total\nHandoff (4 tok)']
    means  = [0.39,  0.033, 344.0,  1376.0]
    mins_  = [0.319, 0.030, 238.0,   952.0]
    maxs_  = [0.483, 0.040, 356.0,  1424.0]
    colors = [SAFE_GREEN, TEAL, CHASM_RED, '#8c1c28']
    yerr_lo = [m - mn for m, mn in zip(means, mins_)]
    yerr_hi = [mx - m  for m, mx in zip(means, maxs_)]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    bars = ax.bar(phases, means, color=colors, alpha=0.85,
                  width=0.5, zorder=3,
                  yerr=[yerr_lo, yerr_hi],
                  error_kw=dict(elinewidth=1.2, capsize=4, ecolor='#444'))

    # Value labels on bars
    for bar, val in zip(bars, means):
        label = f'{val:.3f} ms' if val < 1 else f'{val:.0f} ms'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.03,
                label, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Budget line
    ax.axhline(92, color=ORANGE, linewidth=1.4, linestyle='--', alpha=0.9)
    ax.text(3.35, 95, '92 ms\nBudget', color=ORANGE, fontsize=8, ha='right')

    ax.set_ylabel('Latency (ms, log scale)')
    # ax.set_title('Figure 3: Hardware Cascade Latency Breakdown\n'
    #              '(RTX 2000 Ada GPU + Xeon W5 + Intel AMX)')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 8000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f'{x:.3f}' if x < 1 else f'{x:.0f}'))

    # Annotation: PCIe is << budget
    ax.annotate('PCIe: 0.033 ms\n(350× below 92 ms budget)',
                xy=(1, 0.033), xytext=(1.6, 0.15),
                fontsize=8, color=TEAL,
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.0))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig3_latency_breakdown.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    figure1_density_chasm()
    figure2_efficiency_frontier()
    figure3_latency_breakdown()
    print("\n[OK]  All 3 figures generated successfully.")
    print(f"   Output directory: {OUT_DIR}")
    for f in ['fig1_density_chasm', 'fig2_efficiency_frontier', 'fig3_latency_breakdown']:
        for ext in ['pdf', 'png']:
            path = os.path.join(OUT_DIR, f'{f}.{ext}')
            if os.path.exists(path):
                sz = os.path.getsize(path)
                print(f"   {f}.{ext}: {sz/1024:.0f} KB")
