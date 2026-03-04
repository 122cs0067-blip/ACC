#!/usr/bin/env python3
"""
UAI 2026 Data Analysis & Visualization Suite

Generates publication-quality figures and tables from campaign results:
1. Table 1: Density Chasm Detection Rates (baseline comparison)
2. Figure 1: Drift Trajectory Plots (w(x) vs. token index)
3. Figure 2: Intervention Precision Analysis
4. Figure 3: Compute-Normalized Quality Metrics

Usage:
    python analyze_campaign_results.py --results-dir uai_2026_results
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
sns.set_palette("colorblind")


class CampaignAnalyzer:
    """Analyzes campaign results and generates UAI 2026 figures."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "cross_baseline_results.json"
        self.drift_logs_dir = self.results_dir / "drift_logs"
        
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file) as f:
            self.data = json.load(f)
        
        self.results = self.data.get('results', [])
        print(f"Loaded {len(self.results)} audit results from {self.results_file}")
    
    def generate_table1(self, output_path: Path = None):
        """
        Generate Table 1: Density Chasm Detection Rates
        
        Format:
        Agent | Model | ALFWorld | GSM8K | Mean
        """
        print("\n" + "="*80)
        print("TABLE 1: DENSITY CHASM DETECTION RATES")
        print("="*80)
        
        # Aggregate by agent and benchmark
        by_agent = {}
        for result in self.results:
            agent = result['agent']
            model = result['model']
            benchmark = result['benchmark']
            rate = result['chasm_detection_rate']
            
            key = (agent, model)
            if key not in by_agent:
                by_agent[key] = {}
            by_agent[key][benchmark] = rate
        
        # Print table
        print(f"{'Agent':<20} {'Model':<15} {'ALFWorld':<12} {'GSM8K':<12} {'Mean':<12}")
        print("-"*80)
        
        table_data = []
        for (agent, model), rates in sorted(by_agent.items()):
            alfworld_rate = rates.get('alfworld', 0.0)
            gsm8k_rate = rates.get('gsm8k', 0.0)
            mean_rate = np.mean([alfworld_rate, gsm8k_rate])
            
            print(f"{agent:<20} {model:<15} {alfworld_rate:>10.1%}  {gsm8k_rate:>10.1%}  {mean_rate:>10.1%}")
            table_data.append({
                'agent': agent,
                'model': model,
                'alfworld': alfworld_rate,
                'gsm8k': gsm8k_rate,
                'mean': mean_rate
            })
        
        # Save as CSV for LaTeX import
        if output_path:
            import csv
            csv_path = output_path / "table1_detection_rates.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['agent', 'model', 'alfworld', 'gsm8k', 'mean'])
                writer.writeheader()
                writer.writerows(table_data)
            print(f"\n[OK] Table saved to {csv_path}")
        
        return table_data
    
    def plot_drift_trajectories(self, output_path: Path = None):
        """
        Generate Figure 1: Drift Trajectory Plots
        
        Shows w(x) vs. token index for ACC vs. baselines on sample tasks.
        """
        if not self.drift_logs_dir.exists():
            print("[ALERT]️  No drift logs found. Skipping trajectory plots.")
            return
        
        drift_logs = list(self.drift_logs_dir.glob("*.json"))
        if not drift_logs:
            print("[ALERT]️  No drift log files found. Skipping trajectory plots.")
            return
        
        print(f"\nGenerating drift trajectory plots from {len(drift_logs)} logs...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, log_file in enumerate(drift_logs[:4]):  # Plot first 4 samples
            with open(log_file) as f:
                log_data = json.load(f)
            
            steps = [entry['step'] for entry in log_data]
            drift_scores = [entry['w_x'] for entry in log_data]
            threshold = log_data[0].get('threshold', 0.0) if log_data else 0.0
            interventions = [entry.get('intervention', False) for entry in log_data]
            
            ax = axes[idx]
            ax.plot(steps, drift_scores, 'b-', linewidth=1.5, label='Drift Score w(x)')
            ax.axhline(threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold lambda*={threshold:.4f}')
            
            # Mark interventions
            intervention_steps = [s for s, i in zip(steps, interventions) if i]
            intervention_scores = [w for w, i in zip(drift_scores, interventions) if i]
            if intervention_steps:
                ax.scatter(intervention_steps, intervention_scores, color='red', s=50, 
                          marker='x', linewidths=2, label='Intervention', zorder=5)
            
            ax.set_xlabel('Token Index')
            ax.set_ylabel('Drift Score w(x)')
            ax.set_title(f'Sample {log_file.stem}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig_path = output_path / "figure1_drift_trajectories.pdf"
            plt.savefig(fig_path, bbox_inches='tight')
            print(f"[OK] Figure saved to {fig_path}")
        
        plt.show()
    
    def analyze_intervention_precision(self, output_path: Path = None):
        """
        Generate Figure 2: Intervention Precision Analysis
        
        Plots:
        - True Positive Rate: Interventions during actual drift
        - False Positive Rate: Interventions during safe generation
        """
        print("\nAnalyzing intervention precision...")
        
        # Extract intervention data
        agent_stats = {}
        for result in self.results:
            agent = result['agent']
            if agent not in agent_stats:
                agent_stats[agent] = {
                    'total_interventions': 0,
                    'total_samples': 0,
                    'chasm_detected': 0
                }
            
            agent_stats[agent]['total_samples'] += result['samples_tested']
            agent_stats[agent]['chasm_detected'] += result['chasm_detection_count']
            
            # Count interventions from sample results
            for sample in result.get('results', []):
                interventions = len(sample.get('handoff_events', []))
                agent_stats[agent]['total_interventions'] += interventions
        
        # Calculate precision metrics
        agents = []
        intervention_rates = []
        detection_rates = []
        
        for agent, stats in agent_stats.items():
            agents.append(agent)
            intervention_rate = stats['total_interventions'] / max(stats['total_samples'], 1)
            detection_rate = stats['chasm_detected'] / max(stats['total_samples'], 1)
            intervention_rates.append(intervention_rate)
            detection_rates.append(detection_rate)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(agents))
        width = 0.35
        
        ax.bar(x - width/2, intervention_rates, width, label='Intervention Rate', alpha=0.8)
        ax.bar(x + width/2, detection_rates, width, label='Detection Rate', alpha=0.8)
        
        ax.set_xlabel('Agent')
        ax.set_ylabel('Rate')
        ax.set_title('Intervention Precision: Detection vs. Intervention Rates')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            fig_path = output_path / "figure2_intervention_precision.pdf"
            plt.savefig(fig_path, bbox_inches='tight')
            print(f"[OK] Figure saved to {fig_path}")
        
        plt.show()
    
    def compute_statistical_significance(self):
        """
        Compute statistical significance tests for baseline comparisons.
        
        Uses Mann-Whitney U test (non-parametric) for detection rate comparisons.
        """
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        # Group results by agent
        by_agent = {}
        for result in self.results:
            agent = result['agent']
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(result['chasm_detection_rate'])
        
        # Compare ACC vs. each baseline
        if 'acc' not in by_agent:
            print("[ALERT]️  ACC results not found. Skipping significance tests.")
            return
        
        acc_rates = by_agent['acc']
        
        for agent, rates in by_agent.items():
            if agent == 'acc':
                continue
            
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(acc_rates, rates, alternative='greater')
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"ACC vs. {agent:15} | U={statistic:6.1f} | p={p_value:.4f} | {significance}")
        
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    def generate_all_figures(self, output_dir: Path = None):
        """Generate all publication figures and tables."""
        if output_dir is None:
            output_dir = self.results_dir / "figures"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*80}")
        print(f"GENERATING UAI 2026 PUBLICATION FIGURES")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}")
        
        # Table 1
        self.generate_table1(output_dir)
        
        # Figure 1
        self.plot_drift_trajectories(output_dir)
        
        # Figure 2
        self.analyze_intervention_precision(output_dir)
        
        # Statistical tests
        self.compute_statistical_significance()
        
        print(f"\n{'='*80}")
        print(f"[OK] All figures generated successfully!")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze campaign results and generate UAI 2026 figures"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("cross_baseline_results"),
        help="Directory containing cross_baseline_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: results_dir/figures)"
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Generate only Table 1 (skip plots)"
    )
    
    args = parser.parse_args()
    
    analyzer = CampaignAnalyzer(args.results_dir)
    
    if args.table_only:
        analyzer.generate_table1(args.output_dir or args.results_dir)
    else:
        analyzer.generate_all_figures(args.output_dir)


if __name__ == "__main__":
    main()
