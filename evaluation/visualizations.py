"""
Result Visualization
Visualization tools for MAS decision-making results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path


class ResultVisualizer:
    """
    Creates visualizations for multi-agent system decision-making results.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def plot_consensus_evolution(self, consensus_history: List[Dict[str, Any]],
                                 save_path: Optional[str] = None):
        """
        Plot how consensus evolves over iterations.

        Args:
            consensus_history: List of consensus results over iterations
            save_path: Optional path to save the plot
        """
        if not consensus_history:
            print("No consensus history to plot")
            return

        iterations = [h.get('iteration', i) for i, h in enumerate(consensus_history)]
        agreement_levels = [h.get('agreement_level', 0) for h in consensus_history]

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, agreement_levels, marker='o', linewidth=2, markersize=8)
        plt.axhline(y=0.7, color='r', linestyle='--', label='Consensus Threshold (0.7)')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Agreement Level', fontsize=12)
        plt.title('Consensus Evolution Over Iterations', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_agent_contributions(self, agent_metrics: Dict[str, Dict[str, float]],
                                save_path: Optional[str] = None):
        """
        Plot agent contribution metrics.

        Args:
            agent_metrics: Dictionary with metrics for each agent
            save_path: Optional path to save the plot
        """
        if not agent_metrics:
            print("No agent metrics to plot")
            return

        agents = list(agent_metrics.keys())
        confidences = [m.get('average_confidence', 0) for m in agent_metrics.values()]
        scores = [m.get('average_score', 0) for m in agent_metrics.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Confidence plot
        colors = sns.color_palette("husl", len(agents))
        ax1.bar(agents, confidences, color=colors)
        ax1.set_xlabel('Agent', fontsize=12)
        ax1.set_ylabel('Average Confidence', fontsize=12)
        ax1.set_title('Agent Confidence Levels', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Score plot
        ax2.bar(agents, scores, color=colors)
        ax2.set_xlabel('Agent', fontsize=12)
        ax2.set_ylabel('Average Action Score', fontsize=12)
        ax2.set_title('Agent Action Scores', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_action_comparison(self, actions: List[Dict[str, Any]],
                              criteria: List[str],
                              save_path: Optional[str] = None):
        """
        Plot radar chart comparing actions across criteria.

        Args:
            actions: List of actions with criteria scores
            criteria: List of criteria names
            save_path: Optional path to save the plot
        """
        if not actions or not criteria:
            print("No actions or criteria to plot")
            return

        # Prepare data
        num_vars = len(criteria)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for action in actions[:5]:  # Plot up to 5 actions
            values = [action.get('criteria_scores', {}).get(c, 0) for c in criteria]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2,
                   label=action.get('name', action.get('id', 'Unknown')))
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)
        ax.set_ylim(0, 1.0)
        ax.set_title('Action Comparison Across Criteria',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_decision_distribution(self, decisions: List[Dict[str, Any]],
                                  save_path: Optional[str] = None):
        """
        Plot distribution of decisions made.

        Args:
            decisions: List of decisions
            save_path: Optional path to save the plot
        """
        if not decisions:
            print("No decisions to plot")
            return

        # Count decisions by action
        from collections import Counter
        action_ids = [
            d.get('proposed_action', {}).get('id', 'unknown')
            for d in decisions
        ]
        action_counts = Counter(action_ids)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        ax1.bar(actions, counts, color=sns.color_palette("husl", len(actions)))
        ax1.set_xlabel('Action', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Decision Distribution', fontsize=14, fontweight='bold')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Pie chart
        ax2.pie(counts, labels=actions, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Decision Proportion', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sensitivity_analysis(self, sensitivity_results: Dict[str, Any],
                                 save_path: Optional[str] = None):
        """
        Plot sensitivity analysis results.

        Args:
            sensitivity_results: Results from sensitivity analysis
            save_path: Optional path to save the plot
        """
        if not sensitivity_results or 'results_by_weight' not in sensitivity_results:
            print("No sensitivity results to plot")
            return

        results = sensitivity_results['results_by_weight']
        weights = [r['weight'] for r in results]
        top_actions = [r['top_alternative'] for r in results]
        scores = [r['top_score'] for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Score evolution
        ax1.plot(weights, scores, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel(f'{sensitivity_results.get("criterion", "Criterion")} Weight', fontsize=12)
        ax1.set_ylabel('Top Alternative Score', fontsize=12)
        ax1.set_title('Score Sensitivity to Weight Changes', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Threshold changes
        unique_actions = sorted(set(top_actions))
        action_to_num = {action: i for i, action in enumerate(unique_actions)}
        numeric_actions = [action_to_num[action] for action in top_actions]

        ax2.plot(weights, numeric_actions, marker='s', linewidth=2, markersize=6)
        ax2.set_xlabel(f'{sensitivity_results.get("criterion", "Criterion")} Weight', fontsize=12)
        ax2.set_ylabel('Top Alternative', fontsize=12)
        ax2.set_yticks(range(len(unique_actions)))
        ax2.set_yticklabels(unique_actions)
        ax2.set_title('Ranking Changes with Weight Variation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metrics_comparison(self, comparison_data: Dict[str, Any],
                               save_path: Optional[str] = None):
        """
        Plot comparison of metrics across configurations.

        Args:
            comparison_data: Comparison data from metrics
            save_path: Optional path to save the plot
        """
        if not comparison_data or 'metrics_comparison' not in comparison_data:
            print("No comparison data to plot")
            return

        # Convert to DataFrame for easier plotting
        data = []
        for metric_name, config_values in comparison_data['metrics_comparison'].items():
            for config, value in config_values.items():
                data.append({
                    'Metric': metric_name,
                    'Configuration': config,
                    'Value': value
                })

        df = pd.DataFrame(data)

        # Plot grouped bar chart
        metrics = df['Metric'].unique()[:10]  # Limit to 10 metrics
        df_subset = df[df['Metric'].isin(metrics)]

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create grouped bar chart
        x = np.arange(len(metrics))
        configurations = df_subset['Configuration'].unique()
        width = 0.8 / len(configurations)

        for i, config in enumerate(configurations):
            config_data = df_subset[df_subset['Configuration'] == config]
            values = [config_data[config_data['Metric'] == m]['Value'].values[0]
                     if len(config_data[config_data['Metric'] == m]) > 0 else 0
                     for m in metrics]
            ax.bar(x + i * width, values, width, label=config)

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Configuration Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(configurations) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(self, metrics_report: Dict[str, Any],
                                save_path: str = "summary_dashboard.png"):
        """
        Create a comprehensive dashboard with multiple metrics.

        Args:
            metrics_report: Comprehensive metrics report
            save_path: Path to save the dashboard
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Overall performance score
        ax1 = fig.add_subplot(gs[0, :])
        score = metrics_report.get('overall_performance_score', 0)
        ax1.barh(['Overall Performance'], [score], color='green' if score >= 0.7 else 'orange')
        ax1.set_xlim(0, 1.0)
        ax1.set_title('Overall Performance Score', fontsize=14, fontweight='bold')
        ax1.text(score/2, 0, f'{score:.2f}', ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')

        # Consensus metrics
        if 'consensus' in metrics_report.get('metrics', {}):
            ax2 = fig.add_subplot(gs[1, 0])
            consensus = metrics_report['metrics']['consensus']
            metrics_to_plot = ['consensus_rate', 'average_agreement_level', 'convergence_rate']
            values = [consensus.get(m, 0) for m in metrics_to_plot]
            ax2.bar(range(len(metrics_to_plot)), values, color=sns.color_palette("husl", 3))
            ax2.set_xticks(range(len(metrics_to_plot)))
            ax2.set_xticklabels(['Consensus\nRate', 'Avg\nAgreement', 'Convergence\nRate'])
            ax2.set_ylim(0, 1.0)
            ax2.set_title('Consensus Metrics', fontsize=12, fontweight='bold')

        # Decision quality
        if 'decision_quality' in metrics_report.get('metrics', {}):
            ax3 = fig.add_subplot(gs[1, 1])
            quality = metrics_report['metrics']['decision_quality']
            avg_conf = quality.get('average_confidence', 0)
            std_conf = quality.get('std_confidence', 0)
            ax3.bar(['Confidence'], [avg_conf], yerr=[std_conf], capsize=10, color='blue')
            ax3.set_ylim(0, 1.0)
            ax3.set_title('Decision Quality', fontsize=12, fontweight='bold')

        # Diversity
        if 'diversity' in metrics_report.get('metrics', {}):
            ax4 = fig.add_subplot(gs[1, 2])
            diversity = metrics_report['metrics']['diversity']
            div_score = diversity.get('diversity_score', 0)
            ax4.pie([div_score, 1-div_score], labels=['Diverse', 'Consensus'],
                   autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgray'])
            ax4.set_title('Opinion Diversity', fontsize=12, fontweight='bold')

        # Summary text
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        summary_text = f"""
        Experiment Summary
        {'=' * 50}
        Timestamp: {metrics_report.get('timestamp', 'N/A')}
        Scenarios Evaluated: {metrics_report.get('num_scenarios', 0)}
        Overall Performance: {score:.2%}

        Key Findings:
        - Consensus achieved at {consensus.get('consensus_rate', 0):.1%} rate
        - Average agreement level: {consensus.get('average_agreement_level', 0):.2f}
        - Decision confidence: {quality.get('average_confidence', 0):.2f} ± {quality.get('std_confidence', 0):.2f}
        """
        ax5.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
                verticalalignment='center')

        plt.suptitle('Multi-Agent System Performance Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Dashboard saved to: {self.output_dir / save_path}")
