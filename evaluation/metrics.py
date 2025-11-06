"""
Performance Metrics
Evaluation metrics for MAS decision-making performance
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


class PerformanceMetrics:
    """
    Calculates various performance metrics for the multi-agent system.
    """

    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics_history = []

    def calculate_consensus_metrics(self, consensus_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate metrics related to consensus building.

        Args:
            consensus_results: List of consensus results from multiple scenarios

        Returns:
            Dictionary with consensus metrics
        """
        if not consensus_results:
            return {
                'consensus_rate': 0.0,
                'average_agreement_level': 0.0,
                'average_iterations': 0.0,
                'convergence_rate': 0.0
            }

        consensus_achieved = [
            r.get('consensus_achieved', False) for r in consensus_results
        ]
        agreement_levels = [
            r.get('agreement_level', 0.0) for r in consensus_results
        ]
        iterations = [
            r.get('iterations_required', 1) for r in consensus_results
            if 'iterations_required' in r
        ]

        metrics = {
            'consensus_rate': sum(consensus_achieved) / len(consensus_achieved),
            'average_agreement_level': float(np.mean(agreement_levels)),
            'std_agreement_level': float(np.std(agreement_levels)),
            'min_agreement_level': float(np.min(agreement_levels)),
            'max_agreement_level': float(np.max(agreement_levels)),
            'average_iterations': float(np.mean(iterations)) if iterations else 0.0,
            'convergence_rate': sum(1 for i in iterations if i <= 3) / len(iterations) if iterations else 0.0
        }

        return metrics

    def calculate_decision_quality_metrics(self, decisions: List[Dict[str, Any]],
                                          ground_truth: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        Calculate decision quality metrics.

        Args:
            decisions: List of decisions made by the system
            ground_truth: Optional ground truth decisions for comparison

        Returns:
            Dictionary with decision quality metrics
        """
        metrics = {}

        # Calculate confidence statistics
        confidences = [d.get('confidence', 0.0) for d in decisions]
        metrics['average_confidence'] = float(np.mean(confidences))
        metrics['std_confidence'] = float(np.std(confidences))

        # Calculate action score statistics
        action_scores = [
            d.get('action_score', 0.0)
            for d in decisions
            if 'action_score' in d
        ]
        if action_scores:
            metrics['average_action_score'] = float(np.mean(action_scores))
            metrics['std_action_score'] = float(np.std(action_scores))

        # If ground truth is available, calculate accuracy metrics
        if ground_truth and len(ground_truth) == len(decisions):
            correct_decisions = sum(
                1 for d, gt in zip(decisions, ground_truth)
                if d.get('proposed_action', {}).get('id') == gt.get('correct_action_id')
            )
            metrics['accuracy'] = correct_decisions / len(decisions)

        return metrics

    def calculate_agent_contribution_metrics(self, agent_decisions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Calculate metrics showing individual agent contributions.

        Args:
            agent_decisions: Dictionary mapping agent IDs to their decisions

        Returns:
            Dictionary with agent contribution metrics
        """
        agent_metrics = {}

        for agent_id, decisions in agent_decisions.items():
            if not decisions:
                continue

            confidences = [d.get('confidence', 0.0) for d in decisions]
            scores = [d.get('action_score', 0.0) for d in decisions if 'action_score' in d]

            agent_metrics[agent_id] = {
                'num_decisions': len(decisions),
                'average_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'average_score': float(np.mean(scores)) if scores else 0.0,
                'consistency': self._calculate_consistency(decisions)
            }

        return agent_metrics

    def _calculate_consistency(self, decisions: List[Dict[str, Any]]) -> float:
        """
        Calculate decision consistency for an agent.

        Args:
            decisions: List of decisions from one agent

        Returns:
            Consistency score (higher = more consistent)
        """
        if len(decisions) < 2:
            return 1.0

        # Calculate variance in confidence levels (lower variance = more consistent)
        confidences = [d.get('confidence', 0.0) for d in decisions]
        confidence_variance = np.var(confidences)

        # Normalize to 0-1 scale (1 = most consistent)
        consistency = 1.0 / (1.0 + confidence_variance)

        return float(consistency)

    def calculate_diversity_metrics(self, agent_proposals: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate diversity metrics for agent proposals.

        Args:
            agent_proposals: List of proposals from different agents

        Returns:
            Dictionary with diversity metrics
        """
        if not agent_proposals:
            return {'diversity_score': 0.0}

        # Count unique proposed actions
        proposed_actions = [
            p.get('proposed_action', {}).get('id', '')
            for p in agent_proposals
        ]
        unique_actions = len(set(proposed_actions))
        total_proposals = len(agent_proposals)

        # Diversity ratio
        diversity_ratio = unique_actions / total_proposals if total_proposals > 0 else 0

        # Calculate variance in confidence levels (higher variance = more diverse opinions)
        confidences = [p.get('confidence', 0.0) for p in agent_proposals]
        confidence_variance = float(np.var(confidences))

        # Calculate entropy of action distribution
        from collections import Counter
        action_counts = Counter(proposed_actions)
        probabilities = [count / total_proposals for count in action_counts.values()]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)

        return {
            'diversity_ratio': diversity_ratio,
            'unique_actions': unique_actions,
            'total_proposals': total_proposals,
            'confidence_variance': confidence_variance,
            'opinion_entropy': float(entropy),
            'diversity_score': (diversity_ratio + min(1.0, entropy / 2)) / 2  # Combined metric
        }

    def calculate_efficiency_metrics(self, session_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate efficiency metrics for the decision-making process.

        Args:
            session_data: Data from a decision-making session

        Returns:
            Dictionary with efficiency metrics
        """
        metrics = {}

        # Time-based metrics
        if 'start_time' in session_data and 'end_time' in session_data:
            duration = session_data['end_time'] - session_data['start_time']
            metrics['total_duration_seconds'] = duration

        # Iteration-based metrics
        if 'iterations' in session_data:
            metrics['num_iterations'] = len(session_data['iterations'])

        # Agent utilization
        if 'agent_calls' in session_data:
            metrics['total_agent_calls'] = sum(session_data['agent_calls'].values())
            metrics['average_calls_per_agent'] = float(np.mean(list(session_data['agent_calls'].values())))

        # LLM usage metrics
        if 'llm_stats' in session_data:
            metrics['llm_requests'] = session_data['llm_stats'].get('total_requests', 0)
            metrics['llm_tokens'] = session_data['llm_stats'].get('total_tokens', 0)

        return metrics

    def calculate_robustness_metrics(self, sensitivity_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate robustness metrics based on sensitivity analysis.

        Args:
            sensitivity_results: Results from sensitivity analysis

        Returns:
            Dictionary with robustness metrics
        """
        metrics = {}

        if 'ranking_stability' in sensitivity_results:
            metrics['ranking_stability'] = sensitivity_results['ranking_stability']

        if 'expert_sensitivities' in sensitivity_results:
            sensitivities = [
                e['score_sensitivity']
                for e in sensitivity_results['expert_sensitivities']
            ]
            metrics['average_sensitivity'] = float(np.mean(sensitivities))
            metrics['max_sensitivity'] = float(np.max(sensitivities))
            metrics['robustness_score'] = 1.0 - metrics['average_sensitivity']

        return metrics

    def generate_summary_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all metrics.

        Args:
            all_results: Dictionary containing all experimental results

        Returns:
            Comprehensive summary report
        """
        report = {
            'timestamp': all_results.get('timestamp', 'unknown'),
            'num_scenarios': all_results.get('num_scenarios', 0),
            'metrics': {}
        }

        # Consensus metrics
        if 'consensus_results' in all_results:
            report['metrics']['consensus'] = self.calculate_consensus_metrics(
                all_results['consensus_results']
            )

        # Decision quality metrics
        if 'decisions' in all_results:
            report['metrics']['decision_quality'] = self.calculate_decision_quality_metrics(
                all_results['decisions'],
                all_results.get('ground_truth')
            )

        # Agent contribution metrics
        if 'agent_decisions' in all_results:
            report['metrics']['agent_contributions'] = self.calculate_agent_contribution_metrics(
                all_results['agent_decisions']
            )

        # Diversity metrics
        if 'agent_proposals' in all_results:
            report['metrics']['diversity'] = self.calculate_diversity_metrics(
                all_results['agent_proposals']
            )

        # Efficiency metrics
        if 'session_data' in all_results:
            report['metrics']['efficiency'] = self.calculate_efficiency_metrics(
                all_results['session_data']
            )

        # Overall performance score (composite metric)
        report['overall_performance_score'] = self._calculate_overall_score(report['metrics'])

        return report

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate an overall performance score from multiple metrics.

        Args:
            metrics: Dictionary of all calculated metrics

        Returns:
            Overall performance score (0-1)
        """
        scores = []

        # Consensus score
        if 'consensus' in metrics:
            consensus_score = (
                metrics['consensus'].get('consensus_rate', 0) * 0.4 +
                metrics['consensus'].get('average_agreement_level', 0) * 0.6
            )
            scores.append(consensus_score)

        # Decision quality score
        if 'decision_quality' in metrics:
            quality_score = metrics['decision_quality'].get('average_confidence', 0)
            scores.append(quality_score)

        # Diversity score (moderate diversity is good, not too high or low)
        if 'diversity' in metrics:
            diversity = metrics['diversity'].get('diversity_score', 0.5)
            # Optimal diversity around 0.5
            diversity_score = 1.0 - abs(diversity - 0.5) * 2
            scores.append(diversity_score)

        # Calculate average
        overall = float(np.mean(scores)) if scores else 0.0

        return overall

    def compare_configurations(self, results_list: List[Dict[str, Any]],
                             configuration_names: List[str]) -> Dict[str, Any]:
        """
        Compare performance across different system configurations.

        Args:
            results_list: List of result dictionaries from different configurations
            configuration_names: Names of each configuration

        Returns:
            Comparison report
        """
        comparison = {
            'configurations': configuration_names,
            'metrics_comparison': defaultdict(dict)
        }

        for config_name, results in zip(configuration_names, results_list):
            summary = self.generate_summary_report(results)

            # Extract key metrics for comparison
            for metric_category, metric_values in summary['metrics'].items():
                for metric_name, value in metric_values.items():
                    if isinstance(value, (int, float)):
                        comparison['metrics_comparison'][f"{metric_category}.{metric_name}"][config_name] = value

        # Identify best configuration for each metric
        comparison['best_configurations'] = {}
        for metric_name, config_values in comparison['metrics_comparison'].items():
            best_config = max(config_values.items(), key=lambda x: x[1])
            comparison['best_configurations'][metric_name] = {
                'configuration': best_config[0],
                'value': best_config[1]
            }

        return comparison
