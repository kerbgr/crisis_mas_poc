"""
Multi-Criteria Decision Analysis (MCDA) Engine
Implements various MCDA methods for crisis decision-making
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class MCDAEngine:
    """
    Multi-Criteria Decision Analysis engine implementing various MCDA methods.
    Supports TOPSIS, Weighted Sum, and Simple Additive Weighting (SAW).
    """

    def __init__(self, criteria_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the MCDA engine.

        Args:
            criteria_weights: Dictionary mapping criteria names to weights
        """
        self.criteria_weights = criteria_weights or {}
        self.analysis_history = []

    def set_criteria_weights(self, criteria_weights: Dict[str, float]):
        """
        Set or update criteria weights.

        Args:
            criteria_weights: Dictionary mapping criteria names to weights
        """
        # Normalize weights to sum to 1
        total_weight = sum(criteria_weights.values())
        if total_weight > 0:
            self.criteria_weights = {
                k: v / total_weight for k, v in criteria_weights.items()
            }
        else:
            self.criteria_weights = criteria_weights

    def weighted_sum_method(self, alternatives: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Simple Weighted Sum Method (WSM).

        Args:
            alternatives: List of alternatives with criteria scores

        Returns:
            List of (alternative_id, weighted_score) tuples, sorted by score
        """
        scores = []

        for alternative in alternatives:
            alt_id = alternative.get('id', 'unknown')
            criteria_scores = alternative.get('criteria_scores', {})

            weighted_score = 0.0
            total_weight = 0.0

            for criterion, weight in self.criteria_weights.items():
                if criterion in criteria_scores:
                    weighted_score += criteria_scores[criterion] * weight
                    total_weight += weight

            # Normalize by total weight used
            if total_weight > 0:
                weighted_score /= total_weight

            scores.append((alt_id, weighted_score))

        # Sort by score (descending)
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        self.analysis_history.append({
            'method': 'weighted_sum',
            'num_alternatives': len(alternatives),
            'best_alternative': ranked[0] if ranked else None
        })

        return ranked

    def topsis_method(self, alternatives: List[Dict[str, Any]],
                     beneficial_criteria: Optional[List[str]] = None,
                     cost_criteria: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

        Args:
            alternatives: List of alternatives with criteria scores
            beneficial_criteria: Criteria where higher is better (default: all)
            cost_criteria: Criteria where lower is better

        Returns:
            List of (alternative_id, closeness_score) tuples, sorted by closeness
        """
        if not alternatives:
            return []

        # Extract criteria matrix
        alt_ids = [alt.get('id', f'alt_{i}') for i, alt in enumerate(alternatives)]
        criteria = list(self.criteria_weights.keys())

        # Build decision matrix
        matrix = np.zeros((len(alternatives), len(criteria)))

        for i, alternative in enumerate(alternatives):
            criteria_scores = alternative.get('criteria_scores', {})
            for j, criterion in enumerate(criteria):
                matrix[i, j] = criteria_scores.get(criterion, 0.0)

        # Normalize the decision matrix (vector normalization)
        normalized_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

        # Apply weights
        weights = np.array([self.criteria_weights[c] for c in criteria])
        weighted_matrix = normalized_matrix * weights

        # Determine ideal and negative-ideal solutions
        beneficial = beneficial_criteria or criteria
        cost = cost_criteria or []

        ideal_solution = np.zeros(len(criteria))
        negative_ideal = np.zeros(len(criteria))

        for j, criterion in enumerate(criteria):
            if criterion in cost:
                # For cost criteria, lower is better
                ideal_solution[j] = weighted_matrix[:, j].min()
                negative_ideal[j] = weighted_matrix[:, j].max()
            else:
                # For beneficial criteria, higher is better
                ideal_solution[j] = weighted_matrix[:, j].max()
                negative_ideal[j] = weighted_matrix[:, j].min()

        # Calculate distances
        distance_to_ideal = np.sqrt(((weighted_matrix - ideal_solution) ** 2).sum(axis=1))
        distance_to_negative = np.sqrt(((weighted_matrix - negative_ideal) ** 2).sum(axis=1))

        # Calculate closeness coefficient
        closeness = distance_to_negative / (distance_to_ideal + distance_to_negative)

        # Handle division by zero
        closeness = np.nan_to_num(closeness, nan=0.0)

        # Create ranking
        scores = list(zip(alt_ids, closeness))
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        self.analysis_history.append({
            'method': 'topsis',
            'num_alternatives': len(alternatives),
            'best_alternative': ranked[0] if ranked else None
        })

        return ranked

    def saw_method(self, alternatives: List[Dict[str, Any]],
                   normalize: bool = True) -> List[Tuple[str, float]]:
        """
        Simple Additive Weighting (SAW) method.

        Args:
            alternatives: List of alternatives with criteria scores
            normalize: Whether to normalize criteria scores

        Returns:
            List of (alternative_id, saw_score) tuples, sorted by score
        """
        if not alternatives:
            return []

        alt_ids = [alt.get('id', f'alt_{i}') for i, alt in enumerate(alternatives)]
        criteria = list(self.criteria_weights.keys())

        # Build decision matrix
        matrix = np.zeros((len(alternatives), len(criteria)))

        for i, alternative in enumerate(alternatives):
            criteria_scores = alternative.get('criteria_scores', {})
            for j, criterion in enumerate(criteria):
                matrix[i, j] = criteria_scores.get(criterion, 0.0)

        # Normalize if requested
        if normalize:
            # Max normalization for each criterion
            max_values = matrix.max(axis=0)
            max_values[max_values == 0] = 1  # Avoid division by zero
            normalized_matrix = matrix / max_values
        else:
            normalized_matrix = matrix

        # Apply weights and sum
        weights = np.array([self.criteria_weights[c] for c in criteria])
        weighted_scores = (normalized_matrix * weights).sum(axis=1)

        # Create ranking
        scores = list(zip(alt_ids, weighted_scores))
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        self.analysis_history.append({
            'method': 'saw',
            'num_alternatives': len(alternatives),
            'normalized': normalize,
            'best_alternative': ranked[0] if ranked else None
        })

        return ranked

    def compare_methods(self, alternatives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare results from different MCDA methods.

        Args:
            alternatives: List of alternatives with criteria scores

        Returns:
            Dictionary with comparison results
        """
        # Run all methods
        wsm_results = self.weighted_sum_method(alternatives)
        topsis_results = self.topsis_method(alternatives)
        saw_results = self.saw_method(alternatives)

        # Find consensus ranking (most frequently top-ranked)
        top_choices = [
            wsm_results[0][0] if wsm_results else None,
            topsis_results[0][0] if topsis_results else None,
            saw_results[0][0] if saw_results else None
        ]

        # Count occurrences
        from collections import Counter
        choice_counts = Counter(top_choices)
        most_common = choice_counts.most_common(1)
        consensus_choice = most_common[0][0] if most_common else None

        return {
            'wsm_ranking': wsm_results,
            'topsis_ranking': topsis_results,
            'saw_ranking': saw_results,
            'consensus_choice': consensus_choice,
            'agreement_level': choice_counts[consensus_choice] / 3 if consensus_choice else 0.0,
            'all_methods_agree': len(set(filter(None, top_choices))) == 1
        }

    def sensitivity_analysis(self, alternatives: List[Dict[str, Any]],
                           criterion_to_vary: str,
                           weight_range: Tuple[float, float] = (0.0, 1.0),
                           steps: int = 10) -> Dict[str, Any]:
        """
        Perform sensitivity analysis by varying one criterion's weight.

        Args:
            alternatives: List of alternatives
            criterion_to_vary: Name of criterion to vary
            weight_range: Range of weights to test
            steps: Number of steps in the range

        Returns:
            Sensitivity analysis results
        """
        if criterion_to_vary not in self.criteria_weights:
            raise ValueError(f"Criterion '{criterion_to_vary}' not found in weights")

        original_weight = self.criteria_weights[criterion_to_vary]
        results = []

        weight_values = np.linspace(weight_range[0], weight_range[1], steps)

        for weight in weight_values:
            # Temporarily modify weight
            self.criteria_weights[criterion_to_vary] = weight

            # Re-normalize other weights
            other_criteria_sum = sum(
                w for c, w in self.criteria_weights.items() if c != criterion_to_vary
            )

            if other_criteria_sum > 0:
                remaining_weight = 1.0 - weight
                for c in self.criteria_weights:
                    if c != criterion_to_vary:
                        self.criteria_weights[c] *= (remaining_weight / other_criteria_sum)

            # Run analysis
            ranking = self.weighted_sum_method(alternatives)
            top_alternative = ranking[0][0] if ranking else None

            results.append({
                'weight': float(weight),
                'top_alternative': top_alternative,
                'top_score': ranking[0][1] if ranking else 0.0,
                'full_ranking': ranking
            })

        # Restore original weight
        self.criteria_weights[criterion_to_vary] = original_weight

        # Analyze stability
        top_alternatives = [r['top_alternative'] for r in results]
        stability = len(set(top_alternatives)) / len(top_alternatives) if top_alternatives else 0

        return {
            'criterion': criterion_to_vary,
            'original_weight': original_weight,
            'results_by_weight': results,
            'ranking_stability': 1.0 - stability,  # Higher = more stable
            'weight_threshold_changes': self._find_threshold_changes(results)
        }

    def _find_threshold_changes(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find weight thresholds where the top alternative changes.

        Args:
            results: List of results from sensitivity analysis

        Returns:
            List of threshold points
        """
        thresholds = []

        for i in range(1, len(results)):
            if results[i]['top_alternative'] != results[i-1]['top_alternative']:
                thresholds.append({
                    'weight_threshold': (results[i-1]['weight'] + results[i]['weight']) / 2,
                    'previous_top': results[i-1]['top_alternative'],
                    'new_top': results[i]['top_alternative']
                })

        return thresholds
