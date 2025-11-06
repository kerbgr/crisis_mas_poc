"""
Evidential Reasoning Implementation
Simplified ER approach for combining uncertain information from multiple experts
Based on Dempster-Shafer theory and evidential reasoning rule
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class EvidentialReasoning:
    """
    Implements a simplified Evidential Reasoning approach for decision-making under uncertainty.
    Combines beliefs from multiple experts considering their reliability and confidence.
    """

    def __init__(self):
        """Initialize the Evidential Reasoning engine."""
        self.belief_history = []

    def create_belief_structure(self, assessment: float, confidence: float) -> Dict[str, float]:
        """
        Create a basic belief structure from an assessment and confidence level.

        Args:
            assessment: Assessment value (0-1)
            confidence: Confidence in the assessment (0-1)

        Returns:
            Dictionary with belief assignments
        """
        # Simple belief structure:
        # - Belief mass assigned to the assessment value
        # - Remaining mass assigned to uncertainty
        belief = {
            'belief': assessment * confidence,
            'disbelief': (1 - assessment) * confidence,
            'uncertainty': 1 - confidence
        }

        return belief

    def combine_beliefs(self, beliefs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Combine multiple belief structures using Dempster's rule of combination.

        Args:
            beliefs: List of belief structures from different experts

        Returns:
            Combined belief structure
        """
        if not beliefs:
            return {'belief': 0.0, 'disbelief': 0.0, 'uncertainty': 1.0}

        if len(beliefs) == 1:
            return beliefs[0]

        # Start with the first belief
        combined = beliefs[0].copy()

        # Sequentially combine with remaining beliefs
        for i in range(1, len(beliefs)):
            combined = self._combine_two_beliefs(combined, beliefs[i])

        self.belief_history.append({
            'num_beliefs': len(beliefs),
            'combined_result': combined
        })

        return combined

    def _combine_two_beliefs(self, belief1: Dict[str, float],
                            belief2: Dict[str, float]) -> Dict[str, float]:
        """
        Combine two belief structures using Dempster's rule.

        Args:
            belief1: First belief structure
            belief2: Second belief structure

        Returns:
            Combined belief structure
        """
        # Extract components
        b1, d1, u1 = belief1.get('belief', 0), belief1.get('disbelief', 0), belief1.get('uncertainty', 0)
        b2, d2, u2 = belief2.get('belief', 0), belief2.get('disbelief', 0), belief2.get('uncertainty', 0)

        # Calculate conflict
        conflict = b1 * d2 + d1 * b2

        # Normalization factor (handling conflict)
        if conflict >= 1.0:
            # Complete conflict - use averaging
            return {
                'belief': (b1 + b2) / 2,
                'disbelief': (d1 + d2) / 2,
                'uncertainty': (u1 + u2) / 2
            }

        k = 1 / (1 - conflict) if conflict < 1.0 else 1.0

        # Dempster's combination rule
        combined_belief = k * (b1 * b2 + b1 * u2 + u1 * b2)
        combined_disbelief = k * (d1 * d2 + d1 * u2 + u1 * d2)
        combined_uncertainty = k * (u1 * u2)

        # Normalize to ensure sum = 1
        total = combined_belief + combined_disbelief + combined_uncertainty
        if total > 0:
            combined_belief /= total
            combined_disbelief /= total
            combined_uncertainty /= total

        return {
            'belief': combined_belief,
            'disbelief': combined_disbelief,
            'uncertainty': combined_uncertainty
        }

    def aggregate_expert_assessments(self, expert_assessments: List[Dict[str, Any]],
                                    reliability_weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Aggregate assessments from multiple experts using ER.

        Args:
            expert_assessments: List of expert assessments with scores and confidence
            reliability_weights: Optional reliability weights for each expert

        Returns:
            Aggregated assessment with combined belief structure
        """
        if not expert_assessments:
            return {
                'aggregated_score': 0.5,
                'belief': 0.0,
                'disbelief': 0.0,
                'uncertainty': 1.0,
                'confidence': 0.0
            }

        # Create belief structures for each expert
        beliefs = []
        for i, assessment in enumerate(expert_assessments):
            score = assessment.get('score', 0.5)
            confidence = assessment.get('confidence', 0.5)

            # Apply reliability weight if provided
            if reliability_weights and i < len(reliability_weights):
                confidence *= reliability_weights[i]

            belief = self.create_belief_structure(score, confidence)
            beliefs.append(belief)

        # Combine all beliefs
        combined = self.combine_beliefs(beliefs)

        # Calculate expected value from belief structure
        aggregated_score = combined['belief'] + 0.5 * combined['uncertainty']

        return {
            'aggregated_score': aggregated_score,
            'belief': combined['belief'],
            'disbelief': combined['disbelief'],
            'uncertainty': combined['uncertainty'],
            'confidence': combined['belief'] + combined['disbelief']
        }

    def rank_alternatives(self, alternatives: Dict[str, List[Dict[str, Any]]],
                         criteria_weights: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Rank alternatives using ER across multiple criteria.

        Args:
            alternatives: Dict mapping alternative IDs to lists of criterion assessments
            criteria_weights: Weights for each criterion

        Returns:
            List of (alternative_id, score) tuples, sorted by score
        """
        alternative_scores = {}

        for alt_id, assessments in alternatives.items():
            # Weight assessments by criteria weights
            weighted_assessments = []

            for assessment in assessments:
                criterion = assessment.get('criterion', '')
                weight = criteria_weights.get(criterion, 1.0)

                weighted_assessment = assessment.copy()
                weighted_assessment['confidence'] *= weight

                weighted_assessments.append(weighted_assessment)

            # Aggregate using ER
            result = self.aggregate_expert_assessments(weighted_assessments)
            alternative_scores[alt_id] = result['aggregated_score']

        # Sort by score
        ranked = sorted(alternative_scores.items(), key=lambda x: x[1], reverse=True)

        return ranked

    def calculate_belief_distance(self, belief1: Dict[str, float],
                                 belief2: Dict[str, float]) -> float:
        """
        Calculate distance between two belief structures.
        Useful for measuring disagreement between experts.

        Args:
            belief1: First belief structure
            belief2: Second belief structure

        Returns:
            Distance measure (0 = identical, higher = more different)
        """
        b1 = np.array([belief1.get('belief', 0),
                      belief1.get('disbelief', 0),
                      belief1.get('uncertainty', 0)])

        b2 = np.array([belief2.get('belief', 0),
                      belief2.get('disbelief', 0),
                      belief2.get('uncertainty', 0)])

        # Euclidean distance
        distance = np.linalg.norm(b1 - b2)

        return float(distance)

    def sensitivity_analysis(self, expert_assessments: List[Dict[str, Any]],
                           perturbation: float = 0.1) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on the aggregated result.

        Args:
            expert_assessments: List of expert assessments
            perturbation: Amount to perturb confidence values (0-1)

        Returns:
            Sensitivity analysis results
        """
        # Original aggregation
        original = self.aggregate_expert_assessments(expert_assessments)

        # Perturb each expert's confidence and re-aggregate
        perturbed_results = []

        for i in range(len(expert_assessments)):
            # Create perturbed assessment
            perturbed_assessments = [a.copy() for a in expert_assessments]

            # Increase confidence
            perturbed_assessments[i]['confidence'] = min(
                1.0,
                perturbed_assessments[i]['confidence'] * (1 + perturbation)
            )
            result_up = self.aggregate_expert_assessments(perturbed_assessments)

            # Decrease confidence
            perturbed_assessments[i]['confidence'] = max(
                0.0,
                expert_assessments[i]['confidence'] * (1 - perturbation)
            )
            result_down = self.aggregate_expert_assessments(perturbed_assessments)

            sensitivity = abs(result_up['aggregated_score'] - result_down['aggregated_score'])

            perturbed_results.append({
                'expert_index': i,
                'original_confidence': expert_assessments[i]['confidence'],
                'score_sensitivity': sensitivity
            })

        return {
            'original_score': original['aggregated_score'],
            'original_uncertainty': original['uncertainty'],
            'expert_sensitivities': perturbed_results,
            'most_influential_expert': max(perturbed_results, key=lambda x: x['score_sensitivity'])['expert_index']
        }
