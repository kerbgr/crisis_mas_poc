"""
Consensus Model
Implements consensus detection and convergence algorithms for multi-agent decision-making
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


class ConsensusModel:
    """
    Implements consensus-building and detection mechanisms for multi-agent systems.
    Supports various consensus metrics and convergence algorithms.
    """

    def __init__(self, consensus_threshold: float = 0.7):
        """
        Initialize the consensus model.

        Args:
            consensus_threshold: Minimum agreement level for consensus (0-1)
        """
        self.consensus_threshold = consensus_threshold
        self.consensus_history = []

    def calculate_agreement_level(self, preferences: List[Dict[str, Any]]) -> float:
        """
        Calculate overall agreement level among agents.

        Args:
            preferences: List of agent preferences/rankings

        Returns:
            Agreement level (0-1, where 1 = perfect agreement)
        """
        if not preferences or len(preferences) < 2:
            return 1.0

        # Extract top choices from each agent
        top_choices = [
            pref.get('top_choice') or pref.get('proposed_action', {}).get('id')
            for pref in preferences
        ]

        # Remove None values
        top_choices = [c for c in top_choices if c is not None]

        if not top_choices:
            return 0.0

        # Calculate agreement as proportion of most common choice
        from collections import Counter
        choice_counts = Counter(top_choices)
        most_common_count = choice_counts.most_common(1)[0][1]

        agreement = most_common_count / len(top_choices)

        return agreement

    def detect_consensus(self, preferences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect if consensus exists among agents.

        Args:
            preferences: List of agent preferences

        Returns:
            Dictionary with consensus detection results
        """
        agreement_level = self.calculate_agreement_level(preferences)
        consensus_achieved = agreement_level >= self.consensus_threshold

        # Get the consensus choice if exists
        top_choices = [
            pref.get('top_choice') or pref.get('proposed_action', {}).get('id')
            for pref in preferences
        ]
        top_choices = [c for c in top_choices if c is not None]

        from collections import Counter
        if top_choices:
            choice_counts = Counter(top_choices)
            consensus_choice = choice_counts.most_common(1)[0][0]
            support_count = choice_counts[consensus_choice]
        else:
            consensus_choice = None
            support_count = 0

        result = {
            'consensus_achieved': consensus_achieved,
            'agreement_level': agreement_level,
            'consensus_threshold': self.consensus_threshold,
            'consensus_choice': consensus_choice,
            'support_count': support_count,
            'total_agents': len(preferences),
            'dissenting_agents': len(preferences) - support_count
        }

        self.consensus_history.append(result)

        return result

    def calculate_kendall_tau(self, ranking1: List[str], ranking2: List[str]) -> float:
        """
        Calculate Kendall's Tau correlation between two rankings.
        Measures the similarity between two ranking lists.

        Args:
            ranking1: First ranking list
            ranking2: Second ranking list

        Returns:
            Kendall's Tau value (-1 to 1, where 1 = identical rankings)
        """
        # Find common elements
        common = set(ranking1).intersection(set(ranking2))

        if len(common) < 2:
            return 0.0

        # Create filtered rankings with only common elements
        filtered1 = [item for item in ranking1 if item in common]
        filtered2 = [item for item in ranking2 if item in common]

        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0
        n = len(filtered1)

        for i in range(n):
            for j in range(i + 1, n):
                # Get positions in both rankings
                pos1_i = filtered1.index(filtered1[i])
                pos1_j = filtered1.index(filtered1[j])
                pos2_i = filtered2.index(filtered1[i])
                pos2_j = filtered2.index(filtered1[j])

                # Check if pair is concordant or discordant
                if (pos1_i < pos1_j and pos2_i < pos2_j) or (pos1_i > pos1_j and pos2_i > pos2_j):
                    concordant += 1
                else:
                    discordant += 1

        # Calculate Kendall's Tau
        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 0.0

        tau = (concordant - discordant) / total_pairs

        return tau

    def calculate_ranking_similarity(self, rankings: List[List[str]]) -> float:
        """
        Calculate average pairwise ranking similarity across all agents.

        Args:
            rankings: List of ranking lists from different agents

        Returns:
            Average similarity score (0-1)
        """
        if len(rankings) < 2:
            return 1.0

        similarities = []

        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                tau = self.calculate_kendall_tau(rankings[i], rankings[j])
                # Convert from [-1, 1] to [0, 1]
                similarity = (tau + 1) / 2
                similarities.append(similarity)

        return float(np.mean(similarities)) if similarities else 0.0

    def iterative_consensus_building(self, preferences: List[Dict[str, Any]],
                                   max_iterations: int = 5,
                                   convergence_rate: float = 0.3) -> Dict[str, Any]:
        """
        Simulate iterative consensus building where agents adjust their preferences.

        Args:
            preferences: Initial agent preferences
            max_iterations: Maximum number of iterations
            convergence_rate: Rate at which agents adjust toward group opinion

        Returns:
            Results of consensus building process
        """
        current_preferences = [p.copy() for p in preferences]
        iteration_history = []

        for iteration in range(max_iterations):
            # Detect current consensus
            consensus_result = self.detect_consensus(current_preferences)
            iteration_history.append({
                'iteration': iteration,
                'agreement_level': consensus_result['agreement_level'],
                'consensus_achieved': consensus_result['consensus_achieved']
            })

            # Check if consensus achieved
            if consensus_result['consensus_achieved']:
                break

            # Simulate preference adjustment
            # In real system, this would involve agent communication
            current_preferences = self._adjust_preferences(
                current_preferences,
                consensus_result['consensus_choice'],
                convergence_rate
            )

        final_consensus = self.detect_consensus(current_preferences)

        return {
            'initial_agreement': iteration_history[0]['agreement_level'],
            'final_agreement': final_consensus['agreement_level'],
            'consensus_achieved': final_consensus['consensus_achieved'],
            'iterations_required': len(iteration_history),
            'max_iterations': max_iterations,
            'iteration_history': iteration_history,
            'final_preferences': current_preferences
        }

    def _adjust_preferences(self, preferences: List[Dict[str, Any]],
                          consensus_choice: str,
                          convergence_rate: float) -> List[Dict[str, Any]]:
        """
        Simulate agents adjusting their preferences toward consensus.

        Args:
            preferences: Current preferences
            consensus_choice: The emerging consensus choice
            convergence_rate: Rate of adjustment

        Returns:
            Adjusted preferences
        """
        adjusted = []

        for pref in preferences:
            adjusted_pref = pref.copy()

            # Agents may adjust toward consensus based on convergence rate
            if np.random.random() < convergence_rate:
                # Agent moves toward consensus
                adjusted_pref['top_choice'] = consensus_choice
            else:
                # Agent maintains preference
                pass

            adjusted.append(adjusted_pref)

        return adjusted

    def identify_opinion_leaders(self, preferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify influential agents (opinion leaders) based on preference alignment.

        Args:
            preferences: Agent preferences with agent IDs

        Returns:
            List of agents ranked by influence
        """
        # Count how many agents support each agent's top choice
        influence_scores = []

        for i, pref in enumerate(preferences):
            agent_id = pref.get('agent_id', f'agent_{i}')
            agent_choice = pref.get('top_choice') or pref.get('proposed_action', {}).get('id')

            if agent_choice is None:
                continue

            # Count supporters
            supporters = sum(
                1 for other_pref in preferences
                if (other_pref.get('top_choice') or other_pref.get('proposed_action', {}).get('id')) == agent_choice
            )

            # Calculate influence (proportion of supporters)
            influence = supporters / len(preferences) if preferences else 0

            influence_scores.append({
                'agent_id': agent_id,
                'choice': agent_choice,
                'supporters': supporters,
                'influence_score': influence,
                'confidence': pref.get('confidence', 0.5)
            })

        # Sort by influence
        ranked = sorted(influence_scores, key=lambda x: x['influence_score'], reverse=True)

        return ranked

    def calculate_consensus_quality(self, final_decision: Dict[str, Any],
                                   preferences: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Assess the quality of the reached consensus.

        Args:
            final_decision: The final consensus decision
            preferences: Original agent preferences

        Returns:
            Dictionary with quality metrics
        """
        decision_id = final_decision.get('recommended_action_id') or final_decision.get('consensus_choice')

        # Calculate support level
        support_count = sum(
            1 for pref in preferences
            if (pref.get('top_choice') or pref.get('proposed_action', {}).get('id')) == decision_id
        )
        support_level = support_count / len(preferences) if preferences else 0

        # Calculate average confidence of supporters
        supporter_confidences = [
            pref.get('confidence', 0.5)
            for pref in preferences
            if (pref.get('top_choice') or pref.get('proposed_action', {}).get('id')) == decision_id
        ]
        avg_confidence = float(np.mean(supporter_confidences)) if supporter_confidences else 0

        # Calculate disagreement intensity (average confidence of dissenters)
        dissenter_confidences = [
            pref.get('confidence', 0.5)
            for pref in preferences
            if (pref.get('top_choice') or pref.get('proposed_action', {}).get('id')) != decision_id
        ]
        disagreement_intensity = float(np.mean(dissenter_confidences)) if dissenter_confidences else 0

        # Overall consensus quality
        quality = support_level * avg_confidence * (1 - 0.5 * disagreement_intensity)

        return {
            'support_level': support_level,
            'supporter_confidence': avg_confidence,
            'disagreement_intensity': disagreement_intensity,
            'consensus_quality': quality,
            'is_strong_consensus': quality >= 0.7
        }

    def get_consensus_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about consensus history.

        Returns:
            Dictionary with consensus statistics
        """
        if not self.consensus_history:
            return {
                'total_attempts': 0,
                'success_rate': 0.0,
                'average_agreement': 0.0
            }

        total_attempts = len(self.consensus_history)
        successes = sum(1 for result in self.consensus_history if result['consensus_achieved'])
        agreement_levels = [result['agreement_level'] for result in self.consensus_history]

        return {
            'total_attempts': total_attempts,
            'successes': successes,
            'success_rate': successes / total_attempts,
            'average_agreement': float(np.mean(agreement_levels)),
            'min_agreement': float(np.min(agreement_levels)),
            'max_agreement': float(np.max(agreement_levels))
        }
