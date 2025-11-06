"""
Coordinator Agent Implementation
Orchestrates expert agents, aggregates recommendations, and builds consensus
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .base_agent import BaseAgent


class CoordinatorAgent(BaseAgent):
    """
    Coordinator agent that manages the multi-agent decision-making process.
    Aggregates expert opinions and facilitates consensus building.
    """

    def __init__(self, agent_id: str, profile: Dict[str, Any],
                 consensus_threshold: float = 0.7):
        """
        Initialize a coordinator agent.

        Args:
            agent_id: Unique identifier for the coordinator
            profile: Agent profile
            consensus_threshold: Minimum agreement level for consensus (0-1)
        """
        super().__init__(agent_id, 'coordinator', profile)
        self.consensus_threshold = consensus_threshold
        self.expert_agents: List[BaseAgent] = []
        self.consensus_history: List[Dict[str, Any]] = []

    def register_expert(self, expert: BaseAgent):
        """
        Register an expert agent with the coordinator.

        Args:
            expert: Expert agent to register
        """
        self.expert_agents.append(expert)

    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate scenario evaluation across all experts.

        Args:
            scenario: Crisis scenario data

        Returns:
            Aggregated evaluation from all experts
        """
        if not self.expert_agents:
            return {
                'status': 'error',
                'message': 'No expert agents registered'
            }

        # Collect evaluations from all experts
        expert_evaluations = []
        for expert in self.expert_agents:
            try:
                evaluation = expert.evaluate_scenario(scenario)
                expert_evaluations.append(evaluation)
            except Exception as e:
                print(f"Error getting evaluation from {expert.agent_id}: {e}")

        # Aggregate evaluations
        aggregated = self._aggregate_evaluations(expert_evaluations)

        self.log_decision({
            'action': 'scenario_evaluation',
            'scenario_id': scenario.get('id', 'unknown'),
            'num_experts': len(expert_evaluations),
            'aggregated_result': aggregated
        })

        return aggregated

    def propose_action(self, scenario: Dict[str, Any],
                       criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate action proposal across all experts and build consensus.

        Args:
            scenario: Crisis scenario data
            criteria: Decision criteria and weights

        Returns:
            Consensus action proposal or multiple proposals if no consensus
        """
        if not self.expert_agents:
            return {
                'status': 'error',
                'message': 'No expert agents registered'
            }

        # Collect proposals from all experts
        expert_proposals = []
        for expert in self.expert_agents:
            try:
                proposal = expert.propose_action(scenario, criteria)
                expert_proposals.append(proposal)
            except Exception as e:
                print(f"Error getting proposal from {expert.agent_id}: {e}")

        # Attempt to build consensus
        consensus_result = self._build_consensus(expert_proposals, scenario, criteria)

        self.log_decision({
            'action': 'action_proposal',
            'scenario_id': scenario.get('id', 'unknown'),
            'num_proposals': len(expert_proposals),
            'consensus_achieved': consensus_result['consensus_achieved'],
            'final_recommendation': consensus_result.get('recommended_action')
        })

        self.consensus_history.append(consensus_result)
        return consensus_result

    def _aggregate_evaluations(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate scenario evaluations from multiple experts.

        Args:
            evaluations: List of expert evaluations

        Returns:
            Aggregated evaluation
        """
        if not evaluations:
            return {'status': 'error', 'message': 'No evaluations to aggregate'}

        # Calculate weighted averages based on confidence and domain relevance
        total_weight = 0.0
        weighted_severity = 0.0
        all_concerns = []

        for eval in evaluations:
            weight = eval.get('confidence', 0.5) * eval.get('domain_relevance', 0.5)
            total_weight += weight
            weighted_severity += eval.get('severity_assessment', 0.5) * weight
            all_concerns.extend(eval.get('key_concerns', []))

        avg_severity = weighted_severity / total_weight if total_weight > 0 else 0.5

        # Identify unique concerns
        unique_concerns = list(set(all_concerns))

        return {
            'aggregated_severity': avg_severity,
            'num_experts_consulted': len(evaluations),
            'key_concerns': unique_concerns,
            'expert_evaluations': evaluations,
            'aggregation_method': 'weighted_average',
            'total_confidence': total_weight / len(evaluations) if evaluations else 0
        }

    def _build_consensus(self, proposals: List[Dict[str, Any]],
                        scenario: Dict[str, Any],
                        criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build consensus from expert proposals using various strategies.

        Args:
            proposals: List of expert proposals
            scenario: Crisis scenario data
            criteria: Decision criteria

        Returns:
            Consensus result with recommended action
        """
        if not proposals:
            return {
                'consensus_achieved': False,
                'status': 'error',
                'message': 'No proposals to build consensus from'
            }

        # Count action preferences
        action_votes = {}
        action_weighted_votes = {}
        action_details = {}

        for proposal in proposals:
            action = proposal.get('proposed_action')
            if action is None:
                continue

            action_id = action.get('id', str(action))
            confidence = proposal.get('confidence', 0.5)

            # Simple voting
            action_votes[action_id] = action_votes.get(action_id, 0) + 1

            # Weighted voting by confidence
            action_weighted_votes[action_id] = (
                action_weighted_votes.get(action_id, 0.0) + confidence
            )

            # Store action details
            if action_id not in action_details:
                action_details[action_id] = action

        if not action_votes:
            return {
                'consensus_achieved': False,
                'status': 'no_valid_proposals',
                'message': 'No valid action proposals received'
            }

        # Determine consensus
        total_votes = len(proposals)
        top_action_id = max(action_votes, key=action_votes.get)
        top_votes = action_votes[top_action_id]
        consensus_level = top_votes / total_votes

        consensus_achieved = consensus_level >= self.consensus_threshold

        # Use evidential reasoning for more sophisticated aggregation
        aggregated_scores = self._aggregate_action_scores(proposals)

        result = {
            'consensus_achieved': consensus_achieved,
            'consensus_level': consensus_level,
            'consensus_threshold': self.consensus_threshold,
            'recommended_action': action_details.get(top_action_id),
            'recommended_action_id': top_action_id,
            'vote_distribution': action_votes,
            'weighted_votes': action_weighted_votes,
            'aggregated_scores': aggregated_scores,
            'expert_proposals': proposals,
            'total_experts': total_votes,
            'decision_strategy': 'majority_vote_with_confidence_weighting'
        }

        # If no consensus, provide alternatives
        if not consensus_achieved:
            sorted_actions = sorted(action_weighted_votes.items(),
                                   key=lambda x: x[1], reverse=True)
            result['alternative_actions'] = [
                {
                    'action_id': action_id,
                    'action': action_details.get(action_id),
                    'support_level': action_weighted_votes[action_id] / total_votes,
                    'votes': action_votes[action_id]
                }
                for action_id, _ in sorted_actions[:3]
            ]
            result['recommendation'] = 'No strong consensus - further deliberation recommended'

        return result

    def _aggregate_action_scores(self, proposals: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate action scores across all proposals.

        Args:
            proposals: List of expert proposals

        Returns:
            Dictionary mapping action IDs to aggregated scores
        """
        # Collect all action scores from proposals
        action_scores_map = {}

        for proposal in proposals:
            all_scores = proposal.get('all_scores', {})
            confidence = proposal.get('confidence', 0.5)

            for action_id, score in all_scores.items():
                if action_id not in action_scores_map:
                    action_scores_map[action_id] = []
                action_scores_map[action_id].append(score * confidence)

        # Compute average scores
        aggregated = {}
        for action_id, scores in action_scores_map.items():
            aggregated[action_id] = float(np.mean(scores))

        return aggregated

    def get_consensus_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about consensus-building performance.

        Returns:
            Dictionary with consensus statistics
        """
        if not self.consensus_history:
            return {
                'total_decisions': 0,
                'consensus_rate': 0.0,
                'average_consensus_level': 0.0
            }

        total_decisions = len(self.consensus_history)
        consensus_achieved_count = sum(
            1 for result in self.consensus_history
            if result.get('consensus_achieved', False)
        )

        consensus_levels = [
            result.get('consensus_level', 0.0)
            for result in self.consensus_history
        ]

        return {
            'total_decisions': total_decisions,
            'consensus_achieved_count': consensus_achieved_count,
            'consensus_rate': consensus_achieved_count / total_decisions,
            'average_consensus_level': float(np.mean(consensus_levels)),
            'min_consensus_level': float(np.min(consensus_levels)),
            'max_consensus_level': float(np.max(consensus_levels)),
            'threshold': self.consensus_threshold
        }

    def reset(self):
        """Reset coordinator state for new scenario."""
        self.expert_agents.clear()
        self.consensus_history.clear()
