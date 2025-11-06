"""
Expert Agent Implementation
Specialized agents with domain expertise (medical, logistics, public safety, etc.)
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .base_agent import BaseAgent


class ExpertAgent(BaseAgent):
    """
    Expert agent with specialized domain knowledge.
    Uses LLM-enhanced reasoning for domain-specific evaluations.
    """

    def __init__(self, agent_id: str, profile: Dict[str, Any], llm_client=None):
        """
        Initialize an expert agent.

        Args:
            agent_id: Unique identifier for the agent
            profile: Agent profile with expertise and preferences
            llm_client: Optional LLM client for enhanced reasoning
        """
        super().__init__(agent_id, 'expert', profile)
        self.llm_client = llm_client
        self.criteria_weights = profile.get('criteria_weights', {})
        self.risk_tolerance = profile.get('risk_tolerance', 0.5)

    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a crisis scenario from the expert's domain perspective.

        Args:
            scenario: Crisis scenario data

        Returns:
            Dictionary containing the expert's evaluation
        """
        scenario_type = scenario.get('type', 'unknown')
        severity = scenario.get('severity', 0.5)
        affected_population = scenario.get('affected_population', 0)

        # Domain-specific evaluation
        domain_relevance = self._calculate_domain_relevance(scenario)

        evaluation = {
            'agent_id': self.agent_id,
            'expertise_domain': self.expertise_domain,
            'scenario_type': scenario_type,
            'domain_relevance': domain_relevance,
            'severity_assessment': severity,
            'confidence': self.confidence_level * domain_relevance,
            'key_concerns': self._identify_key_concerns(scenario),
            'assessment': None  # Will be filled by LLM if available
        }

        # Use LLM for deeper analysis if available
        if self.llm_client:
            evaluation['assessment'] = self._llm_enhanced_evaluation(scenario)

        self.log_decision(evaluation)
        return evaluation

    def propose_action(self, scenario: Dict[str, Any],
                       criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose an action based on expert knowledge and MCDA.

        Args:
            scenario: Crisis scenario data
            criteria: Decision criteria and weights

        Returns:
            Dictionary containing proposed action and justification
        """
        # Extract available actions from scenario
        available_actions = scenario.get('available_actions', [])

        if not available_actions:
            return {
                'agent_id': self.agent_id,
                'proposed_action': None,
                'justification': 'No available actions to evaluate',
                'confidence': 0.0
            }

        # Evaluate each action against criteria
        action_scores = self._evaluate_actions(available_actions, criteria)

        # Select best action based on weighted scores
        best_action_idx = np.argmax(action_scores)
        best_action = available_actions[best_action_idx]
        best_score = action_scores[best_action_idx]

        proposal = {
            'agent_id': self.agent_id,
            'expertise_domain': self.expertise_domain,
            'proposed_action': best_action,
            'action_score': float(best_score),
            'all_scores': {action['id']: float(score)
                          for action, score in zip(available_actions, action_scores)},
            'justification': self._generate_justification(best_action, criteria),
            'confidence': self.confidence_level
        }

        # Use LLM for detailed justification if available
        if self.llm_client:
            proposal['detailed_justification'] = self._llm_enhanced_justification(
                scenario, best_action, criteria
            )

        self.log_decision(proposal)
        return proposal

    def _calculate_domain_relevance(self, scenario: Dict[str, Any]) -> float:
        """
        Calculate how relevant the scenario is to this expert's domain.

        Args:
            scenario: Crisis scenario data

        Returns:
            Relevance score between 0 and 1
        """
        scenario_tags = set(scenario.get('tags', []))
        expertise_tags = set(self.profile.get('expertise_tags', []))

        if not expertise_tags:
            return 0.5  # Default relevance

        overlap = len(scenario_tags.intersection(expertise_tags))
        relevance = overlap / len(expertise_tags) if expertise_tags else 0.5

        return min(1.0, relevance + 0.3)  # Minimum 30% relevance

    def _identify_key_concerns(self, scenario: Dict[str, Any]) -> List[str]:
        """
        Identify key concerns based on expert's domain.

        Args:
            scenario: Crisis scenario data

        Returns:
            List of key concern strings
        """
        concerns = []

        if self.expertise_domain == 'medical':
            if scenario.get('casualties', 0) > 0:
                concerns.append('Casualty management and triage')
            if scenario.get('affected_population', 0) > 1000:
                concerns.append('Public health response scaling')

        elif self.expertise_domain == 'logistics':
            if scenario.get('infrastructure_damage', False):
                concerns.append('Supply chain disruption')
            concerns.append('Resource allocation optimization')

        elif self.expertise_domain == 'public_safety':
            if scenario.get('severity', 0) > 0.7:
                concerns.append('Emergency evacuation procedures')
            concerns.append('Public communication strategy')

        elif self.expertise_domain == 'environmental':
            if scenario.get('type') in ['flood', 'wildfire', 'earthquake']:
                concerns.append('Environmental impact assessment')
                concerns.append('Long-term ecological effects')

        return concerns if concerns else ['General crisis response']

    def _evaluate_actions(self, actions: List[Dict[str, Any]],
                         criteria: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate all available actions against decision criteria.

        Args:
            actions: List of available actions
            criteria: Decision criteria and weights

        Returns:
            Numpy array of scores for each action
        """
        scores = []

        for action in actions:
            # Calculate weighted score based on criteria
            action_score = 0.0
            total_weight = 0.0

            for criterion, weight in self.criteria_weights.items():
                if criterion in action.get('criteria_scores', {}):
                    criterion_score = action['criteria_scores'][criterion]
                    action_score += criterion_score * weight
                    total_weight += weight

            # Normalize by total weight
            if total_weight > 0:
                action_score /= total_weight

            # Apply risk tolerance adjustment
            risk_level = action.get('risk_level', 0.5)
            risk_adjustment = 1.0 - abs(risk_level - self.risk_tolerance)
            action_score *= risk_adjustment

            scores.append(action_score)

        return np.array(scores)

    def _generate_justification(self, action: Dict[str, Any],
                               criteria: Dict[str, Any]) -> str:
        """
        Generate a justification for the proposed action.

        Args:
            action: Selected action
            criteria: Decision criteria

        Returns:
            Justification string
        """
        justification = (
            f"As a {self.expertise_domain} expert, I recommend '{action.get('name', action.get('id', 'this action'))}' "
            f"based on the following considerations:\n"
        )

        # Add criteria-based reasoning
        for criterion, weight in sorted(self.criteria_weights.items(),
                                       key=lambda x: x[1], reverse=True)[:3]:
            if criterion in action.get('criteria_scores', {}):
                score = action['criteria_scores'][criterion]
                justification += f"- {criterion.replace('_', ' ').title()}: Score {score:.2f} (weight: {weight:.2f})\n"

        return justification

    def _llm_enhanced_evaluation(self, scenario: Dict[str, Any]) -> str:
        """
        Use LLM to provide enhanced scenario evaluation.

        Args:
            scenario: Crisis scenario data

        Returns:
            LLM-generated assessment
        """
        # This will be implemented when integrated with llm_client
        # For now, return a placeholder
        return f"[LLM evaluation pending for {self.expertise_domain} domain]"

    def _llm_enhanced_justification(self, scenario: Dict[str, Any],
                                   action: Dict[str, Any],
                                   criteria: Dict[str, Any]) -> str:
        """
        Use LLM to provide detailed justification.

        Args:
            scenario: Crisis scenario data
            action: Selected action
            criteria: Decision criteria

        Returns:
            LLM-generated justification
        """
        # This will be implemented when integrated with llm_client
        return f"[Detailed LLM justification pending for {self.expertise_domain} domain]"
