"""
Base Agent Class
Defines the core structure and interface for all agents in the MAS
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the crisis management MAS.
    Provides common functionality and defines the interface for agent operations.
    """

    def __init__(self, agent_id: str, agent_type: str, profile: Dict[str, Any]):
        """
        Initialize a base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/role of the agent (e.g., 'expert', 'coordinator')
            profile: Agent profile containing expertise, weights, and preferences
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.profile = profile
        self.expertise_domain = profile.get('expertise_domain', 'general')
        self.confidence_level = profile.get('confidence_level', 0.7)
        self.decision_history: List[Dict[str, Any]] = []
        self.created_at = datetime.now()

    def get_profile(self) -> Dict[str, Any]:
        """Return agent profile information."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'expertise_domain': self.expertise_domain,
            'confidence_level': self.confidence_level,
            'profile': self.profile
        }

    def log_decision(self, decision: Dict[str, Any]):
        """
        Log a decision made by this agent.

        Args:
            decision: Dictionary containing decision details
        """
        decision_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'decision': decision
        }
        self.decision_history.append(decision_entry)

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return the agent's decision history."""
        return self.decision_history

    @abstractmethod
    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a crisis scenario and provide assessment.

        Args:
            scenario: Crisis scenario data

        Returns:
            Dictionary containing the agent's evaluation
        """
        pass

    @abstractmethod
    def propose_action(self, scenario: Dict[str, Any],
                       criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose an action based on the scenario and criteria.

        Args:
            scenario: Crisis scenario data
            criteria: Decision criteria and weights

        Returns:
            Dictionary containing proposed action and justification
        """
        pass

    def update_confidence(self, feedback: Dict[str, Any]):
        """
        Update agent's confidence level based on feedback.

        Args:
            feedback: Feedback data from previous decisions
        """
        if 'accuracy' in feedback:
            accuracy = feedback['accuracy']
            # Simple confidence adjustment based on accuracy
            self.confidence_level = 0.7 * self.confidence_level + 0.3 * accuracy
            self.confidence_level = max(0.1, min(1.0, self.confidence_level))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type}, expertise={self.expertise_domain})"
