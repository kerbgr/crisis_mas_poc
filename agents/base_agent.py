"""
Base Agent Class
Defines the core structure and interface for all agents in the MAS
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the crisis management MAS.
    Loads agent profiles from JSON configuration and provides common functionality.
    """

    def __init__(self, agent_id: str, profile_path: str = "agents/agent_profiles.json"):
        """
        Initialize a base agent by loading its profile from JSON.

        Args:
            agent_id: Unique identifier for the agent (must match ID in profile file)
            profile_path: Path to the agent profiles JSON file

        Raises:
            FileNotFoundError: If profile file doesn't exist
            ValueError: If agent_id is not found in profiles
            KeyError: If required fields are missing from profile
        """
        self.agent_id = agent_id
        self.profile_path = profile_path

        # Load the agent's profile
        profile = self.load_profile()

        # Set properties from profile with validation
        self.name: str = profile.get('name', f"Agent {agent_id}")
        self.role: str = profile.get('role', 'Unknown Role')
        self.expertise: str = profile.get('expertise', 'general')
        self.experience_years: int = profile.get('experience_years', 0)
        self.risk_tolerance: float = profile.get('risk_tolerance', 0.5)
        self.weight_preferences: Dict[str, float] = profile.get('weight_preferences', {})

        # Additional properties from profile
        self.description: str = profile.get('description', '')
        self.confidence_level: float = profile.get('confidence_level', 0.7)
        self.expertise_tags: List[str] = profile.get('expertise_tags', [])

        # Store the full profile for reference
        self._full_profile = profile

        # Runtime properties
        self.decision_history: List[Dict[str, Any]] = []
        self.created_at: datetime = datetime.now()

        # Validate weight preferences
        self._validate_weight_preferences()

    def load_profile(self) -> Dict[str, Any]:
        """
        Load agent profile from JSON file.

        Returns:
            Dictionary containing the agent's profile data

        Raises:
            FileNotFoundError: If the profile file doesn't exist
            ValueError: If agent_id is not found in the profiles
            json.JSONDecodeError: If the JSON file is malformed
        """
        # Convert to absolute path if relative
        profile_path = Path(self.profile_path)

        # Check if file exists
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Profile file not found: {profile_path}\n"
                f"Please ensure the file exists at the specified location."
            )

        try:
            # Load JSON file
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Find agent profile by agent_id
            agents = data.get('agents', [])

            for agent_profile in agents:
                if agent_profile.get('agent_id') == self.agent_id:
                    return agent_profile

            # Agent ID not found
            available_ids = [a.get('agent_id', 'unknown') for a in agents]
            raise ValueError(
                f"Agent ID '{self.agent_id}' not found in profile file.\n"
                f"Available agent IDs: {', '.join(available_ids)}"
            )

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Malformed JSON in profile file: {profile_path}",
                e.doc,
                e.pos
            )
        except Exception as e:
            raise Exception(f"Error loading profile for agent '{self.agent_id}': {str(e)}")

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the agent.

        Returns:
            Dictionary containing agent metadata including:
                - agent_id: Unique identifier
                - name: Agent's name
                - role: Agent's role
                - expertise: Area of expertise
                - experience_years: Years of experience
                - risk_tolerance: Risk tolerance level (0-1)
                - weight_preferences: Decision criteria weights
                - description: Agent description
                - confidence_level: Confidence level (0-1)
                - created_at: Timestamp when agent was created
                - decisions_made: Number of decisions in history
        """
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'role': self.role,
            'expertise': self.expertise,
            'experience_years': self.experience_years,
            'risk_tolerance': self.risk_tolerance,
            'weight_preferences': self.weight_preferences,
            'description': self.description,
            'confidence_level': self.confidence_level,
            'expertise_tags': self.expertise_tags,
            'created_at': self.created_at.isoformat(),
            'decisions_made': len(self.decision_history)
        }

    def _validate_weight_preferences(self):
        """
        Validate that weight preferences are properly formatted.

        Raises:
            ValueError: If weights are invalid
        """
        if not self.weight_preferences:
            return  # Empty weights are allowed

        # Check that all weights are between 0 and 1
        for criterion, weight in self.weight_preferences.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(
                    f"Invalid weight for criterion '{criterion}': {weight}. "
                    f"Weights must be numeric."
                )
            if not 0 <= weight <= 1:
                raise ValueError(
                    f"Invalid weight for criterion '{criterion}': {weight}. "
                    f"Weights must be between 0 and 1."
                )

        # Check if weights sum to approximately 1 (allow small floating point errors)
        total_weight = sum(self.weight_preferences.values())
        if not (0.99 <= total_weight <= 1.01):
            print(f"Warning: Agent '{self.agent_id}' weight preferences sum to {total_weight:.3f}, "
                  f"expected ~1.0")

    def get_profile(self) -> Dict[str, Any]:
        """
        Get the agent's complete profile.

        Returns:
            Dictionary containing the full agent profile
        """
        return self._full_profile.copy()

    def log_decision(self, decision: Dict[str, Any]):
        """
        Log a decision made by this agent.

        Args:
            decision: Dictionary containing decision details
        """
        decision_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'decision': decision
        }
        self.decision_history.append(decision_entry)

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """
        Get the agent's complete decision history.

        Returns:
            List of decision entries with timestamps
        """
        return self.decision_history.copy()

    def update_confidence(self, feedback: Dict[str, Any]):
        """
        Update agent's confidence level based on feedback.

        Args:
            feedback: Dictionary containing feedback data with 'accuracy' key
        """
        if 'accuracy' in feedback:
            accuracy = feedback['accuracy']
            # Weighted average: 70% old confidence, 30% new accuracy
            self.confidence_level = 0.7 * self.confidence_level + 0.3 * accuracy
            # Ensure confidence stays within bounds
            self.confidence_level = max(0.1, min(1.0, self.confidence_level))

    @abstractmethod
    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a crisis scenario and provide assessment.
        This method must be implemented by all subclasses.

        Args:
            scenario: Dictionary containing crisis scenario data

        Returns:
            Dictionary containing the agent's evaluation

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"Agent class {self.__class__.__name__} must implement evaluate_scenario()"
        )

    @abstractmethod
    def propose_action(self, scenario: Dict[str, Any],
                       criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose an action based on the scenario and criteria.
        This method must be implemented by all subclasses.

        Args:
            scenario: Dictionary containing crisis scenario data
            criteria: Dictionary containing decision criteria and weights

        Returns:
            Dictionary containing proposed action and justification

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"Agent class {self.__class__.__name__} must implement propose_action()"
        )

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"id='{self.agent_id}', "
            f"name='{self.name}', "
            f"role='{self.role}', "
            f"expertise='{self.expertise}')"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"{self.name} ({self.role})\n"
            f"  Expertise: {self.expertise}\n"
            f"  Experience: {self.experience_years} years\n"
            f"  Risk Tolerance: {self.risk_tolerance:.2f}\n"
            f"  Confidence: {self.confidence_level:.2f}"
        )


def load_agent_by_id(agent_id: str, profile_path: str = "agents/agent_profiles.json") -> Dict[str, Any]:
    """
    Utility function to load an agent profile by ID without instantiating an agent.

    Args:
        agent_id: Unique identifier for the agent
        profile_path: Path to the agent profiles JSON file

    Returns:
        Dictionary containing the agent's profile

    Raises:
        FileNotFoundError: If profile file doesn't exist
        ValueError: If agent_id is not found
    """
    profile_path = Path(profile_path)

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    with open(profile_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    agents = data.get('agents', [])

    for agent_profile in agents:
        if agent_profile.get('agent_id') == agent_id:
            return agent_profile

    available_ids = [a.get('agent_id', 'unknown') for a in agents]
    raise ValueError(
        f"Agent ID '{agent_id}' not found in profile file.\n"
        f"Available agent IDs: {', '.join(available_ids)}"
    )


def list_available_agents(profile_path: str = "agents/agent_profiles.json") -> List[Dict[str, str]]:
    """
    List all available agents in the profile file.

    Args:
        profile_path: Path to the agent profiles JSON file

    Returns:
        List of dictionaries with agent_id, name, and role

    Raises:
        FileNotFoundError: If profile file doesn't exist
    """
    profile_path = Path(profile_path)

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    with open(profile_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    agents = data.get('agents', [])

    return [
        {
            'agent_id': agent.get('agent_id', 'unknown'),
            'name': agent.get('name', 'Unknown'),
            'role': agent.get('role', 'Unknown'),
            'expertise': agent.get('expertise', 'general')
        }
        for agent in agents
    ]
