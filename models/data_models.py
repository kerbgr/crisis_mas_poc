"""
Pydantic data models for Crisis MAS.

This module provides type-safe, validated data structures for the multi-agent system.
All models use Pydantic for runtime validation and type checking.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class BeliefDistribution(BaseModel):
    """
    Validated belief distribution over alternatives.

    Ensures:
    - All values are floats between 0.0 and 1.0
    - Sum of all beliefs is approximately 1.0 (within tolerance)
    - Alternative IDs are non-empty strings
    """

    beliefs: Dict[str, float] = Field(
        description="Mapping of alternative_id to belief value [0.0, 1.0]"
    )

    @field_validator('beliefs')
    @classmethod
    def validate_beliefs(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate belief values are in [0.0, 1.0] range."""
        if not v:
            raise ValueError("Belief distribution cannot be empty")

        for alt_id, belief in v.items():
            if not alt_id or not isinstance(alt_id, str):
                raise ValueError(f"Alternative ID must be a non-empty string, got: {alt_id}")

            if not isinstance(belief, (float, int)):
                raise ValueError(
                    f"Belief for '{alt_id}' must be a number, got {type(belief).__name__}: {belief}"
                )

            if not (0.0 <= belief <= 1.0):
                raise ValueError(
                    f"Belief for '{alt_id}' must be in [0.0, 1.0], got: {belief}"
                )

        return v

    @model_validator(mode='after')
    def validate_sum(self) -> 'BeliefDistribution':
        """Validate that beliefs sum to approximately 1.0."""
        total = sum(self.beliefs.values())
        tolerance = 0.01

        if abs(total - 1.0) > tolerance:
            logger.warning(
                f"Belief distribution sum is {total:.4f}, expected ~1.0. "
                f"Normalizing beliefs."
            )
            # Normalize beliefs to sum to 1.0
            if total > 0:
                self.beliefs = {
                    alt_id: belief / total
                    for alt_id, belief in self.beliefs.items()
                }

        return self

    def to_dict(self) -> Dict[str, float]:
        """Convert to plain dictionary for backward compatibility."""
        return self.beliefs

    def __getitem__(self, key: str) -> float:
        """Allow dictionary-style access."""
        return self.beliefs[key]

    def __setitem__(self, key: str, value: float) -> None:
        """Allow dictionary-style assignment."""
        self.beliefs[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.beliefs

    def get(self, key: str, default: float = 0.0) -> float:
        """Allow dict.get() style access."""
        return self.beliefs.get(key, default)

    def keys(self):
        """Allow dict.keys() style access."""
        return self.beliefs.keys()

    def values(self):
        """Allow dict.values() style access."""
        return self.beliefs.values()

    def items(self):
        """Allow dict.items() style access."""
        return self.beliefs.items()

    def __len__(self) -> int:
        """Return the number of alternatives in the distribution."""
        return len(self.beliefs)

    def __bool__(self) -> bool:
        """Return False if beliefs dictionary is empty, True otherwise."""
        return bool(self.beliefs)


class Alternative(BaseModel):
    """
    A decision alternative with associated data.
    """

    alternative_id: str = Field(description="Unique identifier for the alternative")
    name: str = Field(description="Human-readable name")
    description: Optional[str] = Field(default=None, description="Detailed description")

    @field_validator('alternative_id', 'name')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure strings are non-empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class AlternativeRanking(BaseModel):
    """
    Ranking of an alternative with score and reasoning.
    """

    alternative_id: str = Field(description="ID of the alternative being ranked")
    rank: int = Field(ge=1, description="Rank position (1 = highest)")
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized score")
    reasoning: Optional[str] = Field(default=None, description="Why this rank was assigned")

    @field_validator('alternative_id')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure alternative_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("alternative_id cannot be empty")
        return v.strip()


class AgentAssessment(BaseModel):
    """
    Complete assessment from an expert agent.

    This is the primary data structure returned by ExpertAgent.evaluate_scenario()
    """

    agent_id: str = Field(description="Unique identifier of the agent")
    agent_name: str = Field(description="Human-readable agent name")
    role: str = Field(description="Agent's expertise/role")

    belief_distribution: BeliefDistribution = Field(
        description="Validated belief distribution over alternatives"
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Agent's confidence in their assessment [0.0, 1.0]"
    )

    reasoning: str = Field(description="Detailed reasoning for the assessment")

    key_concerns: List[str] = Field(
        default_factory=list,
        description="List of main concerns identified"
    )

    recommended_actions: List[str] = Field(
        default_factory=list,
        description="List of recommended actions"
    )

    risk_assessment: Optional[str] = Field(
        default=None,
        description="Risk level assessment (e.g., 'high', 'medium', 'low')"
    )

    timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp of assessment"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @field_validator('agent_id', 'agent_name', 'role', 'reasoning')
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Ensure critical string fields are non-empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get attribute value with default (dict-like interface).

        Provides backward compatibility with code expecting dict.get().
        """
        try:
            return getattr(self, key, default)
        except AttributeError:
            return default

    def __getitem__(self, key: str) -> Any:
        """
        Get attribute value (dict-like interface).

        Provides backward compatibility with code expecting dict[key].
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' not found in AgentAssessment")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set attribute value (dict-like interface).

        Provides backward compatibility with code expecting dict[key] = value.
        """
        setattr(self, key, value)

    def keys(self):
        """Return keys (dict-like interface)."""
        return self.model_dump().keys()

    def values(self):
        """Return values (dict-like interface)."""
        return self.model_dump().values()

    def items(self):
        """Return items (dict-like interface)."""
        return self.model_dump().items()

    def __contains__(self, key: str) -> bool:
        """Check if key exists (dict-like interface)."""
        return hasattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to plain dictionary for backward compatibility.

        This allows existing code to continue using dict-style access.
        """
        data = self.model_dump()
        # Convert nested BeliefDistribution to plain dict
        data['belief_distribution'] = self.belief_distribution.to_dict()
        return data

    model_config = {
        "extra": "allow",  # Allow additional fields for backward compatibility
        "validate_assignment": True  # Validate on attribute assignment
    }


class LLMResponse(BaseModel):
    """
    Validated response from an LLM client.

    This structure is returned by parse_json_response() in LLM clients.
    Matches the actual format used by Claude, OpenAI, and LM Studio clients.
    """

    alternative_rankings: Dict[str, float] = Field(
        default_factory=dict,
        description="Alternative IDs mapped to scores (0.0-1.0)"
    )

    reasoning: str = Field(description="Reasoning for the rankings")

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the response [0.0, 1.0]"
    )

    key_concerns: List[str] = Field(
        default_factory=list,
        description="Main concerns identified"
    )

    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )

    risk_assessment: Optional[str] = Field(
        default=None,
        description="Risk level (high/medium/low)"
    )

    raw_response: Optional[str] = Field(
        default=None,
        description="Original LLM response text"
    )

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v: str) -> str:
        """Ensure reasoning is non-empty."""
        if not v or not v.strip():
            raise ValueError("Reasoning cannot be empty")
        return v.strip()

    @field_validator('alternative_rankings')
    @classmethod
    def validate_rankings(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate ranking scores are in valid range."""
        if not v:
            return v

        for alt_id, score in v.items():
            if not isinstance(score, (int, float)):
                raise ValueError(f"Score for {alt_id} must be numeric, got {type(score).__name__}")
            if not (0.0 <= score <= 1.0):
                logger.warning(f"Score {score} for {alt_id} outside [0.0, 1.0] range")

        return v

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get attribute value with default (dict-like interface).

        Provides backward compatibility with code expecting dict.get().
        """
        try:
            return getattr(self, key, default)
        except AttributeError:
            return default

    def __getitem__(self, key: str) -> Any:
        """
        Get attribute value (dict-like interface).

        Provides backward compatibility with code expecting dict[key].
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' not found in LLMResponse")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set attribute value (dict-like interface).

        Provides backward compatibility with code expecting dict[key] = value.
        Uses setattr to allow setting extra fields dynamically.
        """
        setattr(self, key, value)

    def keys(self):
        """
        Return keys (dict-like interface).

        Provides backward compatibility with code expecting dict.keys().
        """
        return self.model_dump().keys()

    def values(self):
        """
        Return values (dict-like interface).

        Provides backward compatibility with code expecting dict.values().
        """
        return self.model_dump().values()

    def items(self):
        """
        Return items (dict-like interface).

        Provides backward compatibility with code expecting dict.items().
        """
        return self.model_dump().items()

    def __contains__(self, key: str) -> bool:
        """
        Check if key exists (dict-like interface).

        Provides backward compatibility with 'key in response' checks.
        """
        return hasattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary."""
        return self.model_dump()

    model_config = {
        "extra": "allow",
        "validate_assignment": True
    }


class ConflictResolution(BaseModel):
    """
    Structured conflict resolution recommendation.

    Returned by ConsensusModel.suggest_resolution()
    """

    resolution_strategy: str = Field(
        description="Strategy to resolve conflict (e.g., 'weighted_aggregation', 'compromise', 'escalation_required')"
    )

    conflicting_agents: List[str] = Field(
        default_factory=list,
        description="IDs of agents in conflict"
    )

    compromise_alternatives: List[str] = Field(
        default_factory=list,
        description="Alternative IDs that could serve as compromises"
    )

    suggested_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions to resolve the conflict"
    )

    rationale: str = Field(
        description="Explanation of why this resolution is suggested"
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Confidence in the resolution recommendation"
    )

    requires_human_intervention: bool = Field(
        default=False,
        description="Whether human decision-making is required"
    )

    @field_validator('resolution_strategy', 'rationale')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure critical fields are non-empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary."""
        return self.model_dump()

    model_config = {
        "extra": "allow",
        "validate_assignment": True
    }


# Utility functions for validation and conversion

def validate_belief_distribution(data: Union[Dict[str, float], BeliefDistribution]) -> BeliefDistribution:
    """
    Validate and convert belief distribution data.

    Args:
        data: Either a dict or BeliefDistribution instance

    Returns:
        Validated BeliefDistribution instance

    Raises:
        ValueError: If validation fails
    """
    if isinstance(data, BeliefDistribution):
        return data

    if isinstance(data, dict):
        try:
            return BeliefDistribution(beliefs=data)
        except Exception as e:
            logger.error(f"Failed to validate belief distribution: {e}")
            raise ValueError(f"Invalid belief distribution: {e}")

    raise ValueError(f"Expected dict or BeliefDistribution, got {type(data).__name__}")


def validate_agent_assessment(data: Union[Dict[str, Any], AgentAssessment]) -> AgentAssessment:
    """
    Validate and convert agent assessment data.

    Args:
        data: Either a dict or AgentAssessment instance

    Returns:
        Validated AgentAssessment instance

    Raises:
        ValueError: If validation fails
    """
    if isinstance(data, AgentAssessment):
        return data

    if isinstance(data, dict):
        try:
            # Convert belief_distribution if it's a plain dict
            if 'belief_distribution' in data and isinstance(data['belief_distribution'], dict):
                data = data.copy()
                data['belief_distribution'] = BeliefDistribution(beliefs=data['belief_distribution'])

            return AgentAssessment(**data)
        except Exception as e:
            logger.error(f"Failed to validate agent assessment: {e}")
            raise ValueError(f"Invalid agent assessment: {e}")

    raise ValueError(f"Expected dict or AgentAssessment, got {type(data).__name__}")


def safe_to_dict(obj: Union[BaseModel, Dict, Any]) -> Dict[str, Any]:
    """
    Safely convert Pydantic models or dicts to plain dicts.

    This is useful for backward compatibility with code expecting plain dicts.

    Args:
        obj: Pydantic model, dict, or other object

    Returns:
        Plain dictionary representation
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return obj
    else:
        logger.warning(f"Unexpected type in safe_to_dict: {type(obj).__name__}")
        return {}
