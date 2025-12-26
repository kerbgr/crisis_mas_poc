"""
Unit tests for Pydantic data models.

Tests validation, edge cases, and backward compatibility.
"""

import pytest
from pydantic import ValidationError
from models.data_models import (
    BeliefDistribution,
    AgentAssessment,
    LLMResponse,
    ConflictResolution,
    Alternative,
    AlternativeRanking,
    validate_belief_distribution,
    validate_agent_assessment,
    safe_to_dict
)


class TestBeliefDistribution:
    """Test BeliefDistribution model."""

    def test_valid_belief_distribution(self):
        """Test creating a valid belief distribution."""
        beliefs = {"alt1": 0.6, "alt2": 0.4}
        bd = BeliefDistribution(beliefs=beliefs)

        assert bd.beliefs == beliefs
        assert bd["alt1"] == 0.6
        assert bd.get("alt2") == 0.4
        assert bd.get("alt3", 0.0) == 0.0

    def test_belief_distribution_normalization(self):
        """Test automatic normalization when sum != 1.0."""
        beliefs = {"alt1": 0.6, "alt2": 0.5}  # Sum = 1.1
        bd = BeliefDistribution(beliefs=beliefs)

        # Should be normalized
        assert abs(sum(bd.beliefs.values()) - 1.0) < 0.001
        assert abs(bd["alt1"] - 0.545) < 0.01  # 0.6/1.1 ≈ 0.545
        assert abs(bd["alt2"] - 0.455) < 0.01  # 0.5/1.1 ≈ 0.455

    def test_belief_distribution_dict_access(self):
        """Test dict-like access methods."""
        beliefs = {"alt1": 0.7, "alt2": 0.3}
        bd = BeliefDistribution(beliefs=beliefs)

        # Test dict-like access
        assert "alt1" in bd.keys()
        assert 0.7 in bd.values()
        assert ("alt1", 0.7) in bd.items()
        assert bd.to_dict() == beliefs

    def test_empty_belief_distribution(self):
        """Test that empty belief distribution raises error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            BeliefDistribution(beliefs={})

    def test_invalid_belief_value_type(self):
        """Test that non-numeric belief values raise error."""
        with pytest.raises(ValidationError):
            # Pydantic v2 raises ValidationError with "unable to parse string as a number"
            BeliefDistribution(beliefs={"alt1": "high"})

    def test_belief_value_out_of_range(self):
        """Test that belief values outside [0, 1] raise error."""
        with pytest.raises(ValidationError, match="must be in"):
            BeliefDistribution(beliefs={"alt1": 1.5})

        with pytest.raises(ValidationError, match="must be in"):
            BeliefDistribution(beliefs={"alt1": -0.1})

    def test_empty_alternative_id(self):
        """Test that empty alternative IDs raise error."""
        with pytest.raises(ValidationError, match="non-empty string"):
            BeliefDistribution(beliefs={"": 1.0})

    def test_integer_beliefs(self):
        """Test that integer belief values are accepted."""
        bd = BeliefDistribution(beliefs={"alt1": 1, "alt2": 0})
        assert bd["alt1"] == 1.0
        assert bd["alt2"] == 0.0


class TestAlternative:
    """Test Alternative model."""

    def test_valid_alternative(self):
        """Test creating a valid alternative."""
        alt = Alternative(
            alternative_id="alt_001",
            name="Evacuate Immediately",
            description="Full evacuation of the area"
        )

        assert alt.alternative_id == "alt_001"
        assert alt.name == "Evacuate Immediately"
        assert alt.description == "Full evacuation of the area"

    def test_alternative_without_description(self):
        """Test alternative without optional description."""
        alt = Alternative(alternative_id="alt_001", name="Evacuate")
        assert alt.description is None

    def test_empty_alternative_id(self):
        """Test that empty alternative_id raises error."""
        with pytest.raises(ValidationError):
            Alternative(alternative_id="", name="Test")

    def test_empty_name(self):
        """Test that empty name raises error."""
        with pytest.raises(ValidationError):
            Alternative(alternative_id="alt_001", name="")

    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed."""
        alt = Alternative(alternative_id="  alt_001  ", name="  Test  ")
        assert alt.alternative_id == "alt_001"
        assert alt.name == "Test"


class TestAlternativeRanking:
    """Test AlternativeRanking model."""

    def test_valid_ranking(self):
        """Test creating a valid ranking."""
        ranking = AlternativeRanking(
            alternative_id="alt_001",
            rank=1,
            score=0.95,
            reasoning="Best option due to safety"
        )

        assert ranking.alternative_id == "alt_001"
        assert ranking.rank == 1
        assert ranking.score == 0.95

    def test_ranking_without_optional_fields(self):
        """Test ranking with only required fields."""
        ranking = AlternativeRanking(alternative_id="alt_001", rank=1)
        assert ranking.score is None
        assert ranking.reasoning is None

    def test_invalid_rank(self):
        """Test that rank < 1 raises error."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            AlternativeRanking(alternative_id="alt_001", rank=0)

    def test_invalid_score_range(self):
        """Test that score outside [0, 1] raises error."""
        with pytest.raises(ValidationError):
            AlternativeRanking(alternative_id="alt_001", rank=1, score=1.5)

        with pytest.raises(ValidationError):
            AlternativeRanking(alternative_id="alt_001", rank=1, score=-0.1)


class TestAgentAssessment:
    """Test AgentAssessment model."""

    def test_valid_assessment(self):
        """Test creating a valid agent assessment."""
        assessment = AgentAssessment(
            agent_id="agent_001",
            agent_name="Medical Expert",
            role="Medical Infrastructure Commander",
            belief_distribution=BeliefDistribution(beliefs={"alt1": 0.6, "alt2": 0.4}),
            confidence=0.85,
            reasoning="Based on medical capacity analysis",
            key_concerns=["Hospital capacity", "Medical supplies"],
            recommended_actions=["Activate emergency protocols"]
        )

        assert assessment.agent_id == "agent_001"
        assert assessment.confidence == 0.85
        assert len(assessment.key_concerns) == 2
        assert isinstance(assessment.belief_distribution, BeliefDistribution)

    def test_assessment_with_dict_belief_distribution(self):
        """Test that dict beliefs are converted to BeliefDistribution."""
        assessment = AgentAssessment(
            agent_id="agent_001",
            agent_name="Expert",
            role="Commander",
            belief_distribution=BeliefDistribution(beliefs={"alt1": 1.0}),
            confidence=0.8,
            reasoning="Test"
        )

        assert isinstance(assessment.belief_distribution, BeliefDistribution)
        assert assessment.belief_distribution["alt1"] == 1.0

    def test_assessment_to_dict(self):
        """Test converting assessment to dict."""
        assessment = AgentAssessment(
            agent_id="agent_001",
            agent_name="Expert",
            role="Commander",
            belief_distribution=BeliefDistribution(beliefs={"alt1": 0.7, "alt2": 0.3}),
            confidence=0.8,
            reasoning="Test"
        )

        data = assessment.to_dict()
        assert isinstance(data, dict)
        assert isinstance(data['belief_distribution'], dict)
        assert data['belief_distribution']['alt1'] == 0.7

    def test_invalid_confidence_range(self):
        """Test that confidence outside [0, 1] raises error."""
        with pytest.raises(ValidationError):
            AgentAssessment(
                agent_id="agent_001",
                agent_name="Expert",
                role="Commander",
                belief_distribution=BeliefDistribution(beliefs={"alt1": 1.0}),
                confidence=1.5,
                reasoning="Test"
            )

    def test_empty_required_fields(self):
        """Test that empty required string fields raise error."""
        with pytest.raises(ValidationError):
            AgentAssessment(
                agent_id="",
                agent_name="Expert",
                role="Commander",
                belief_distribution=BeliefDistribution(beliefs={"alt1": 1.0}),
                confidence=0.8,
                reasoning="Test"
            )

    def test_assessment_with_metadata(self):
        """Test assessment with additional metadata."""
        assessment = AgentAssessment(
            agent_id="agent_001",
            agent_name="Expert",
            role="Commander",
            belief_distribution=BeliefDistribution(beliefs={"alt1": 1.0}),
            confidence=0.8,
            reasoning="Test",
            metadata={"timestamp": "2025-01-01", "version": "1.0"}
        )

        assert assessment.metadata["timestamp"] == "2025-01-01"


class TestLLMResponse:
    """Test LLMResponse model."""

    def test_valid_llm_response(self):
        """Test creating a valid LLM response."""
        response = LLMResponse(
            alternative_rankings=[
                AlternativeRanking(alternative_id="alt1", rank=1, score=0.9),
                AlternativeRanking(alternative_id="alt2", rank=2, score=0.7)
            ],
            reasoning="Alt1 is better due to safety",
            confidence=0.85,
            key_concerns=["Safety", "Cost"],
            recommended_actions=["Proceed with alt1"]
        )

        assert len(response.alternative_rankings) == 2
        assert response.confidence == 0.85
        assert response.alternative_rankings[0].rank == 1

    def test_llm_response_without_rankings(self):
        """Test LLM response without rankings."""
        response = LLMResponse(
            reasoning="Unable to rank alternatives",
            confidence=0.5
        )

        assert len(response.alternative_rankings) == 0

    def test_duplicate_alternative_ids(self):
        """Test that duplicate alternative IDs raise error."""
        with pytest.raises(ValidationError, match="Duplicate alternative_id"):
            LLMResponse(
                alternative_rankings=[
                    AlternativeRanking(alternative_id="alt1", rank=1),
                    AlternativeRanking(alternative_id="alt1", rank=2)
                ],
                reasoning="Test",
                confidence=0.8
            )

    def test_empty_reasoning(self):
        """Test that empty reasoning raises error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            LLMResponse(
                reasoning="",
                confidence=0.8
            )

    def test_llm_response_to_dict(self):
        """Test converting LLM response to dict."""
        response = LLMResponse(
            alternative_rankings=[AlternativeRanking(alternative_id="alt1", rank=1)],
            reasoning="Test",
            confidence=0.8
        )

        data = response.to_dict()
        assert isinstance(data, dict)
        assert isinstance(data['alternative_rankings'], list)


class TestConflictResolution:
    """Test ConflictResolution model."""

    def test_valid_conflict_resolution(self):
        """Test creating a valid conflict resolution."""
        resolution = ConflictResolution(
            resolution_strategy="weighted_aggregation",
            conflicting_agents=["agent_001", "agent_002"],
            compromise_alternatives=["alt_middle"],
            suggested_actions=["Use weighted voting", "Consult human"],
            rationale="Agents disagree on risk assessment",
            confidence=0.75,
            requires_human_intervention=False
        )

        assert resolution.resolution_strategy == "weighted_aggregation"
        assert len(resolution.conflicting_agents) == 2
        assert resolution.confidence == 0.75
        assert not resolution.requires_human_intervention

    def test_conflict_resolution_requires_escalation(self):
        """Test conflict resolution requiring escalation."""
        resolution = ConflictResolution(
            resolution_strategy="escalation_required",
            suggested_actions=["Escalate to human decision-maker"],
            rationale="Agents have fundamentally different risk assessments",
            requires_human_intervention=True
        )

        assert resolution.requires_human_intervention

    def test_empty_strategy(self):
        """Test that empty strategy raises error."""
        with pytest.raises(ValidationError):
            ConflictResolution(
                resolution_strategy="",
                rationale="Test"
            )

    def test_empty_rationale(self):
        """Test that empty rationale raises error."""
        with pytest.raises(ValidationError):
            ConflictResolution(
                resolution_strategy="compromise",
                rationale=""
            )

    def test_conflict_resolution_to_dict(self):
        """Test converting conflict resolution to dict."""
        resolution = ConflictResolution(
            resolution_strategy="compromise",
            rationale="Best middle ground"
        )

        data = resolution.to_dict()
        assert isinstance(data, dict)
        assert data['resolution_strategy'] == "compromise"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_belief_distribution_with_dict(self):
        """Test validate_belief_distribution with dict input."""
        beliefs = {"alt1": 0.6, "alt2": 0.4}
        bd = validate_belief_distribution(beliefs)

        assert isinstance(bd, BeliefDistribution)
        assert bd["alt1"] == 0.6

    def test_validate_belief_distribution_with_model(self):
        """Test validate_belief_distribution with BeliefDistribution input."""
        bd_input = BeliefDistribution(beliefs={"alt1": 1.0})
        bd_output = validate_belief_distribution(bd_input)

        assert bd_output is bd_input

    def test_validate_belief_distribution_with_invalid_type(self):
        """Test validate_belief_distribution with invalid input."""
        with pytest.raises(ValueError, match="Expected dict or BeliefDistribution"):
            validate_belief_distribution("invalid")

    def test_validate_agent_assessment_with_dict(self):
        """Test validate_agent_assessment with dict input."""
        data = {
            "agent_id": "agent_001",
            "agent_name": "Expert",
            "role": "Commander",
            "belief_distribution": {"alt1": 1.0},
            "confidence": 0.8,
            "reasoning": "Test"
        }
        assessment = validate_agent_assessment(data)

        assert isinstance(assessment, AgentAssessment)
        assert isinstance(assessment.belief_distribution, BeliefDistribution)

    def test_validate_agent_assessment_with_model(self):
        """Test validate_agent_assessment with AgentAssessment input."""
        assessment_input = AgentAssessment(
            agent_id="agent_001",
            agent_name="Expert",
            role="Commander",
            belief_distribution=BeliefDistribution(beliefs={"alt1": 1.0}),
            confidence=0.8,
            reasoning="Test"
        )
        assessment_output = validate_agent_assessment(assessment_input)

        assert assessment_output is assessment_input

    def test_safe_to_dict_with_model(self):
        """Test safe_to_dict with Pydantic model."""
        bd = BeliefDistribution(beliefs={"alt1": 0.6, "alt2": 0.4})
        data = safe_to_dict(bd)

        assert isinstance(data, dict)
        assert "beliefs" in data

    def test_safe_to_dict_with_dict(self):
        """Test safe_to_dict with plain dict."""
        input_dict = {"key": "value"}
        output_dict = safe_to_dict(input_dict)

        assert output_dict is input_dict

    def test_safe_to_dict_with_other_type(self):
        """Test safe_to_dict with unexpected type."""
        result = safe_to_dict("string")
        assert result == {}


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_belief_distribution_as_dict(self):
        """Test that BeliefDistribution can be used like a dict."""
        bd = BeliefDistribution(beliefs={"alt1": 0.7, "alt2": 0.3})

        # Dict-style iteration
        for alt_id in bd.keys():
            assert alt_id in ["alt1", "alt2"]

        # Dict-style unpacking
        plain_dict = {**bd.beliefs}
        assert plain_dict == {"alt1": 0.7, "alt2": 0.3}

    def test_assessment_extra_fields_allowed(self):
        """Test that AgentAssessment allows extra fields."""
        assessment = AgentAssessment(
            agent_id="agent_001",
            agent_name="Expert",
            role="Commander",
            belief_distribution=BeliefDistribution(beliefs={"alt1": 1.0}),
            confidence=0.8,
            reasoning="Test",
            custom_field="custom_value"  # Extra field
        )

        # Extra field should be preserved
        data = assessment.model_dump()
        assert data.get('custom_field') == "custom_value"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_alternative_belief_distribution(self):
        """Test belief distribution with single alternative."""
        bd = BeliefDistribution(beliefs={"alt1": 1.0})
        assert bd["alt1"] == 1.0

    def test_many_alternatives_belief_distribution(self):
        """Test belief distribution with many alternatives."""
        beliefs = {f"alt_{i}": 1.0/10 for i in range(10)}
        bd = BeliefDistribution(beliefs=beliefs)
        assert len(bd.beliefs) == 10

    def test_confidence_boundary_values(self):
        """Test confidence at boundary values."""
        # Min confidence
        assessment1 = AgentAssessment(
            agent_id="agent_001",
            agent_name="Expert",
            role="Commander",
            belief_distribution=BeliefDistribution(beliefs={"alt1": 1.0}),
            confidence=0.0,
            reasoning="Test"
        )
        assert assessment1.confidence == 0.0

        # Max confidence
        assessment2 = AgentAssessment(
            agent_id="agent_001",
            agent_name="Expert",
            role="Commander",
            belief_distribution=BeliefDistribution(beliefs={"alt1": 1.0}),
            confidence=1.0,
            reasoning="Test"
        )
        assert assessment2.confidence == 1.0

    def test_very_small_belief_values(self):
        """Test belief distribution with very small values."""
        bd = BeliefDistribution(beliefs={"alt1": 0.999, "alt2": 0.001})
        assert abs(bd["alt2"] - 0.001) < 0.0001

    def test_unicode_in_strings(self):
        """Test that Unicode characters work in string fields."""
        assessment = AgentAssessment(
            agent_id="agent_001",
            agent_name="Expert médical 🏥",
            role="Commander",
            belief_distribution=BeliefDistribution(beliefs={"alt1": 1.0}),
            confidence=0.8,
            reasoning="Test avec caractères spéciaux: αβγ"
        )
        assert "médical" in assessment.agent_name
        assert "αβγ" in assessment.reasoning
