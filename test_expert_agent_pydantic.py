#!/usr/bin/env python3
"""
Quick test to verify ExpertAgent works with Pydantic models.

This tests:
1. ExpertAgent can generate BeliefDistribution
2. ExpertAgent can create AgentAssessment
3. Assessment has proper type validation
4. Backward compatibility with dict-like access
"""

import json
import sys
from agents.expert_agent import ExpertAgent
from llm_integration.claude_client import ClaudeClient
from models.data_models import BeliefDistribution, AgentAssessment

def test_expert_agent_pydantic():
    """Test ExpertAgent with Pydantic integration."""

    print("=" * 60)
    print("Testing ExpertAgent with Pydantic Models")
    print("=" * 60)

    # Load scenario
    print("\n1. Loading flood scenario...")
    with open('scenarios/flood_scenario.json', 'r') as f:
        scenario = json.load(f)

    # Create mock alternatives (not in scenario file)
    alternatives = [
        {"id": "alt_evacuate", "name": "Immediate Evacuation"},
        {"id": "alt_barriers", "name": "Deploy Barriers"},
        {"id": "alt_monitor", "name": "Monitor and Wait"}
    ]
    print(f"   ✓ Loaded scenario: {scenario['type']}")
    print(f"   ✓ Created {len(alternatives)} mock alternatives")

    # Initialize agent (use mock mode to avoid API calls)
    print("\n2. Initializing ExpertAgent...")
    try:
        # Try to use Claude, but it might not be configured
        client = ClaudeClient()
        agent = ExpertAgent("agent_meteorologist", llm_client=client)
        print(f"   ✓ Created agent: {agent.name}")
        print(f"   ✓ Role: {agent.role}")
    except Exception as e:
        print(f"   ✗ Failed to initialize agent: {e}")
        print("   Note: This might be expected if API keys are not configured")
        return False

    # Test generate_belief_distribution with mock data
    print("\n3. Testing generate_belief_distribution()...")
    mock_llm_response = {
        'alternative_rankings': {
            'alt_evacuate': 0.6,
            'alt_barriers': 0.3,
            'alt_monitor': 0.1
        },
        'reasoning': 'Test reasoning',
        'confidence': 0.85,
        'key_concerns': ['Test concern']
    }

    try:
        belief_dist = agent.generate_belief_distribution(mock_llm_response)
        print(f"   ✓ Type: {type(belief_dist).__name__}")
        print(f"   ✓ Is BeliefDistribution: {isinstance(belief_dist, BeliefDistribution)}")
        print(f"   ✓ Keys: {list(belief_dist.keys())}")
        print(f"   ✓ Sum: {sum(belief_dist.values()):.4f}")

        # Test dict-like access
        print(f"   ✓ Dict access works: belief_dist['alt_evacuate'] = {belief_dist['alt_evacuate']}")
        print(f"   ✓ .get() works: belief_dist.get('alt_evacuate') = {belief_dist.get('alt_evacuate')}")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test _build_assessment
    print("\n4. Testing _build_assessment()...")
    try:
        assessment = agent._build_assessment(
            belief_distribution=belief_dist,
            criteria_scores={},
            llm_response=mock_llm_response,
            scenario=scenario
        )

        print(f"   ✓ Type: {type(assessment).__name__}")
        print(f"   ✓ Is AgentAssessment: {isinstance(assessment, AgentAssessment)}")
        print(f"   ✓ Agent ID: {assessment.agent_id}")
        print(f"   ✓ Confidence: {assessment.confidence}")
        print(f"   ✓ Reasoning: {assessment.reasoning[:50]}...")

        # Test attribute access
        print(f"   ✓ Attribute access works: assessment.confidence = {assessment.confidence}")

        # Test belief_distribution access
        print(f"   ✓ Belief distribution type: {type(assessment.belief_distribution).__name__}")
        print(f"   ✓ Can iterate beliefs: {list(assessment.belief_distribution.keys())[:2]}")

        # Test .to_dict() for backward compatibility
        print("\n5. Testing backward compatibility...")
        assessment_dict = assessment.to_dict()
        print(f"   ✓ .to_dict() returns dict: {isinstance(assessment_dict, dict)}")
        print(f"   ✓ Dict keys: {list(assessment_dict.keys())[:5]}")
        print(f"   ✓ belief_distribution in dict is dict: {isinstance(assessment_dict['belief_distribution'], dict)}")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test validation catches errors
    print("\n6. Testing validation...")
    try:
        # Try to create invalid belief distribution
        try:
            invalid_belief = BeliefDistribution(beliefs={"alt1": 1.5})  # > 1.0
            print(f"   ✗ Validation did not catch invalid belief value")
            return False
        except Exception:
            print(f"   ✓ Validation correctly rejects belief > 1.0")

        try:
            invalid_belief = BeliefDistribution(beliefs={"alt1": -0.1})  # < 0.0
            print(f"   ✗ Validation did not catch negative belief value")
            return False
        except Exception:
            print(f"   ✓ Validation correctly rejects belief < 0.0")

        try:
            invalid_belief = BeliefDistribution(beliefs={})  # Empty
            print(f"   ✗ Validation did not catch empty beliefs")
            return False
        except Exception:
            print(f"   ✓ Validation correctly rejects empty beliefs")

    except Exception as e:
        print(f"   ✗ Unexpected error during validation testing: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_expert_agent_pydantic()
    sys.exit(0 if success else 1)
