#!/usr/bin/env python3
"""
Simple test to verify Pydantic models work correctly.

This bypasses LLM API calls and directly tests the data models.
"""

import sys
from models.data_models import BeliefDistribution, AgentAssessment
from pydantic import ValidationError


def test_belief_distribution():
    """Test BeliefDistribution model."""
    print("=" * 60)
    print("Testing BeliefDistribution")
    print("=" * 60)

    # Test valid distribution
    print("\n1. Creating valid belief distribution...")
    try:
        beliefs = {"alt_evacuate": 0.6, "alt_barriers": 0.3, "alt_monitor": 0.1}
        bd = BeliefDistribution(beliefs=beliefs)
        print(f"   ✓ Created: {bd}")
        print(f"   ✓ Type: {type(bd).__name__}")
        print(f"   ✓ Sum: {sum(bd.values()):.4f}")
        print(f"   ✓ Dict access: bd['alt_evacuate'] = {bd['alt_evacuate']}")
        print(f"   ✓ .get(): bd.get('alt_evacuate') = {bd.get('alt_evacuate')}")
        print(f"   ✓ Keys: {list(bd.keys())}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test normalization
    print("\n2. Testing automatic normalization...")
    try:
        beliefs = {"alt1": 0.6, "alt2": 0.5}  # Sum = 1.1, should normalize
        bd = BeliefDistribution(beliefs=beliefs)
        total = sum(bd.values())
        print(f"   ✓ Normalized sum: {total:.4f}")
        assert abs(total - 1.0) < 0.001, "Sum should be ~1.0"
        print(f"   ✓ Values normalized correctly")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test validation catches errors
    print("\n3. Testing validation...")
    try:
        BeliefDistribution(beliefs={"alt1": 1.5})
        print(f"   ✗ Should have rejected belief > 1.0")
        return False
    except ValidationError:
        print(f"   ✓ Correctly rejects belief > 1.0")

    try:
        BeliefDistribution(beliefs={"alt1": -0.1})
        print(f"   ✗ Should have rejected belief < 0.0")
        return False
    except ValidationError:
        print(f"   ✓ Correctly rejects belief < 0.0")

    try:
        BeliefDistribution(beliefs={})
        print(f"   ✗ Should have rejected empty beliefs")
        return False
    except ValidationError:
        print(f"   ✓ Correctly rejects empty beliefs")

    print("\n   ✓ BeliefDistribution tests PASSED")
    return True


def test_agent_assessment():
    """Test AgentAssessment model."""
    print("\n" + "=" * 60)
    print("Testing AgentAssessment")
    print("=" * 60)

    # Create valid assessment
    print("\n1. Creating valid agent assessment...")
    try:
        belief_dist = BeliefDistribution(beliefs={"alt1": 0.7, "alt2": 0.3})

        assessment = AgentAssessment(
            agent_id="test_agent_001",
            agent_name="Test Medical Expert",
            role="Medical Commander",
            belief_distribution=belief_dist,
            confidence=0.85,
            reasoning="This is test reasoning for the assessment",
            key_concerns=["Concern 1", "Concern 2"],
            recommended_actions=["Action 1", "Action 2"]
        )

        print(f"   ✓ Created assessment")
        print(f"   ✓ Type: {type(assessment).__name__}")
        print(f"   ✓ Agent ID: {assessment.agent_id}")
        print(f"   ✓ Confidence: {assessment.confidence}")
        print(f"   ✓ Role: {assessment.role}")
        print(f"   ✓ Belief distribution type: {type(assessment.belief_distribution).__name__}")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test attribute access
    print("\n2. Testing attribute access...")
    try:
        print(f"   ✓ assessment.agent_id = {assessment.agent_id}")
        print(f"   ✓ assessment.confidence = {assessment.confidence}")
        print(f"   ✓ assessment.reasoning[:30] = {assessment.reasoning[:30]}...")
        print(f"   ✓ len(assessment.key_concerns) = {len(assessment.key_concerns)}")
        print(f"   ✓ assessment.belief_distribution['alt1'] = {assessment.belief_distribution['alt1']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test .to_dict() for backward compatibility
    print("\n3. Testing backward compatibility (.to_dict())...")
    try:
        data = assessment.to_dict()
        print(f"   ✓ Returns dict: {isinstance(data, dict)}")
        print(f"   ✓ Has agent_id: {'agent_id' in data}")
        print(f"   ✓ Has belief_distribution: {'belief_distribution' in data}")
        print(f"   ✓ belief_distribution is dict: {isinstance(data['belief_distribution'], dict)}")
        print(f"   ✓ Dict access: data['confidence'] = {data['confidence']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test validation
    print("\n4. Testing validation...")
    try:
        AgentAssessment(
            agent_id="test",
            agent_name="Test",
            role="Role",
            belief_distribution=belief_dist,
            confidence=1.5,  # Invalid: > 1.0
            reasoning="Test"
        )
        print(f"   ✗ Should have rejected confidence > 1.0")
        return False
    except ValidationError:
        print(f"   ✓ Correctly rejects confidence > 1.0")

    try:
        AgentAssessment(
            agent_id="",  # Invalid: empty
            agent_name="Test",
            role="Role",
            belief_distribution=belief_dist,
            confidence=0.8,
            reasoning="Test"
        )
        print(f"   ✗ Should have rejected empty agent_id")
        return False
    except ValidationError:
        print(f"   ✓ Correctly rejects empty agent_id")

    print("\n   ✓ AgentAssessment tests PASSED")
    return True


def test_integration():
    """Test that models work together."""
    print("\n" + "=" * 60)
    print("Testing Model Integration")
    print("=" * 60)

    print("\n1. Creating complete assessment with all fields...")
    try:
        belief_dist = BeliefDistribution(beliefs={
            "alt_evacuate": 0.5,
            "alt_barriers": 0.3,
            "alt_monitor": 0.2
        })

        assessment = AgentAssessment(
            agent_id="agent_medical_001",
            agent_name="Dr. Emergency Medicine",
            role="Medical Infrastructure Commander",
            belief_distribution=belief_dist,
            confidence=0.92,
            reasoning="Based on hospital capacity and patient safety, immediate evacuation is recommended.",
            key_concerns=[
                "Hospital capacity at 95%",
                "Limited evacuation routes",
                "Vulnerable patient population"
            ],
            recommended_actions=[
                "Activate emergency medical protocols",
                "Begin patient transfer to regional facilities",
                "Establish mobile triage units"
            ],
            risk_assessment="high",
            timestamp="2025-11-10T12:00:00",
            metadata={
                "scenario_type": "flood",
                "assessment_number": 1,
                "llm_provider": "test"
            }
        )

        print(f"   ✓ Created complex assessment")
        print(f"   ✓ Belief distribution sum: {sum(assessment.belief_distribution.values()):.4f}")
        print(f"   ✓ Number of concerns: {len(assessment.key_concerns)}")
        print(f"   ✓ Number of actions: {len(assessment.recommended_actions)}")
        print(f"   ✓ Risk assessment: {assessment.risk_assessment}")

        # Test iteration
        print("\n2. Testing iteration over belief distribution...")
        for alt_id, belief in assessment.belief_distribution.items():
            print(f"   ✓ {alt_id}: {belief:.2f}")

        # Test model_dump()
        print("\n3. Testing Pydantic model_dump()...")
        dump = assessment.model_dump()
        print(f"   ✓ model_dump() returns dict: {isinstance(dump, dict)}")
        print(f"   ✓ Keys: {list(dump.keys())[:5]}...")

        print("\n   ✓ Integration tests PASSED")
        return True

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🚀 " * 30)
    print("Pydantic Models Test Suite")
    print("🚀 " * 30)

    results = []

    results.append(("BeliefDistribution", test_belief_distribution()))
    results.append(("AgentAssessment", test_agent_assessment()))
    results.append(("Integration", test_integration()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests PASSED! 🎉")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
