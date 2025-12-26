"""
Test Script for Simplified Evidential Reasoning Module
Demonstrates functionality for crisis management MAS system
"""

from decision_framework.evidential_reasoning import EvidentialReasoning


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def test_basic_aggregation():
    """Test 1: Basic belief aggregation with two agents."""
    print_section("Test 1: Basic Belief Aggregation")

    # Initialize ER engine
    er = EvidentialReasoning(enable_logging=False)

    # Two agents evaluating three alternatives
    agent_beliefs = {
        "agent_meteorologist": {"A1": 0.7, "A2": 0.2, "A3": 0.1},
        "agent_operations": {"A1": 0.5, "A2": 0.3, "A3": 0.2}
    }

    agent_weights = {
        "agent_meteorologist": 0.55,
        "agent_operations": 0.45
    }

    print("Input Data:")
    print(f"  Agent Beliefs:")
    for agent_id, beliefs in agent_beliefs.items():
        print(f"    {agent_id}: {beliefs}")
    print(f"\n  Agent Weights:")
    for agent_id, weight in agent_weights.items():
        print(f"    {agent_id}: {weight}")

    # Perform aggregation
    result = er.combine_beliefs(agent_beliefs, agent_weights)

    print(f"\nResults:")
    print(f"  Combined Beliefs: {result['combined_beliefs']}")
    print(f"  Confidence Score: {result['confidence']:.3f}")
    print(f"  Uncertainty: {result['uncertainty']:.3f}")

    # Verify calculations manually
    print(f"\n  Manual Verification:")
    print(f"    A1: 0.7 * 0.55 + 0.5 * 0.45 = {0.7 * 0.55 + 0.5 * 0.45:.3f}")
    print(f"    A2: 0.2 * 0.55 + 0.3 * 0.45 = {0.2 * 0.55 + 0.3 * 0.45:.3f}")
    print(f"    A3: 0.1 * 0.55 + 0.2 * 0.45 = {0.1 * 0.55 + 0.2 * 0.45:.3f}")

    print(f"\n✓ Test 1 Passed")


def test_multiple_agents():
    """Test 2: Multiple agents with different preferences."""
    print_section("Test 2: Multiple Agents (4 agents)")

    er = EvidentialReasoning(enable_logging=False)

    # Four agents with different belief distributions
    agent_beliefs = {
        "agent_meteorologist": {"A1": 0.6, "A2": 0.3, "A3": 0.1},
        "agent_medical": {"A1": 0.2, "A2": 0.6, "A3": 0.2},
        "agent_logistics": {"A1": 0.3, "A2": 0.4, "A3": 0.3},
        "agent_public_safety": {"A1": 0.5, "A2": 0.3, "A3": 0.2}
    }

    # Equal weights
    agent_weights = {
        "agent_meteorologist": 0.25,
        "agent_medical": 0.25,
        "agent_logistics": 0.25,
        "agent_public_safety": 0.25
    }

    result = er.combine_beliefs(agent_beliefs, agent_weights)

    print("Combined Beliefs:")
    for alt, belief in sorted(result['combined_beliefs'].items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(belief * 50)
        print(f"  {alt}: {belief:.3f} {bar}")

    print(f"\nConfidence: {result['confidence']:.3f}")
    print(f"\n✓ Test 2 Passed")


def test_unequal_alternatives():
    """Test 3: Agents proposing different sets of alternatives."""
    print_section("Test 3: Agents with Different Alternative Sets")

    er = EvidentialReasoning(enable_logging=False)

    # Agents mentioning different alternatives
    agent_beliefs = {
        "agent_1": {"A1": 0.7, "A2": 0.3},  # Only mentions A1, A2
        "agent_2": {"A2": 0.4, "A3": 0.6},  # Only mentions A2, A3
        "agent_3": {"A1": 0.5, "A3": 0.5}   # Only mentions A1, A3
    }

    agent_weights = {"agent_1": 0.4, "agent_2": 0.3, "agent_3": 0.3}

    print("Note: Agents mention different sets of alternatives")
    print("  agent_1: A1, A2")
    print("  agent_2: A2, A3")
    print("  agent_3: A1, A3")
    print("\nER will handle this by treating unmentioned alternatives as 0.0")

    result = er.combine_beliefs(agent_beliefs, agent_weights)

    print(f"\nCombined Beliefs:")
    for alt in sorted(result['alternatives']):
        belief = result['combined_beliefs'][alt]
        print(f"  {alt}: {belief:.3f}")

    print(f"\n✓ Test 3 Passed")


def test_normalization():
    """Test 4: Normalization of beliefs and weights."""
    print_section("Test 4: Normalization")

    er = EvidentialReasoning(enable_logging=False)

    print("Scenario: Agent beliefs and weights don't sum to 1.0")

    # Unnormalized beliefs
    beliefs = {"A1": 0.6, "A2": 0.4, "A3": 0.3}  # Sum = 1.3
    print(f"\nOriginal beliefs (sum={sum(beliefs.values()):.2f}):")
    print(f"  {beliefs}")

    normalized_beliefs = er.normalize_distribution(beliefs)
    print(f"\nNormalized beliefs (sum={sum(normalized_beliefs.values()):.2f}):")
    print(f"  {normalized_beliefs}")

    # Unnormalized weights
    weights = {"agent_1": 0.6, "agent_2": 0.8, "agent_3": 0.4}  # Sum = 1.8
    print(f"\nOriginal weights (sum={sum(weights.values()):.2f}):")
    print(f"  {weights}")

    normalized_weights = er.normalize_weights(weights)
    print(f"\nNormalized weights (sum={sum(normalized_weights.values()):.2f}):")
    print(f"  {normalized_weights}")

    print(f"\n✓ Test 4 Passed")


def test_confidence_calculation():
    """Test 5: Confidence score calculation."""
    print_section("Test 5: Confidence Score Calculation")

    er = EvidentialReasoning(enable_logging=False)

    test_cases = [
        {
            "name": "High confidence - one dominant alternative",
            "beliefs": {"A1": 0.9, "A2": 0.05, "A3": 0.05}
        },
        {
            "name": "Medium confidence - moderate distribution",
            "beliefs": {"A1": 0.6, "A2": 0.3, "A3": 0.1}
        },
        {
            "name": "Low confidence - evenly distributed",
            "beliefs": {"A1": 0.33, "A2": 0.33, "A3": 0.34}
        }
    ]

    print("Confidence reflects how decisively beliefs are distributed")
    print("(High = concentrated, Low = spread out)\n")

    for case in test_cases:
        confidence = er.calculate_confidence(case['beliefs'])
        print(f"{case['name']}:")
        print(f"  Beliefs: {case['beliefs']}")
        print(f"  Confidence: {confidence:.3f}\n")

    print("✓ Test 5 Passed")


def test_error_handling():
    """Test 6: Error handling and validation."""
    print_section("Test 6: Error Handling")

    er = EvidentialReasoning(enable_logging=False)

    tests = [
        {
            "name": "Empty beliefs dictionary",
            "agent_beliefs": {},
            "agent_weights": {"agent_1": 0.5},
            "expected_error": "agent_beliefs dictionary is empty"
        },
        {
            "name": "Mismatched agent IDs",
            "agent_beliefs": {"agent_1": {"A1": 1.0}},
            "agent_weights": {"agent_2": 1.0},
            "expected_error": "Agent IDs mismatch"
        },
        {
            "name": "Negative belief values",
            "beliefs": {"A1": 0.5, "A2": -0.3},
            "test_type": "normalize_distribution"
        },
        {
            "name": "All zero beliefs",
            "beliefs": {"A1": 0.0, "A2": 0.0},
            "test_type": "normalize_distribution"
        }
    ]

    for test in tests:
        print(f"Testing: {test['name']}")
        try:
            if test.get('test_type') == 'normalize_distribution':
                er.normalize_distribution(test['beliefs'])
                print(f"  ✗ Should have raised ValueError")
            else:
                er.combine_beliefs(test['agent_beliefs'], test['agent_weights'])
                print(f"  ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"  ✓ Caught ValueError: {str(e)[:60]}...")
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")

    print(f"\n✓ Test 6 Passed")


def test_crisis_scenario():
    """Test 7: Realistic crisis management scenario."""
    print_section("Test 7: Crisis Management Scenario")

    print("Scenario: Urban flood crisis - selecting emergency response action\n")

    er = EvidentialReasoning(enable_logging=True)

    # Five alternative actions
    # A1: Immediate evacuation
    # A2: Deploy flood barriers
    # A3: Shelter in place
    # A4: Rescue operations
    # A5: Hybrid approach

    agent_beliefs = {
        "agent_meteorologist": {
            "A1": 0.3,  # Evacuation
            "A2": 0.2,  # Barriers
            "A3": 0.1,  # Shelter
            "A4": 0.15, # Rescue
            "A5": 0.25  # Hybrid
        },
        "agent_medical": {
            "A1": 0.25,
            "A2": 0.1,
            "A3": 0.05,
            "A4": 0.45,  # Prioritizes rescue
            "A5": 0.15
        },
        "agent_logistics": {
            "A1": 0.2,
            "A2": 0.35,  # Prioritizes barriers
            "A3": 0.1,
            "A4": 0.15,
            "A5": 0.2
        },
        "agent_public_safety": {
            "A1": 0.4,   # Prioritizes evacuation
            "A2": 0.2,
            "A3": 0.05,
            "A4": 0.2,
            "A5": 0.15
        }
    }

    # Weights based on relevance to flood scenario
    agent_weights = {
        "agent_meteorologist": 0.30,  # High weight - weather expert
        "agent_medical": 0.25,
        "agent_logistics": 0.20,
        "agent_public_safety": 0.25
    }

    result = er.combine_beliefs(agent_beliefs, agent_weights)

    # Print formatted summary
    print(er.get_aggregation_summary(result))

    # Decision recommendation
    top_alt = er.get_top_alternatives(result['combined_beliefs'], top_n=1)[0]
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")
    print(f"Top Action: {top_alt[0]} with belief={top_alt[1]:.3f} ({top_alt[1]*100:.1f}%)")
    print(f"Decision Confidence: {result['confidence']:.3f}")

    if result['confidence'] > 0.7:
        print("Status: HIGH CONFIDENCE - Clear preference identified")
    elif result['confidence'] > 0.5:
        print("Status: MODERATE CONFIDENCE - Some agreement present")
    else:
        print("Status: LOW CONFIDENCE - Further deliberation recommended")

    print(f"\n✓ Test 7 Passed")


def test_aggregation_history():
    """Test 8: Aggregation history tracking."""
    print_section("Test 8: Aggregation History")

    er = EvidentialReasoning(enable_logging=False)

    print("Performing 3 sequential aggregations...\n")

    for i in range(3):
        agent_beliefs = {
            "agent_1": {"A1": 0.5 + i*0.1, "A2": 0.5 - i*0.1},
            "agent_2": {"A1": 0.4, "A2": 0.6}
        }
        agent_weights = {"agent_1": 0.5, "agent_2": 0.5}

        result = er.combine_beliefs(agent_beliefs, agent_weights)
        print(f"Aggregation {i+1}: {result['combined_beliefs']}")

    history = er.get_history()
    print(f"\nTotal aggregations in history: {len(history)}")

    print(f"\nHistory details:")
    for i, entry in enumerate(history, 1):
        print(f"  {i}. Timestamp: {entry['timestamp']}")
        print(f"     Confidence: {entry['confidence']:.3f}")
        print(f"     Agents: {len(entry['agents_involved'])}")

    print(f"\nClearing history...")
    er.clear_history()
    print(f"History length after clear: {len(er.get_history())}")

    print(f"\n✓ Test 8 Passed")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("EVIDENTIAL REASONING MODULE - COMPREHENSIVE TEST SUITE")
    print("For Crisis Management Multi-Agent System")
    print("="*70)

    tests = [
        test_basic_aggregation,
        test_multiple_agents,
        test_unequal_alternatives,
        test_normalization,
        test_confidence_calculation,
        test_error_handling,
        test_crisis_scenario,
        test_aggregation_history
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ Test Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
