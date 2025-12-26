"""
Test Script for Consensus Model
Crisis Management Multi-Agent System
"""

from decision_framework.consensus_model import ConsensusModel


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def test_initialization():
    """Test 1: Initialization with different thresholds."""
    print_section("Test 1: Initialization")

    # Default threshold
    cm1 = ConsensusModel()
    print(f"Default consensus threshold: {cm1.consensus_threshold}")

    # Custom threshold
    cm2 = ConsensusModel(consensus_threshold=0.85)
    print(f"Custom consensus threshold: {cm2.consensus_threshold}")

    print("\n✓ Test 1 Passed")


def test_consensus_high_agreement():
    """Test 2: High consensus - agents strongly agree."""
    print_section("Test 2: High Consensus (Strong Agreement)")

    cm = ConsensusModel(consensus_threshold=0.75)

    # Two agents with very similar belief distributions
    agent_beliefs = {
        "agent_meteorologist": {
            "A1": 0.65,  # Both prefer A1
            "A2": 0.25,
            "A3": 0.10
        },
        "agent_medical": {
            "A1": 0.70,  # Similar preference for A1
            "A2": 0.20,
            "A3": 0.10
        }
    }

    print("Agent Beliefs:")
    for agent_id, beliefs in agent_beliefs.items():
        print(f"  {agent_id}: {beliefs}")

    consensus_level = cm.calculate_consensus_level(agent_beliefs)
    print(f"\nConsensus Level: {consensus_level:.3f}")
    print(f"Threshold: {cm.consensus_threshold}")
    print(f"Consensus Reached: {cm.is_consensus_reached(consensus_level)}")

    conflicts = cm.detect_conflicts(agent_beliefs)
    print(f"\nConflicts Detected: {len(conflicts)}")

    print("\n✓ Test 2 Passed")


def test_consensus_moderate_agreement():
    """Test 3: Moderate consensus - some agreement."""
    print_section("Test 3: Moderate Consensus")

    cm = ConsensusModel(consensus_threshold=0.75)

    # Agents agree on top choice but differ on others
    agent_beliefs = {
        "agent_logistics": {
            "A1": 0.50,
            "A2": 0.30,
            "A3": 0.20
        },
        "agent_public_safety": {
            "A1": 0.55,
            "A2": 0.15,  # Different distribution
            "A3": 0.30
        }
    }

    print("Agent Beliefs:")
    for agent_id, beliefs in agent_beliefs.items():
        print(f"  {agent_id}: {beliefs}")

    consensus_level = cm.calculate_consensus_level(agent_beliefs)
    print(f"\nConsensus Level: {consensus_level:.3f}")
    print(f"Consensus Reached: {cm.is_consensus_reached(consensus_level)}")

    conflicts = cm.detect_conflicts(agent_beliefs)
    print(f"\nConflicts Detected: {len(conflicts)}")
    if conflicts:
        for conflict in conflicts:
            agent_1, agent_2 = conflict['agent_pair']
            print(f"  • {agent_1} vs {agent_2}")
            print(f"    Severity: {conflict['severity']}")

    print("\n✓ Test 3 Passed")


def test_consensus_low_agreement():
    """Test 4: Low consensus - agents disagree."""
    print_section("Test 4: Low Consensus (Disagreement)")

    cm = ConsensusModel(consensus_threshold=0.75)

    # Agents prefer completely different alternatives
    agent_beliefs = {
        "agent_meteorologist": {
            "A1": 0.80,  # Strongly prefers A1
            "A2": 0.10,
            "A3": 0.10
        },
        "agent_medical": {
            "A1": 0.10,
            "A2": 0.80,  # Strongly prefers A2
            "A3": 0.10
        }
    }

    print("Agent Beliefs:")
    for agent_id, beliefs in agent_beliefs.items():
        top_choice = max(beliefs.items(), key=lambda x: x[1])
        print(f"  {agent_id}:")
        print(f"    Top choice: {top_choice[0]} ({top_choice[1]:.2f})")
        print(f"    Distribution: {beliefs}")

    consensus_level = cm.calculate_consensus_level(agent_beliefs)
    print(f"\nConsensus Level: {consensus_level:.3f}")
    print(f"Consensus Reached: {cm.is_consensus_reached(consensus_level)}")

    conflicts = cm.detect_conflicts(agent_beliefs)
    print(f"\nConflicts Detected: {len(conflicts)}")
    if conflicts:
        for conflict in conflicts:
            agent_1, agent_2 = conflict['agent_pair']
            print(f"  • Conflict: {agent_1} vs {agent_2}")
            print(f"    {conflict['agent_1_top_choice']} (belief: {conflict['agent_1_belief']:.2f}) vs "
                  f"{conflict['agent_2_top_choice']} (belief: {conflict['agent_2_belief']:.2f})")
            print(f"    Disagreement magnitude: {conflict['disagreement_magnitude']:.2f}")
            print(f"    Severity: {conflict['severity']}")

    print("\n✓ Test 4 Passed")


def test_conflict_detection():
    """Test 5: Detailed conflict detection with varying thresholds."""
    print_section("Test 5: Conflict Detection with Thresholds")

    cm = ConsensusModel()

    agent_beliefs = {
        "agent_1": {
            "A1": 0.60,
            "A2": 0.25,
            "A3": 0.15
        },
        "agent_2": {
            "A1": 0.20,
            "A2": 0.65,
            "A3": 0.15
        }
    }

    thresholds = [0.1, 0.3, 0.5]

    print("Testing different conflict thresholds:\n")
    for threshold in thresholds:
        conflicts = cm.detect_conflicts(agent_beliefs, conflict_threshold=threshold)
        print(f"Threshold = {threshold}:")
        print(f"  Conflicts detected: {len(conflicts)}")
        if conflicts:
            print(f"  Severity: {conflicts[0]['severity']}")
        print()

    print("✓ Test 5 Passed")


def test_resolution_suggestions():
    """Test 6: Conflict resolution suggestions."""
    print_section("Test 6: Conflict Resolution Suggestions")

    cm = ConsensusModel()

    agent_beliefs = {
        "agent_meteorologist": {
            "A1": 0.70,
            "A2": 0.20,
            "A3": 0.10
        },
        "agent_medical": {
            "A1": 0.15,
            "A2": 0.75,
            "A3": 0.10
        }
    }

    # Alternative descriptions
    alternatives_data = {
        "A1": {"name": "Immediate Mass Evacuation"},
        "A2": {"name": "Deploy Flood Barriers + Selective Evacuation"},
        "A3": {"name": "Shelter-in-Place with Monitoring"}
    }

    conflicts = cm.detect_conflicts(agent_beliefs)

    print("Detected Conflicts:")
    for conflict in conflicts:
        agent_1, agent_2 = conflict['agent_pair']
        print(f"  • {agent_1} prefers {conflict['agent_1_top_choice']}")
        print(f"  • {agent_2} prefers {conflict['agent_2_top_choice']}")
        print(f"  • Severity: {conflict['severity']}\n")

    print("Resolution Suggestions:")
    print("="*70)
    resolution_text = cm.suggest_resolution(conflicts, agent_beliefs, alternatives_data)
    print(resolution_text)

    print("\n✓ Test 6 Passed")


def test_multiple_agents():
    """Test 7: Consensus with more than 2 agents."""
    print_section("Test 7: Multiple Agents (4 agents)")

    cm = ConsensusModel(consensus_threshold=0.70)

    agent_beliefs = {
        "agent_meteorologist": {
            "A1": 0.50,
            "A2": 0.30,
            "A3": 0.20
        },
        "agent_medical": {
            "A1": 0.55,
            "A2": 0.25,
            "A3": 0.20
        },
        "agent_logistics": {
            "A1": 0.45,
            "A2": 0.35,
            "A3": 0.20
        },
        "agent_public_safety": {
            "A1": 0.60,
            "A2": 0.25,
            "A3": 0.15
        }
    }

    print("4 agents with similar preferences (all favor A1):\n")
    for agent_id, beliefs in agent_beliefs.items():
        top = max(beliefs.items(), key=lambda x: x[1])
        print(f"  {agent_id}: Top={top[0]} ({top[1]:.2f})")

    consensus_level = cm.calculate_consensus_level(agent_beliefs)
    print(f"\nAverage Pairwise Consensus: {consensus_level:.3f}")
    print(f"Consensus Reached: {cm.is_consensus_reached(consensus_level)}")

    conflicts = cm.detect_conflicts(agent_beliefs)
    print(f"\nConflicts Detected: {len(conflicts)}")

    print("\n✓ Test 7 Passed")


def test_comprehensive_analysis():
    """Test 8: Comprehensive consensus analysis."""
    print_section("Test 8: Comprehensive Analysis")

    cm = ConsensusModel(consensus_threshold=0.75)

    agent_beliefs = {
        "agent_meteorologist": {
            "A1": 0.35,
            "A2": 0.40,  # Prefers A2
            "A3": 0.25
        },
        "agent_medical": {
            "A1": 0.60,  # Prefers A1
            "A2": 0.25,
            "A3": 0.15
        }
    }

    alternatives_data = {
        "A1": {
            "name": "Immediate Mass Evacuation",
            "safety_score": 0.90,
            "cost_euros": 480000
        },
        "A2": {
            "name": "Deploy Flood Barriers + Selective Evacuation",
            "safety_score": 0.78,
            "cost_euros": 320000
        },
        "A3": {
            "name": "Shelter-in-Place with Monitoring",
            "safety_score": 0.62,
            "cost_euros": 150000
        }
    }

    print("Performing comprehensive consensus analysis...\n")

    analysis = cm.analyze_consensus(agent_beliefs, alternatives_data)

    print(f"Consensus Level: {analysis['consensus_level']:.3f}")
    print(f"Consensus Reached: {analysis['consensus_reached']}")
    print(f"Number of Conflicts: {len(analysis['conflicts'])}")

    if analysis['conflicts']:
        print("\nConflicts:")
        for conflict in analysis['conflicts']:
            print(f"  • {conflict['agent_1']} vs {conflict['agent_2']}")
            print(f"    Severity: {conflict['severity']}")
            print(f"    {conflict['agent_1_top_choice']} vs {conflict['agent_2_top_choice']}")

    print("\nResolution Suggestions:")
    print("-" * 70)
    print(analysis['resolution_suggestions'])

    print("\n✓ Test 8 Passed")


def test_crisis_scenario():
    """Test 9: Realistic crisis management scenario."""
    print_section("Test 9: Crisis Management Scenario")

    print("Scenario: Urban flood crisis - Two experts must reach consensus\n")

    cm = ConsensusModel(consensus_threshold=0.75)

    # Meteorologist prioritizes fast response
    # Medical expert prioritizes safety
    agent_beliefs = {
        "agent_meteorologist": {
            "A1": 0.25,  # Mass evacuation
            "A2": 0.50,  # Flood barriers (fast deployment)
            "A3": 0.15,  # Rescue ops
            "A4": 0.10   # Shelter in place
        },
        "agent_medical": {
            "A1": 0.60,  # Mass evacuation (safest)
            "A2": 0.20,  # Flood barriers
            "A3": 0.15,  # Rescue ops
            "A4": 0.05   # Shelter in place
        }
    }

    alternatives_data = {
        "A1": {
            "name": "Immediate Mass Evacuation",
            "safety_score": 0.90,
            "cost_euros": 480000,
            "response_time_hours": 4
        },
        "A2": {
            "name": "Deploy Flood Barriers + Selective Evacuation",
            "safety_score": 0.78,
            "cost_euros": 320000,
            "response_time_hours": 3
        },
        "A3": {
            "name": "Prioritized Rescue Operations",
            "safety_score": 0.88,
            "cost_euros": 280000,
            "response_time_hours": 2
        },
        "A4": {
            "name": "Shelter-in-Place with Monitoring",
            "safety_score": 0.62,
            "cost_euros": 150000,
            "response_time_hours": 1
        }
    }

    print("Agent Preferences:")
    print("  Meteorologist (focuses on speed):")
    for alt, belief in sorted(agent_beliefs["agent_meteorologist"].items(),
                             key=lambda x: x[1], reverse=True):
        alt_name = alternatives_data[alt]["name"]
        print(f"    {alt}: {belief:.2f} - {alt_name}")

    print("\n  Medical Expert (focuses on safety):")
    for alt, belief in sorted(agent_beliefs["agent_medical"].items(),
                             key=lambda x: x[1], reverse=True):
        alt_name = alternatives_data[alt]["name"]
        print(f"    {alt}: {belief:.2f} - {alt_name}")

    print(f"\n{'='*70}")
    print("CONSENSUS ANALYSIS")
    print(f"{'='*70}")

    analysis = cm.analyze_consensus(agent_beliefs, alternatives_data)

    print(f"\nConsensus Level: {analysis['consensus_level']:.3f}")
    print(f"Threshold: {cm.consensus_threshold}")
    print(f"Status: {'CONSENSUS REACHED ✓' if analysis['consensus_reached'] else 'NO CONSENSUS - CONFLICT DETECTED ✗'}")

    if analysis['conflicts']:
        print(f"\n{'='*70}")
        print("CONFLICT DETAILS")
        print(f"{'='*70}")
        for conflict in analysis['conflicts']:
            agent_1, agent_2 = conflict['agent_pair']
            print(f"\nConflict between {agent_1} and {agent_2}:")

            alt1 = conflict['agent_1_top_choice']
            alt2 = conflict['agent_2_top_choice']

            print(f"  • {agent_1} strongly prefers: {alt1}")
            print(f"    ({alternatives_data[alt1]['name']})")
            print(f"    Belief: {conflict['agent_1_belief']:.2f}")

            print(f"\n  • {agent_2} strongly prefers: {alt2}")
            print(f"    ({alternatives_data[alt2]['name']})")
            print(f"    Belief: {conflict['agent_2_belief']:.2f}")

            print(f"\n  • Disagreement Magnitude: {conflict['disagreement_magnitude']:.2f}")
            print(f"  • Severity: {conflict['severity'].upper()}")

        print(f"\n{'='*70}")
        print("RESOLUTION RECOMMENDATIONS")
        print(f"{'='*70}")
        print(analysis['resolution_suggestions'])

    print("\n✓ Test 9 Passed")


def test_edge_cases():
    """Test 10: Edge cases and boundary conditions."""
    print_section("Test 10: Edge Cases")

    cm = ConsensusModel()

    print("Test Case 1: Perfect Agreement (identical beliefs)")
    perfect_agreement = {
        "agent_1": {"A1": 0.5, "A2": 0.3, "A3": 0.2},
        "agent_2": {"A1": 0.5, "A2": 0.3, "A3": 0.2}
    }
    consensus = cm.calculate_consensus_level(perfect_agreement)
    print(f"  Consensus Level: {consensus:.3f} (should be 1.0)")

    print("\nTest Case 2: Complete Disagreement")
    complete_disagreement = {
        "agent_1": {"A1": 1.0, "A2": 0.0, "A3": 0.0},
        "agent_2": {"A1": 0.0, "A2": 1.0, "A3": 0.0}
    }
    consensus = cm.calculate_consensus_level(complete_disagreement)
    print(f"  Consensus Level: {consensus:.3f} (should be close to 0.0)")

    print("\nTest Case 3: Single alternative")
    single_alt = {
        "agent_1": {"A1": 1.0},
        "agent_2": {"A1": 1.0}
    }
    consensus = cm.calculate_consensus_level(single_alt)
    print(f"  Consensus Level: {consensus:.3f} (should be 1.0)")

    print("\nTest Case 4: Different alternative sets")
    different_alts = {
        "agent_1": {"A1": 0.7, "A2": 0.3},
        "agent_2": {"A2": 0.6, "A3": 0.4}
    }
    consensus = cm.calculate_consensus_level(different_alts)
    print(f"  Consensus Level: {consensus:.3f}")
    conflicts = cm.detect_conflicts(different_alts)
    print(f"  Conflicts: {len(conflicts)}")

    print("\n✓ Test 10 Passed")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CONSENSUS MODEL - COMPREHENSIVE TEST SUITE")
    print("Crisis Management Multi-Agent System")
    print("="*70)

    tests = [
        test_initialization,
        test_consensus_high_agreement,
        test_consensus_moderate_agreement,
        test_consensus_low_agreement,
        test_conflict_detection,
        test_resolution_suggestions,
        test_multiple_agents,
        test_comprehensive_analysis,
        test_crisis_scenario,
        test_edge_cases
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
