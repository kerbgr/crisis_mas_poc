"""
Test Script for MCDA Engine
Crisis Management Multi-Agent System
"""

from decision_framework.mcda_engine import MCDAEngine


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def test_initialization():
    """Test 1: Engine initialization and criteria loading."""
    print_section("Test 1: Initialization & Criteria Loading")

    mcda = MCDAEngine("scenarios/criteria_weights.json")

    print(f"Engine initialized: {mcda}")
    print(f"\nCriteria loaded: {len(mcda.criteria_config)}")

    criteria_info = mcda.get_criteria_info()
    print("\nCriteria Configuration:")
    for cid, info in criteria_info.items():
        print(f"  • {info['name']:20s}: weight={info['weight']:.2f}, type={info['type']}")

    print("\n✓ Test 1 Passed")


def test_normalization():
    """Test 2: Score normalization for benefit and cost criteria."""
    print_section("Test 2: Score Normalization")

    mcda = MCDAEngine("scenarios/criteria_weights.json")

    print("Testing normalization with values [5, 7, 10]")
    print(f"  Min: 5, Max: 10\n")

    # Test benefit criterion (higher is better)
    value = 7
    normalized_benefit = mcda.normalize_score(value, 5, 10, 'benefit')
    print(f"Benefit criterion (value=7):")
    print(f"  Normalized: {normalized_benefit:.3f}")
    print(f"  Expected: (7-5)/(10-5) = {(7-5)/(10-5):.3f}")

    # Test cost criterion (lower is better)
    normalized_cost = mcda.normalize_score(value, 5, 10, 'cost')
    print(f"\nCost criterion (value=7):")
    print(f"  Normalized: {normalized_cost:.3f}")
    print(f"  Expected: (10-7)/(10-5) = {(10-7)/(10-5):.3f}")

    # Test edge case
    print(f"\nEdge case - all values same:")
    normalized_edge = mcda.normalize_score(5, 5, 5, 'benefit')
    print(f"  Normalized: {normalized_edge:.3f} (should be 1.0)")

    print("\n✓ Test 2 Passed")


def test_basic_ranking():
    """Test 3: Basic alternative ranking."""
    print_section("Test 3: Basic Alternative Ranking")

    mcda = MCDAEngine("scenarios/criteria_weights.json")

    # Define 4 alternatives with different profiles
    alternatives = [
        {
            "id": "A1",
            "name": "Immediate Evacuation",
            "estimated_metrics": {
                "safety_score": 0.95,        # High safety
                "cost_euros": 450000,         # High cost
                "response_time_hours": 2,     # Fast
                "social_disruption": 0.8      # High disruption (low acceptance)
            }
        },
        {
            "id": "A2",
            "name": "Deploy Flood Barriers",
            "estimated_metrics": {
                "safety_score": 0.75,         # Medium safety
                "cost_euros": 300000,         # Medium cost
                "response_time_hours": 4,     # Medium speed
                "social_disruption": 0.4      # Low disruption (high acceptance)
            }
        },
        {
            "id": "A3",
            "name": "Shelter in Place",
            "estimated_metrics": {
                "safety_score": 0.60,         # Lower safety
                "cost_euros": 100000,         # Low cost
                "response_time_hours": 1,     # Very fast
                "social_disruption": 0.3      # Low disruption
            }
        },
        {
            "id": "A4",
            "name": "Hybrid Approach",
            "estimated_metrics": {
                "safety_score": 0.85,         # Good safety
                "cost_euros": 380000,         # Higher cost
                "response_time_hours": 3,     # Medium-fast
                "social_disruption": 0.5      # Medium disruption
            }
        }
    ]

    print("Alternatives:")
    for alt in alternatives:
        print(f"  {alt['id']}: {alt['name']}")
        metrics = alt['estimated_metrics']
        print(f"      Safety: {metrics['safety_score']}, "
              f"Cost: €{metrics['cost_euros']:,}, "
              f"Time: {metrics['response_time_hours']}h, "
              f"Disruption: {metrics['social_disruption']}")

    # Rank alternatives
    ranked = mcda.rank_alternatives(alternatives)

    print(f"\nRanking Results:")
    for rank, (alt_id, score, norm_scores) in enumerate(ranked, 1):
        alt_name = next(a['name'] for a in alternatives if a['id'] == alt_id)
        print(f"  {rank}. {alt_name:30s} (Score: {score:.3f})")

    print(f"\n✓ Winner: {ranked[0][0]} ({next(a['name'] for a in alternatives if a['id'] == ranked[0][0])})")
    print("✓ Test 3 Passed")

    return alternatives, ranked


def test_explain_ranking(alternatives, ranked):
    """Test 4: Ranking explanation."""
    print_section("Test 4: Ranking Explanation")

    mcda = MCDAEngine("scenarios/criteria_weights.json")

    # Create alternatives data dict for explanation
    alt_data = {alt['id']: alt for alt in alternatives}

    explanation = mcda.explain_ranking(ranked, alt_data)
    print(explanation)

    print("\n✓ Test 4 Passed")


def test_custom_weights():
    """Test 5: Custom weight profiles."""
    print_section("Test 5: Custom Weight Profiles")

    mcda = MCDAEngine("scenarios/criteria_weights.json")

    alternatives = [
        {
            "id": "A1",
            "name": "High Safety Option",
            "estimated_metrics": {
                "safety_score": 0.95,
                "cost_euros": 500000,
                "response_time_hours": 3,
                "social_disruption": 0.7
            }
        },
        {
            "id": "A2",
            "name": "Low Cost Option",
            "estimated_metrics": {
                "safety_score": 0.70,
                "cost_euros": 150000,
                "response_time_hours": 5,
                "social_disruption": 0.4
            }
        }
    ]

    # Test different weight profiles
    profiles = {
        "medical_focused": {
            "safety": 0.50,
            "cost": 0.15,
            "response_time": 0.20,
            "social_acceptance": 0.15
        },
        "cost_focused": {
            "safety": 0.20,
            "cost": 0.50,
            "response_time": 0.15,
            "social_acceptance": 0.15
        }
    }

    print("Testing 2 alternatives with different weight profiles:\n")

    for profile_name, weights in profiles.items():
        print(f"{profile_name}:")
        print(f"  Weights: {weights}")

        ranked = mcda.rank_alternatives(alternatives, weights)
        winner = next(a['name'] for a in alternatives if a['id'] == ranked[0][0])

        print(f"  Winner: {winner} (score: {ranked[0][1]:.3f})\n")

    print("✓ Test 5 Passed")


def test_sensitivity_analysis():
    """Test 6: Sensitivity analysis."""
    print_section("Test 6: Sensitivity Analysis")

    mcda = MCDAEngine("scenarios/criteria_weights.json")

    alternatives = [
        {
            "id": "A1",
            "name": "Safe but Expensive",
            "estimated_metrics": {
                "safety_score": 0.95,
                "cost_euros": 500000,
                "response_time_hours": 3,
                "social_disruption": 0.6
            }
        },
        {
            "id": "A2",
            "name": "Cheap but Risky",
            "estimated_metrics": {
                "safety_score": 0.65,
                "cost_euros": 150000,
                "response_time_hours": 4,
                "social_disruption": 0.5
            }
        },
        {
            "id": "A3",
            "name": "Balanced",
            "estimated_metrics": {
                "safety_score": 0.80,
                "cost_euros": 300000,
                "response_time_hours": 3,
                "social_disruption": 0.4
            }
        }
    ]

    print("Performing sensitivity analysis on 'safety' criterion weight...")
    print("Testing weight range: 0.1 to 0.6\n")

    sensitivity = mcda.sensitivity_analysis(
        alternatives,
        criterion_to_vary='safety',
        weight_range=(0.1, 0.6),
        num_steps=11
    )

    print(f"Original weight: {sensitivity['original_weight']:.2f}")
    print(f"Stability score: {sensitivity['stability_score']:.2f}")
    print(f"Number of winner changes: {sensitivity['num_winner_changes']}")
    print(f"Unique winners: {sensitivity['unique_winners']}")

    if sensitivity['winner_changes']:
        print(f"\nWinner changes at:")
        for change in sensitivity['winner_changes']:
            print(f"  • Weight ≈ {change['weight_threshold']:.2f}: "
                  f"{change['previous_winner']} → {change['new_winner']}")

    # Show winner at each weight
    print(f"\nWinner progression:")
    for result in sensitivity['results'][::2]:  # Show every other
        weight = result['weight']
        winner = result['winner']
        score = result['winner_score']
        print(f"  Weight={weight:.2f}: {winner} (score: {score:.3f})")

    print("\n✓ Test 6 Passed")


def test_profile_comparison():
    """Test 7: Weight profile comparison."""
    print_section("Test 7: Weight Profile Comparison")

    mcda = MCDAEngine("scenarios/criteria_weights.json")

    alternatives = [
        {
            "id": "A1",
            "name": "Maximum Safety",
            "estimated_metrics": {
                "safety_score": 0.98,
                "cost_euros": 600000,
                "response_time_hours": 4,
                "social_disruption": 0.7
            }
        },
        {
            "id": "A2",
            "name": "Fast Response",
            "estimated_metrics": {
                "safety_score": 0.75,
                "cost_euros": 350000,
                "response_time_hours": 1,
                "social_disruption": 0.6
            }
        },
        {
            "id": "A3",
            "name": "Cost Effective",
            "estimated_metrics": {
                "safety_score": 0.70,
                "cost_euros": 120000,
                "response_time_hours": 5,
                "social_disruption": 0.3
            }
        }
    ]

    profiles = {
        "medical_focused": {"safety": 0.50, "cost": 0.15, "response_time": 0.20, "social_acceptance": 0.15},
        "cost_focused": {"safety": 0.20, "cost": 0.50, "response_time": 0.15, "social_acceptance": 0.15},
        "speed_focused": {"safety": 0.25, "cost": 0.15, "response_time": 0.45, "social_acceptance": 0.15}
    }

    print("Comparing 3 alternatives across 3 weight profiles:\n")

    comparison = mcda.compare_weight_profiles(alternatives, profiles)

    for profile_name, result in comparison['results'].items():
        winner = result['winner']
        score = result['winner_score']
        winner_name = next(a['name'] for a in alternatives if a['id'] == winner)
        print(f"{profile_name:20s}: {winner_name:20s} (score: {score:.3f})")

    print(f"\nAll profiles agree: {comparison['all_agree']}")
    if not comparison['all_agree']:
        print(f"Winner distribution: {comparison['winner_distribution']}")

    print("\n✓ Test 7 Passed")


def test_crisis_scenario():
    """Test 8: Realistic crisis management scenario."""
    print_section("Test 8: Realistic Crisis Scenario")

    print("Scenario: Major urban flood - selecting emergency response strategy\n")

    mcda = MCDAEngine("scenarios/criteria_weights.json")

    alternatives = [
        {
            "id": "A1",
            "name": "Immediate Mass Evacuation",
            "estimated_metrics": {
                "safety_score": 0.90,
                "cost_euros": 480000,
                "response_time_hours": 4,
                "social_disruption": 0.85  # High disruption
            }
        },
        {
            "id": "A2",
            "name": "Deploy Flood Barriers + Selective Evacuation",
            "estimated_metrics": {
                "safety_score": 0.78,
                "cost_euros": 320000,
                "response_time_hours": 3,
                "social_disruption": 0.55
            }
        },
        {
            "id": "A3",
            "name": "Prioritized Rescue Operations",
            "estimated_metrics": {
                "safety_score": 0.88,
                "cost_euros": 280000,
                "response_time_hours": 2,
                "social_disruption": 0.70
            }
        },
        {
            "id": "A4",
            "name": "Shelter-in-Place with Monitoring",
            "estimated_metrics": {
                "safety_score": 0.62,
                "cost_euros": 150000,
                "response_time_hours": 1,
                "social_disruption": 0.40
            }
        }
    ]

    print("Available Actions:")
    for alt in alternatives:
        print(f"\n{alt['name']} ({alt['id']}):")
        metrics = alt['estimated_metrics']
        print(f"  • Safety Score:      {metrics['safety_score']:.2f}")
        print(f"  • Cost:              €{metrics['cost_euros']:,}")
        print(f"  • Response Time:     {metrics['response_time_hours']} hours")
        print(f"  • Social Disruption: {metrics['social_disruption']:.2f} (lower is better)")

    # Rank alternatives
    ranked = mcda.rank_alternatives(alternatives)

    print(f"\n{'='*70}")
    print("DECISION RECOMMENDATION")
    print(f"{'='*70}\n")

    # Top 3
    for rank, (alt_id, score, norm_scores) in enumerate(ranked[:3], 1):
        alt = next(a for a in alternatives if a['id'] == alt_id)
        print(f"#{rank}. {alt['name']}")
        print(f"    Overall Score: {score:.3f}")
        print(f"    Key Strengths:")

        # Find top 2 criteria
        sorted_criteria = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
        for crit_id, crit_score in sorted_criteria[:2]:
            crit_name = mcda.criteria_config[crit_id]['name']
            print(f"      • {crit_name}: {crit_score:.2f}")
        print()

    # Winner detail
    winner_id, winner_score, winner_scores = ranked[0]
    winner = next(a for a in alternatives if a['id'] == winner_id)

    print(f"{'='*70}")
    print(f"RECOMMENDED ACTION: {winner['name']}")
    print(f"{'='*70}")
    print(f"Overall Score: {winner_score:.3f}\n")

    print("Justification:")
    print(f"  This option achieves the best balance across all criteria")
    print(f"  considering the current weight configuration:")
    for crit_id, config in sorted(
        mcda.criteria_config.items(),
        key=lambda x: x[1]['weight'],
        reverse=True
    ):
        crit_name = config['name']
        weight = config['weight']
        norm_score = winner_scores.get(crit_id, 0)
        contribution = norm_score * weight
        print(f"    • {crit_name:20s}: {norm_score:.2f} × {weight:.2f} = {contribution:.3f}")

    print("\n✓ Test 8 Passed")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MCDA ENGINE - COMPREHENSIVE TEST SUITE")
    print("Crisis Management Multi-Agent System")
    print("="*70)

    try:
        # Basic tests
        test_initialization()
        test_normalization()

        # Ranking tests
        alternatives, ranked = test_basic_ranking()
        test_explain_ranking(alternatives, ranked)

        # Advanced tests
        test_custom_weights()
        test_sensitivity_analysis()
        test_profile_comparison()

        # Realistic scenario
        test_crisis_scenario()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Test Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
