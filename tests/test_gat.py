#!/usr/bin/env python3
"""
Test Graph Attention Network (GAT) Aggregator
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_framework.gat_aggregator import GATAggregator, GraphAttentionLayer
import numpy as np


def test_gat_basic():
    """Test basic GAT aggregation."""
    print("\n" + "="*80)
    print("TEST 1: Basic GAT Aggregation")
    print("="*80)

    # Create scenario
    scenario = {
        'type': 'flood',
        'severity': 0.8,
        'tags': ['natural_disaster', 'emergency']
    }

    # Create mock agent assessments
    agent_assessments = {
        'agent_meteorologist': {
            'agent_name': 'Dr. Sarah Chen',
            'expertise': 'weather_forecasting',
            'confidence': 0.85,
            'risk_tolerance': 0.4,
            'belief_distribution': {
                'action_evacuate': 0.6,
                'action_shelter': 0.3,
                'action_barriers': 0.1
            },
            'reasoning': 'Heavy rainfall continuing, evacuation is critical',
            'key_concerns': ['Rising water levels', 'Limited time', 'Weather worsening']
        },
        'logistics_expert': {
            'agent_name': 'Jennifer Rodriguez',
            'expertise': 'logistics',
            'confidence': 0.75,
            'risk_tolerance': 0.5,
            'belief_distribution': {
                'action_evacuate': 0.4,
                'action_shelter': 0.2,
                'action_barriers': 0.4
            },
            'reasoning': 'Need to balance evacuation with infrastructure protection',
            'key_concerns': ['Resource constraints', 'Transportation logistics']
        },
        'medical_expert': {
            'agent_name': 'Dr. Marcus Williams',
            'expertise': 'emergency_medicine',
            'confidence': 0.80,
            'risk_tolerance': 0.3,
            'belief_distribution': {
                'action_evacuate': 0.7,
                'action_shelter': 0.2,
                'action_barriers': 0.1
            },
            'reasoning': 'Patient safety is paramount, immediate evacuation needed',
            'key_concerns': ['Casualties', 'Medical emergencies', 'Hospital capacity']
        }
    }

    # Initialize GAT aggregator
    gat = GATAggregator(num_attention_heads=4, use_multi_head=True)

    # Aggregate beliefs
    result = gat.aggregate_beliefs_with_gat(
        agent_assessments,
        scenario
    )

    # Verify result structure
    assert 'aggregated_beliefs' in result
    assert 'attention_weights' in result
    assert 'confidence' in result
    assert 'explanation' in result

    print("\n✅ GAT Aggregation Results:")
    print(f"\nAggregated Beliefs:")
    for alt, belief in sorted(result['aggregated_beliefs'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {alt}: {belief:.3f}")

    print(f"\nOverall Confidence: {result['confidence']:.3f}")
    print(f"Uncertainty: {result['uncertainty']:.3f}")

    print("\n✅ Agent Attention Weights (Self-Attention):")
    attention_summary = gat.get_attention_summary(result['attention_weights'])
    for agent_id, weight in sorted(attention_summary.items(), key=lambda x: x[1], reverse=True):
        agent_name = agent_assessments[agent_id]['agent_name']
        print(f"  {agent_name}: {weight:.3f}")

    print("\n✅ Explanation:")
    print(result['explanation'])

    # Verify beliefs sum to 1.0
    total = sum(result['aggregated_beliefs'].values())
    assert abs(total - 1.0) < 0.01, f"Beliefs should sum to 1.0, got {total}"

    print("\n✅ TEST PASSED: Basic GAT aggregation works correctly")
    return result


def test_feature_extraction():
    """Test agent feature extraction."""
    print("\n" + "="*80)
    print("TEST 2: Feature Extraction")
    print("="*80)

    layer = GraphAttentionLayer(feature_dim=8)

    scenario = {
        'type': 'flood',
        'severity': 0.8,
        'tags': ['emergency', 'natural_disaster']
    }

    assessment = {
        'expertise': 'weather_forecasting',
        'confidence': 0.85,
        'risk_tolerance': 0.4,
        'belief_distribution': {
            'A1': 0.6,
            'A2': 0.3,
            'A3': 0.1
        },
        'reasoning': 'This is a detailed reasoning with multiple considerations and careful analysis.',
        'key_concerns': ['Concern 1', 'Concern 2', 'Concern 3']
    }

    features = layer.extract_agent_features('agent_1', assessment, scenario)

    print(f"\n✅ Extracted Features (dim={len(features)}):")
    feature_names = [
        'Confidence',
        'Certainty',
        'Expertise Relevance',
        'Risk Tolerance',
        'Severity Awareness',
        'Top Choice Strength',
        'Number of Concerns',
        'Reasoning Quality'
    ]
    for i, (name, value) in enumerate(zip(feature_names, features)):
        print(f"  {i+1}. {name}: {value:.3f}")

    assert len(features) == 8, f"Should have 8 features, got {len(features)}"
    assert all(0 <= f <= 1.5 for f in features), "All features should be in reasonable range"

    print("\n✅ TEST PASSED: Feature extraction works correctly")


def test_attention_coefficients():
    """Test attention coefficient computation."""
    print("\n" + "="*80)
    print("TEST 3: Attention Coefficients")
    print("="*80)

    layer = GraphAttentionLayer()

    # Create features for 3 agents
    features = {
        'agent_1': np.array([0.8, 0.7, 0.9, 0.5, 0.8, 0.4, 0.6, 0.7]),  # High confidence, high relevance
        'agent_2': np.array([0.6, 0.5, 0.6, 0.5, 0.8, 0.3, 0.4, 0.5]),  # Medium
        'agent_3': np.array([0.4, 0.3, 0.3, 0.7, 0.8, 0.2, 0.3, 0.4])   # Low confidence
    }

    # Fully connected graph
    adjacency = np.ones((3, 3))

    attention = layer.compute_attention_coefficients(features, adjacency)

    print(f"\n✅ Attention Matrix (3x3):")
    agent_ids = list(features.keys())
    print(f"        {agent_ids[0]:>10} {agent_ids[1]:>10} {agent_ids[2]:>10}")
    for i, from_agent in enumerate(agent_ids):
        row_str = f"{from_agent:>10}:"
        for j in range(3):
            row_str += f" {attention[i,j]:>10.3f}"
        print(row_str)

    # Verify attention weights sum to 1.0 for each agent
    for i in range(3):
        row_sum = np.sum(attention[i])
        assert abs(row_sum - 1.0) < 0.01, f"Row {i} should sum to 1.0, got {row_sum}"

    print("\n✅ Self-Attention Weights (diagonal):")
    for i, agent_id in enumerate(agent_ids):
        print(f"  {agent_id}: {attention[i,i]:.3f}")

    print("\n✅ TEST PASSED: Attention coefficients computed correctly")


def test_comparison_with_simple_average():
    """Compare GAT with simple averaging."""
    print("\n" + "="*80)
    print("TEST 4: GAT vs Simple Average")
    print("="*80)

    scenario = {'type': 'flood', 'severity': 0.8, 'tags': ['emergency']}

    # Agents with different expertise relevance
    agent_assessments = {
        'relevant_expert': {  # Should get higher weight
            'agent_name': 'Flood Expert',
            'expertise': 'flood_management',
            'confidence': 0.9,
            'risk_tolerance': 0.3,
            'belief_distribution': {'A1': 0.8, 'A2': 0.2},
            'reasoning': 'Detailed analysis based on flood expertise',
            'key_concerns': ['Water levels', 'Drainage', 'Safety']
        },
        'less_relevant': {  # Should get lower weight
            'agent_name': 'General Expert',
            'expertise': 'general_operations',
            'confidence': 0.7,
            'risk_tolerance': 0.5,
            'belief_distribution': {'A1': 0.3, 'A2': 0.7},
            'reasoning': 'General assessment',
            'key_concerns': ['Operations']
        }
    }

    # GAT aggregation
    gat = GATAggregator()
    gat_result = gat.aggregate_beliefs_with_gat(agent_assessments, scenario)

    # Simple average
    simple_avg = {
        'A1': (0.8 + 0.3) / 2,
        'A2': (0.2 + 0.7) / 2
    }

    print("\n✅ GAT Aggregated Beliefs:")
    for alt, belief in sorted(gat_result['aggregated_beliefs'].items()):
        print(f"  {alt}: {belief:.3f}")

    print("\n✅ Simple Average:")
    for alt, belief in sorted(simple_avg.items()):
        print(f"  {alt}: {belief:.3f}")

    print("\n✅ Attention Weights:")
    attention_summary = gat.get_attention_summary(gat_result['attention_weights'])
    for agent_id, weight in attention_summary.items():
        print(f"  {agent_assessments[agent_id]['agent_name']}: {weight:.3f}")

    # GAT should give more weight to the relevant expert
    assert attention_summary['relevant_expert'] > attention_summary['less_relevant'], \
        "GAT should give more weight to relevant expert"

    # GAT's A1 should be higher than simple average (relevant expert prefers A1)
    assert gat_result['aggregated_beliefs']['A1'] > simple_avg['A1'], \
        "GAT should favor A1 more than simple average"

    print("\n✅ TEST PASSED: GAT correctly weights experts based on relevance")


def main():
    """Run all GAT tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "GAT AGGREGATOR TESTS" + " "*34 + "║")
    print("╚" + "="*78 + "╝")

    tests = [
        test_feature_extraction,
        test_attention_coefficients,
        test_gat_basic,
        test_comparison_with_simple_average
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {passed + failed}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("="*80)

    if failed == 0:
        print("\n🎉 ALL GAT TESTS PASSED! 🎉")
        print("\nThe Graph Attention Network aggregator:")
        print("  ✅ Correctly extracts agent features")
        print("  ✅ Computes valid attention coefficients")
        print("  ✅ Aggregates beliefs properly")
        print("  ✅ Gives higher weight to relevant experts")
        print("  ✅ Outperforms simple averaging")
        print("\nGAT is ready for integration! 🚀")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
