#!/usr/bin/env python3
"""
Test script to verify single-agent vs multi-agent baseline comparison.
This creates mock decisions to test the comparison logic.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import evaluate_decision


def test_baseline_comparison():
    """Test that baseline comparison works correctly with mock data."""

    # Mock multi-agent decision (better performance)
    multi_agent_decision = {
        'recommended_alternative': 'action_hybrid_approach',
        'confidence': 0.85,
        'consensus_level': 0.78,
        'consensus_reached': True,
        'final_scores': {
            'action_evacuate_immediate': 0.20,
            'action_deploy_barriers': 0.15,
            'action_rescue_operations': 0.25,
            'action_shelter_in_place': 0.05,
            'action_hybrid_approach': 0.35
        },
        'agent_opinions': {
            'agent_meteorologist': {
                'agent_name': 'Dr. Sarah Chen',
                'preference': 'action_hybrid_approach',
                'confidence': 0.82
            },
            'logistics_expert_01': {
                'agent_name': 'Jennifer Rodriguez',
                'preference': 'action_hybrid_approach',
                'confidence': 0.88
            },
            'medical_expert_01': {
                'agent_name': 'Dr. Marcus Williams',
                'preference': 'action_rescue_operations',
                'confidence': 0.85
            }
        },
        'collection_info': {
            'assessments': {
                'agent_meteorologist': {
                    'confidence': 0.82,
                    'belief_distribution': {
                        'action_evacuate_immediate': 0.15,
                        'action_deploy_barriers': 0.20,
                        'action_rescue_operations': 0.30,
                        'action_shelter_in_place': 0.05,
                        'action_hybrid_approach': 0.30
                    }
                },
                'logistics_expert_01': {
                    'confidence': 0.88,
                    'belief_distribution': {
                        'action_evacuate_immediate': 0.20,
                        'action_deploy_barriers': 0.10,
                        'action_rescue_operations': 0.20,
                        'action_shelter_in_place': 0.05,
                        'action_hybrid_approach': 0.45
                    }
                },
                'medical_expert_01': {
                    'confidence': 0.85,
                    'belief_distribution': {
                        'action_evacuate_immediate': 0.25,
                        'action_deploy_barriers': 0.15,
                        'action_rescue_operations': 0.35,
                        'action_shelter_in_place': 0.05,
                        'action_hybrid_approach': 0.20
                    }
                }
            }
        },
        'decision_time_seconds': 2.5,
        'timestamp': '2025-01-15T10:00:00'
    }

    # Mock single-agent baseline (lower performance)
    single_agent_baseline = {
        'recommended_alternative': 'action_rescue_operations',
        'confidence': 0.70,
        'final_scores': {
            'action_evacuate_immediate': 0.20,
            'action_deploy_barriers': 0.15,
            'action_rescue_operations': 0.40,
            'action_shelter_in_place': 0.10,
            'action_hybrid_approach': 0.15
        },
        'belief_distribution': {
            'action_evacuate_immediate': 0.20,
            'action_deploy_barriers': 0.15,
            'action_rescue_operations': 0.40,
            'action_shelter_in_place': 0.10,
            'action_hybrid_approach': 0.15
        },
        'reasoning': 'Immediate rescue is critical to save lives.',
        'agent_info': {
            'agent_id': 'agent_meteorologist',
            'agent_name': 'Dr. Sarah Chen',
            'agent_role': 'Meteorologist',
            'expertise': 'weather_forecasting'
        },
        'decision_time_seconds': 1.2,
        'timestamp': '2025-01-15T10:00:00',
        'baseline_type': 'single_agent'
    }

    print("="*80)
    print("TESTING BASELINE COMPARISON")
    print("="*80)
    print()
    print("Multi-Agent Decision:")
    print(f"  Recommendation: {multi_agent_decision['recommended_alternative']}")
    print(f"  Confidence: {multi_agent_decision['confidence']:.3f}")
    print(f"  Consensus: {multi_agent_decision['consensus_level']:.3f}")
    print()
    print("Single-Agent Baseline:")
    print(f"  Recommendation: {single_agent_baseline['recommended_alternative']}")
    print(f"  Confidence: {single_agent_baseline['confidence']:.3f}")
    print(f"  Agent: {single_agent_baseline['agent_info']['agent_name']}")
    print()

    # Run evaluation with baseline comparison
    metrics = evaluate_decision(
        multi_agent_decision,
        baseline_assessment=single_agent_baseline
    )

    print()
    print("="*80)
    print("TEST RESULTS")
    print("="*80)

    if 'baseline_comparison' in metrics:
        print("✓ Baseline comparison was performed")

        if 'baseline_metrics' in metrics:
            print("✓ Baseline metrics calculated")

            ma_quality = metrics['decision_quality']['weighted_score']
            sa_quality = metrics['baseline_metrics']['decision_quality']['weighted_score']

            print()
            print(f"Multi-Agent Quality:  {ma_quality:.3f}")
            print(f"Single-Agent Quality: {sa_quality:.3f}")

            if ma_quality > sa_quality:
                improvement = ((ma_quality - sa_quality) / sa_quality) * 100
                print(f"✓ Multi-agent shows {improvement:.1f}% improvement")
            else:
                print("✗ Multi-agent did not outperform single-agent")
        else:
            print("✗ Baseline metrics missing")
    else:
        print("✗ Baseline comparison not performed")

    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_baseline_comparison()
