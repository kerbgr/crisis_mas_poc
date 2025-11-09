#!/usr/bin/env python3
"""
Debug script to test single-agent criteria scores evaluation.
"""

import sys
import importlib.util

# Import MetricsEvaluator directly
spec = importlib.util.spec_from_file_location(
    "metrics_module",
    "/home/user/crisis_mas_poc/evaluation/metrics.py"
)
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)
MetricsEvaluator = metrics_module.MetricsEvaluator

# Create evaluator
evaluator = MetricsEvaluator()

print("=" * 80)
print("DEBUGGING SINGLE-AGENT CRITERIA SCORES")
print("=" * 80)
print()

# Test Case 1: Nested format (from expert agent)
print("Test 1: Nested criteria_scores format (expert agent format)")
print("-" * 80)

single_agent_nested = {
    'recommended_alternative': 'A1',
    'confidence': 0.80,
    'final_scores': {'A1': 0.7, 'A2': 0.5, 'A3': 0.3},
    'criteria_scores': {
        'safety': {'A1': 0.90, 'A2': 0.75, 'A3': 0.60},
        'cost': {'A1': 0.50, 'A2': 0.80, 'A3': 0.70},
        'speed': {'A1': 0.95, 'A2': 0.65, 'A3': 0.50}
    }
}

print(f"Recommended: {single_agent_nested['recommended_alternative']}")
print(f"Criteria scores: {single_agent_nested['criteria_scores']}")
print()

dq1 = evaluator.calculate_decision_quality(single_agent_nested)
print(f"Result:")
print(f"  weighted_score: {dq1['weighted_score']:.3f}")
print(f"  criteria_satisfaction: {dq1['criteria_satisfaction']}")
print(f"  Expected: (0.90 + 0.50 + 0.95) / 3 = {(0.90 + 0.50 + 0.95) / 3:.3f}")
print()

# Test Case 2: Empty criteria_scores
print("=" * 80)
print("Test 2: Empty criteria_scores")
print("-" * 80)

single_agent_empty = {
    'recommended_alternative': 'A1',
    'confidence': 0.80,
    'final_scores': {'A1': 0.7, 'A2': 0.5, 'A3': 0.3},
    'criteria_scores': {}
}

print(f"Recommended: {single_agent_empty['recommended_alternative']}")
print(f"Criteria scores: {single_agent_empty['criteria_scores']}")
print()

dq2 = evaluator.calculate_decision_quality(single_agent_empty)
print(f"Result:")
print(f"  weighted_score: {dq2['weighted_score']:.3f}")
print(f"  Should fallback to final_scores['A1'] = 0.7")
print()

# Test Case 3: Mismatched alternative ID
print("=" * 80)
print("Test 3: Alternative ID mismatch")
print("-" * 80)

single_agent_mismatch = {
    'recommended_alternative': 'action_1',  # Different ID
    'confidence': 0.80,
    'final_scores': {'action_1': 0.7, 'action_2': 0.5},
    'criteria_scores': {
        'safety': {'A1': 0.90, 'A2': 0.75},  # Different IDs!
        'cost': {'A1': 0.50, 'A2': 0.80},
        'speed': {'A1': 0.95, 'A2': 0.65}
    }
}

print(f"Recommended: {single_agent_mismatch['recommended_alternative']}")
print(f"Criteria scores use IDs: {list(next(iter(single_agent_mismatch['criteria_scores'].values())).keys())}")
print()

dq3 = evaluator.calculate_decision_quality(single_agent_mismatch)
print(f"Result:")
print(f"  weighted_score: {dq3['weighted_score']:.3f}")
print(f"  Should fallback to final_scores['action_1'] = 0.7")
print()

# Test Case 4: What we expect from actual expert agent
print("=" * 80)
print("Test 4: Typical expert agent assessment")
print("-" * 80)

# This simulates what actually comes from expert_agent.evaluate_scenario()
typical_assessment = {
    'agent_id': 'agent_medical_01',
    'agent_name': 'Medical Expert',
    'belief_distribution': {
        'action_rescue_operations': 0.65,
        'action_infrastructure': 0.25,
        'action_hybrid_approach': 0.10
    },
    'criteria_scores': {
        'criterion_safety': {
            'action_rescue_operations': 0.90,
            'action_infrastructure': 0.60,
            'action_hybrid_approach': 0.85
        },
        'criterion_cost': {
            'action_rescue_operations': 0.50,
            'action_infrastructure': 0.70,
            'action_hybrid_approach': 0.80
        },
        'criterion_speed': {
            'action_rescue_operations': 0.95,
            'action_infrastructure': 0.40,
            'action_hybrid_approach': 0.90
        }
    },
    'confidence': 0.82,
    'reasoning': '...'
}

# Single-agent baseline decision
baseline = {
    'recommended_alternative': 'action_rescue_operations',
    'confidence': typical_assessment['confidence'],
    'final_scores': typical_assessment['belief_distribution'],
    'criteria_scores': typical_assessment['criteria_scores']
}

print(f"Recommended: {baseline['recommended_alternative']}")
print(f"Has criteria_scores: {bool(baseline['criteria_scores'])}")
print(f"Criteria count: {len(baseline['criteria_scores'])}")
print()

dq4 = evaluator.calculate_decision_quality(baseline)
print(f"Result:")
print(f"  weighted_score: {dq4['weighted_score']:.3f}")
print(f"  criteria_satisfaction: {dq4['criteria_satisfaction']}")
print(f"  Expected: (0.90 + 0.50 + 0.95) / 3 = {(0.90 + 0.50 + 0.95) / 3:.3f}")
print()

print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if dq2['weighted_score'] == 0.0:
    print("✗ Empty criteria_scores is falling through to 0.0 instead of using fallback")
else:
    print("✓ Empty criteria_scores properly falls back to final_scores")

if dq3['weighted_score'] == 0.0:
    print("✗ ID mismatch returns 0.0 instead of using fallback")
elif dq3['weighted_score'] == 0.7:
    print("✓ ID mismatch properly falls back to final_scores")

if dq4['weighted_score'] > 0.0:
    print(f"✓ Typical expert agent assessment works: {dq4['weighted_score']:.3f}")
else:
    print(f"✗ Typical expert agent assessment returns 0.0")

print()
print("If single-agent DQS is 0.0 in actual run, check:")
print("1. Is criteria_scores actually empty in the assessment?")
print("2. Is there an alternative ID mismatch?")
print("3. Is final_scores also missing the recommended alternative?")
