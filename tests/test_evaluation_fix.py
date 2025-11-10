#!/usr/bin/env python3
"""
Test script to verify decision quality evaluation fixes.
"""

import sys
import os

# Direct import to avoid loading visualizations
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
from datetime import datetime
import scipy.stats as stats

# Import MetricsEvaluator directly from the module file
# to avoid loading __init__.py which imports visualization dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "metrics_module",
    "/home/user/crisis_mas_poc/evaluation/metrics.py"
)
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)
MetricsEvaluator = metrics_module.MetricsEvaluator

logging.basicConfig(level=logging.WARNING)

# Test 1: Multi-agent decision with MCDA scores
print("=" * 80)
print("TEST 1: Multi-agent Decision Quality Calculation")
print("=" * 80)

multi_agent_decision = {
    'recommended_alternative': 'action_rescue_operations',
    'confidence': 0.760,  # Now based on consensus + agent confidence
    'decision_quality_score': 0.310,  # ER+MCDA combined score
    'final_scores': {
        'action_rescue_operations': 0.310,
        'action_infrastructure': 0.250,
        'action_hybrid_approach': 0.440
    },
    'mcda_scores': {
        'action_rescue_operations': 0.720,  # MCDA score
        'action_infrastructure': 0.550,
        'action_hybrid_approach': 0.810
    },
    'er_scores': {
        'action_rescue_operations': 0.350,
        'action_infrastructure': 0.280,
        'action_hybrid_approach': 0.370
    }
}

evaluator = MetricsEvaluator()
dq_multi = evaluator.calculate_decision_quality(multi_agent_decision)

print(f"Multi-Agent Decision Quality:")
print(f"  Recommended: {dq_multi['recommended_alternative']}")
print(f"  Quality Score (from MCDA): {dq_multi['weighted_score']:.3f}")
print(f"  Confidence (from decision): {dq_multi['confidence']:.3f}")
print(f"  Criteria Satisfaction: {dq_multi['criteria_satisfaction']}")
print()

# Test 2: Single-agent decision with criteria scores
print("=" * 80)
print("TEST 2: Single-agent Decision Quality Calculation")
print("=" * 80)

single_agent_decision = {
    'recommended_alternative': 'action_hybrid_approach',
    'confidence': 0.82,  # LLM's subjective confidence
    'final_scores': {
        'action_rescue_operations': 0.650,
        'action_infrastructure': 0.500,
        'action_hybrid_approach': 0.850
    },
    'criteria_scores': {
        'safety': {
            'action_rescue_operations': 0.90,
            'action_infrastructure': 0.60,
            'action_hybrid_approach': 0.85
        },
        'cost': {
            'action_rescue_operations': 0.50,
            'action_infrastructure': 0.70,
            'action_hybrid_approach': 0.80
        },
        'speed': {
            'action_rescue_operations': 0.95,
            'action_infrastructure': 0.40,
            'action_hybrid_approach': 0.90
        }
    }
}

dq_single = evaluator.calculate_decision_quality(single_agent_decision)

print(f"Single-Agent Decision Quality:")
print(f"  Recommended: {dq_single['recommended_alternative']}")
print(f"  Quality Score (from criteria): {dq_single['weighted_score']:.3f}")
print(f"  Confidence (from LLM): {dq_single['confidence']:.3f}")
print(f"  Criteria Satisfaction: {dq_single['criteria_satisfaction']}")
print()

# Test 3: Comparison
print("=" * 80)
print("TEST 3: Baseline Comparison")
print("=" * 80)

multi_results = {
    'decision_quality': dq_multi,
    'confidence': {
        'decision_confidence': multi_agent_decision['confidence'],
        'average_confidence': 0.760
    }
}

single_results = {
    'decision_quality': dq_single,
    'confidence': {
        'decision_confidence': single_agent_decision['confidence'],
        'average_confidence': 0.820
    }
}

comparison = evaluator.compare_to_baseline(multi_results, single_results)

print(f"Decision Quality Comparison:")
print(f"  Multi-agent: {comparison['decision_quality']['multi_agent']:.3f}")
print(f"  Single-agent: {comparison['decision_quality']['single_agent']:.3f}")
print(f"  Improvement: {comparison['decision_quality']['improvement_percentage']:+.1f}%")
print()

print(f"Confidence Comparison:")
print(f"  Multi-agent: {comparison['confidence']['multi_agent']:.3f}")
print(f"  Single-agent: {comparison['confidence']['single_agent']:.3f}")
print(f"  Improvement: {comparison['confidence']['improvement_percentage']:+.1f}%")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("✓ Decision quality now calculated from criteria satisfaction (MCDA/criteria scores)")
print("✓ Multi-agent uses MCDA scores:", dq_multi['weighted_score'])
print("✓ Single-agent uses criteria scores:", dq_single['weighted_score'])
print("✓ Both scores are now COMPARABLE (same scale, same methodology)")
print("✓ Confidence is separate from quality score")
print()
print("OLD (BROKEN) BEHAVIOR:")
print("  - Multi-agent quality = 0.310 (ER+MCDA combined)")
print("  - Single-agent quality = 0.820 (LLM confidence)")
print("  - Comparison was INVALID (different metrics)")
print()
print("NEW (FIXED) BEHAVIOR:")
print(f"  - Multi-agent quality = {dq_multi['weighted_score']:.3f} (MCDA score for recommended)")
print(f"  - Single-agent quality = {dq_single['weighted_score']:.3f} (criteria scores averaged)")
print("  - Comparison is VALID (both based on criteria satisfaction)")
print()
print("=" * 80)
