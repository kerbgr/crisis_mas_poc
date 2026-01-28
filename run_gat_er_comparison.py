#!/usr/bin/env python3
"""
GAT vs ER Comparative Evaluation Script

This script compares Graph Attention Network (GAT) and Evidential Reasoning (ER)
aggregation methods using simulated agent assessments that mirror the structure
and quality of real LLM-based agent outputs.

Usage:
    python run_gat_er_comparison.py

Output:
    - results/er/ : ER experiment results
    - results/gat/ : GAT experiment results
    - Comparative analysis printed to console
"""

import json
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import decision framework components
from decision_framework.evidential_reasoning import EvidentialReasoning
from decision_framework.gat_aggregator import GATAggregator
from decision_framework.mcda_engine import MCDAEngine
from decision_framework.consensus_model import ConsensusModel


def load_scenario(scenario_name: str) -> Dict[str, Any]:
    """Load scenario from JSON file."""
    scenarios_dir = Path("scenarios")
    scenario_path = scenarios_dir / f"{scenario_name}.json"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")

    with open(scenario_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_simulated_assessments(
    scenario: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Generate simulated agent assessments based on scenario characteristics.

    These assessments mirror the structure and quality of real LLM-based
    agent outputs, with belief distributions, confidence scores, and reasoning.
    """
    np.random.seed(seed)

    scenario_type = scenario.get('type', 'unknown')
    severity = scenario.get('severity', 0.5)
    alternatives = scenario.get('available_actions', [])

    if not alternatives:
        raise ValueError("No alternatives found in scenario")

    alt_ids = [a['id'] for a in alternatives]

    # Define agent profiles with expertise mapping
    agent_profiles = {
        'meteorology_silver_advisory': {
            'name': 'Meteorology-Silver-Advisory',
            'expertise': 'weather_environment',
            'relevance': {'flood': 0.95, 'wildfire': 0.90, 'hazmat': 0.70}
        },
        'fire_silver_tactical': {
            'name': 'Fire-Silver-Tactical',
            'expertise': 'fire_rescue',
            'relevance': {'flood': 0.60, 'wildfire': 0.95, 'hazmat': 0.85}
        },
        'fire_gold_strategic': {
            'name': 'Fire-Gold-Strategic',
            'expertise': 'fire_rescue',
            'relevance': {'flood': 0.55, 'wildfire': 0.90, 'hazmat': 0.80}
        },
        'medical_bronze_operational': {
            'name': 'Medical-Bronze-Operational',
            'expertise': 'medical_health',
            'relevance': {'flood': 0.75, 'wildfire': 0.70, 'hazmat': 0.90}
        },
        'medical_gold_strategic': {
            'name': 'Medical-Gold-Strategic',
            'expertise': 'medical_health',
            'relevance': {'flood': 0.70, 'wildfire': 0.65, 'hazmat': 0.85}
        },
        'logistics_silver_tactical': {
            'name': 'Logistics-Silver-Tactical',
            'expertise': 'logistics',
            'relevance': {'flood': 0.80, 'wildfire': 0.75, 'hazmat': 0.75}
        },
        'police_silver_tactical': {
            'name': 'Police-Silver-Tactical',
            'expertise': 'law_enforcement',
            'relevance': {'flood': 0.70, 'wildfire': 0.65, 'hazmat': 0.80}
        },
        'police_gold_strategic': {
            'name': 'Police-Gold-Strategic',
            'expertise': 'law_enforcement',
            'relevance': {'flood': 0.65, 'wildfire': 0.60, 'hazmat': 0.75}
        },
        'civilprotection_gold_strategic': {
            'name': 'CivilProtection-Gold-Strategic',
            'expertise': 'civil_protection',
            'relevance': {'flood': 0.85, 'wildfire': 0.85, 'hazmat': 0.85}
        },
        'psap_gold_strategic': {
            'name': 'PSAP-Gold-Strategic',
            'expertise': 'emergency_communications',
            'relevance': {'flood': 0.75, 'wildfire': 0.75, 'hazmat': 0.80}
        },
        'coastguard_silver_tactical': {
            'name': 'CoastGuard-Silver-Tactical',
            'expertise': 'maritime_operations',
            'relevance': {'flood': 0.60, 'wildfire': 0.30, 'hazmat': 0.55}
        },
        'coastguard_gold_strategic': {
            'name': 'CoastGuard-Gold-Strategic',
            'expertise': 'maritime_operations',
            'relevance': {'flood': 0.55, 'wildfire': 0.25, 'hazmat': 0.50}
        },
        'environmental_silver': {
            'name': 'Environmental-Silver',
            'expertise': 'environmental',
            'relevance': {'flood': 0.80, 'wildfire': 0.85, 'hazmat': 0.90}
        }
    }

    assessments = {}

    for agent_id, profile in agent_profiles.items():
        # Get relevance for this scenario type
        relevance = profile['relevance'].get(scenario_type, 0.5)

        # Generate belief distribution based on alternative scores
        beliefs = {}
        raw_scores = []

        for alt in alternatives:
            alt_id = alt['id']
            criteria_scores = alt.get('criteria_scores', {})

            # Calculate base score from criteria
            if criteria_scores:
                base_score = np.mean(list(criteria_scores.values()))
            else:
                base_score = 0.5

            # Add agent-specific variation
            noise = np.random.normal(0, 0.1)
            score = np.clip(base_score + noise * (1 - relevance), 0.05, 0.95)
            raw_scores.append((alt_id, score))

        # Normalize to sum to 1.0
        total = sum(s for _, s in raw_scores)
        for alt_id, score in raw_scores:
            beliefs[alt_id] = score / total

        # Calculate confidence based on relevance and severity
        confidence = np.clip(0.6 + relevance * 0.3 + np.random.normal(0, 0.05), 0.5, 0.95)

        # Generate assessment
        top_alt = max(beliefs.items(), key=lambda x: x[1])

        assessments[agent_id] = {
            'agent_id': agent_id,
            'agent_name': profile['name'],
            'expertise': profile['expertise'],
            'belief_distribution': beliefs,
            'confidence': confidence,
            'risk_tolerance': np.clip(0.5 + np.random.normal(0, 0.15), 0.2, 0.8),
            'key_concerns': [
                f"Concern related to {scenario_type}",
                f"Resource availability for {top_alt[0]}",
                "Time constraints and coordination"
            ],
            'reasoning': f"Based on {profile['expertise']} expertise, recommending {top_alt[0]} "
                        f"with {top_alt[1]:.1%} belief due to scenario severity {severity:.1%}.",
            'reliability_score': np.clip(0.75 + np.random.normal(0, 0.1), 0.6, 0.95)
        }

    return assessments


def run_er_aggregation(
    assessments: Dict[str, Dict[str, Any]],
    scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """Run Evidential Reasoning aggregation."""
    er_engine = EvidentialReasoning()

    start_time = datetime.now()

    # Extract belief distributions
    agent_beliefs = {
        agent_id: assessment['belief_distribution']
        for agent_id, assessment in assessments.items()
    }

    # Equal weights for all agents
    weights = {agent_id: 1.0 / len(assessments) for agent_id in assessments}

    # Combine beliefs
    er_result = er_engine.combine_beliefs(agent_beliefs, weights)

    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    return {
        'method': 'ER',
        'aggregated_beliefs': er_result.get('combined_beliefs', {}),
        'confidence': er_result.get('confidence', 0.0),
        'uncertainty': er_result.get('uncertainty', 1.0),
        'processing_time_ms': processing_time,
        'er_details': er_result
    }


def run_gat_aggregation(
    assessments: Dict[str, Dict[str, Any]],
    scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """Run Graph Attention Network aggregation."""
    gat = GATAggregator(num_attention_heads=4, use_multi_head=True)

    start_time = datetime.now()

    gat_result = gat.aggregate_beliefs_with_gat(assessments, scenario)

    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    # Extract top 3 influential agents
    attention_weights = gat_result.get('attention_weights', {})
    agent_importance = []
    for agent_id, weights in attention_weights.items():
        self_attention = weights.get(agent_id, 0.0)
        agent_importance.append((agent_id, self_attention))

    agent_importance.sort(key=lambda x: x[1], reverse=True)
    top_3_agents = agent_importance[:3]

    return {
        'method': 'GAT',
        'aggregated_beliefs': gat_result.get('aggregated_beliefs', {}),
        'confidence': gat_result.get('confidence', 0.0),
        'uncertainty': gat_result.get('uncertainty', 1.0),
        'processing_time_ms': processing_time,
        'attention_weights': attention_weights,
        'top_3_influential_agents': top_3_agents,
        'gat_details': gat_result
    }


def calculate_consensus(assessments: Dict[str, Dict[str, Any]]) -> float:
    """Calculate consensus level using cosine similarity."""
    consensus_model = ConsensusModel()

    agent_beliefs = {
        agent_id: assessment['belief_distribution']
        for agent_id, assessment in assessments.items()
    }

    return consensus_model.calculate_consensus_level(agent_beliefs)


def run_mcda(alternatives: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run MCDA scoring on alternatives."""
    mcda = MCDAEngine()
    rankings = mcda.rank_alternatives(alternatives)
    return {alt_id: score for alt_id, score, _ in rankings}


def calculate_decision_quality_score(
    aggregated_beliefs: Dict[str, float],
    mcda_scores: Dict[str, float]
) -> float:
    """Calculate combined decision quality score (60% ER/GAT + 40% MCDA)."""
    combined_scores = {}

    for alt_id in aggregated_beliefs:
        belief_score = aggregated_beliefs.get(alt_id, 0.0)
        mcda_score = mcda_scores.get(alt_id, 0.0)
        combined_scores[alt_id] = 0.6 * belief_score + 0.4 * mcda_score

    if combined_scores:
        return max(combined_scores.values())
    return 0.0


def run_experiment(
    scenario_name: str,
    method: str,
    seed: int
) -> Dict[str, Any]:
    """Run single experiment for one scenario with one method."""
    logger.info(f"Running {method.upper()} experiment for {scenario_name} (seed={seed})")

    # Load scenario
    scenario = load_scenario(scenario_name)
    alternatives = scenario.get('available_actions', [])

    # Generate assessments
    assessments = generate_simulated_assessments(scenario, seed=seed)

    # Run aggregation
    if method.lower() == 'er':
        agg_result = run_er_aggregation(assessments, scenario)
    else:
        agg_result = run_gat_aggregation(assessments, scenario)

    # Calculate metrics
    consensus_level = calculate_consensus(assessments)
    mcda_scores = run_mcda(alternatives)

    # Find recommended alternative
    aggregated_beliefs = agg_result['aggregated_beliefs']
    if aggregated_beliefs:
        recommended_alt = max(aggregated_beliefs.items(), key=lambda x: x[1])[0]
    else:
        recommended_alt = None

    # Calculate DQS
    dqs = calculate_decision_quality_score(aggregated_beliefs, mcda_scores)

    # Calculate average agent confidence
    avg_confidence = np.mean([a['confidence'] for a in assessments.values()])

    # Overall decision confidence (60% consensus + 40% avg agent confidence)
    decision_confidence = 0.6 * consensus_level + 0.4 * avg_confidence

    # Build result
    result = {
        'scenario': scenario_name,
        'scenario_type': scenario.get('type', 'unknown'),
        'aggregation_method': method.upper(),
        'seed': seed,
        'timestamp': datetime.now().isoformat(),

        'recommended_alternative': recommended_alt,
        'decision_quality_score': dqs,
        'consensus_level': consensus_level,
        'confidence': decision_confidence,
        'uncertainty': agg_result['uncertainty'],

        'aggregated_beliefs': aggregated_beliefs,
        'mcda_scores': mcda_scores,
        'processing_time_ms': agg_result['processing_time_ms'],

        'agents_participated': len(assessments),
        'agent_confidences': {
            agent_id: a['confidence']
            for agent_id, a in assessments.items()
        }
    }

    # Add method-specific details
    if method.lower() == 'gat':
        result['gat_metrics'] = {
            'attention_weights': agg_result.get('attention_weights', {}),
            'top_3_influential_agents': agg_result.get('top_3_influential_agents', [])
        }

    return result


def save_result(result: Dict[str, Any], output_dir: Path) -> str:
    """Save result to JSON file."""
    output_dir.mkdir(exist_ok=True, parents=True)

    scenario = result['scenario']
    method = result['aggregation_method'].lower()
    seed = result['seed']

    filename = f"{scenario}_run{seed}.json"
    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved result to: {filepath}")
    return str(filepath)


def print_comparison_table(results: Dict[str, List[Dict[str, Any]]]):
    """Print comparative analysis table."""
    print("\n" + "=" * 120)
    print("COMPARATIVE ANALYSIS: ER vs GAT AGGREGATION METHODS")
    print("=" * 120)

    # Group by scenario
    scenarios = set()
    for method_results in results.values():
        for r in method_results:
            scenarios.add(r['scenario'])

    # Header
    print(f"\n{'Metric':<30} | ", end="")
    for scenario in sorted(scenarios):
        print(f"{'ER':>12} {'GAT':>12} | ", end="")
    print()
    print("-" * 120)

    # Metrics to compare
    metrics = [
        ('Decision Quality Score', 'decision_quality_score'),
        ('Consensus Level', 'consensus_level'),
        ('Confidence', 'confidence'),
        ('Processing Time (ms)', 'processing_time_ms')
    ]

    for metric_name, metric_key in metrics:
        print(f"{metric_name:<30} | ", end="")

        for scenario in sorted(scenarios):
            er_vals = [r[metric_key] for r in results.get('er', []) if r['scenario'] == scenario]
            gat_vals = [r[metric_key] for r in results.get('gat', []) if r['scenario'] == scenario]

            er_mean = np.mean(er_vals) if er_vals else 0.0
            gat_mean = np.mean(gat_vals) if gat_vals else 0.0

            if metric_key == 'processing_time_ms':
                print(f"{er_mean:>10.1f}ms {gat_mean:>10.1f}ms | ", end="")
            else:
                print(f"{er_mean:>11.3f} {gat_mean:>11.3f} | ", end="")
        print()

    print("-" * 120)

    # Recommended alternatives
    print(f"\n{'Recommended Alternative':<30} | ", end="")
    for scenario in sorted(scenarios):
        er_alts = [r['recommended_alternative'] for r in results.get('er', []) if r['scenario'] == scenario]
        gat_alts = [r['recommended_alternative'] for r in results.get('gat', []) if r['scenario'] == scenario]

        er_alt = er_alts[0] if er_alts else "N/A"
        gat_alt = gat_alts[0] if gat_alts else "N/A"

        # Truncate long names
        er_short = er_alt[:12] if er_alt else "N/A"
        gat_short = gat_alt[:12] if gat_alt else "N/A"

        print(f"{er_short:>12} {gat_short:>12} | ", end="")
    print()

    print("\n" + "=" * 120)


def print_gat_attention_analysis(results: Dict[str, List[Dict[str, Any]]]):
    """Print GAT attention weight analysis."""
    print("\n" + "=" * 80)
    print("GAT ATTENTION WEIGHT ANALYSIS")
    print("=" * 80)

    gat_results = results.get('gat', [])

    for result in gat_results:
        scenario = result['scenario']
        gat_metrics = result.get('gat_metrics', {})
        top_agents = gat_metrics.get('top_3_influential_agents', [])

        print(f"\n{scenario}:")
        print("-" * 40)

        if top_agents:
            for i, (agent_id, weight) in enumerate(top_agents, 1):
                agent_name = agent_id.replace('_', ' ').title()
                print(f"  {i}. {agent_name}: {weight:.3f}")
        else:
            print("  No attention weights available")

    print("\n" + "=" * 80)


def print_statistical_summary(results: Dict[str, List[Dict[str, Any]]]):
    """Print statistical summary across all runs."""
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY (Mean ± Std across runs)")
    print("=" * 80)

    metrics = [
        ('Decision Quality Score', 'decision_quality_score'),
        ('Consensus Level', 'consensus_level'),
        ('Confidence', 'confidence'),
        ('Processing Time (ms)', 'processing_time_ms')
    ]

    for method in ['er', 'gat']:
        method_results = results.get(method, [])
        if not method_results:
            continue

        print(f"\n{method.upper()} Method:")
        print("-" * 40)

        for metric_name, metric_key in metrics:
            vals = [r[metric_key] for r in method_results]
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            if metric_key == 'processing_time_ms':
                print(f"  {metric_name}: {mean_val:.1f} ± {std_val:.1f} ms")
            else:
                print(f"  {metric_name}: {mean_val:.3f} ± {std_val:.3f}")

    # GAT vs ER comparison
    print("\n" + "-" * 40)
    print("GAT vs ER Difference (Δ = GAT - ER):")
    print("-" * 40)

    er_results = results.get('er', [])
    gat_results = results.get('gat', [])

    if er_results and gat_results:
        for metric_name, metric_key in metrics:
            er_mean = np.mean([r[metric_key] for r in er_results])
            gat_mean = np.mean([r[metric_key] for r in gat_results])
            delta = gat_mean - er_mean

            if metric_key == 'processing_time_ms':
                print(f"  {metric_name}: Δ = {delta:+.1f} ms")
            else:
                pct_change = (delta / er_mean * 100) if er_mean != 0 else 0
                print(f"  {metric_name}: Δ = {delta:+.4f} ({pct_change:+.1f}%)")

    print("\n" + "=" * 80)


def main():
    """Main entry point for comparative evaluation."""
    print("=" * 80)
    print("GAT vs ER COMPARATIVE EVALUATION")
    print("Crisis Management Multi-Agent System")
    print("=" * 80)

    # Define experiments
    scenarios = [
        'flood_scenario',
        'forest_fire_evia',
        'ammonia_leak_elefsina'
    ]
    methods = ['er', 'gat']
    seeds = [1, 2, 3]  # 3 runs per condition

    # Results storage
    all_results = {'er': [], 'gat': []}

    # Output directories
    er_dir = Path('results/er')
    gat_dir = Path('results/gat')

    # Run experiments
    total_experiments = len(scenarios) * len(methods) * len(seeds)
    completed = 0

    print(f"\nRunning {total_experiments} experiments ({len(scenarios)} scenarios × {len(methods)} methods × {len(seeds)} runs)...")
    print("-" * 80)

    for scenario in scenarios:
        for method in methods:
            for seed in seeds:
                try:
                    result = run_experiment(scenario, method, seed)
                    all_results[method].append(result)

                    # Save result
                    output_dir = er_dir if method == 'er' else gat_dir
                    save_result(result, output_dir)

                    completed += 1
                    logger.info(f"Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.0f}%)")

                except Exception as e:
                    logger.error(f"Experiment failed: {scenario}/{method}/seed{seed}: {e}")

    # Print results
    print("\n")
    print_comparison_table(all_results)
    print_gat_attention_analysis(all_results)
    print_statistical_summary(all_results)

    # Summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {completed}/{total_experiments}")
    print(f"ER runs completed: {len(all_results['er'])}")
    print(f"GAT runs completed: {len(all_results['gat'])}")
    print(f"\nResults saved to:")
    print(f"  - ER results: {er_dir}/")
    print(f"  - GAT results: {gat_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
