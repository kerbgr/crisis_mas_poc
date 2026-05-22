#!/usr/bin/env python3
"""
Main Orchestration Script for Crisis Management Multi-Agent System

This script runs the complete multi-agent decision-making workflow:
1. Load configuration and scenario
2. Initialize all components (agents, engines, models)
3. Execute coordinated decision-making process
4. Evaluate performance and compare to baseline
5. Generate visualizations and reports
6. Save results to disk

Usage:
    python main.py --scenario flood_scenario --output-dir results/run_1
    python main.py --scenario wildfire --verbose
    python main.py --help
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root into os.environ

# Import Crisis MAS components
from agents.expert_agent import ExpertAgent
from agents.coordinator_agent import CoordinatorAgent
from llm_integration.claude_client import ClaudeClient
from llm_integration.openai_client import OpenAIClient
from llm_integration.lmstudio_client import LMStudioClient
from decision_framework.evidential_reasoning import EvidentialReasoning
from decision_framework.mcda_engine import MCDAEngine
from decision_framework.consensus_model import ConsensusModel
from evaluation.metrics import MetricsEvaluator
from evaluation.visualizations import SystemVisualizer
from scenarios.expert_selector import ExpertSelector


# ============================================================================
# Configuration and Setup
# ============================================================================

def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """
    Configure logging for the application.

    Args:
        verbose: Enable verbose (DEBUG level) logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True  # override any handlers set by imported libs (e.g. numexpr)
    )

    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("CRISIS MANAGEMENT MULTI-AGENT SYSTEM")
    logger.info("="*80)

    return logger


def load_environment():
    """
    Load and validate environment variables.

    Returns:
        Dictionary of environment variables
    """
    logger = logging.getLogger(__name__)

    # Check for API keys
    api_keys = {
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY')
    }

    if not api_keys['anthropic'] and not api_keys['openai']:
        logger.warning(
            "No API keys found in environment. "
            "Will automatically fall back to LM Studio (local, no API key required). "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY for cloud LLM providers."
        )

    logger.info("Environment loaded")
    return api_keys


def load_scenario(scenario_path: str) -> Dict[str, Any]:
    """
    Load scenario data from JSON file.

    Args:
        scenario_path: Path to scenario JSON file

    Returns:
        Scenario dictionary
    """
    logger = logging.getLogger(__name__)

    path = Path(scenario_path)
    if not path.exists():
        # Try scenarios directory
        path = Path("scenarios") / f"{scenario_path}.json"

    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")

    with open(path, 'r') as f:
        scenario = json.load(f)

    logger.info(f"Loaded scenario: {scenario.get('id', scenario.get('scenario_id', 'unknown'))}")
    logger.info(f"  Type: {scenario.get('type', scenario.get('crisis_type', 'unknown'))}")
    logger.info(f"  Location: {scenario.get('location', 'unknown')}")
    logger.info(f"  Severity: {scenario.get('severity', 0):.2f}")

    return scenario


def load_alternatives(alternatives_path: str) -> List[Dict[str, Any]]:
    """
    Load response alternatives from JSON file.

    Args:
        alternatives_path: Path to alternatives JSON file

    Returns:
        List of alternative dictionaries
    """
    logger = logging.getLogger(__name__)

    path = Path(alternatives_path)
    if not path.exists():
        path = Path("scenarios") / f"{alternatives_path}.json"

    if not path.exists():
        raise FileNotFoundError(f"Alternatives not found: {alternatives_path}")

    with open(path, 'r') as f:
        data = json.load(f)

    # Extract alternatives list
    alternatives = data.get('alternatives', data) if isinstance(data, dict) else data

    logger.info(f"Loaded {len(alternatives)} response alternatives")

    return alternatives


# ============================================================================
# Component Initialization
# ============================================================================

def prompt_llm_provider() -> str:
    """
    Interactively ask the user which LLM provider to use.

    Returns:
        Provider string: 'lmstudio', 'claude', or 'openai'
    """
    print("\n" + "=" * 50)
    print("Select LLM Provider:")
    print("  1. LM Studio  (local, default)")
    print("  2. Claude API  (Anthropic)")
    print("  3. OpenAI API")
    print("=" * 50)

    choice = input("Enter choice [1]: ").strip()

    provider_map = {"1": "lmstudio", "2": "claude", "3": "openai", "": "lmstudio"}
    provider = provider_map.get(choice)

    if provider is None:
        print(f"Invalid choice '{choice}'. Defaulting to LM Studio.")
        provider = "lmstudio"

    print(f"Selected provider: {provider}\n")
    return provider


def initialize_llm_client(
    provider: str = "lmstudio",
    api_keys: Optional[Dict[str, str]] = None
):
    """
    Initialize LLM client based on provider.

    Automatically falls back to LM Studio if no API keys are available.

    Args:
        provider: LLM provider ('claude', 'openai', 'lmstudio')
        api_keys: Dictionary of API keys

    Returns:
        LLM client instance
    """
    logger = logging.getLogger(__name__)

    # Try requested provider first
    if provider == "claude":
        api_key = api_keys.get('anthropic') if api_keys else None
        if api_key:
            client = ClaudeClient(api_key=api_key)
            logger.info("Initialized Claude client")
            return client
        else:
            logger.warning("No ANTHROPIC_API_KEY found. Falling back to LM Studio...")
            provider = "lmstudio"

    elif provider == "openai":
        api_key = api_keys.get('openai') if api_keys else None
        if api_key:
            client = OpenAIClient(api_key=api_key)
            logger.info("Initialized OpenAI client")
            return client
        else:
            logger.warning("No OPENAI_API_KEY found. Falling back to LM Studio...")
            provider = "lmstudio"

    # LM Studio doesn't require API key
    if provider == "lmstudio":
        client = LMStudioClient()
        logger.info("Initialized LM Studio client (local, no API key required)")
        return client

    raise ValueError(f"Unknown LLM provider: {provider}")


def initialize_expert_agents(
    llm_client,
    agent_ids: Optional[List[str]] = None
) -> List[ExpertAgent]:
    """
    Initialize expert agents.

    Args:
        llm_client: LLM client to use for all agents
        agent_ids: Optional list of specific agent IDs to load.
                   Supported agent IDs (13 expert agents total):
                   - Core: meteorology_silver_advisory, logistics_silver_advisory, medical_silver_tactical
                   - Civil Protection: civilprotection_gold_strategic
                   - Environmental: environment_silver_advisory
                   - Emergency Response: psap_silver_coordination
                   - Police: police_silver_tactical, police_gold_strategic
                   - Fire: fire_silver_tactical, fire_gold_strategic
                   - Medical: medical_gold_strategic
                   - Coast Guard: coastguard_silver_tactical, coastguard_gold_strategic

    Returns:
        List of ExpertAgent instances
    """
    logger = logging.getLogger(__name__)

    # Default agents if not specified - use 3 core agents for backward compatibility
    if agent_ids is None:
        agent_ids = [
            "meteorology_silver_advisory",
            "logistics_silver_advisory",
            "medical_silver_tactical"
        ]

    # If user passes 'all', load all 13 expert agents
    if agent_ids == ['all']:
        agent_ids = [
            "meteorology_silver_advisory",
            "medical_silver_tactical",
            "logistics_silver_advisory",
            "civilprotection_gold_strategic",
            "environment_silver_advisory",
            "psap_silver_coordination",
            "police_silver_tactical",
            "police_gold_strategic",
            "fire_silver_tactical",
            "fire_gold_strategic",
            "medical_gold_strategic",
            "coastguard_silver_tactical",
            "coastguard_gold_strategic"
        ]
        logger.info("Loading all 13 expert agents...")

    agents = []
    for agent_id in agent_ids:
        try:
            agent = ExpertAgent(
                agent_id=agent_id,
                llm_client=llm_client
            )
            if agent.load_reliability_data():
                logger.info(f"Loaded historical reliability for {agent_id}")
            agents.append(agent)
            logger.info(f"Initialized agent: {agent.name} ({agent.role})")
        except Exception as e:
            logger.error(f"Failed to initialize agent {agent_id}: {e}")

    logger.info(f"Initialized {len(agents)} expert agents")
    return agents


def initialize_decision_framework() -> Dict[str, Any]:
    """
    Initialize decision framework components (ER, MCDA, Consensus).

    Returns:
        Dictionary of framework components
    """
    logger = logging.getLogger(__name__)

    framework = {
        'er_engine': EvidentialReasoning(),
        'mcda_engine': MCDAEngine(),
        'consensus_model': ConsensusModel(consensus_threshold=0.75)
    }

    logger.info("Initialized decision framework:")
    logger.info("  - Evidential Reasoning engine")
    logger.info("  - MCDA engine")
    logger.info("  - Consensus model (threshold: 0.75)")

    return framework


def initialize_coordinator(
    expert_agents: List[ExpertAgent],
    framework: Dict[str, Any],
    aggregation_method: str = "ER"
) -> CoordinatorAgent:
    """
    Initialize coordinator agent.

    Args:
        expert_agents: List of expert agents
        framework: Decision framework components
        aggregation_method: Aggregation method ("ER" or "GAT")

    Returns:
        CoordinatorAgent instance
    """
    logger = logging.getLogger(__name__)

    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=framework['er_engine'],
        mcda_engine=framework['mcda_engine'],
        consensus_model=framework['consensus_model'],
        parallel_assessment=True,
        aggregation_method=aggregation_method.upper()
    )

    logger.info(f"Initialized coordinator with {len(expert_agents)} agents (aggregation={aggregation_method.upper()})")

    return coordinator


# ============================================================================
# Decision-Making Workflow
# ============================================================================

def run_decision_process(
    coordinator: CoordinatorAgent,
    scenario: Dict[str, Any],
    alternatives: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Execute the complete multi-agent decision-making process.

    Args:
        coordinator: Coordinator agent
        scenario: Crisis scenario
        alternatives: Response alternatives

    Returns:
        Decision results dictionary
    """
    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("="*80)
    logger.info("EXECUTING MULTI-AGENT DECISION PROCESS")
    logger.info("="*80)

    start_time = datetime.now()

    # Run coordinated decision-making
    decision = coordinator.make_final_decision(scenario, alternatives)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("")
    logger.info("Decision process completed:")
    logger.info(f"  Recommended: {decision['recommended_alternative']}")
    logger.info(f"  Confidence: {decision['confidence']:.3f}")
    logger.info(f"  Consensus: {decision['consensus_level']:.3f}")
    logger.info(f"  Time: {duration:.2f}s")

    return decision


def run_single_agent_baseline(
    expert_agents: List[ExpertAgent],
    scenario: Dict[str, Any],
    alternatives: List[Dict[str, Any]],
    agent_type: str = "first"
) -> Dict[str, Any]:
    """
    Run single-agent baseline for comparison with multi-agent approach.

    This provides a baseline to demonstrate the value of multi-agent consensus.
    Uses only ONE agent with NO consensus building - just direct decision from
    a single perspective.

    Args:
        expert_agents: List of available expert agents
        scenario: Crisis scenario
        alternatives: Response alternatives
        agent_type: Which agent to use for baseline:
            - "first": Use first agent in list (default)
            - "meteorologist": Use meteorologist if available
            - "logistics": Use logistics coordinator if available
            - "medical": Use medical expert if available
            - "operations": Use operations expert if available

    Returns:
        Single-agent decision dictionary in similar format to coordinator output
        for fair comparison, containing:
            - recommended_alternative: str
            - confidence: float
            - final_scores: Dict[str, float]
            - belief_distribution: Dict[str, float]
            - reasoning: str
            - agent_info: Dict with agent details
            - decision_time_seconds: float
    """
    logger = logging.getLogger(__name__)
    import time

    logger.info("")
    logger.info("="*80)
    logger.info("RUNNING SINGLE-AGENT BASELINE")
    logger.info("="*80)

    # Select agent based on type
    selected_agent = None

    if agent_type == "first":
        selected_agent = expert_agents[0]
    else:
        # Map agent types to role/expertise keywords
        type_keywords = {
            "meteorologist": ["meteorologist", "weather", "environmental"],
            "logistics": ["logistics", "supply", "operations"],
            "medical": ["medical", "health", "emergency"],
            "operations": ["operations", "coordinator", "emergency"]
        }

        keywords = type_keywords.get(agent_type.lower(), [])

        # Find matching agent
        for agent in expert_agents:
            for keyword in keywords:
                if (keyword.lower() in agent.role.lower() or
                    keyword.lower() in agent.expertise.lower()):
                    selected_agent = agent
                    break
            if selected_agent:
                break

        # Fallback to first agent if no match
        if not selected_agent:
            logger.warning(f"No agent found matching type '{agent_type}', using first agent")
            selected_agent = expert_agents[0]

    logger.info(f"Selected baseline agent: {selected_agent.name} ({selected_agent.role})")
    logger.info(f"Expertise: {selected_agent.expertise}")
    logger.info("")

    # Time the decision
    start_time = time.time()

    # Get single-agent assessment
    try:
        assessment = selected_agent.evaluate_scenario(scenario, alternatives)
    except Exception as e:
        logger.error(f"Single-agent baseline failed: {e}")
        # Return error decision
        return {
            'recommended_alternative': None,
            'confidence': 0.0,
            'final_scores': {},
            'belief_distribution': {},
            'reasoning': f"ERROR: Single-agent baseline failed - {str(e)}",
            'agent_info': {
                'agent_id': selected_agent.agent_id,
                'agent_name': selected_agent.name,
                'agent_role': selected_agent.role
            },
            'decision_time_seconds': time.time() - start_time,
            'error': str(e)
        }

    end_time = time.time()
    decision_time = end_time - start_time

    # Extract belief distribution and find top alternative
    belief_distribution = assessment.get('belief_distribution', {})

    if belief_distribution:
        # Find alternative with highest belief
        top_alternative = max(belief_distribution.items(), key=lambda x: x[1])
        recommended_alternative = top_alternative[0]
        top_score = top_alternative[1]
    else:
        recommended_alternative = None
        top_score = 0.0

    # Build decision structure similar to coordinator output
    baseline_decision = {
        'recommended_alternative': recommended_alternative,
        'confidence': assessment.get('confidence', 0.0),
        'final_scores': belief_distribution,  # Single agent's beliefs are the final scores
        'belief_distribution': belief_distribution,
        'reasoning': assessment.get('reasoning', ''),
        'key_concerns': assessment.get('key_concerns', []),
        'criteria_scores': assessment.get('criteria_scores', {}),
        'agent_info': {
            'agent_id': assessment.get('agent_id'),
            'agent_name': assessment.get('agent_name'),
            'agent_role': assessment.get('agent_role'),
            'expertise': assessment.get('expertise')
        },
        'decision_time_seconds': decision_time,
        'timestamp': assessment.get('timestamp'),
        'scenario_type': assessment.get('scenario_type'),
        'baseline_type': 'single_agent'
    }

    logger.info("Single-agent baseline completed:")
    logger.info(f"  Agent: {assessment.get('agent_name')}")
    logger.info(f"  Recommended: {recommended_alternative}")
    logger.info(f"  Confidence: {assessment.get('confidence', 0.0):.3f}")
    logger.info(f"  Time: {decision_time:.2f}s")
    logger.info("")

    return baseline_decision


def run_individual_agent_comparisons(
    expert_agents: List[ExpertAgent],
    scenario: Dict[str, Any],
    alternatives: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluate EACH agent individually for comparison with multi-agent consensus.

    This provides comprehensive analysis showing how each expert's individual
    judgment compares to the collaborative consensus decision.

    Args:
        expert_agents: List of all expert agents
        scenario: Crisis scenario
        alternatives: Response alternatives

    Returns:
        List of individual agent decisions, each containing:
            - agent_id, agent_name, agent_role, expertise
            - recommended_alternative
            - confidence
            - belief_distribution
            - decision_quality_score
            - reasoning
    """
    logger = logging.getLogger(__name__)
    import time

    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATING EACH AGENT INDIVIDUALLY (for comparison)")
    logger.info("="*80)

    individual_decisions = []

    for i, agent in enumerate(expert_agents, 1):
        logger.info(f"\n[{i}/{len(expert_agents)}] Evaluating: {agent.name} ({agent.role})")

        start_time = time.time()

        try:
            # Get individual agent assessment
            assessment = agent.evaluate_scenario(scenario, alternatives)

            # Extract belief distribution and find top alternative
            belief_distribution = assessment.get('belief_distribution', {})

            if belief_distribution:
                top_alternative = max(belief_distribution.items(), key=lambda x: x[1])
                recommended_alternative = top_alternative[0]
                top_score = top_alternative[1]
            else:
                recommended_alternative = None
                top_score = 0.0

            decision_time = time.time() - start_time

            # Build decision structure
            decision = {
                'agent_id': assessment.get('agent_id'),
                'agent_name': assessment.get('agent_name'),
                'agent_role': assessment.get('agent_role'),
                'expertise': agent.expertise,
                'recommended_alternative': recommended_alternative,
                'confidence': assessment.get('confidence', 0.0),
                'belief_distribution': belief_distribution,
                'final_scores': belief_distribution,
                'reasoning': assessment.get('reasoning', ''),
                'key_concerns': assessment.get('key_concerns', []),
                'criteria_scores': assessment.get('criteria_scores', {}),
                'decision_time_seconds': decision_time,
                'timestamp': assessment.get('timestamp')
            }

            individual_decisions.append(decision)

            logger.info(f"  → Recommended: {recommended_alternative} (confidence: {decision.get('confidence', 0):.1%})")

        except Exception as e:
            logger.error(f"  → Failed: {str(e)}")
            # Add error decision
            individual_decisions.append({
                'agent_id': agent.agent_id,
                'agent_name': agent.name,
                'agent_role': agent.role,
                'expertise': agent.expertise,
                'recommended_alternative': None,
                'confidence': 0.0,
                'error': str(e)
            })

    logger.info(f"\nCompleted individual evaluations for {len(individual_decisions)} agents")
    logger.info("")

    return individual_decisions


# ============================================================================
# Evaluation and Metrics
# ============================================================================

def evaluate_decision(
    decision: Dict[str, Any],
    individual_decisions: Optional[List[Dict[str, Any]]] = None,
    baseline_assessment: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate decision quality and calculate metrics.

    Args:
        decision: Multi-agent decision
        individual_decisions: List of individual agent decisions for comparison (NEW - preferred)
        baseline_assessment: Single-agent baseline assessment (LEGACY - for backward compatibility)
        ground_truth: Optional ground truth for validation

    Returns:
        Evaluation metrics dictionary
    """
    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATING DECISION QUALITY")
    logger.info("="*80)

    evaluator = MetricsEvaluator()

    # Calculate all metrics
    metrics = {}

    # 1. Decision Quality Score
    metrics['decision_quality'] = evaluator.calculate_decision_quality(
        decision,
        ground_truth=ground_truth
    )
    logger.info(f"Decision Quality Score: {metrics['decision_quality']['weighted_score']:.3f}")

    # 2. Consensus Level
    if 'collection_info' in decision and 'assessments' in decision['collection_info']:
        metrics['consensus'] = evaluator.calculate_consensus_metrics(
            decision['collection_info']['assessments']
        )
        logger.info(f"Consensus Level: {metrics['consensus']['consensus_level']:.3f}")
    else:
        logger.warning("No assessment data available for consensus calculation")
        metrics['consensus'] = {'consensus_level': 0.0, 'pairwise_agreements': {}}

    # 3. Confidence Score
    metrics['confidence'] = evaluator.calculate_confidence_metrics(decision)
    logger.info(f"Confidence Score: {metrics['confidence']['decision_confidence']:.3f}")

    # 4. Expert Contribution Balance
    if 'collection_info' in decision and 'assessments' in decision['collection_info']:
        metrics['expert_contribution_balance'] = evaluator.calculate_expert_contribution_balance(
            decision['collection_info']['assessments']
        )
        logger.info(f"Expert Balance: {metrics['expert_contribution_balance']['balance_score']:.3f}")
    else:
        logger.warning("No assessment data available for expert balance calculation")
        metrics['expert_contribution_balance'] = {'balance_score': 0.0, 'gini_coefficient': 1.0}

    # 5. Compare to individual agents if available (NEW - comprehensive comparison)
    if individual_decisions:
        logger.info("")
        logger.info("Comparing multi-agent consensus with EACH individual agent...")

        individual_metrics_list = []
        multi_agent_quality = metrics['decision_quality']['weighted_score']
        multi_agent_confidence = metrics['confidence']['decision_confidence']
        multi_agent_recommendation = decision.get('recommended_alternative')

        # Calculate metrics for each individual agent
        for ind_decision in individual_decisions:
            if 'error' in ind_decision:
                continue  # Skip failed evaluations

            ind_metrics = {
                'agent_id': ind_decision.get('agent_id'),
                'agent_name': ind_decision.get('agent_name'),
                'agent_role': ind_decision.get('agent_role'),
                'expertise': ind_decision.get('expertise'),
                'recommended_alternative': ind_decision.get('recommended_alternative'),
                'confidence': ind_decision.get('confidence', 0.0),
                'decision_quality': evaluator.calculate_decision_quality(
                    ind_decision, ground_truth=ground_truth
                ),
                'agrees_with_consensus': ind_decision.get('recommended_alternative') == multi_agent_recommendation
            }

            individual_metrics_list.append(ind_metrics)

        # Calculate statistics across all individual agents
        qualities = [m['decision_quality']['weighted_score'] for m in individual_metrics_list]
        confidences = [m['confidence'] for m in individual_metrics_list]
        agreements = sum(1 for m in individual_metrics_list if m['agrees_with_consensus'])

        avg_individual_quality = sum(qualities) / len(qualities) if qualities else 0.0
        min_individual_quality = min(qualities) if qualities else 0.0
        max_individual_quality = max(qualities) if qualities else 0.0

        avg_individual_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        agreement_rate = (agreements / len(individual_metrics_list)) * 100 if individual_metrics_list else 0.0

        # Store comprehensive comparison
        metrics['individual_comparisons'] = {
            'multi_agent_quality': multi_agent_quality,
            'multi_agent_confidence': multi_agent_confidence,
            'multi_agent_recommendation': multi_agent_recommendation,
            'individual_agents': individual_metrics_list,
            'statistics': {
                'avg_quality': avg_individual_quality,
                'min_quality': min_individual_quality,
                'max_quality': max_individual_quality,
                'avg_confidence': avg_individual_confidence,
                'agreement_rate_percent': agreement_rate,
                'num_agents_agree': agreements,
                'total_agents': len(individual_metrics_list)
            }
        }

        # Log comprehensive comparison
        logger.info("")
        logger.info("="*80)
        logger.info("MULTI-AGENT vs INDIVIDUAL AGENTS COMPARISON")
        logger.info("="*80)

        logger.info("\nDecision Quality:")
        logger.info(f"  Multi-agent consensus: {multi_agent_quality:.3f}")
        logger.info(f"  Individual agents:")
        logger.info(f"    Average:  {avg_individual_quality:.3f}")
        logger.info(f"    Range:    {min_individual_quality:.3f} - {max_individual_quality:.3f}")
        logger.info(f"  Improvement over average: {((multi_agent_quality - avg_individual_quality) / max(avg_individual_quality, 0.001)) * 100:+.1f}%")

        logger.info("\nConfidence Levels:")
        logger.info(f"  Multi-agent consensus: {multi_agent_confidence:.3f}")
        logger.info(f"  Individual agents (avg): {avg_individual_confidence:.3f}")

        logger.info("\nConsensus Agreement:")
        logger.info(f"  Agents agreeing with consensus: {agreements}/{len(individual_metrics_list)} ({agreement_rate:.1f}%)")
        logger.info(f"  Multi-agent recommendation: {multi_agent_recommendation}")

        # Show individual agent recommendations
        logger.info("\nIndividual Agent Decisions:")
        for ind_metrics in sorted(individual_metrics_list, key=lambda x: x['decision_quality']['weighted_score'], reverse=True):
            agree_marker = "✓" if ind_metrics['agrees_with_consensus'] else "✗"
            logger.info(
                f"  [{agree_marker}] {ind_metrics['agent_name']}: "
                f"{ind_metrics['recommended_alternative']} "
                f"(quality: {ind_metrics['decision_quality']['weighted_score']:.3f}, "
                f"confidence: {ind_metrics['confidence']:.2f})"
            )

        logger.info("")

    # 5b. Compare to baseline if available (LEGACY - for backward compatibility)
    elif baseline_assessment:
        logger.info("")
        logger.info("Comparing with single-agent baseline...")

        # Calculate baseline metrics (same metrics as multi-agent)
        baseline_metrics = {}

        # Decision quality for baseline
        baseline_metrics['decision_quality'] = evaluator.calculate_decision_quality(
            baseline_assessment,
            ground_truth=ground_truth
        )

        # Confidence for baseline
        baseline_metrics['confidence'] = evaluator.calculate_confidence_metrics(
            baseline_assessment
        )

        # Store baseline metrics for comparison
        metrics['baseline_metrics'] = baseline_metrics

        # Perform comparison
        multi_agent_results = {
            'decision_quality': metrics['decision_quality'],
            'confidence': metrics['confidence']
        }

        baseline_results = {
            'decision_quality': baseline_metrics['decision_quality'],
            'confidence': baseline_metrics['confidence']
        }

        metrics['baseline_comparison'] = evaluator.compare_to_baseline(
            multi_agent_results,
            baseline_results
        )

        # Log comparison results
        logger.info("")
        logger.info("="*80)
        logger.info("SINGLE-AGENT vs MULTI-AGENT COMPARISON")
        logger.info("="*80)

        ma_quality = metrics['decision_quality']['weighted_score']
        sa_quality = baseline_metrics['decision_quality']['weighted_score']
        quality_improvement = ((ma_quality - sa_quality) / max(sa_quality, 0.001)) * 100

        ma_confidence = metrics['confidence']['decision_confidence']
        sa_confidence = baseline_metrics['confidence']['decision_confidence']
        confidence_improvement = ((ma_confidence - sa_confidence) / max(sa_confidence, 0.001)) * 100

        logger.info("Decision Quality:")
        logger.info(f"  Single-agent: {sa_quality:.3f}")
        logger.info(f"  Multi-agent:  {ma_quality:.3f}")
        logger.info(f"  Improvement:  {quality_improvement:+.1f}%")
        logger.info("")
        logger.info("Confidence:")
        logger.info(f"  Single-agent: {sa_confidence:.3f}")
        logger.info(f"  Multi-agent:  {ma_confidence:.3f}")
        logger.info(f"  Improvement:  {confidence_improvement:+.1f}%")
        logger.info("")

        # Check if decisions differ
        ma_recommendation = decision.get('recommended_alternative')
        sa_recommendation = baseline_assessment.get('recommended_alternative')

        if ma_recommendation and sa_recommendation:
            if ma_recommendation == sa_recommendation:
                logger.info(f"Both approaches recommend: {ma_recommendation}")
            else:
                logger.info(f"Different recommendations:")
                logger.info(f"  Single-agent: {sa_recommendation}")
                logger.info(f"  Multi-agent:  {ma_recommendation}")

        logger.info("="*80)

    return metrics


# ============================================================================
# Visualization and Reporting
# ============================================================================

def generate_visualizations(
    decision: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: Path,
    aggregation_method: Optional[str] = None,
    comparative_results: Optional[Dict[str, Any]] = None,
    comparison_only: bool = False
) -> Dict[str, str]:
    """
    Generate all visualization plots.

    Args:
        decision: Decision results
        metrics: Evaluation metrics
        output_dir: Output directory
        aggregation_method: Aggregation method used ('er' or 'gat') - shown in plot titles
        comparative_results: Optional comparative analysis results for ER vs GAT comparison plots
        comparison_only: If True, only generate comparison plots (skip standard plots)

    Returns:
        Dictionary mapping plot type to file path
    """
    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)

    viz = SystemVisualizer(output_dir=str(output_dir), dpi=300)
    saved_paths = {}

    # Generate standard plots only if not comparison_only mode
    if not comparison_only:
        # Prepare visualization data
        viz_data = {
            'agent_assessments': decision['collection_info']['assessments'],
            'consensus_history': [decision['consensus_level']],  # Single point for now
            'criteria_weights': {
                'Safety': 0.35,
                'Cost': 0.25,
                'Response Time': 0.20,
                'Effectiveness': 0.20
            },
            'metrics': metrics,
            'decision': decision,
            'agent_profiles': {
                agent_id: {
                    'name': assessment.get('agent_name', agent_id),
                    'expertise': assessment.get('expertise', 'General')
                }
                for agent_id, assessment in decision['collection_info']['assessments'].items()
            }
        }

        # Generate all standard plots (with aggregation method label in titles)
        saved_paths = viz.generate_all_plots(viz_data, aggregation_method=aggregation_method)

        logger.info(f"Generated {len(saved_paths)} standard visualizations:")
        for plot_type, path in saved_paths.items():
            logger.info(f"  - {plot_type}: {path}")

    # Generate comparison visualizations if comparative results are provided
    if comparative_results:
        logger.info("")
        logger.info("Generating ER vs GAT comparison visualizations...")

        try:
            # Method comparison chart (metrics side-by-side)
            path = viz.plot_method_comparison(
                comparative_results,
                "er_vs_gat_metrics.png"
            )
            if path:
                saved_paths['method_comparison'] = path
                logger.info(f"  - method_comparison: {path}")
        except Exception as e:
            logger.error(f"Failed to plot method comparison: {e}")

        try:
            # Recommendation comparison chart
            path = viz.plot_recommendation_comparison(
                comparative_results,
                "er_vs_gat_recommendations.png"
            )
            if path:
                saved_paths['recommendation_comparison'] = path
                logger.info(f"  - recommendation_comparison: {path}")
        except Exception as e:
            logger.error(f"Failed to plot recommendation comparison: {e}")

        try:
            # Comprehensive comparative summary
            path = viz.plot_comparative_summary(
                comparative_results,
                "er_vs_gat_summary.png"
            )
            if path:
                saved_paths['comparative_summary'] = path
                logger.info(f"  - comparative_summary: {path}")
        except Exception as e:
            logger.error(f"Failed to plot comparative summary: {e}")

        try:
            # Per-alternative DQS deviation ER vs GAT
            path = viz.plot_dqs_er_gat_deviation(
                comparative_results,
                "er_vs_gat_dqs_deviation.png"
            )
            if path:
                saved_paths['dqs_deviation'] = path
                logger.info(f"  - dqs_deviation: {path}")
        except Exception as e:
            logger.error(f"Failed to plot DQS ER vs GAT deviation: {e}")

    return saved_paths


def save_results(
    decision: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: Path,
    llm_provider: str = "unknown",
    filename: str = "results.json"
) -> str:
    """
    Save results to JSON file.

    Args:
        decision: Decision results
        metrics: Evaluation metrics
        output_dir: Output directory
        llm_provider: LLM provider used for this run (e.g. 'claude', 'openai', 'lmstudio')
        filename: Output filename

    Returns:
        Path to saved file
    """
    logger = logging.getLogger(__name__)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / filename

    results = {
        'timestamp': datetime.now().isoformat(),
        'llm_provider': llm_provider,
        'decision': decision,
        'metrics': metrics
    }

    def _json_default(obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, 'dict'):
            return obj.dict()
        return str(obj)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)

    logger.info(f"Results saved to: {output_path}")

    return str(output_path)


def generate_summary_report(
    decision: Dict[str, Any],
    metrics: Dict[str, Any]
) -> str:
    """
    Generate human-readable summary report.

    Args:
        decision: Decision results
        metrics: Evaluation metrics

    Returns:
        Summary report string
    """
    evaluator = MetricsEvaluator()
    report = evaluator.generate_report(metrics)

    return report


def print_summary(
    decision: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: Path
):
    """
    Print summary to console.

    Args:
        decision: Decision results
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info("")

    # Decision summary
    logger.info("DECISION")
    logger.info(f"  Recommended Alternative: {decision.get('recommended_alternative', 'None')}")
    logger.info(f"  Confidence: {decision.get('confidence', 0.0):.1%}")
    logger.info(f"  Consensus Level: {decision.get('consensus_level', 0.0):.1%}")
    logger.info(f"  Consensus Reached: {'Yes' if decision.get('consensus_reached', False) else 'No'}")
    if 'decision_time_seconds' in decision:
        logger.info(f"  Processing Time: {decision['decision_time_seconds']:.2f}s")
    if 'error' in decision:
        logger.warning(f"  ERROR: {decision['error']}")
    logger.info("")

    # Agent opinions
    if decision.get('agent_opinions'):
        logger.info("AGENT OPINIONS")
        for agent_id, opinion in decision['agent_opinions'].items():
            logger.info(
                f"  {opinion.get('agent_name', agent_id)}: "
                f"{opinion.get('preference', 'N/A')} "
                f"(confidence: {opinion.get('confidence', 0.0):.1%})"
            )
        logger.info("")

    # Metrics summary
    logger.info("METRICS")
    logger.info(f"  Decision Quality: {metrics['decision_quality']['weighted_score']:.3f}")
    logger.info(f"  Consensus Level: {metrics['consensus']['consensus_level']:.3f}")
    logger.info(f"  Expert Balance: {metrics['expert_contribution_balance']['balance_score']:.3f}")

    if 'individual_comparisons' in metrics:
        logger.info("")
        logger.info("MULTI-AGENT vs INDIVIDUAL AGENTS COMPARISON")

        comp = metrics['individual_comparisons']
        stats = comp['statistics']

        ma_quality = comp['multi_agent_quality']
        avg_quality = stats['avg_quality']
        quality_improvement = ((ma_quality - avg_quality) / max(avg_quality, 0.001)) * 100

        logger.info(f"  Quality:     {avg_quality:.3f} (avg individual) → {ma_quality:.3f} (multi) [{quality_improvement:+.1f}%]")
        logger.info(f"               Range: {stats['min_quality']:.3f} - {stats['max_quality']:.3f}")
        logger.info(f"  Confidence:  {stats['avg_confidence']:.3f} (avg individual) → {comp['multi_agent_confidence']:.3f} (multi)")
        logger.info(f"  Agreement:   {stats['num_agents_agree']}/{stats['total_agents']} agents ({stats['agreement_rate_percent']:.1f}%)")

    elif 'baseline_comparison' in metrics and 'baseline_metrics' in metrics:
        # Legacy single-agent baseline comparison
        logger.info("")
        logger.info("BASELINE COMPARISON")

        ma_quality = metrics['decision_quality']['weighted_score']
        sa_quality = metrics['baseline_metrics']['decision_quality']['weighted_score']
        quality_improvement = ((ma_quality - sa_quality) / max(sa_quality, 0.001)) * 100

        ma_confidence = metrics['confidence']['decision_confidence']
        sa_confidence = metrics['baseline_metrics']['confidence']['decision_confidence']

        logger.info(f"  Quality:     {sa_quality:.3f} (single) → {ma_quality:.3f} (multi) [{quality_improvement:+.1f}%]")
        logger.info(f"  Confidence:  {sa_confidence:.3f} (single) → {ma_confidence:.3f} (multi)")

    logger.info("")

    # Per-alternative DQS breakdown
    final_scores = decision.get('final_scores', {})
    er_scores    = decision.get('er_scores', {})
    mcda_scores  = decision.get('mcda_scores', {})
    recommended  = decision.get('recommended_alternative')

    if final_scores:
        logger.info("ALTERNATIVE RANKINGS  (DQS = 60% ER + 40% MCDA)")
        logger.info(f"  {'Rank':<5} {'Alternative':<35} {'DQS':>7} {'ER':>7} {'MCDA':>7}")
        logger.info("  " + "-"*65)
        for rank, (alt_id, combined) in enumerate(
            sorted(final_scores.items(), key=lambda x: x[1], reverse=True), 1
        ):
            er_s    = er_scores.get(alt_id, 0.0)
            mcda_s  = mcda_scores.get(alt_id, 0.0)
            marker  = "  ← RECOMMENDED" if alt_id == recommended else ""
            logger.info(
                f"  {rank:<5} {alt_id:<35} {combined:>7.4f} {er_s:>7.4f} {mcda_s:>7.4f}{marker}"
            )
        logger.info("")

    # Output files
    logger.info("OUTPUT")
    logger.info(f"  Results Directory: {output_dir}")
    logger.info(f"  Results File: results.json")
    logger.info(f"  Visualizations: *.png")
    logger.info("")

    logger.info("="*80)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("="*80)


# ============================================================================
# Comparative Analysis
# ============================================================================

def _run_mcda_only(mcda_engine, alternatives: list) -> dict:
    """Build a synthetic decision dict using pure TOPSIS (no agent beliefs)."""
    rankings = mcda_engine.rank_alternatives(alternatives)
    mcda_scores = {alt_id: score for alt_id, score, _ in rankings}

    # L1-normalise
    total = sum(mcda_scores.values())
    mcda_norm = {k: v / total for k, v in mcda_scores.items()} if total > 0 else mcda_scores

    recommended = max(mcda_norm, key=mcda_norm.get) if mcda_norm else None
    top_score = mcda_norm.get(recommended, 0.0) if recommended else 0.0

    return {
        'recommended_alternative': recommended,
        'confidence': top_score,
        'decision_quality_score': top_score,
        'consensus_level': 0.0,
        'consensus_reached': False,
        'final_scores': mcda_norm,
        'er_scores': {},
        'mcda_scores': mcda_scores,
        'agent_opinions': {},
        'agents_participated': 0,
        'conflicts': [],
        'resolution': {},
        'explanation': 'Pure MCDA (TOPSIS) ranking - no agent beliefs used.',
        'scenario_id': 'unknown',
        'timestamp': datetime.now().isoformat(),
        'decision_time_seconds': 0.0,
        'collection_info': {'assessments': {}}
    }


def run_comparative_analysis(
    coordinator: CoordinatorAgent,
    expert_agents: List[ExpertAgent],
    framework: Dict[str, Any],
    scenario: Dict[str, Any],
    alternatives: List[Dict[str, Any]],
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run comparative analysis of ER vs GAT aggregation methods.

    This function runs the scenario with both aggregation methods and
    produces a side-by-side comparison for transparency.

    Args:
        coordinator: Initial coordinator agent (will be re-initialized for each method)
        expert_agents: List of expert agents
        framework: Decision framework components
        scenario: Crisis scenario
        alternatives: Response alternatives
        output_dir: Output directory for results
        verbose: Enable verbose output

    Returns:
        Comparative analysis results dictionary
    """
    import time
    logger = logging.getLogger(__name__)
    evaluator = MetricsEvaluator()

    logger.info("")
    logger.info("="*80)
    logger.info("COMPARATIVE ANALYSIS: ER vs GAT AGGREGATION METHODS")
    logger.info("="*80)

    results = {
        'scenario': scenario.get('id', scenario.get('scenario_id', 'unknown')),
        'scenario_type': scenario.get('type', scenario.get('crisis_type', 'unknown')),
        'num_agents': len(expert_agents),
        'methods': {}
    }

    # Determine which methods to run (GAT_TRAINED only if weights exist)
    _weights_path = Path("models/gat_weights/gat_trained_weights.json")
    _weights_available = _weights_path.exists()
    if _weights_available:
        methods_to_run = ['ER', 'GAT', 'GAT_TRAINED', 'MCDA']
    else:
        methods_to_run = ['ER', 'GAT', 'MCDA']
        logger.warning(
            "GAT trained weights not found at %s -- skipping GAT_TRAINED. "
            "Run 'python scripts/train_gat.py' to generate them.", _weights_path
        )

    # Run with each aggregation method
    for method in methods_to_run:
        logger.info(f"\n--- Running with {method} aggregation ---")

        start_time = time.time()

        if method == 'MCDA':
            # MCDA-only: pure TOPSIS ranking, no agent beliefs involved
            decision = _run_mcda_only(framework['mcda_engine'], alternatives)
        else:
            method_coordinator = CoordinatorAgent(
                expert_agents=expert_agents,
                er_engine=framework['er_engine'],
                mcda_engine=framework['mcda_engine'],
                consensus_model=framework['consensus_model'],
                parallel_assessment=True,
                aggregation_method=method
            )
            decision = method_coordinator.make_final_decision(scenario, alternatives)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Calculate metrics
        metrics = {}
        metrics['decision_quality'] = evaluator.calculate_decision_quality(decision)

        if 'collection_info' in decision and 'assessments' in decision['collection_info']:
            metrics['consensus'] = evaluator.calculate_consensus_metrics(
                decision['collection_info']['assessments']
            )
        else:
            metrics['consensus'] = {'consensus_level': decision.get('consensus_level', 0.0)}

        metrics['confidence'] = evaluator.calculate_confidence_metrics(decision)

        # Store results
        method_result = {
            'recommended_alternative': decision.get('recommended_alternative'),
            'confidence': decision.get('confidence', 0.0),
            'consensus_level': decision.get('consensus_level', 0.0),
            'decision_quality_score': metrics['decision_quality']['weighted_score'],
            'processing_time_ms': processing_time,
            'decision': decision,
            'metrics': metrics
        }

        if method in ('GAT', 'GAT_TRAINED') and 'gat_analysis' in decision:
            method_result['attention_weights'] = decision['gat_analysis'].get('attention_weights', {})
            method_result['top_influential_agents'] = decision['gat_analysis'].get('top_influential_agents', [])

        results['methods'][method] = method_result

        logger.info(f"  Recommended: {method_result['recommended_alternative']}")
        logger.info(f"  Confidence: {method_result['confidence']:.3f}")
        logger.info(f"  Consensus: {method_result['consensus_level']:.3f}")
        logger.info(f"  DQS: {method_result['decision_quality_score']:.3f}")
        logger.info(f"  Processing time: {processing_time:.1f} ms")

    # Build generic comparison metrics (all methods vs ER baseline)
    er_result = results['methods']['ER']
    comparison = {'same_recommendation_all': len({v['recommended_alternative'] for v in results['methods'].values()}) == 1}
    for m, mdata in results['methods'].items():
        if m == 'ER':
            continue
        comparison[f'{m}_vs_ER'] = {
            'decision_quality_delta': mdata['decision_quality_score'] - er_result['decision_quality_score'],
            'consensus_delta': mdata['consensus_level'] - er_result['consensus_level'],
            'confidence_delta': mdata['confidence'] - er_result['confidence'],
            'same_recommendation': mdata['recommended_alternative'] == er_result['recommended_alternative'],
        }
    # Keep legacy keys for backward compatibility with older result readers
    if 'GAT' in results['methods']:
        gat_result = results['methods']['GAT']
        comparison['decision_quality_delta'] = gat_result['decision_quality_score'] - er_result['decision_quality_score']
        comparison['consensus_delta'] = gat_result['consensus_level'] - er_result['consensus_level']
        comparison['confidence_delta'] = gat_result['confidence'] - er_result['confidence']
        comparison['processing_time_delta_ms'] = gat_result['processing_time_ms'] - er_result['processing_time_ms']
        comparison['same_recommendation'] = er_result['recommended_alternative'] == gat_result['recommended_alternative']
    results['comparison'] = comparison

    # Print 4-way comparison summary table
    logger.info("")
    logger.info("="*80)
    logger.info("4-WAY COMPARATIVE RESULTS SUMMARY")
    logger.info("="*80)
    logger.info("")
    col_w = 14
    header_methods = list(results['methods'].keys())
    hdr = f"{'Metric':<26}" + "".join(f"{m:>{col_w}}" for m in header_methods)
    logger.info(hdr)
    logger.info("-" * (26 + col_w * len(header_methods)))

    def _row(label, key, fmt=".3f"):
        vals = [results['methods'][m][key] for m in header_methods]
        return f"{label:<26}" + "".join(f"{v:{col_w}{fmt}}" for v in vals)

    logger.info(_row("Decision Quality Score", "decision_quality_score"))
    logger.info(_row("Consensus Level",        "consensus_level"))
    logger.info(_row("Confidence",             "confidence"))
    logger.info(_row("Processing Time (ms)",   "processing_time_ms", ".1f"))
    logger.info("-" * (26 + col_w * len(header_methods)))
    for m in header_methods:
        logger.info(f"  {m} -> {results['methods'][m]['recommended_alternative']}")

    # Per-alternative DQS scores across all methods
    all_alts = sorted(
        set().union(*[set(results['methods'][m]['decision'].get('final_scores', {}).keys())
                      for m in header_methods]),
        key=lambda a: er_result['decision'].get('final_scores', {}).get(a, 0.0),
        reverse=True
    )
    if all_alts:
        logger.info("")
        logger.info("PER-ALTERNATIVE DQS SCORES")
        score_hdr = f"  {'Alternative':<35}" + "".join(f"{m:>{col_w}}" for m in header_methods)
        logger.info(score_hdr)
        logger.info("  " + "-" * (35 + col_w * len(header_methods)))
        for alt_id in all_alts:
            tags = [m + "✓" for m in header_methods
                    if results['methods'][m]['recommended_alternative'] == alt_id]
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            row_vals = "".join(
                f"{results['methods'][m]['decision'].get('final_scores', {}).get(alt_id, 0.0):>{col_w}.4f}"
                for m in header_methods
            )
            logger.info(f"  {alt_id:<35}{row_vals}{tag_str}")

    logger.info("")
    logger.info("="*80)

    # Save comparative results JSON
    comparison_file = output_dir / "comparative_analysis.json"
    serializable_results = {
        'scenario': results['scenario'],
        'scenario_type': results['scenario_type'],
        'num_agents': results['num_agents'],
        'comparison': results['comparison'],
        'methods': {
            method: {
                'recommended_alternative': data['recommended_alternative'],
                'confidence': data['confidence'],
                'consensus_level': data['consensus_level'],
                'decision_quality_score': data['decision_quality_score'],
                'processing_time_ms': data['processing_time_ms'],
                'decision': {
                    'final_scores': data.get('decision', {}).get('final_scores', {}),
                    'recommended_alternative': data['recommended_alternative'],
                }
            }
            for method, data in results['methods'].items()
        }
    }
    with open(comparison_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    logger.info(f"Comparative analysis saved to: {comparison_file}")

    # Save full results for each method in its own subdirectory
    for method, method_data in results['methods'].items():
        method_dir = output_dir / method.lower()
        method_dir.mkdir(exist_ok=True, parents=True)
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'scenario': results['scenario'],
            'scenario_type': results['scenario_type'],
            'aggregation_method': method,
            'decision': method_data['decision'],
            'metrics': method_data['metrics']
        }
        if method in ('GAT', 'GAT_TRAINED'):
            if 'attention_weights' in method_data:
                full_results['gat_attention_weights'] = method_data['attention_weights']
            if 'top_influential_agents' in method_data:
                full_results['gat_top_influential_agents'] = method_data['top_influential_agents']
        with open(method_dir / "results.json", 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        logger.info(f"{method} full results saved to: {method_dir / 'results.json'}")

    # Persist reliability data after comparative runs
    for agent in expert_agents:
        try:
            agent.save_reliability_data()
        except Exception as e:
            logger.warning(f"Failed to save reliability for {agent.agent_id}: {e}")

    return results


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main orchestration function.

    Runs the complete Crisis MAS workflow from start to finish.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Crisis Management Multi-Agent System - Decision Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (3 agents: meteorologist, logistics, medical)
  python main.py --scenario flood_scenario

  # Run with AUTOMATIC expert selection based on scenario metadata
  python main.py --scenario flood_scenario --expert-selection auto

  # Run with all 13 expert agents for comprehensive multi-perspective analysis
  python main.py --scenario flood_scenario --agents all

  # Run with specific expert agents (manual selection)
  python main.py --scenario flood_scenario --agents psap_commander_01 police_onscene_01 fire_onscene_01

  # Run with specific baseline agent type
  python main.py --scenario flood_scenario --baseline-agent meteorologist

  # Run without baseline comparison
  python main.py --scenario wildfire --no-baseline

  # Run with custom output directory and verbose logging
  python main.py --scenario wildfire --output-dir results/test_1 --verbose

  # Run with different LLM provider
  python main.py --scenario flood --llm-provider openai

  # Run with auto-selection and verbose mode to see selection reasoning
  python main.py --scenario flood_scenario --expert-selection auto --verbose

  # Run comparative analysis of ER vs GAT aggregation methods
  python main.py --scenario flood_scenario --compare-methods

  # Run comparative analysis with all 13 agents
  python main.py --scenario flood_scenario --agents all --compare-methods

For more information, see README.md
        """
    )

    parser.add_argument(
        '--scenario',
        type=str,
        default='flood_scenario',
        help='Scenario name or path to scenario JSON file (default: flood_scenario)'
    )

    parser.add_argument(
        '--alternatives',
        type=str,
        default=None,
        help='Path to alternatives JSON file (default: scenarios/<scenario>_alternatives.json)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results and visualizations (default: results)'
    )

    parser.add_argument(
        '--llm-provider',
        type=str,
        default=None,
        choices=['claude', 'openai', 'lmstudio'],
        help='LLM provider to use (claude, openai, lmstudio). '
             'If omitted, prompts interactively when run in a terminal; '
             'defaults to lmstudio in non-interactive (scripted) mode.'
    )

    parser.add_argument(
        '--agents',
        nargs='+',
        default=None,
        help='Specific agent IDs to use (default: 3 core agents). '
             'Use "all" to load all 13 expert agents, or specify agent IDs: '
             'meteorology_silver_advisory, logistics_silver_advisory, medical_silver_tactical, '
             'civilprotection_gold_strategic, environment_silver_advisory, psap_silver_coordination, '
             'police_silver_tactical, police_gold_strategic, fire_silver_tactical, fire_gold_strategic, '
             'medical_gold_strategic, coastguard_silver_tactical, coastguard_gold_strategic'
    )

    parser.add_argument(
        '--expert-selection',
        type=str,
        default='manual',
        choices=['manual', 'auto'],
        help='Expert selection mode: '
             '"manual" (use --agents flag), '
             '"auto" (automatic selection based on scenario metadata). '
             'Default: manual for backward compatibility'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )

    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip single-agent baseline comparison'
    )

    parser.add_argument(
        '--baseline-agent',
        type=str,
        default='first',
        choices=['first', 'meteorologist', 'logistics', 'medical', 'operations'],
        help='Agent type to use for single-agent baseline (default: first)'
    )

    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )

    parser.add_argument(
        '--aggregation-method',
        type=str,
        default='er',
        choices=['er', 'gat', 'gat_trained', 'mcda'],
        help='Aggregation method: "er", "gat", "gat_trained" (trained weights), "mcda" (pure TOPSIS). Default: er'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--compare-methods',
        action='store_true',
        help='Run comparative analysis of ER vs GAT aggregation methods. '
             'This runs the scenario with both methods and produces a side-by-side comparison.'
    )

    args = parser.parse_args()

    # Extract scenario name for directory structure
    scenario_name = Path(args.scenario).stem

    # Determine LLM provider: use CLI arg if given, prompt when interactive, else default
    if args.llm_provider is not None:
        llm_provider = args.llm_provider
    elif sys.stdin.isatty():
        llm_provider = prompt_llm_provider()
    else:
        llm_provider = 'lmstudio'

    # Setup output directory with structure: results/scenario_name/run_X_<provider>
    base_output_dir = Path(args.output_dir)
    scenario_output_dir = base_output_dir / scenario_name

    # Find next run number
    scenario_output_dir.mkdir(exist_ok=True, parents=True)
    existing_runs = [d for d in scenario_output_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if existing_runs:
        run_numbers = []
        for d in existing_runs:
            try:
                run_numbers.append(int(d.name.split('_')[1]))
            except (IndexError, ValueError):
                pass
        next_run = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_run = 1

    output_dir = scenario_output_dir / f"run_{next_run}_{llm_provider}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging
    log_file = output_dir / "run.log"
    logger = setup_logging(verbose=args.verbose, log_file=str(log_file))

    try:
        # ===== 1. SETUP =====
        logger.info("Step 1/6: Environment Setup")
        logger.info(f"Output directory: {output_dir}")
        api_keys = load_environment()

        logger.info("Step 2/6: Loading Configuration")
        scenario = load_scenario(args.scenario)

        # Load alternatives
        if args.alternatives:
            alternatives_path = args.alternatives
        else:
            # Try default location
            scenario_name = Path(args.scenario).stem
            alternatives_path = f"scenarios/{scenario_name}_alternatives.json"

        try:
            alternatives = load_alternatives(alternatives_path)
        except FileNotFoundError:
            logger.warning(f"Alternatives file not found: {alternatives_path}")
            logger.info("Using default alternatives from scenario")
            # Check both 'alternatives' and 'available_actions' keys
            alternatives = scenario.get('alternatives', scenario.get('available_actions', []))

            if not alternatives:
                raise ValueError("No alternatives found in scenario or alternatives file")

        # ===== 2. INITIALIZE COMPONENTS =====
        logger.info("Step 3/6: Initializing Components")

        llm_client = initialize_llm_client(llm_provider, api_keys)

        # Determine which agents to use based on expert selection mode
        selected_agent_ids = args.agents

        if args.expert_selection == 'auto':
            # Use ExpertSelector to automatically choose experts based on scenario
            logger.info("Using automatic expert selection based on scenario metadata...")
            selector = ExpertSelector(verbose=args.verbose)
            selected_agent_ids = selector.select_experts(scenario)
            logger.info(f"Auto-selected {len(selected_agent_ids)} experts: {', '.join(selected_agent_ids)}")
        elif args.agents:
            logger.info(f"Using manually specified agents: {', '.join(args.agents)}")
        else:
            logger.info("Using default 3 core experts (backward compatibility)")

        expert_agents = initialize_expert_agents(llm_client, selected_agent_ids)

        if not expert_agents:
            raise ValueError("No expert agents initialized")

        # Set random seed if specified (for reproducibility)
        if args.seed is not None:
            import random
            import numpy as np
            random.seed(args.seed)
            np.random.seed(args.seed)
            logger.info(f"Random seed set to: {args.seed}")

        framework = initialize_decision_framework()
        coordinator = initialize_coordinator(
            expert_agents,
            framework,
            aggregation_method=args.aggregation_method
        )

        # ===== 3. RUN DECISION PROCESS =====
        # Initialize comparative_results (set if --compare-methods is used)
        comparative_results = None
        # Track which aggregation method is used for visualizations
        selected_method = args.aggregation_method

        # Check if comparative analysis is requested
        if args.compare_methods:
            logger.info("Step 4/6: Running Comparative Analysis (ER vs GAT)")

            comparative_results = run_comparative_analysis(
                coordinator=coordinator,
                expert_agents=expert_agents,
                framework=framework,
                scenario=scenario,
                alternatives=alternatives,
                output_dir=output_dir,
                verbose=args.verbose
            )

            # Use the better-performing method's decision for subsequent evaluation
            er_dqs = comparative_results['methods']['ER']['decision_quality_score']
            gat_dqs = comparative_results['methods']['GAT']['decision_quality_score']

            if gat_dqs >= er_dqs:
                decision = comparative_results['methods']['GAT']['decision']
                selected_method = 'gat'
                logger.info("Using GAT decision for evaluation (equal or better DQS)")
            else:
                decision = comparative_results['methods']['ER']['decision']
                selected_method = 'er'
                logger.info("Using ER decision for evaluation (better DQS)")

        else:
            logger.info("Step 4/6: Running Decision Process")
            decision = run_decision_process(coordinator, scenario, alternatives)

        # Run individual agent comparisons if requested (NEW - comprehensive)
        individual_decisions = None
        if not args.no_baseline:
            individual_decisions = run_individual_agent_comparisons(
                expert_agents,
                scenario,
                alternatives
            )

        # ===== 4. EVALUATE =====
        logger.info("Step 5/6: Evaluating Performance")

        metrics = evaluate_decision(
            decision,
            individual_decisions=individual_decisions
        )

        # ===== 5. OUTPUT =====
        logger.info("Step 6/6: Generating Output")

        # Save results
        save_results(decision, metrics, output_dir, llm_provider=llm_provider)

        # Persist reliability data for all agents
        for agent in expert_agents:
            try:
                agent.save_reliability_data()
            except Exception as e:
                logger.warning(f"Failed to save reliability for {agent.agent_id}: {e}")

        # Generate visualizations
        if not args.no_viz:
            if args.compare_methods and comparative_results:
                # Generate separate visualizations for each method in subdirectories
                for method in list(comparative_results['methods'].keys()):
                    method_lower = method.lower()
                    method_dir = output_dir / method_lower
                    method_dir.mkdir(exist_ok=True, parents=True)

                    method_decision = comparative_results['methods'][method]['decision']
                    method_metrics = comparative_results['methods'][method]['metrics']

                    # Enrich method metrics with individual comparisons so the
                    # decision_comparison plot has data to render
                    if individual_decisions:
                        full_method_metrics = evaluate_decision(
                            method_decision,
                            individual_decisions=individual_decisions
                        )
                        method_metrics['individual_comparisons'] = full_method_metrics.get(
                            'individual_comparisons', {}
                        )

                    logger.info(f"Generating {method} visualizations in {method_dir}")
                    generate_visualizations(
                        method_decision,
                        method_metrics,
                        method_dir,
                        aggregation_method=method_lower,
                        comparative_results=None  # No comparison plots in subdirs
                    )

                # Generate comparison visualizations only in main output directory
                logger.info("Generating comparison visualizations in main output directory")
                generate_visualizations(
                    decision,
                    metrics,
                    output_dir,
                    aggregation_method=selected_method,
                    comparative_results=comparative_results,
                    comparison_only=True
                )
            else:
                # Standard single-method visualization
                generate_visualizations(
                    decision,
                    metrics,
                    output_dir,
                    aggregation_method=selected_method,
                    comparative_results=comparative_results
                )

        # Generate and save report
        report = generate_summary_report(decision, metrics)
        report_path = output_dir / "report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")

        # ===== 6. SUMMARY =====
        print_summary(decision, metrics, output_dir)

        return 0

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
