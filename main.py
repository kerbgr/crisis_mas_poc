#!/usr/bin/env python3
"""
Main Orchestration Script for Crisis Management Multi-Agent System
Master's Thesis PoC - Operational Research & Decision Making

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
        handlers=handlers
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
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY for LLM functionality."
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

    logger.info(f"Loaded scenario: {scenario.get('scenario_id', 'unknown')}")
    logger.info(f"  Type: {scenario.get('crisis_type', 'unknown')}")
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

def initialize_llm_client(
    provider: str = "claude",
    api_keys: Optional[Dict[str, str]] = None
):
    """
    Initialize LLM client based on provider.

    Args:
        provider: LLM provider ('claude', 'openai', 'lmstudio')
        api_keys: Dictionary of API keys

    Returns:
        LLM client instance
    """
    logger = logging.getLogger(__name__)

    if provider == "claude":
        api_key = api_keys.get('anthropic') if api_keys else None
        client = ClaudeClient(api_key=api_key)
        logger.info("Initialized Claude client")

    elif provider == "openai":
        api_key = api_keys.get('openai') if api_keys else None
        client = OpenAIClient(api_key=api_key)
        logger.info("Initialized OpenAI client")

    elif provider == "lmstudio":
        client = LMStudioClient()
        logger.info("Initialized LM Studio client (local)")

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return client


def initialize_expert_agents(
    llm_client,
    agent_ids: Optional[List[str]] = None
) -> List[ExpertAgent]:
    """
    Initialize expert agents.

    Args:
        llm_client: LLM client to use for all agents
        agent_ids: Optional list of specific agent IDs to load

    Returns:
        List of ExpertAgent instances
    """
    logger = logging.getLogger(__name__)

    # Default agents if not specified
    if agent_ids is None:
        agent_ids = [
            "agent_meteorologist",
            "logistics_expert_01",
            "medical_expert_01"
        ]

    agents = []
    for agent_id in agent_ids:
        try:
            agent = ExpertAgent(
                agent_id=agent_id,
                llm_client=llm_client
            )
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
    framework: Dict[str, Any]
) -> CoordinatorAgent:
    """
    Initialize coordinator agent.

    Args:
        expert_agents: List of expert agents
        framework: Decision framework components

    Returns:
        CoordinatorAgent instance
    """
    logger = logging.getLogger(__name__)

    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=framework['er_engine'],
        mcda_engine=framework['mcda_engine'],
        consensus_model=framework['consensus_model'],
        parallel_assessment=True
    )

    logger.info(f"Initialized coordinator with {len(expert_agents)} agents")

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
    expert_agent: ExpertAgent,
    scenario: Dict[str, Any],
    alternatives: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run single-agent baseline for comparison.

    Args:
        expert_agent: Single expert agent to use
        scenario: Crisis scenario
        alternatives: Response alternatives

    Returns:
        Single-agent assessment
    """
    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("Running single-agent baseline...")

    assessment = expert_agent.evaluate_scenario(scenario, alternatives)

    logger.info(f"  Baseline from: {expert_agent.name}")
    logger.info(f"  Confidence: {assessment['confidence']:.3f}")

    return assessment


# ============================================================================
# Evaluation and Metrics
# ============================================================================

def evaluate_decision(
    decision: Dict[str, Any],
    baseline_assessment: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate decision quality and calculate metrics.

    Args:
        decision: Multi-agent decision
        baseline_assessment: Single-agent baseline assessment
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

    # 5. Compare to baseline if available
    if baseline_assessment:
        # Convert baseline to same format as multi-agent results
        baseline_results = {
            'decision_quality': {
                'weighted_score': baseline_assessment['confidence'],
                'confidence': baseline_assessment['confidence']
            },
            'confidence': {
                'decision_confidence': baseline_assessment['confidence']
            }
        }

        multi_agent_results = {
            'decision_quality': metrics['decision_quality'],
            'confidence': metrics['confidence']
        }

        metrics['baseline_comparison'] = evaluator.compare_to_baseline(
            multi_agent_results,
            baseline_results
        )

        logger.info("")
        logger.info("Baseline Comparison:")
        if 'decision_quality' in metrics['baseline_comparison']:
            improvement = metrics['baseline_comparison']['decision_quality'].get('improvement_percentage', 0)
            logger.info(f"  Quality Improvement: {improvement:+.1f}%")

    return metrics


# ============================================================================
# Visualization and Reporting
# ============================================================================

def generate_visualizations(
    decision: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: Path
) -> Dict[str, str]:
    """
    Generate all visualization plots.

    Args:
        decision: Decision results
        metrics: Evaluation metrics
        output_dir: Output directory

    Returns:
        Dictionary mapping plot type to file path
    """
    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)

    viz = SystemVisualizer(output_dir=str(output_dir), dpi=300)

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
        'agent_profiles': {
            agent_id: {
                'name': assessment.get('agent_name', agent_id),
                'expertise': assessment.get('expertise', 'General')
            }
            for agent_id, assessment in decision['collection_info']['assessments'].items()
        }
    }

    # Generate all plots
    saved_paths = viz.generate_all_plots(viz_data)

    logger.info(f"Generated {len(saved_paths)} visualizations:")
    for plot_type, path in saved_paths.items():
        logger.info(f"  - {plot_type}: {path}")

    return saved_paths


def save_results(
    decision: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: Path,
    filename: str = "results.json"
) -> str:
    """
    Save results to JSON file.

    Args:
        decision: Decision results
        metrics: Evaluation metrics
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    logger = logging.getLogger(__name__)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / filename

    results = {
        'timestamp': datetime.now().isoformat(),
        'decision': decision,
        'metrics': metrics
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

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

    if 'baseline_comparison' in metrics:
        comp = metrics['baseline_comparison']
        if 'decision_quality' in comp:
            improvement = comp['decision_quality'].get('improvement_percentage', 0)
            logger.info(f"  vs Baseline: {improvement:+.1f}% improvement")

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
  python main.py --scenario flood_scenario
  python main.py --scenario wildfire --output-dir results/test_1 --verbose
  python main.py --scenario flood --llm-provider openai

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
        default='claude',
        choices=['claude', 'openai', 'lmstudio'],
        help='LLM provider to use (default: claude)'
    )

    parser.add_argument(
        '--agents',
        nargs='+',
        default=None,
        help='Specific agent IDs to use (default: all available)'
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
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging
    log_file = output_dir / "run.log"
    logger = setup_logging(verbose=args.verbose, log_file=str(log_file))

    try:
        # ===== 1. SETUP =====
        logger.info("Step 1/6: Environment Setup")
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

        llm_client = initialize_llm_client(args.llm_provider, api_keys)
        expert_agents = initialize_expert_agents(llm_client, args.agents)

        if not expert_agents:
            raise ValueError("No expert agents initialized")

        framework = initialize_decision_framework()
        coordinator = initialize_coordinator(expert_agents, framework)

        # ===== 3. RUN DECISION PROCESS =====
        logger.info("Step 4/6: Running Decision Process")

        decision = run_decision_process(coordinator, scenario, alternatives)

        # Run baseline comparison if requested
        baseline_assessment = None
        if not args.no_baseline:
            baseline_assessment = run_single_agent_baseline(
                expert_agents[0],
                scenario,
                alternatives
            )

        # ===== 4. EVALUATE =====
        logger.info("Step 5/6: Evaluating Performance")

        metrics = evaluate_decision(
            decision,
            baseline_assessment=baseline_assessment
        )

        # ===== 5. OUTPUT =====
        logger.info("Step 6/6: Generating Output")

        # Save results
        save_results(decision, metrics, output_dir)

        # Generate visualizations
        if not args.no_viz:
            generate_visualizations(decision, metrics, output_dir)

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
