"""
Main Orchestration Script
Crisis Management Multi-Agent System (MAS) - Proof of Concept

This script orchestrates the entire MAS system:
1. Loads scenarios and agent profiles
2. Initializes expert and coordinator agents
3. Runs decision-making simulations
4. Evaluates performance
5. Generates visualizations

Author: Crisis MAS Research Team
For: Master's Thesis in Operational Research & Decision Making
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import MAS components
from agents import ExpertAgent, CoordinatorAgent
from scenarios import ScenarioLoader
from decision_framework import EvidentialReasoning, MCDAEngine, ConsensusModel
from llm_integration import ClaudeClient, PromptTemplates
from evaluation import PerformanceMetrics, ResultVisualizer
from utils import Config


class CrisisMAS:
    """
    Main orchestrator for the Crisis Management Multi-Agent System.
    """

    def __init__(self, config: Config, use_llm: bool = True):
        """
        Initialize the Crisis MAS system.

        Args:
            config: Configuration object
            use_llm: Whether to use LLM for enhanced reasoning
        """
        self.config = config
        self.use_llm = use_llm

        # Initialize components
        self.scenario_loader = ScenarioLoader(
            scenarios_dir=config.get('scenarios.scenarios_dir', 'scenarios')
        )

        # Initialize LLM client if enabled
        self.llm_client = None
        if use_llm:
            try:
                self.llm_client = ClaudeClient(
                    model=config.get('llm.model', 'claude-3-5-sonnet-20241022')
                )
                print("✓ LLM client initialized")
            except Exception as e:
                print(f"Warning: Could not initialize LLM client: {e}")
                print("Running without LLM enhancement")

        # Initialize decision framework
        self.er_engine = EvidentialReasoning()
        self.mcda_engine = MCDAEngine()
        self.consensus_model = ConsensusModel(
            consensus_threshold=config.get('decision_framework.consensus_threshold', 0.7)
        )

        # Initialize evaluation tools
        self.metrics = PerformanceMetrics()
        self.visualizer = ResultVisualizer(
            output_dir=config.get('evaluation.output_dir', 'results')
        )

        # Agent storage
        self.expert_agents: List[ExpertAgent] = []
        self.coordinator: Optional[CoordinatorAgent] = None

        print(f"\n{'='*60}")
        print("Crisis Management Multi-Agent System")
        print(f"{'='*60}")
        print(f"Initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"LLM Enhancement: {'Enabled' if self.llm_client else 'Disabled'}")
        print(f"{'='*60}\n")

    def load_agents(self, profiles_file: str = "agents/agent_profiles.json"):
        """
        Load agent profiles and initialize agents.

        Args:
            profiles_file: Path to agent profiles JSON file
        """
        print(f"Loading agent profiles from: {profiles_file}")

        with open(profiles_file, 'r') as f:
            profiles_data = json.load(f)

        agents_list = profiles_data.get('agents', [])

        for profile in agents_list:
            agent_type = profile.get('agent_type', 'expert')

            if agent_type == 'expert':
                agent = ExpertAgent(
                    agent_id=profile['agent_id'],
                    profile=profile,
                    llm_client=self.llm_client
                )
                self.expert_agents.append(agent)
                print(f"  ✓ Loaded expert agent: {agent.agent_id} ({agent.expertise_domain})")

            elif agent_type == 'coordinator':
                self.coordinator = CoordinatorAgent(
                    agent_id=profile['agent_id'],
                    profile=profile,
                    consensus_threshold=profile.get('consensus_threshold', 0.7)
                )
                print(f"  ✓ Loaded coordinator agent: {self.coordinator.agent_id}")

        # Register experts with coordinator
        if self.coordinator:
            for expert in self.expert_agents:
                self.coordinator.register_expert(expert)

        print(f"\nTotal agents loaded: {len(self.expert_agents)} experts + 1 coordinator\n")

    def run_scenario(self, scenario_file: str) -> Dict[str, Any]:
        """
        Run a complete decision-making cycle for a scenario.

        Args:
            scenario_file: Name of the scenario file

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Running Scenario: {scenario_file}")
        print(f"{'='*60}\n")

        # Load scenario
        scenario = self.scenario_loader.load_scenario(scenario_file)
        print(self.scenario_loader.get_scenario_summary(scenario))

        # Load criteria weights
        criteria = self.scenario_loader.load_criteria_weights()

        start_time = time.time()

        # Step 1: Expert Evaluation
        print("\n--- Phase 1: Expert Scenario Evaluation ---")
        expert_evaluations = []
        for expert in self.expert_agents:
            print(f"Consulting {expert.agent_id}...")
            evaluation = expert.evaluate_scenario(scenario)
            expert_evaluations.append(evaluation)
            print(f"  Confidence: {evaluation['confidence']:.2f}, "
                  f"Domain relevance: {evaluation['domain_relevance']:.2f}")

        # Step 2: Expert Proposals
        print("\n--- Phase 2: Expert Action Proposals ---")
        expert_proposals = []
        for expert in self.expert_agents:
            print(f"{expert.agent_id} proposing action...")
            proposal = expert.propose_action(scenario, criteria)
            expert_proposals.append(proposal)

            if proposal.get('proposed_action'):
                print(f"  → {proposal['proposed_action'].get('name', 'Unknown')}")
                print(f"  Score: {proposal.get('action_score', 0):.2f}, "
                      f"Confidence: {proposal.get('confidence', 0):.2f}")

        # Step 3: Consensus Building
        print("\n--- Phase 3: Consensus Building ---")
        if self.coordinator:
            consensus_result = self.coordinator.propose_action(scenario, criteria)

            print(f"Consensus achieved: {consensus_result.get('consensus_achieved', False)}")
            print(f"Agreement level: {consensus_result.get('consensus_level', 0):.2%}")

            if consensus_result.get('recommended_action'):
                action = consensus_result['recommended_action']
                print(f"\nRecommended Action: {action.get('name', action.get('id'))}")
                print(f"Support: {consensus_result.get('vote_distribution', {})}")
        else:
            consensus_result = {'error': 'No coordinator available'}

        end_time = time.time()

        # Compile results
        results = {
            'scenario_id': scenario.get('id'),
            'scenario_file': scenario_file,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': end_time - start_time,
            'scenario': scenario,
            'criteria': criteria,
            'expert_evaluations': expert_evaluations,
            'expert_proposals': expert_proposals,
            'consensus_result': consensus_result,
            'num_experts': len(self.expert_agents),
            'llm_enabled': self.llm_client is not None
        }

        print(f"\n{'='*60}")
        print(f"Scenario completed in {results['duration_seconds']:.2f} seconds")
        print(f"{'='*60}\n")

        return results

    def run_batch_scenarios(self, scenario_files: List[str]) -> List[Dict[str, Any]]:
        """
        Run multiple scenarios in batch.

        Args:
            scenario_files: List of scenario file names

        Returns:
            List of results dictionaries
        """
        print(f"\n{'='*60}")
        print(f"Running Batch Scenarios: {len(scenario_files)} scenarios")
        print(f"{'='*60}\n")

        all_results = []

        for i, scenario_file in enumerate(scenario_files, 1):
            print(f"\n[{i}/{len(scenario_files)}] Processing: {scenario_file}")
            try:
                results = self.run_scenario(scenario_file)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {scenario_file}: {e}")
                continue

        return all_results

    def evaluate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate results and generate metrics.

        Args:
            results: Results from scenario run

        Returns:
            Evaluation metrics
        """
        print("\n--- Evaluating Results ---")

        # Extract data for evaluation
        consensus_results = [results['consensus_result']]
        expert_proposals = results['expert_proposals']

        # Calculate metrics
        consensus_metrics = self.metrics.calculate_consensus_metrics(consensus_results)
        diversity_metrics = self.metrics.calculate_diversity_metrics(expert_proposals)

        # Agent contributions
        agent_decisions = {
            agent.agent_id: agent.get_decision_history()
            for agent in self.expert_agents
        }
        agent_metrics = self.metrics.calculate_agent_contribution_metrics(agent_decisions)

        evaluation = {
            'consensus_metrics': consensus_metrics,
            'diversity_metrics': diversity_metrics,
            'agent_metrics': agent_metrics
        }

        # Print summary
        print(f"\nConsensus Rate: {consensus_metrics.get('consensus_rate', 0):.1%}")
        print(f"Average Agreement: {consensus_metrics.get('average_agreement_level', 0):.2f}")
        print(f"Opinion Diversity: {diversity_metrics.get('diversity_score', 0):.2f}")

        return evaluation

    def generate_visualizations(self, results: Dict[str, Any],
                               evaluation: Dict[str, Any]):
        """
        Generate visualizations for results.

        Args:
            results: Results from scenario run
            evaluation: Evaluation metrics
        """
        print("\n--- Generating Visualizations ---")

        scenario_id = results.get('scenario_id', 'unknown')

        # Agent contributions
        if 'agent_metrics' in evaluation:
            self.visualizer.plot_agent_contributions(
                evaluation['agent_metrics'],
                save_path=f"{scenario_id}_agent_contributions.png"
            )
            print("✓ Agent contributions plot saved")

        # Action comparison
        if 'available_actions' in results.get('scenario', {}):
            self.visualizer.plot_action_comparison(
                results['scenario']['available_actions'],
                ['effectiveness', 'safety', 'speed', 'cost', 'public_acceptance'],
                save_path=f"{scenario_id}_action_comparison.png"
            )
            print("✓ Action comparison plot saved")

        # Decision distribution
        self.visualizer.plot_decision_distribution(
            results['expert_proposals'],
            save_path=f"{scenario_id}_decision_distribution.png"
        )
        print("✓ Decision distribution plot saved")

    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save results to JSON file.

        Args:
            results: Results to save
            output_file: Output file path
        """
        output_path = Path(self.config.get('evaluation.output_dir', 'results')) / output_file

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main entry point for the Crisis MAS system."""

    parser = argparse.ArgumentParser(
        description='Crisis Management Multi-Agent System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--scenario',
        type=str,
        default='flood_scenario.json',
        help='Scenario file to run (default: flood_scenario.json)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Configuration file (optional)'
    )

    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM enhancement'
    )

    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization generation'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (default: auto-generated)'
    )

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)
    config.ensure_directories()

    # Override LLM setting if specified
    use_llm = not args.no_llm and config.get('llm.enable_llm', True)

    # Initialize system
    mas = CrisisMAS(config, use_llm=use_llm)

    # Load agents
    mas.load_agents(config.get('agents.profiles_file', 'agents/agent_profiles.json'))

    # Run scenario
    results = mas.run_scenario(args.scenario)

    # Evaluate results
    evaluation = mas.evaluate_results(results)

    # Generate visualizations
    if not args.no_viz and config.get('evaluation.save_visualizations', True):
        mas.generate_visualizations(results, evaluation)

    # Save results
    if args.output:
        output_file = args.output
    else:
        scenario_id = results.get('scenario_id', 'scenario')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{scenario_id}_{timestamp}_results.json"

    results['evaluation'] = evaluation
    mas.save_results(results, output_file)

    # Print LLM statistics if used
    if mas.llm_client:
        llm_stats = mas.llm_client.get_statistics()
        print(f"\nLLM Usage Statistics:")
        print(f"  Total requests: {llm_stats['total_requests']}")
        print(f"  Total tokens: {llm_stats['total_tokens']}")

    print("\n" + "="*60)
    print("Crisis MAS Execution Complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
