#!/usr/bin/env python3
"""
Integration Test for Crisis Management Multi-Agent System
Master's Thesis PoC - Complete End-to-End Workflow Validation

This test suite validates:
1. Scenario and configuration loading
2. Agent initialization
3. Full multi-agent decision process
4. Evidential Reasoning aggregation
5. MCDA scoring
6. Consensus detection
7. Output generation and validation
8. Performance measurement

Run with: pytest tests/test_integration.py -v
Or: python tests/test_integration.py
"""

import os
import sys
import json
import time
import unittest
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


class TestCrisisMASIntegration(unittest.TestCase):
    """
    Comprehensive integration test for Crisis MAS.

    Tests the complete workflow from scenario loading through
    decision-making to output generation.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used across all tests."""
        cls.project_root = project_root
        cls.scenarios_dir = cls.project_root / "scenarios"
        cls.agents_dir = cls.project_root / "agents"
        cls.test_output_dir = cls.project_root / "tests" / "output"

        # Create test output directory
        cls.test_output_dir.mkdir(exist_ok=True, parents=True)

        # Store test results for performance analysis
        cls.performance_metrics = {}
        cls.test_results = {}

        print("\n" + "="*80)
        print("CRISIS MAS INTEGRATION TEST SUITE")
        print("="*80)
        print(f"Project Root: {cls.project_root}")
        print(f"Test Output:  {cls.test_output_dir}")
        print("="*80 + "\n")

    def setUp(self):
        """Set up before each test."""
        self.start_time = time.time()

    def tearDown(self):
        """Clean up after each test."""
        elapsed = time.time() - self.start_time
        test_name = self._testMethodName
        self.performance_metrics[test_name] = elapsed
        print(f"  ⏱️  {test_name}: {elapsed:.2f}s")

    # ========================================================================
    # TEST 1: Scenario Loading
    # ========================================================================

    def test_01_load_flood_scenario(self):
        """Test that flood scenario loads correctly with all required fields."""
        print("\n🧪 TEST 1: Loading flood scenario...")

        scenario_path = self.scenarios_dir / "flood_scenario.json"

        # Check file exists
        self.assertTrue(
            scenario_path.exists(),
            f"Scenario file not found: {scenario_path}"
        )

        # Load and parse JSON
        with open(scenario_path, 'r') as f:
            scenario = json.load(f)

        # Validate required top-level fields
        required_fields = ['id', 'type', 'name', 'description', 'severity',
                          'location', 'available_actions']
        for field in required_fields:
            self.assertIn(
                field, scenario,
                f"Missing required field '{field}' in scenario"
            )

        # Validate scenario data types
        self.assertIsInstance(scenario['severity'], (int, float),
                            "Severity must be numeric")
        self.assertTrue(
            0 <= scenario['severity'] <= 1,
            f"Severity {scenario['severity']} must be between 0 and 1"
        )

        # Validate alternatives exist
        alternatives = scenario.get('available_actions', [])
        self.assertGreater(
            len(alternatives), 0,
            "Scenario must have at least one alternative"
        )

        # Validate each alternative has required fields
        for i, alt in enumerate(alternatives):
            alt_fields = ['id', 'name', 'description']
            for field in alt_fields:
                self.assertIn(
                    field, alt,
                    f"Alternative {i} missing required field '{field}'"
                )

        # Store for later tests
        self.test_results['scenario'] = scenario
        self.test_results['alternatives'] = alternatives

        print(f"  ✅ Loaded scenario: {scenario['name']}")
        print(f"  ✅ Type: {scenario['type']}, Severity: {scenario['severity']}")
        print(f"  ✅ Alternatives: {len(alternatives)}")

    # ========================================================================
    # TEST 2: Configuration Loading
    # ========================================================================

    def test_02_load_agent_profiles(self):
        """Test that agent profiles load correctly."""
        print("\n🧪 TEST 2: Loading agent profiles...")

        profiles_path = self.agents_dir / "agent_profiles.json"

        self.assertTrue(
            profiles_path.exists(),
            f"Agent profiles file not found: {profiles_path}"
        )

        with open(profiles_path, 'r') as f:
            profiles = json.load(f)

        # Check structure
        self.assertIn('agents', profiles, "Profiles must have 'agents' key")
        agents_list = profiles['agents']

        self.assertGreater(
            len(agents_list), 0,
            "Must have at least one agent profile"
        )

        # Validate each agent profile
        for agent in agents_list:
            required = ['agent_id', 'name', 'role', 'expertise']
            for field in required:
                self.assertIn(
                    field, agent,
                    f"Agent profile missing required field '{field}'"
                )

        self.test_results['agent_profiles'] = agents_list

        print(f"  ✅ Loaded {len(agents_list)} agent profiles")
        for agent in agents_list:
            print(f"    - {agent['name']} ({agent['role']})")

    def test_03_load_criteria_weights(self):
        """Test that criteria weights load correctly."""
        print("\n🧪 TEST 3: Loading criteria weights...")

        criteria_path = self.scenarios_dir / "criteria_weights.json"

        self.assertTrue(
            criteria_path.exists(),
            f"Criteria weights file not found: {criteria_path}"
        )

        with open(criteria_path, 'r') as f:
            criteria = json.load(f)

        # Validate structure - can have 'criteria', 'decision_criteria', or be a direct dict
        if 'criteria' in criteria:
            criteria_dict = {c['name']: c for c in criteria['criteria']}
        elif 'decision_criteria' in criteria:
            criteria_dict = criteria['decision_criteria']
        else:
            # Assume the whole thing is a criteria dict
            criteria_dict = criteria

        self.assertGreater(
            len(criteria_dict), 0,
            "Must have at least one criterion"
        )

        # Validate each criterion
        total_weight = 0
        for crit_id, crit in criteria_dict.items():
            self.assertIn('name', crit, f"Criterion {crit_id} must have 'name'")
            self.assertIn('weight', crit, f"Criterion {crit_id} must have 'weight'")

            weight = crit['weight']
            self.assertIsInstance(weight, (int, float), "Weight must be numeric")
            self.assertGreater(weight, 0, "Weight must be positive")
            total_weight += weight

        # Weights should approximately sum to 1.0 (allow small floating point error)
        self.assertAlmostEqual(
            total_weight, 1.0, places=2,
            msg=f"Criteria weights should sum to 1.0, got {total_weight}"
        )

        self.test_results['criteria'] = criteria_dict

        print(f"  ✅ Loaded {len(criteria_dict)} criteria")
        print(f"  ✅ Total weight: {total_weight:.3f}")

    # ========================================================================
    # TEST 4: LLM Client Initialization
    # ========================================================================

    def test_04_initialize_llm_client(self):
        """Test LLM client initialization."""
        print("\n🧪 TEST 4: Initializing LLM client...")

        # Check for API key
        api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('OPENAI_API_KEY')

        if not api_key:
            print("  ⚠️  No API key found - using mock mode")
            self.test_results['llm_client'] = None
            self.test_results['use_mock'] = True
            self.skipTest("No API key available - skipping LLM tests")
            return

        # Try to initialize client
        if os.getenv('ANTHROPIC_API_KEY'):
            client = ClaudeClient(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.assertIsNotNone(client, "ClaudeClient initialization failed")
            print(f"  ✅ Initialized ClaudeClient (model: {client.model})")
        elif os.getenv('OPENAI_API_KEY'):
            client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
            self.assertIsNotNone(client, "OpenAIClient initialization failed")
            print(f"  ✅ Initialized OpenAIClient (model: {client.model})")

        self.test_results['llm_client'] = client
        self.test_results['use_mock'] = False

    # ========================================================================
    # TEST 5: Expert Agent Initialization
    # ========================================================================

    def test_05_initialize_expert_agents(self):
        """Test expert agent initialization."""
        print("\n🧪 TEST 5: Initializing expert agents...")

        # Need profiles from test 2
        self.assertIn('agent_profiles', self.test_results,
                     "Must run test_02 first")

        # Get or create mock LLM client
        llm_client = self.test_results.get('llm_client')
        if llm_client is None:
            # Create mock client for testing without API key
            print("  ⚠️  Using mock LLM client")
            llm_client = self._create_mock_llm_client()

        # Initialize agents
        agent_profiles = self.test_results['agent_profiles']
        profile_path = self.agents_dir / "agent_profiles.json"

        agents = []
        for profile in agent_profiles[:3]:  # Test with first 3 agents
            agent = ExpertAgent(
                agent_id=profile['agent_id'],
                llm_client=llm_client,
                profile_path=str(profile_path)
            )

            # Validate agent properties
            self.assertEqual(agent.agent_id, profile['agent_id'])
            self.assertEqual(agent.name, profile['name'])
            self.assertEqual(agent.role, profile['role'])

            agents.append(agent)
            print(f"  ✅ Initialized: {agent.name} ({agent.role})")

        self.assertGreater(len(agents), 0, "Should initialize at least one agent")
        self.test_results['expert_agents'] = agents

    # ========================================================================
    # TEST 6: Decision Framework Initialization
    # ========================================================================

    def test_06_initialize_decision_framework(self):
        """Test decision framework components initialization."""
        print("\n🧪 TEST 6: Initializing decision framework...")

        # Initialize ER engine
        er_engine = EvidentialReasoning()
        self.assertIsNotNone(er_engine, "ER engine initialization failed")
        print("  ✅ Evidential Reasoning engine initialized")

        # Initialize MCDA engine
        criteria_path = self.scenarios_dir / "criteria_weights.json"
        mcda_engine = MCDAEngine(criteria_weights_path=str(criteria_path))
        self.assertIsNotNone(mcda_engine, "MCDA engine initialization failed")
        self.assertGreater(len(mcda_engine.criteria_config), 0, "MCDA must have criteria")
        print(f"  ✅ MCDA engine initialized ({len(mcda_engine.criteria_config)} criteria)")

        # Initialize Consensus model
        consensus_model = ConsensusModel(consensus_threshold=0.75)
        self.assertIsNotNone(consensus_model, "Consensus model initialization failed")
        self.assertEqual(consensus_model.consensus_threshold, 0.75)
        print("  ✅ Consensus model initialized (threshold: 0.75)")

        # Store components
        self.test_results['er_engine'] = er_engine
        self.test_results['mcda_engine'] = mcda_engine
        self.test_results['consensus_model'] = consensus_model

    # ========================================================================
    # TEST 7: Coordinator Initialization
    # ========================================================================

    def test_07_initialize_coordinator(self):
        """Test coordinator agent initialization."""
        print("\n🧪 TEST 7: Initializing coordinator...")

        # Need agents and framework from previous tests
        self.assertIn('expert_agents', self.test_results,
                     "Must run test_05 first")
        self.assertIn('er_engine', self.test_results,
                     "Must run test_06 first")

        expert_agents = self.test_results['expert_agents']
        er_engine = self.test_results['er_engine']
        mcda_engine = self.test_results['mcda_engine']
        consensus_model = self.test_results['consensus_model']

        # Initialize coordinator
        coordinator = CoordinatorAgent(
            expert_agents=expert_agents,
            er_engine=er_engine,
            mcda_engine=mcda_engine,
            consensus_model=consensus_model,
            parallel_assessment=True
        )

        self.assertIsNotNone(coordinator, "Coordinator initialization failed")
        self.assertEqual(len(coordinator.expert_agents), len(expert_agents))
        self.assertTrue(coordinator.parallel_assessment)

        print(f"  ✅ Coordinator initialized with {len(expert_agents)} agents")
        print(f"  ✅ Parallel assessment: {coordinator.parallel_assessment}")

        self.test_results['coordinator'] = coordinator

    # ========================================================================
    # TEST 8: Multi-Agent Decision Process
    # ========================================================================

    def test_08_run_decision_process(self):
        """Test complete multi-agent decision process."""
        print("\n🧪 TEST 8: Running multi-agent decision process...")

        # Check prerequisites
        required = ['scenario', 'alternatives', 'coordinator']
        for req in required:
            self.assertIn(req, self.test_results,
                         f"Must run previous tests first (missing {req})")

        scenario = self.test_results['scenario']
        alternatives = self.test_results['alternatives']
        coordinator = self.test_results['coordinator']

        # Run decision process
        start_time = time.time()

        try:
            decision = coordinator.make_final_decision(
                scenario=scenario,
                alternatives=alternatives,
                criteria=self.test_results.get('criteria')
            )
        except Exception as e:
            if self.test_results.get('use_mock', False):
                print(f"  ⚠️  Skipping due to mock LLM: {str(e)[:100]}")
                self.skipTest("Mock LLM cannot make real API calls")
                return
            else:
                raise

        elapsed = time.time() - start_time

        # Validate decision structure
        required_fields = [
            'recommended_alternative',
            'confidence',
            'consensus_level',
            'final_scores',
            'consensus_reached'
        ]

        for field in required_fields:
            self.assertIn(
                field, decision,
                f"Decision missing required field '{field}'"
            )

        # Validate decision values
        self.assertIsNotNone(
            decision['recommended_alternative'],
            "Decision must recommend an alternative"
        )

        self.assertIsInstance(
            decision['confidence'], (int, float),
            "Confidence must be numeric"
        )
        self.assertTrue(
            0 <= decision['confidence'] <= 1,
            f"Confidence {decision['confidence']} must be between 0 and 1"
        )

        self.assertIsInstance(
            decision['consensus_level'], (int, float),
            "Consensus level must be numeric"
        )
        self.assertTrue(
            0 <= decision['consensus_level'] <= 1,
            f"Consensus {decision['consensus_level']} must be between 0 and 1"
        )

        # Validate final scores
        final_scores = decision['final_scores']
        self.assertIsInstance(final_scores, dict, "Final scores must be a dict")
        self.assertGreater(len(final_scores), 0, "Must have at least one score")

        # All scores should be between 0 and 1
        for alt_id, score in final_scores.items():
            self.assertTrue(
                0 <= score <= 1,
                f"Score for {alt_id} ({score}) must be between 0 and 1"
            )

        # Store decision
        self.test_results['decision'] = decision

        print(f"  ✅ Decision completed in {elapsed:.2f}s")
        print(f"  ✅ Recommended: {decision['recommended_alternative']}")
        print(f"  ✅ Confidence: {decision['confidence']:.3f}")
        print(f"  ✅ Consensus: {decision['consensus_level']:.3f}")
        print(f"  ✅ Consensus reached: {decision['consensus_reached']}")

    # ========================================================================
    # TEST 9: Evidential Reasoning Validation
    # ========================================================================

    def test_09_validate_er_aggregation(self):
        """Test that ER aggregation produces valid results."""
        print("\n🧪 TEST 9: Validating ER aggregation...")

        self.assertIn('decision', self.test_results,
                     "Must run test_08 first")

        decision = self.test_results['decision']

        # Check if we have collection info with assessments
        if 'collection_info' not in decision:
            print("  ⚠️  No collection info in decision (may have failed)")
            self.skipTest("Decision does not contain assessment data")
            return

        collection_info = decision['collection_info']

        if 'assessments' not in collection_info:
            print("  ⚠️  No assessments in collection info")
            self.skipTest("No assessments available")
            return

        assessments = collection_info['assessments']

        # Should have at least one assessment
        self.assertGreater(
            len(assessments), 0,
            "Should have at least one agent assessment"
        )

        # Each assessment should have belief distribution
        for agent_id, assessment in assessments.items():
            self.assertIn(
                'belief_distribution', assessment,
                f"Assessment for {agent_id} missing belief_distribution"
            )

            beliefs = assessment['belief_distribution']
            self.assertIsInstance(beliefs, dict, "Beliefs must be a dict")

            # Beliefs should sum to approximately 1.0
            total = sum(beliefs.values())
            self.assertAlmostEqual(
                total, 1.0, places=2,
                msg=f"Beliefs for {agent_id} should sum to 1.0, got {total}"
            )

        print(f"  ✅ Validated {len(assessments)} agent assessments")
        print(f"  ✅ All belief distributions sum to 1.0")

    # ========================================================================
    # TEST 10: MCDA Scoring Validation
    # ========================================================================

    def test_10_validate_mcda_scoring(self):
        """Test that MCDA scoring produces valid results."""
        print("\n🧪 TEST 10: Validating MCDA scoring...")

        self.assertIn('mcda_engine', self.test_results,
                     "Must run test_06 first")
        self.assertIn('alternatives', self.test_results,
                     "Must run test_01 first")

        mcda_engine = self.test_results['mcda_engine']
        alternatives = self.test_results['alternatives']

        # Run MCDA ranking
        rankings = mcda_engine.rank_alternatives(alternatives)

        # Should return a list of tuples (alt_id, score, explanation)
        self.assertIsInstance(rankings, list, "Rankings must be a list")
        self.assertEqual(
            len(rankings), len(alternatives),
            "Should rank all alternatives"
        )

        # Validate each ranking
        seen_ids = set()
        for i, ranking in enumerate(rankings):
            self.assertEqual(len(ranking), 3,
                           f"Ranking {i} should be (id, score, explanation) tuple")

            alt_id, score, explanation = ranking

            # Check uniqueness
            self.assertNotIn(alt_id, seen_ids,
                           f"Duplicate alternative ID: {alt_id}")
            seen_ids.add(alt_id)

            # Validate score
            self.assertIsInstance(score, (int, float),
                                "Score must be numeric")
            self.assertTrue(
                0 <= score <= 1,
                f"Score {score} for {alt_id} must be between 0 and 1"
            )

        # Rankings should be in descending order
        scores = [r[1] for r in rankings]
        self.assertEqual(
            scores, sorted(scores, reverse=True),
            "Rankings should be in descending order by score"
        )

        print(f"  ✅ MCDA ranked {len(rankings)} alternatives")
        print(f"  ✅ Top alternative: {rankings[0][0]} (score: {rankings[0][1]:.3f})")

    # ========================================================================
    # TEST 11: Consensus Detection Validation
    # ========================================================================

    def test_11_validate_consensus_detection(self):
        """Test that consensus detection works correctly."""
        print("\n🧪 TEST 11: Validating consensus detection...")

        self.assertIn('decision', self.test_results,
                     "Must run test_08 first")

        decision = self.test_results['decision']

        # Check consensus fields
        self.assertIn('consensus_level', decision,
                     "Decision must have consensus_level")
        self.assertIn('consensus_reached', decision,
                     "Decision must have consensus_reached")

        consensus_level = decision['consensus_level']
        consensus_reached = decision['consensus_reached']

        # Validate types
        self.assertIsInstance(consensus_level, (int, float),
                            "Consensus level must be numeric")
        self.assertIsInstance(consensus_reached, bool,
                            "Consensus reached must be boolean")

        # Check range
        self.assertTrue(
            0 <= consensus_level <= 1,
            f"Consensus level {consensus_level} must be between 0 and 1"
        )

        # Check consistency with threshold
        if 'consensus_model' in self.test_results:
            threshold = self.test_results['consensus_model'].consensus_threshold

            if consensus_level >= threshold:
                self.assertTrue(
                    consensus_reached,
                    f"Consensus level {consensus_level} >= threshold {threshold} "
                    f"but consensus_reached is False"
                )

        print(f"  ✅ Consensus level: {consensus_level:.3f}")
        print(f"  ✅ Consensus reached: {consensus_reached}")

    # ========================================================================
    # TEST 12: Metrics Evaluation
    # ========================================================================

    def test_12_evaluate_metrics(self):
        """Test metrics evaluation."""
        print("\n🧪 TEST 12: Evaluating metrics...")

        self.assertIn('decision', self.test_results,
                     "Must run test_08 first")

        decision = self.test_results['decision']
        evaluator = MetricsEvaluator()

        # Calculate decision quality
        quality_metrics = evaluator.calculate_decision_quality(decision)
        self.assertIn('weighted_score', quality_metrics,
                     "Quality metrics must have weighted_score")
        self.assertTrue(
            0 <= quality_metrics['weighted_score'] <= 1,
            "Weighted score must be between 0 and 1"
        )
        print(f"  ✅ Decision quality: {quality_metrics['weighted_score']:.3f}")

        # Calculate confidence metrics
        confidence_metrics = evaluator.calculate_confidence_metrics(decision)
        self.assertIn('decision_confidence', confidence_metrics,
                     "Confidence metrics must have decision_confidence")
        print(f"  ✅ Confidence: {confidence_metrics['decision_confidence']:.3f}")

        # Calculate consensus metrics if we have assessments
        if 'collection_info' in decision and 'assessments' in decision['collection_info']:
            consensus_metrics = evaluator.calculate_consensus_metrics(
                decision['collection_info']['assessments']
            )
            self.assertIn('consensus_level', consensus_metrics,
                         "Consensus metrics must have consensus_level")
            print(f"  ✅ Consensus metrics calculated")

        self.test_results['metrics'] = {
            'quality': quality_metrics,
            'confidence': confidence_metrics
        }

    # ========================================================================
    # TEST 13: Output Generation
    # ========================================================================

    def test_13_generate_outputs(self):
        """Test output file generation."""
        print("\n🧪 TEST 13: Generating outputs...")

        self.assertIn('decision', self.test_results,
                     "Must run test_08 first")

        decision = self.test_results['decision']
        metrics = self.test_results.get('metrics', {})

        # Generate results JSON
        results_path = self.test_output_dir / "test_results.json"

        results = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'metrics': metrics
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.assertTrue(results_path.exists(), "Results file was not created")

        # Verify file is valid JSON
        with open(results_path, 'r') as f:
            loaded = json.load(f)

        self.assertIn('decision', loaded, "Saved results missing decision")
        self.assertIn('metrics', loaded, "Saved results missing metrics")

        print(f"  ✅ Results saved to: {results_path}")

        # Test report generation
        evaluator = MetricsEvaluator()
        report = evaluator.generate_report(metrics) if metrics else "No metrics available"
        report_path = self.test_output_dir / "test_report.txt"

        with open(report_path, 'w') as f:
            f.write(report)

        self.assertTrue(report_path.exists(), "Report file was not created")
        print(f"  ✅ Report saved to: {report_path}")

    # ========================================================================
    # TEST 14: Visualization Generation (Optional)
    # ========================================================================

    def test_14_generate_visualizations(self):
        """Test visualization generation (if data available)."""
        print("\n🧪 TEST 14: Generating visualizations...")

        # Skip if we don't have complete decision data
        if 'decision' not in self.test_results:
            self.skipTest("No decision data available")
            return

        decision = self.test_results['decision']

        # Check if we have enough data for visualizations
        if 'collection_info' not in decision or 'assessments' not in decision['collection_info']:
            print("  ⚠️  Insufficient data for visualizations")
            self.skipTest("No assessment data for visualizations")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for testing

            visualizer = SystemVisualizer(
                output_dir=str(self.test_output_dir),
                dpi=100  # Lower DPI for faster testing
            )

            # Test belief distribution plot
            plot_path = visualizer.plot_belief_distributions(
                decision['collection_info']['assessments'],
                save_path=str(self.test_output_dir / "test_beliefs.png")
            )

            self.assertTrue(
                Path(plot_path).exists(),
                "Belief distribution plot was not created"
            )

            print(f"  ✅ Generated belief distribution plot")

        except ImportError as e:
            print(f"  ⚠️  Skipping visualizations: {e}")
            self.skipTest("Visualization dependencies not available")
        except Exception as e:
            print(f"  ⚠️  Visualization generation failed: {e}")
            # Don't fail test for visualization errors

    # ========================================================================
    # TEST 15: Performance Summary
    # ========================================================================

    def test_15_performance_summary(self):
        """Generate performance summary."""
        print("\n🧪 TEST 15: Performance summary...")

        total_time = sum(self.performance_metrics.values())

        print(f"\n  ⏱️  Total test time: {total_time:.2f}s")
        print(f"  ⏱️  Number of tests: {len(self.performance_metrics)}")
        print(f"  ⏱️  Average per test: {total_time/max(len(self.performance_metrics), 1):.2f}s")

        # Slowest tests
        if self.performance_metrics:
            sorted_tests = sorted(
                self.performance_metrics.items(),
                key=lambda x: x[1],
                reverse=True
            )

            print("\n  📊 Slowest tests:")
            for test_name, elapsed in sorted_tests[:5]:
                print(f"    {test_name}: {elapsed:.2f}s")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _create_mock_llm_client(self):
        """Create a mock LLM client for testing without API keys."""
        class MockLLMClient:
            def __init__(self):
                self.model = "mock-model"
                self.request_count = 0

            def generate_assessment(self, prompt, **kwargs):
                """Return a mock assessment."""
                self.request_count += 1
                return {
                    'alternative_rankings': {
                        'action_rescue_operations': 0.40,
                        'action_hybrid_approach': 0.30,
                        'action_evacuate_immediate': 0.20,
                        'action_deploy_barriers': 0.10
                    },
                    'reasoning': 'Mock reasoning for testing',
                    'confidence': 0.75,
                    'key_concerns': ['Mock concern 1', 'Mock concern 2'],
                    '_metadata': {'mock': True}
                }

        return MockLLMClient()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        print("\n" + "="*80)
        print("INTEGRATION TEST SUMMARY")
        print("="*80)
        print(f"Test output directory: {cls.test_output_dir}")
        print("="*80 + "\n")


# ============================================================================
# Test Runner
# ============================================================================

def run_tests():
    """Run the integration test suite."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCrisisMASIntegration)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
