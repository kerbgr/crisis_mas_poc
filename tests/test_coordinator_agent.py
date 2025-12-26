"""
Test Script for CoordinatorAgent
Crisis Management Multi-Agent System

Tests CoordinatorAgent functionality including assessment collection,
belief aggregation, consensus detection, conflict resolution, and final decision-making.
"""

import os
from typing import Dict, Any, List
from agents.coordinator_agent import CoordinatorAgent
from agents.expert_agent import ExpertAgent
from llm_integration.claude_client import ClaudeClient
from decision_framework.evidential_reasoning import EvidentialReasoning
from decision_framework.mcda_engine import MCDAEngine
from decision_framework.consensus_model import ConsensusModel


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def create_test_scenario() -> Dict[str, Any]:
    """Create a test crisis scenario."""
    return {
        "scenario_id": "SC001",
        "crisis_type": "Urban Flood",
        "location": "Downtown Metropolitan Area",
        "affected_population": 50000,
        "severity": 0.85,
        "response_time_hours": 4,
        "weather_forecast": {
            "precipitation_mm": 200,
            "duration_hours": 48,
            "flood_risk": "HIGH"
        },
        "available_resources": {
            "emergency_vehicles": 50,
            "personnel": 200,
            "shelters": 5,
            "medical_facilities": 3
        }
    }


def create_test_alternatives() -> List[Dict[str, Any]]:
    """Create test response alternatives."""
    return [
        {
            "id": "A1",
            "name": "Immediate Mass Evacuation",
            "description": "Evacuate entire flood zone to designated shelters",
            "estimated_metrics": {
                "safety_score": 0.95,
                "cost_euros": 500000,
                "response_time_hours": 4,
                "effectiveness": 0.90
            }
        },
        {
            "id": "A2",
            "name": "Deploy Flood Barriers + Selective Evacuation",
            "description": "Deploy temporary barriers and evacuate high-risk areas only",
            "estimated_metrics": {
                "safety_score": 0.75,
                "cost_euros": 200000,
                "response_time_hours": 3,
                "effectiveness": 0.70
            }
        },
        {
            "id": "A3",
            "name": "Shelter-in-Place with Monitoring",
            "description": "Residents stay home with continuous monitoring",
            "estimated_metrics": {
                "safety_score": 0.50,
                "cost_euros": 50000,
                "response_time_hours": 1,
                "effectiveness": 0.50
            }
        },
        {
            "id": "A4",
            "name": "Prioritized Rescue Operations",
            "description": "Focus on rescuing vulnerable populations as flood occurs",
            "estimated_metrics": {
                "safety_score": 0.60,
                "cost_euros": 100000,
                "response_time_hours": 2,
                "effectiveness": 0.55
            }
        }
    ]


def create_mock_expert_agents() -> List[ExpertAgent]:
    """Create mock expert agents for testing."""
    os.environ['ANTHROPIC_API_KEY'] = 'test_key'

    # Create LLM clients (won't make real calls in tests)
    claude_client1 = ClaudeClient(api_key="test_key_1")
    claude_client2 = ClaudeClient(api_key="test_key_2")
    claude_client3 = ClaudeClient(api_key="test_key_3")

    # Create expert agents
    meteorologist = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client1
    )

    operations = ExpertAgent(
        agent_id="logistics_expert_01",
        llm_client=claude_client2
    )

    medical = ExpertAgent(
        agent_id="medical_expert_01",
        llm_client=claude_client3
    )

    return [meteorologist, operations, medical]


def test_initialization():
    """Test 1: CoordinatorAgent initialization."""
    print_section("Test 1: CoordinatorAgent Initialization")

    # Create expert agents
    expert_agents = create_mock_expert_agents()

    # Create decision framework components
    er_engine = EvidentialReasoning()
    mcda_engine = MCDAEngine()
    consensus_model = ConsensusModel(consensus_threshold=0.75)

    # Test case 1: Basic initialization
    print("Test Case 1: Basic initialization")
    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model
    )

    print(f"  Number of expert agents: {len(coordinator.expert_agents)}")
    print(f"  Consensus threshold: {coordinator.consensus_threshold}")
    print(f"  Parallel assessment: {coordinator.parallel_assessment}")
    print(f"  Decision count: {coordinator.decision_count}")

    assert len(coordinator.expert_agents) == 3
    assert coordinator.consensus_threshold == 0.75
    assert coordinator.decision_count == 0
    print(f"  ✓ Basic initialization works\n")

    # Test case 2: Custom agent weights
    print("Test Case 2: Custom agent weights")
    custom_weights = {
        "agent_meteorologist": 0.4,
        "logistics_expert_01": 0.35,
        "medical_expert_01": 0.25
    }

    coordinator2 = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model,
        agent_weights=custom_weights
    )

    print(f"  Agent weights: {coordinator2.agent_weights}")
    # Check weights sum to 1.0
    total = sum(coordinator2.agent_weights.values())
    print(f"  Weights sum: {total:.4f}")
    assert abs(total - 1.0) < 1e-10
    print(f"  ✓ Custom weights work\n")

    # Test case 3: Equal weights (default)
    print("Test Case 3: Equal weights (default)")
    coordinator3 = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model,
        agent_weights=None  # Should create equal weights
    )

    print(f"  Agent weights: {coordinator3.agent_weights}")
    # All weights should be approximately equal
    expected_weight = 1.0 / 3
    for weight in coordinator3.agent_weights.values():
        assert abs(weight - expected_weight) < 1e-10
    print(f"  ✓ Equal weights work\n")

    # Test case 4: Invalid initialization
    print("Test Case 4: Invalid initialization (empty agent list)")
    try:
        coordinator_bad = CoordinatorAgent(
            expert_agents=[],  # Empty list
            er_engine=er_engine,
            mcda_engine=mcda_engine,
            consensus_model=consensus_model
        )
        print("  ✗ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:50]}...\n")

    print("✓ Test 1 Passed - Coordinator initialization works")


def test_mock_assessments():
    """Test 2: Create and verify mock assessments."""
    print_section("Test 2: Mock Assessment Creation")

    print("Test Case 1: Create mock assessments for testing")

    # Create mock assessments that mimic what ExpertAgent.evaluate_scenario returns
    mock_assessments = {
        "agent_meteorologist": {
            "agent_id": "agent_meteorologist",
            "agent_name": "Dr. Sarah Chen",
            "belief_distribution": {"A1": 0.70, "A2": 0.20, "A3": 0.08, "A4": 0.02},
            "confidence": 0.87,
            "reasoning": "High precipitation risk requires immediate evacuation"
        },
        "logistics_expert_01": {
            "agent_id": "logistics_expert_01",
            "agent_name": "Jennifer Rodriguez",
            "belief_distribution": {"A1": 0.50, "A2": 0.35, "A3": 0.10, "A4": 0.05},
            "confidence": 0.78,
            "reasoning": "Evacuation feasible but barriers are cost-effective"
        },
        "medical_expert_01": {
            "agent_id": "medical_expert_01",
            "agent_name": "Dr. Marcus Williams",
            "belief_distribution": {"A1": 0.80, "A2": 0.15, "A3": 0.03, "A4": 0.02},
            "confidence": 0.92,
            "reasoning": "Patient safety requires maximum protection"
        }
    }

    print(f"  Number of assessments: {len(mock_assessments)}")
    for agent_id, assessment in mock_assessments.items():
        belief_sum = sum(assessment['belief_distribution'].values())
        print(f"  {agent_id}: confidence={assessment['confidence']:.2f}, belief_sum={belief_sum:.4f}")
        assert abs(belief_sum - 1.0) < 1e-10

    print(f"  ✓ Mock assessments created\n")

    print("✓ Test 2 Passed - Mock assessment creation works")
    return mock_assessments


def test_belief_aggregation(mock_assessments):
    """Test 3: Belief aggregation using ER."""
    print_section("Test 3: Belief Aggregation")

    expert_agents = create_mock_expert_agents()
    er_engine = EvidentialReasoning()
    mcda_engine = MCDAEngine()
    consensus_model = ConsensusModel()

    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model
    )

    print("Test Case 1: Aggregate beliefs from mock assessments")
    aggregated = coordinator.aggregate_beliefs(mock_assessments)

    print(f"  Aggregated beliefs: {aggregated['aggregated_beliefs']}")
    print(f"  Uncertainty: {aggregated['uncertainty']:.4f}")
    print(f"  Confidence: {aggregated['confidence']:.4f}")

    # Check that aggregated beliefs sum to approximately 1.0
    belief_sum = sum(aggregated['aggregated_beliefs'].values())
    print(f"  Belief sum: {belief_sum:.4f}")
    assert abs(belief_sum - 1.0) < 1e-6
    print(f"  ✓ Beliefs aggregated correctly\n")

    # Test case 2: Verify ER details
    print("Test Case 2: Verify ER engine details")
    er_details = aggregated.get('er_details')
    assert er_details is not None
    print(f"  ER details present: {er_details is not None}")
    print(f"  Agents involved: {len(er_details.get('agents_involved', []))}")
    print(f"  ✓ ER details available\n")

    print("✓ Test 3 Passed - Belief aggregation works")


def test_consensus_checking(mock_assessments):
    """Test 4: Consensus level checking."""
    print_section("Test 4: Consensus Checking")

    expert_agents = create_mock_expert_agents()
    er_engine = EvidentialReasoning()
    mcda_engine = MCDAEngine()
    consensus_model = ConsensusModel(consensus_threshold=0.75)

    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model
    )

    print("Test Case 1: Check consensus with mock assessments")
    consensus_info = coordinator.check_consensus(mock_assessments)

    print(f"  Consensus level: {consensus_info['consensus_level']:.4f}")
    print(f"  Consensus reached: {consensus_info['consensus_reached']}")
    print(f"  Number of conflicts: {len(consensus_info.get('conflicts', []))}")

    assert 'consensus_level' in consensus_info
    assert 'consensus_reached' in consensus_info
    print(f"  ✓ Consensus check completed\n")

    # Test case 2: Check pairwise agreements
    print("Test Case 2: Pairwise agreements")
    pairwise = consensus_info.get('pairwise_agreements', {})
    print(f"  Number of pairwise comparisons: {len(pairwise)}")
    if pairwise:
        for pair, similarity in list(pairwise.items())[:3]:
            print(f"    {pair}: {similarity:.4f}")
    print(f"  ✓ Pairwise agreements calculated\n")

    print("✓ Test 4 Passed - Consensus checking works")


def test_conflict_resolution(mock_assessments):
    """Test 5: Conflict resolution."""
    print_section("Test 5: Conflict Resolution")

    expert_agents = create_mock_expert_agents()
    er_engine = EvidentialReasoning()
    mcda_engine = MCDAEngine()
    consensus_model = ConsensusModel(consensus_threshold=0.75)

    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model
    )

    # First check consensus
    consensus_info = coordinator.check_consensus(mock_assessments)
    conflicts = consensus_info.get('conflicts', [])

    print(f"Test Case 1: Resolve {len(conflicts)} conflicts")

    resolution = coordinator.resolve_conflicts(conflicts, mock_assessments)

    print(f"  Resolution strategy: {resolution['resolution_strategy']}")
    print(f"  Number of suggested actions: {len(resolution.get('suggested_actions', []))}")
    print(f"  Rationale: {resolution.get('rationale', 'None')[:100]}...")

    assert 'resolution_strategy' in resolution
    assert 'suggested_actions' in resolution
    print(f"  ✓ Conflict resolution completed\n")

    # Test case 2: No conflicts scenario
    print("Test Case 2: No conflicts scenario")
    resolution_no_conflict = coordinator.resolve_conflicts([], mock_assessments)
    print(f"  Strategy: {resolution_no_conflict['resolution_strategy']}")
    assert resolution_no_conflict['resolution_strategy'] == 'no_action_needed'
    print(f"  ✓ No-conflict case handled\n")

    print("✓ Test 5 Passed - Conflict resolution works")


def test_explanation_generation(mock_assessments):
    """Test 6: Explanation generation."""
    print_section("Test 6: Explanation Generation")

    expert_agents = create_mock_expert_agents()
    er_engine = EvidentialReasoning()
    mcda_engine = MCDAEngine()
    consensus_model = ConsensusModel()

    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model
    )

    # Create a mock decision
    mock_decision = {
        'recommended_alternative': 'A1',
        'confidence': 0.75,
        'consensus_level': 0.82,
        'final_scores': {'A1': 0.75, 'A2': 0.45, 'A3': 0.25, 'A4': 0.15},
        'agent_opinions': {
            'agent_meteorologist': {
                'agent_name': 'Dr. Sarah Chen',
                'preference': 'A1',
                'confidence': 0.87,
                'reasoning': 'High precipitation risk'
            },
            'logistics_expert_01': {
                'agent_name': 'Jennifer Rodriguez',
                'preference': 'A2',
                'confidence': 0.78,
                'reasoning': 'Cost-effective option'
            },
            'medical_expert_01': {
                'agent_name': 'Dr. Marcus Williams',
                'preference': 'A1',
                'confidence': 0.92,
                'reasoning': 'Patient safety priority'
            }
        },
        'consensus_reached': True,
        'conflicts': [],
        'timestamp': '2025-11-06T14:00:00',
        'agents_participated': 3,
        'decision_time_seconds': 2.5
    }

    print("Test Case 1: Generate explanation")
    explanation = coordinator.generate_explanation(mock_decision)

    print(f"  Explanation length: {len(explanation)} chars")
    print(f"  Contains recommended: {'A1' in explanation}")
    print(f"  Contains consensus: {'CONSENSUS' in explanation}")
    print(f"  Contains methodology: {'METHODOLOGY' in explanation}")

    assert len(explanation) > 0
    assert 'A1' in explanation
    assert 'CONSENSUS' in explanation
    print(f"  ✓ Explanation generated\n")

    print("Test Case 2: Display explanation excerpt")
    lines = explanation.split('\n')
    print(f"  Total lines: {len(lines)}")
    print(f"  First 5 lines:")
    for line in lines[:5]:
        print(f"    {line}")
    print(f"  ✓ Explanation formatted correctly\n")

    print("✓ Test 6 Passed - Explanation generation works")


def test_decision_history():
    """Test 7: Decision history tracking."""
    print_section("Test 7: Decision History Tracking")

    expert_agents = create_mock_expert_agents()
    er_engine = EvidentialReasoning()
    mcda_engine = MCDAEngine()
    consensus_model = ConsensusModel()

    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model
    )

    print("Test Case 1: Initial state")
    print(f"  Decision count: {coordinator.decision_count}")
    print(f"  Last decision: {coordinator.get_last_decision()}")
    print(f"  History length: {len(coordinator.get_decision_history())}")

    assert coordinator.decision_count == 0
    assert coordinator.get_last_decision() is None
    assert len(coordinator.get_decision_history()) == 0
    print(f"  ✓ Initial state correct\n")

    # Manually add decisions to history (simulating workflow)
    print("Test Case 2: Simulate decision tracking")
    mock_decision = {
        'recommended_alternative': 'A1',
        'confidence': 0.80,
        'timestamp': '2025-11-06T14:00:00'
    }

    coordinator.last_decision = mock_decision
    coordinator.decision_count = 1
    coordinator.decision_history.append({
        'decision_number': 1,
        'decision': mock_decision,
        'timestamp': mock_decision['timestamp']
    })

    print(f"  Decision count: {coordinator.decision_count}")
    print(f"  Last decision exists: {coordinator.get_last_decision() is not None}")
    print(f"  History length: {len(coordinator.get_decision_history())}")

    assert coordinator.decision_count == 1
    assert coordinator.get_last_decision() is not None
    assert len(coordinator.get_decision_history()) == 1
    print(f"  ✓ Decision tracking works\n")

    print("✓ Test 7 Passed - Decision history tracking works")


def test_string_representations():
    """Test 8: String representations."""
    print_section("Test 8: String Representations")

    expert_agents = create_mock_expert_agents()
    er_engine = EvidentialReasoning()
    mcda_engine = MCDAEngine()
    consensus_model = ConsensusModel()

    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model
    )

    print("Test Case 1: __repr__")
    repr_str = repr(coordinator)
    print(f"  Repr: {repr_str}")
    assert 'CoordinatorAgent' in repr_str
    assert 'experts=3' in repr_str
    print(f"  ✓ __repr__ works\n")

    print("Test Case 2: __str__")
    str_str = str(coordinator)
    print(f"  Str:\n{str_str}")
    assert 'Coordinator Agent' in str_str
    assert 'Expert Agents: 3' in str_str
    print(f"  ✓ __str__ works\n")

    print("✓ Test 8 Passed - String representations work")


def test_error_handling():
    """Test 9: Error handling."""
    print_section("Test 9: Error Handling")

    expert_agents = create_mock_expert_agents()
    er_engine = EvidentialReasoning()
    mcda_engine = MCDAEngine()
    consensus_model = ConsensusModel()

    # Test case 1: Empty assessments in aggregate_beliefs
    print("Test Case 1: Empty assessments")
    coordinator = CoordinatorAgent(
        expert_agents=expert_agents,
        er_engine=er_engine,
        mcda_engine=mcda_engine,
        consensus_model=consensus_model
    )

    empty_result = coordinator.aggregate_beliefs({})
    print(f"  Empty aggregation result: {empty_result['aggregated_beliefs']}")
    print(f"  Uncertainty: {empty_result['uncertainty']}")
    assert empty_result['uncertainty'] == 1.0
    print(f"  ✓ Empty assessments handled\n")

    # Test case 2: Single agent in consensus check
    print("Test Case 2: Single agent consensus")
    single_assessment = {
        "agent_meteorologist": {
            "belief_distribution": {"A1": 0.7, "A2": 0.3}
        }
    }

    consensus_result = coordinator.check_consensus(single_assessment)
    print(f"  Consensus level: {consensus_result['consensus_level']}")
    print(f"  Message: {consensus_result.get('message', 'None')[:50]}")
    assert consensus_result['consensus_level'] == 1.0  # Single agent = full consensus
    print(f"  ✓ Single agent handled\n")

    # Test case 3: Error decision creation
    print("Test Case 3: Error decision creation")
    error_decision = coordinator._create_error_decision(
        "Test error",
        {"scenario_id": "SC001"},
        []
    )

    print(f"  Error decision: {error_decision.get('recommended_alternative')}")
    print(f"  Error message: {error_decision.get('error')}")
    assert error_decision['recommended_alternative'] is None
    assert 'error' in error_decision
    print(f"  ✓ Error decision created\n")

    print("✓ Test 9 Passed - Error handling works")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("COORDINATOR AGENT - COMPREHENSIVE TEST SUITE")
    print("Crisis Management Multi-Agent System")
    print("="*70)

    # Run all tests
    test_initialization()

    mock_assessments = test_mock_assessments()

    test_belief_aggregation(mock_assessments)

    test_consensus_checking(mock_assessments)

    test_conflict_resolution(mock_assessments)

    test_explanation_generation(mock_assessments)

    test_decision_history()

    test_string_representations()

    test_error_handling()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNote: These tests validate CoordinatorAgent logic with mock assessments.")
    print("Full integration with real LLM API calls requires valid API keys")
    print("and is tested separately in integration tests.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
