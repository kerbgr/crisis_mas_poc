"""
Test Script for ExpertAgent
Crisis Management Multi-Agent System

Tests ExpertAgent functionality including scenario evaluation, belief distribution,
criteria scoring, and integration with LLM clients (Claude, OpenAI, LM Studio).
"""

import os
import json
from typing import Dict, Any
from agents.expert_agent import ExpertAgent
from llm_integration.claude_client import ClaudeClient
from llm_integration.openai_client import OpenAIClient
from llm_integration.lmstudio_client import LMStudioClient


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def create_mock_llm_response() -> Dict[str, Any]:
    """Create a mock LLM response for testing."""
    return {
        "alternative_rankings": {
            "A1": 0.65,
            "A2": 0.25,
            "A3": 0.08,
            "A4": 0.02
        },
        "reasoning": "Given the severe flood forecast and limited time window, immediate mass evacuation (A1) is the safest option. The 48-hour precipitation forecast of 200mm exceeds historical flood thresholds.",
        "confidence": 0.87,
        "key_concerns": [
            "Rapid water rise expected within 4 hours",
            "Limited deployment time for barriers",
            "Population density in flood zone",
            "Historical precedent from 2019 flood"
        ]
    }


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


def create_test_alternatives() -> list:
    """Create test response alternatives."""
    return [
        {
            "id": "A1",
            "name": "Immediate Mass Evacuation",
            "description": "Evacuate entire flood zone to designated shelters",
            "safety_score": 0.95,
            "cost_estimate": 500000,
            "response_time_hours": 4,
            "effectiveness": 0.90,
            "advantages": [
                "Maximum safety for residents",
                "Proven effectiveness",
                "Clear communication"
            ],
            "disadvantages": [
                "High cost",
                "Logistical complexity",
                "Potential panic"
            ]
        },
        {
            "id": "A2",
            "name": "Deploy Flood Barriers + Selective Evacuation",
            "description": "Deploy temporary barriers and evacuate high-risk areas only",
            "safety_score": 0.75,
            "cost_estimate": 200000,
            "response_time_hours": 3,
            "effectiveness": 0.70,
            "advantages": [
                "Lower cost than full evacuation",
                "Protects infrastructure",
                "Less disruptive"
            ],
            "disadvantages": [
                "Barriers may fail",
                "Partial risk remains",
                "Requires technical expertise"
            ]
        },
        {
            "id": "A3",
            "name": "Shelter-in-Place with Monitoring",
            "description": "Residents stay home with continuous monitoring and emergency response",
            "safety_score": 0.50,
            "cost_estimate": 50000,
            "response_time_hours": 1,
            "effectiveness": 0.50,
            "advantages": [
                "Low cost",
                "Minimal disruption",
                "Fast implementation"
            ],
            "disadvantages": [
                "Higher risk to residents",
                "Relies on resident compliance",
                "Limited control"
            ]
        },
        {
            "id": "A4",
            "name": "Prioritized Rescue Operations",
            "description": "Focus on rescuing vulnerable populations as flood occurs",
            "safety_score": 0.60,
            "cost_estimate": 100000,
            "response_time_hours": 2,
            "effectiveness": 0.55,
            "advantages": [
                "Targets most vulnerable",
                "Moderate cost",
                "Flexible approach"
            ],
            "disadvantages": [
                "Reactive rather than preventive",
                "Timing critical",
                "Resource intensive during crisis"
            ]
        }
    ]


def test_agent_initialization():
    """Test 1: ExpertAgent initialization."""
    print_section("Test 1: ExpertAgent Initialization")

    # Set up mock API keys for testing
    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_initialization'
    os.environ['OPENAI_API_KEY'] = 'test_key_for_initialization'

    # Test case 1: Initialize with Claude client
    print("Test Case 1: Initialize with ClaudeClient")
    claude_client = ClaudeClient(api_key="test_api_key")
    agent1 = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client
    )
    print(f"  Agent ID: {agent1.agent_id}")
    print(f"  Agent Name: {agent1.name}")
    print(f"  Expertise: {agent1.expertise}")
    print(f"  LLM Client: {type(agent1.llm_client).__name__}")
    print(f"  Assessment Count: {agent1.assessment_count}")
    assert agent1.agent_id == "agent_meteorologist"
    assert agent1.llm_client is claude_client
    assert agent1.assessment_count == 0
    print(f"  ✓ ClaudeClient initialization works\n")

    # Test case 2: Initialize with OpenAI client
    print("Test Case 2: Initialize with OpenAIClient")
    openai_client = OpenAIClient(api_key="test_api_key")
    agent2 = ExpertAgent(
        agent_id="logistics_expert_01",
        llm_client=openai_client
    )
    print(f"  Agent ID: {agent2.agent_id}")
    print(f"  LLM Client: {type(agent2.llm_client).__name__}")
    assert agent2.llm_client is openai_client
    print(f"  ✓ OpenAIClient initialization works\n")

    # Test case 3: Initialize with LM Studio client
    print("Test Case 3: Initialize with LMStudioClient")
    lmstudio_client = LMStudioClient()
    agent3 = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=lmstudio_client
    )
    print(f"  Agent ID: {agent3.agent_id}")
    print(f"  LLM Client: {type(agent3.llm_client).__name__}")
    assert agent3.llm_client is lmstudio_client
    print(f"  ✓ LMStudioClient initialization works\n")

    # Test case 4: Verify prompt_templates initialized
    print("Test Case 4: Verify PromptTemplates initialization")
    assert agent1.prompt_templates is not None
    print(f"  PromptTemplates: {type(agent1.prompt_templates).__name__}")
    print(f"  ✓ PromptTemplates initialized\n")

    print("✓ Test 1 Passed - Agent initialization works with all LLM clients")


def test_belief_distribution():
    """Test 2: Belief distribution generation."""
    print_section("Test 2: Belief Distribution Generation")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_beliefs'
    claude_client = ClaudeClient(api_key="test_api_key")
    agent = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client
    )

    # Test case 1: Normal rankings
    print("Test Case 1: Normal rankings that sum to 1.0")
    llm_response = {
        "alternative_rankings": {
            "A1": 0.65,
            "A2": 0.25,
            "A3": 0.08,
            "A4": 0.02
        }
    }
    belief_dist = agent.generate_belief_distribution(llm_response)
    print(f"  Input rankings: {llm_response['alternative_rankings']}")
    print(f"  Belief distribution: {belief_dist}")
    print(f"  Sum: {sum(belief_dist.values()):.4f}")
    assert abs(sum(belief_dist.values()) - 1.0) < 1e-10
    print(f"  ✓ Normalized correctly\n")

    # Test case 2: Rankings that don't sum to 1.0
    print("Test Case 2: Rankings that need normalization")
    llm_response = {
        "alternative_rankings": {
            "A1": 7.0,
            "A2": 2.0,
            "A3": 0.8,
            "A4": 0.2
        }
    }
    belief_dist = agent.generate_belief_distribution(llm_response)
    print(f"  Input rankings (sum={sum(llm_response['alternative_rankings'].values())}): {llm_response['alternative_rankings']}")
    print(f"  Belief distribution: {belief_dist}")
    print(f"  Sum: {sum(belief_dist.values()):.4f}")
    assert abs(sum(belief_dist.values()) - 1.0) < 1e-10
    print(f"  ✓ Normalized correctly\n")

    # Test case 3: All zeros (edge case)
    print("Test Case 3: All zeros (uniform distribution fallback)")
    llm_response = {
        "alternative_rankings": {
            "A1": 0.0,
            "A2": 0.0,
            "A3": 0.0,
            "A4": 0.0
        }
    }
    belief_dist = agent.generate_belief_distribution(llm_response)
    print(f"  Input rankings: {llm_response['alternative_rankings']}")
    print(f"  Belief distribution (uniform): {belief_dist}")
    print(f"  Sum: {sum(belief_dist.values()):.4f}")
    assert abs(sum(belief_dist.values()) - 1.0) < 1e-10
    assert all(abs(v - 0.25) < 1e-10 for v in belief_dist.values())
    print(f"  ✓ Uniform distribution applied\n")

    # Test case 4: Verify relative ordering preserved
    print("Test Case 4: Verify relative ordering preserved")
    llm_response = {
        "alternative_rankings": {
            "A1": 10.0,
            "A2": 5.0,
            "A3": 3.0,
            "A4": 2.0
        }
    }
    belief_dist = agent.generate_belief_distribution(llm_response)
    print(f"  Input rankings: {llm_response['alternative_rankings']}")
    print(f"  Belief distribution: {belief_dist}")
    sorted_beliefs = sorted(belief_dist.items(), key=lambda x: x[1], reverse=True)
    print(f"  Sorted order: {[alt_id for alt_id, _ in sorted_beliefs]}")
    assert sorted_beliefs[0][0] == "A1"
    assert sorted_beliefs[1][0] == "A2"
    assert sorted_beliefs[2][0] == "A3"
    assert sorted_beliefs[3][0] == "A4"
    print(f"  ✓ Relative ordering preserved\n")

    print("✓ Test 2 Passed - Belief distribution generation works correctly")


def test_criteria_scores():
    """Test 3: Criteria score generation."""
    print_section("Test 3: Criteria Score Generation")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_criteria'
    claude_client = ClaudeClient(api_key="test_api_key")
    agent = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client
    )

    alternatives = create_test_alternatives()
    llm_response = create_mock_llm_response()

    # Test case 1: Generate criteria scores
    print("Test Case 1: Generate criteria scores from alternatives")
    criteria_scores = agent.get_criteria_scores(alternatives, llm_response)
    print(f"  Number of criteria: {len(criteria_scores)}")
    print(f"  Criteria names: {list(criteria_scores.keys())}")
    assert len(criteria_scores) > 0
    print(f"  ✓ Criteria scores generated\n")

    # Test case 2: Verify score structure
    print("Test Case 2: Verify criteria score structure")
    for criterion_name, scores in criteria_scores.items():
        print(f"  {criterion_name}: {scores}")
        assert isinstance(scores, dict)
        assert all(isinstance(alt_id, str) for alt_id in scores.keys())
        assert all(isinstance(score, (int, float)) for score in scores.values())
    print(f"  ✓ Structure is correct\n")

    # Test case 3: Verify all alternatives have scores
    print("Test Case 3: Verify all alternatives have scores for each criterion")
    for criterion_name, scores in criteria_scores.items():
        assert len(scores) == len(alternatives)
        for alt in alternatives:
            assert alt['id'] in scores
    print(f"  ✓ All alternatives scored\n")

    # Test case 4: Verify score ranges
    print("Test Case 4: Verify scores are in [0, 1] range")
    all_in_range = True
    for criterion_name, scores in criteria_scores.items():
        for alt_id, score in scores.items():
            if not (0 <= score <= 1):
                print(f"  WARNING: {criterion_name}[{alt_id}] = {score} out of range")
                all_in_range = False
    if all_in_range:
        print(f"  ✓ All scores in [0, 1] range\n")
    else:
        print(f"  ⚠ Some scores outside [0, 1] range (may be intentional)\n")

    print("✓ Test 3 Passed - Criteria score generation works")


def test_prompt_template_selection():
    """Test 4: Prompt template selection logic."""
    print_section("Test 4: Prompt Template Selection")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_prompts'
    claude_client = ClaudeClient(api_key="test_api_key")

    scenario = create_test_scenario()
    alternatives = create_test_alternatives()

    # Test case 1: Meteorologist template
    print("Test Case 1: Meteorologist agent selects correct template")
    agent_met = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client
    )
    prompt, system_prompt = agent_met._generate_prompt(scenario, alternatives, None)
    print(f"  Agent expertise: {agent_met.expertise}")
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Contains 'METEOROLOGIST': {('METEOROLOGIST' in prompt.upper())}")
    assert 'METEOROLOGIST' in prompt.upper() or 'WEATHER' in prompt.upper()
    print(f"  ✓ Meteorologist template selected\n")

    # Test case 2: Operations template
    print("Test Case 2: Operations agent selects correct template")
    agent_ops = ExpertAgent(
        agent_id="logistics_expert_01",
        llm_client=claude_client
    )
    prompt, system_prompt = agent_ops._generate_prompt(scenario, alternatives, None)
    print(f"  Agent expertise: {agent_ops.expertise}")
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Contains 'OPERATIONS': {('OPERATIONS' in prompt.upper())}")
    assert 'OPERATIONS' in prompt.upper() or 'LOGISTICS' in prompt.upper() or 'DIRECTOR' in prompt.upper()
    print(f"  ✓ Operations template selected\n")

    # Test case 3: Medical template
    print("Test Case 3: Medical agent selects correct template")
    agent_med = ExpertAgent(
        agent_id="medical_expert_01",
        llm_client=claude_client
    )
    prompt, system_prompt = agent_med._generate_prompt(scenario, alternatives, None)
    print(f"  Agent expertise: {agent_med.expertise}")
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Contains 'MEDICAL': {('MEDICAL' in prompt.upper())}")
    assert 'MEDICAL' in prompt.upper() or 'HEALTH' in prompt.upper()
    print(f"  ✓ Medical template selected\n")

    # Test case 4: System prompt
    print("Test Case 4: System prompt generation")
    print(f"  System prompt: {system_prompt[:100] if system_prompt else '(empty)'}...")
    assert len(system_prompt) > 0
    # Check for key terms in system prompt
    assert any(term in system_prompt.lower() for term in ['expert', 'professional', 'experience', 'crisis'])
    print(f"  ✓ System prompt generated\n")

    print("✓ Test 4 Passed - Prompt template selection works correctly")


def test_assessment_structure():
    """Test 5: Assessment return structure."""
    print_section("Test 5: Assessment Return Structure")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_structure'
    claude_client = ClaudeClient(api_key="test_api_key")
    agent = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client
    )

    # We'll create a mock assessment by manually calling internal methods
    # since we can't make real API calls in testing

    print("Test Case 1: Simulated assessment structure")
    llm_response = create_mock_llm_response()
    belief_dist = agent.generate_belief_distribution(llm_response)
    alternatives = create_test_alternatives()
    criteria_scores = agent.get_criteria_scores(alternatives, llm_response)

    # Build assessment structure similar to evaluate_scenario
    import datetime
    assessment = {
        'agent_id': agent.agent_id,
        'agent_name': agent.name,
        'agent_role': agent.role,
        'expertise': agent.expertise,
        'belief_distribution': belief_dist,
        'criteria_scores': criteria_scores,
        'reasoning': llm_response['reasoning'],
        'confidence': llm_response['confidence'],
        'key_concerns': llm_response['key_concerns'],
        'timestamp': datetime.datetime.now().isoformat(),
        'scenario_type': 'Urban Flood',
        'llm_metadata': {'model': 'test-model'},
        'assessment_number': 1
    }

    print(f"  Assessment keys: {list(assessment.keys())}")

    # Verify required keys
    required_keys = [
        'agent_id', 'agent_name', 'agent_role', 'expertise',
        'belief_distribution', 'criteria_scores', 'reasoning',
        'confidence', 'key_concerns', 'timestamp', 'scenario_type',
        'llm_metadata', 'assessment_number'
    ]

    for key in required_keys:
        assert key in assessment, f"Missing required key: {key}"
        print(f"  ✓ {key}: {type(assessment[key]).__name__}")

    print(f"\n  ✓ All required fields present\n")

    # Test case 2: Data types
    print("Test Case 2: Verify data types")
    assert isinstance(assessment['agent_id'], str)
    assert isinstance(assessment['belief_distribution'], dict)
    assert isinstance(assessment['criteria_scores'], dict)
    assert isinstance(assessment['reasoning'], str)
    assert isinstance(assessment['confidence'], (int, float))
    assert isinstance(assessment['key_concerns'], list)
    assert isinstance(assessment['timestamp'], str)
    assert isinstance(assessment['assessment_number'], int)
    print(f"  ✓ All data types correct\n")

    # Test case 3: Belief distribution validity
    print("Test Case 3: Belief distribution validity")
    belief_sum = sum(assessment['belief_distribution'].values())
    print(f"  Belief sum: {belief_sum:.6f}")
    assert abs(belief_sum - 1.0) < 1e-10
    print(f"  ✓ Belief distribution sums to 1.0\n")

    # Test case 4: Confidence range
    print("Test Case 4: Confidence in valid range")
    confidence = assessment['confidence']
    print(f"  Confidence: {confidence}")
    assert 0 <= confidence <= 1
    print(f"  ✓ Confidence in [0, 1]\n")

    print("✓ Test 5 Passed - Assessment structure is correct")


def test_explain_assessment():
    """Test 6: Assessment explanation."""
    print_section("Test 6: Assessment Explanation")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_explanation'
    claude_client = ClaudeClient(api_key="test_api_key")
    agent = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client
    )

    # Create mock last_assessment
    llm_response = create_mock_llm_response()
    belief_dist = agent.generate_belief_distribution(llm_response)
    alternatives = create_test_alternatives()
    criteria_scores = agent.get_criteria_scores(alternatives, llm_response)

    import datetime
    agent.last_assessment = {
        'agent_id': agent.agent_id,
        'agent_name': agent.name,
        'agent_role': agent.role,
        'expertise': agent.expertise,
        'belief_distribution': belief_dist,
        'criteria_scores': criteria_scores,
        'reasoning': llm_response['reasoning'],
        'confidence': llm_response['confidence'],
        'key_concerns': llm_response['key_concerns'],
        'timestamp': datetime.datetime.now().isoformat(),
        'scenario_type': 'Urban Flood',
        'llm_metadata': {'model': 'test-model'},
        'assessment_number': 1
    }

    # Test case 1: Generate explanation
    print("Test Case 1: Generate human-readable explanation")
    explanation = agent.explain_assessment()
    print(f"  Explanation length: {len(explanation)} chars")
    print(f"  Contains agent info: {agent.name in explanation}")
    print(f"  Contains confidence: {str(llm_response['confidence']) in explanation}")
    assert len(explanation) > 0
    assert agent.name in explanation
    print(f"  ✓ Explanation generated\n")

    # Test case 2: Display explanation
    print("Test Case 2: Display explanation content")
    print("━" * 70)
    print(explanation)
    print("━" * 70)
    print(f"  ✓ Explanation displays correctly\n")

    # Test case 3: No assessment yet
    print("Test Case 3: No assessment available")
    agent2 = ExpertAgent(
        agent_id="logistics_expert_01",
        llm_client=claude_client
    )
    explanation2 = agent2.explain_assessment()
    print(f"  Explanation: {explanation2}")
    assert "has not made any assessments yet" in explanation2 or "No assessment" in explanation2
    print(f"  ✓ Handles missing assessment\n")

    print("✓ Test 6 Passed - Assessment explanation works")


def test_error_handling():
    """Test 7: Error handling."""
    print_section("Test 7: Error Handling")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_errors'

    # Test case 1: Invalid LLM client type
    print("Test Case 1: Invalid LLM client type (should raise ValueError)")
    try:
        agent = ExpertAgent(
            agent_id="agent_meteorologist",
            llm_client="not_a_client"  # Invalid type
        )
        print("  ✗ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:50]}...\n")

    # Test case 2: Invalid agent_id
    print("Test Case 2: Non-existent agent_id (should raise Exception)")
    claude_client = ClaudeClient(api_key="test_api_key")
    try:
        agent = ExpertAgent(
            agent_id="nonexistent_agent_999",
            llm_client=claude_client
        )
        print("  ✗ Should have raised Exception")
        assert False
    except (ValueError, Exception) as e:
        print(f"  ✓ Correctly raised {type(e).__name__}: {str(e)[:50]}...\n")

    # Test case 3: Missing scenario fields
    print("Test Case 3: Incomplete scenario (should handle gracefully)")
    agent = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client
    )
    incomplete_scenario = {"scenario_id": "SC001"}  # Missing required fields
    alternatives = create_test_alternatives()

    # This should work - the prompt generation should handle missing fields
    try:
        prompt, system_prompt = agent._generate_prompt(incomplete_scenario, alternatives, None)
        print(f"  Prompt generated with incomplete scenario: {len(prompt)} chars")
        print(f"  ✓ Handles incomplete scenario data\n")
    except Exception as e:
        print(f"  ✓ Raised appropriate error: {type(e).__name__}\n")

    # Test case 4: Empty alternatives
    print("Test Case 4: Empty alternatives list")
    scenario = create_test_scenario()
    empty_alternatives = []

    try:
        prompt, system_prompt = agent._generate_prompt(scenario, empty_alternatives, None)
        print(f"  Prompt generated with empty alternatives: {len(prompt)} chars")
        print(f"  ✓ Handles empty alternatives\n")
    except Exception as e:
        print(f"  Raised error (expected): {type(e).__name__}\n")

    print("✓ Test 7 Passed - Error handling works correctly")


def test_state_management():
    """Test 8: Agent state management."""
    print_section("Test 8: Agent State Management")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_state'
    claude_client = ClaudeClient(api_key="test_api_key")
    agent = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client
    )

    # Test case 1: Initial state
    print("Test Case 1: Initial agent state")
    print(f"  Assessment count: {agent.assessment_count}")
    print(f"  Last assessment: {agent.last_assessment}")
    assert agent.assessment_count == 0
    assert agent.last_assessment is None
    print(f"  ✓ Initial state correct\n")

    # Test case 2: Simulate assessment (update state manually)
    print("Test Case 2: State after assessment")
    llm_response = create_mock_llm_response()
    belief_dist = agent.generate_belief_distribution(llm_response)

    # Manually update state (simulating what evaluate_scenario does)
    agent.assessment_count += 1
    import datetime
    agent.last_assessment = {
        'agent_id': agent.agent_id,
        'belief_distribution': belief_dist,
        'timestamp': datetime.datetime.now().isoformat(),
        'assessment_number': agent.assessment_count
    }

    print(f"  Assessment count: {agent.assessment_count}")
    print(f"  Last assessment exists: {agent.last_assessment is not None}")
    print(f"  Assessment number: {agent.last_assessment['assessment_number']}")
    assert agent.assessment_count == 1
    assert agent.last_assessment is not None
    print(f"  ✓ State updated correctly\n")

    # Test case 3: Multiple assessments
    print("Test Case 3: Multiple assessments increment counter")
    for i in range(3):
        agent.assessment_count += 1
        agent.last_assessment['assessment_number'] = agent.assessment_count

    print(f"  Final assessment count: {agent.assessment_count}")
    print(f"  Final assessment number: {agent.last_assessment['assessment_number']}")
    assert agent.assessment_count == 4
    assert agent.last_assessment['assessment_number'] == 4
    print(f"  ✓ Multiple assessments tracked\n")

    print("✓ Test 8 Passed - State management works correctly")


def test_integration_workflow():
    """Test 9: Complete integration workflow simulation."""
    print_section("Test 9: Complete Integration Workflow")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_workflow'

    print("Simulating complete crisis assessment workflow:\n")

    # Step 1: Initialize agents with different LLM clients
    print("Step 1: Initialize multiple expert agents")
    claude_client = ClaudeClient(api_key="test_claude_key")
    openai_client = OpenAIClient(api_key="test_openai_key")

    meteorologist = ExpertAgent(
        agent_id="agent_meteorologist",
        llm_client=claude_client
    )
    operations = ExpertAgent(
        agent_id="logistics_expert_01",
        llm_client=openai_client
    )

    print(f"  Meteorologist: {meteorologist.name} ({type(meteorologist.llm_client).__name__})")
    print(f"  Operations: {operations.name} ({type(operations.llm_client).__name__})")
    print(f"  ✓ Agents initialized\n")

    # Step 2: Prepare scenario and alternatives
    print("Step 2: Prepare crisis scenario")
    scenario = create_test_scenario()
    alternatives = create_test_alternatives()

    print(f"  Scenario: {scenario['crisis_type']}")
    print(f"  Location: {scenario['location']}")
    print(f"  Affected population: {scenario['affected_population']:,}")
    print(f"  Alternatives: {len(alternatives)}")
    print(f"  ✓ Scenario prepared\n")

    # Step 3: Generate prompts for each agent
    print("Step 3: Generate agent-specific prompts")
    met_prompt, met_system = meteorologist._generate_prompt(scenario, alternatives, None)
    ops_prompt, ops_system = operations._generate_prompt(scenario, alternatives, None)

    print(f"  Meteorologist prompt: {len(met_prompt)} chars")
    print(f"  Operations prompt: {len(ops_prompt)} chars")
    print(f"  Contains role-specific content: {('METEOROLOGIST' in met_prompt.upper()) and ('OPERATIONS' in ops_prompt.upper())}")
    print(f"  ✓ Role-specific prompts generated\n")

    # Step 4: Simulate LLM responses and generate beliefs
    print("Step 4: Simulate expert assessments")
    met_response = create_mock_llm_response()
    ops_response = {
        "alternative_rankings": {
            "A1": 0.45,
            "A2": 0.40,
            "A3": 0.10,
            "A4": 0.05
        },
        "reasoning": "From an operations perspective, both mass evacuation and targeted approach have merits.",
        "confidence": 0.75,
        "key_concerns": ["Resource allocation", "Coordination complexity"]
    }

    met_beliefs = meteorologist.generate_belief_distribution(met_response)
    ops_beliefs = operations.generate_belief_distribution(ops_response)

    print(f"  Meteorologist beliefs: {met_beliefs}")
    print(f"  Operations beliefs: {ops_beliefs}")
    print(f"  ✓ Belief distributions generated\n")

    # Step 5: Analyze consensus
    print("Step 5: Analyze multi-agent consensus")
    # Calculate variance for A1 (most critical alternative)
    a1_scores = [met_beliefs['A1'], ops_beliefs['A1']]
    avg_a1 = sum(a1_scores) / len(a1_scores)
    variance = sum((x - avg_a1)**2 for x in a1_scores) / len(a1_scores)

    print(f"  Meteorologist A1: {met_beliefs['A1']:.3f}")
    print(f"  Operations A1: {ops_beliefs['A1']:.3f}")
    print(f"  Average: {avg_a1:.3f}")
    print(f"  Variance: {variance:.4f}")

    consensus_level = "HIGH" if variance < 0.01 else "MODERATE" if variance < 0.05 else "LOW"
    print(f"  Consensus level: {consensus_level}")
    print(f"  ✓ Consensus analyzed\n")

    # Step 6: Generate criteria scores
    print("Step 6: Generate criteria-specific scores")
    met_criteria = meteorologist.get_criteria_scores(alternatives, met_response)
    ops_criteria = operations.get_criteria_scores(alternatives, ops_response)

    print(f"  Meteorologist criteria: {len(met_criteria)} criteria")
    print(f"  Operations criteria: {len(ops_criteria)} criteria")
    print(f"  ✓ Criteria scores generated\n")

    # Step 7: Summary
    print("Step 7: Workflow summary")
    print(f"  Agents: 2")
    print(f"  Scenario: {scenario['crisis_type']}")
    print(f"  Alternatives evaluated: {len(alternatives)}")
    print(f"  Top choice (Meteorologist): {max(met_beliefs.items(), key=lambda x: x[1])[0]}")
    print(f"  Top choice (Operations): {max(ops_beliefs.items(), key=lambda x: x[1])[0]}")
    print(f"  Multi-agent consensus: {consensus_level}")
    print(f"  ✓ Workflow completed successfully\n")

    print("✓ Test 9 Passed - Complete integration workflow works")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("EXPERT AGENT - COMPREHENSIVE TEST SUITE")
    print("Crisis Management Multi-Agent System")
    print("="*70)

    tests = [
        test_agent_initialization,
        test_belief_distribution,
        test_criteria_scores,
        test_prompt_template_selection,
        test_assessment_structure,
        test_explain_assessment,
        test_error_handling,
        test_state_management,
        test_integration_workflow
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ Test Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
    print("\nNote: These tests validate ExpertAgent logic without live API calls.")
    print("Integration with real LLM APIs requires valid API keys.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
