#!/usr/bin/env python3
"""
Diagnostic script to investigate belief_distribution type issues.

This script traces through the entire assessment collection and consensus
process to identify where belief_distributions become non-dicts.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.expert_agent import ExpertAgent
from llm_integration.claude_client import ClaudeClient


def check_type_deep(obj: Any, path: str = "root") -> None:
    """Recursively check types in a nested structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}"
            print(f"  {new_path}: {type(value).__name__}")
            if key == "belief_distribution":
                if not isinstance(value, dict):
                    print(f"    ⚠️  WARNING: Expected dict, got {type(value).__name__}")
                    print(f"    Value: {value}")
                else:
                    print(f"    ✓ Valid dict with {len(value)} alternatives")
            if isinstance(value, (dict, list)) and len(str(value)) < 1000:
                check_type_deep(value, new_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            check_type_deep(item, f"{path}[{i}]")


def test_single_agent_assessment():
    """Test a single agent assessment to check belief_distribution type."""
    print("="*70)
    print("DIAGNOSTIC TEST: Single Agent Assessment")
    print("="*70)

    # Create test scenario and alternatives
    scenario = {
        "type": "flood",
        "severity": 0.75,
        "location": "coastal_city",
        "affected_population": 10000,
        "description": "Major coastal flooding threatening residential areas"
    }

    alternatives = [
        {"id": "A1", "name": "Immediate Evacuation", "description": "Evacuate all residents"},
        {"id": "A2", "name": "Deploy Barriers", "description": "Deploy temporary flood barriers"},
        {"id": "A3", "name": "Shelter in Place", "description": "Have residents shelter in upper floors"}
    ]

    # Load agent profiles
    profiles_path = Path(__file__).parent / "agents" / "agent_profiles.json"
    with open(profiles_path, 'r') as f:
        profiles_data = json.load(f)

    # Test with first agent
    test_agent_data = profiles_data['agents'][0]
    print(f"\n📋 Testing Agent: {test_agent_data['name']} ({test_agent_data['agent_id']})")

    # Create LLM client (will use env var for API key)
    try:
        llm_client = ClaudeClient()
        print("✓ LLM client created successfully")
    except Exception as e:
        print(f"✗ Failed to create LLM client: {e}")
        print("  Note: This test requires ANTHROPIC_API_KEY environment variable")
        return None

    # Create agent
    agent = ExpertAgent(
        agent_id=test_agent_data['agent_id'],
        name=test_agent_data['name'],
        role=test_agent_data['role'],
        expertise=test_agent_data['expertise'],
        llm_client=llm_client
    )
    print(f"✓ Agent created: {agent.name}")

    # Get assessment
    print("\n🔄 Requesting assessment from agent...")
    try:
        assessment = agent.evaluate_scenario(scenario, alternatives)
        print("✓ Assessment received\n")

        # Check types
        print("📊 Assessment Structure:")
        print("-" * 70)
        check_type_deep(assessment, "assessment")

        # Specifically check belief_distribution
        print("\n🔍 Detailed Belief Distribution Analysis:")
        print("-" * 70)
        belief_dist = assessment.get('belief_distribution')

        print(f"Type: {type(belief_dist).__name__}")
        print(f"Is dict: {isinstance(belief_dist, dict)}")

        if isinstance(belief_dist, dict):
            print(f"✓ Valid dictionary with {len(belief_dist)} entries")
            print("\nContents:")
            for alt_id, belief in belief_dist.items():
                print(f"  {alt_id}: {belief} (type: {type(belief).__name__})")

            total = sum(belief_dist.values())
            print(f"\nSum of beliefs: {total:.6f}")

            if abs(total - 1.0) > 0.01:
                print(f"  ⚠️  WARNING: Beliefs don't sum to 1.0 (off by {abs(total - 1.0):.6f})")
        else:
            print(f"✗ PROBLEM FOUND: belief_distribution is not a dict!")
            print(f"  Actual value: {belief_dist}")
            print(f"  Length: {len(str(belief_dist))}")

        return assessment

    except Exception as e:
        print(f"✗ Assessment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_agents():
    """Test multiple agents to see if issue is agent-specific."""
    print("\n" + "="*70)
    print("DIAGNOSTIC TEST: Multiple Agents")
    print("="*70)

    scenario = {
        "type": "flood",
        "severity": 0.75,
        "location": "coastal_city",
        "affected_population": 10000
    }

    alternatives = [
        {"id": "A1", "name": "Evacuate"},
        {"id": "A2", "name": "Deploy Barriers"},
        {"id": "A3", "name": "Shelter"}
    ]

    # Load agent profiles
    profiles_path = Path(__file__).parent / "agents" / "agent_profiles.json"
    with open(profiles_path, 'r') as f:
        profiles_data = json.load(f)

    # Test with first 3 agents
    test_agents_data = profiles_data['agents'][:3]

    try:
        llm_client = ClaudeClient()
    except Exception as e:
        print(f"✗ Cannot create LLM client: {e}")
        return

    print(f"\n📋 Testing {len(test_agents_data)} agents...\n")

    results = []
    for agent_data in test_agents_data:
        agent = ExpertAgent(
            agent_id=agent_data['agent_id'],
            name=agent_data['name'],
            role=agent_data['role'],
            expertise=agent_data['expertise'],
            llm_client=llm_client
        )

        print(f"Testing: {agent.name} ({agent.agent_id})...")
        try:
            assessment = agent.evaluate_scenario(scenario, alternatives)
            belief_dist = assessment.get('belief_distribution')

            is_dict = isinstance(belief_dist, dict)
            results.append({
                'agent_id': agent.agent_id,
                'agent_name': agent.name,
                'belief_dist_type': type(belief_dist).__name__,
                'is_valid': is_dict,
                'value_sample': str(belief_dist)[:100] if not is_dict else "Valid dict"
            })

            status = "✓" if is_dict else "✗"
            print(f"  {status} belief_distribution is {type(belief_dist).__name__}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                'agent_id': agent.agent_id,
                'agent_name': agent.name,
                'error': str(e)
            })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    valid_count = sum(1 for r in results if r.get('is_valid', False))
    invalid_count = sum(1 for r in results if 'is_valid' in r and not r['is_valid'])
    error_count = sum(1 for r in results if 'error' in r)

    print(f"Total agents tested: {len(results)}")
    print(f"Valid belief_distributions: {valid_count}")
    print(f"Invalid belief_distributions: {invalid_count}")
    print(f"Errors: {error_count}")

    if invalid_count > 0:
        print("\n⚠️  INVALID AGENTS:")
        for r in results:
            if 'is_valid' in r and not r['is_valid']:
                print(f"  - {r['agent_name']} ({r['agent_id']})")
                print(f"    Type: {r['belief_dist_type']}")
                print(f"    Sample: {r['value_sample']}")


def main():
    """Run all diagnostic tests."""
    print("🔍 Starting Belief Distribution Type Diagnostics\n")

    # Test 1: Single agent
    assessment = test_single_agent_assessment()

    # Test 2: Multiple agents (if first test passed)
    if assessment and isinstance(assessment.get('belief_distribution'), dict):
        print("\n✓ Single agent test passed, testing multiple agents...\n")
        test_multiple_agents()
    elif assessment:
        print("\n✗ Single agent test found issue, skipping multiple agent test")
    else:
        print("\n✗ Single agent test failed, cannot continue")

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"\n\nDiagnostic failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
