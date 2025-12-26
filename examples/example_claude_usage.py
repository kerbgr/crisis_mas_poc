"""
Example Usage of Claude API Client
Crisis Management Multi-Agent System

This demonstrates how to use ClaudeClient for expert assessments.
NOTE: Requires valid ANTHROPIC_API_KEY to run actual API calls.
"""

from llm_integration.claude_client import ClaudeClient


def example_basic_assessment():
    """Example 1: Basic expert assessment."""
    print("="*70)
    print("Example 1: Basic Expert Assessment")
    print("="*70 + "\n")

    # Initialize client (using dummy key for demonstration)
    client = ClaudeClient(api_key="demo_api_key_replace_with_real_key")
    print(f"✓ Client initialized with model: {client.model}\n")

    # Create assessment prompt
    prompt = """You are a meteorologist expert assessing flood response alternatives for an urban crisis.

SCENARIO:
- Location: Urban area with 50,000 residents
- Threat: Heavy rainfall forecast (200mm in 48 hours)
- Current flood risk: HIGH
- Available response time: 4 hours

ALTERNATIVES:
- A1: Immediate Mass Evacuation (safest, most expensive, 4h response time)
- A2: Deploy Flood Barriers + Selective Evacuation (moderate safety, moderate cost, 3h response time)
- A3: Prioritized Rescue Operations (reactive approach, lower cost, 2h response time)
- A4: Shelter-in-Place with Monitoring (riskiest, cheapest, 1h response time)

Respond with a JSON object containing:
- alternative_rankings: dict with scores for each alternative (A1-A4), summing to 1.0
- reasoning: your professional analysis as a meteorologist (2-3 sentences)
- confidence: your confidence level (0.0 to 1.0)
- key_concerns: list of 2-4 key concerns from your expertise domain

Example format:
{
    "alternative_rankings": {"A1": 0.7, "A2": 0.2, "A3": 0.08, "A4": 0.02},
    "reasoning": "Based on weather patterns...",
    "confidence": 0.85,
    "key_concerns": ["flood timing", "precipitation intensity"]
}"""

    # Generate assessment (uncomment to make real API call)
    # result = client.generate_assessment(
    #     prompt=prompt,
    #     system_prompt="You are an expert meteorologist providing crisis management assessments.",
    #     max_tokens=2000,
    #     temperature=0.7
    # )

    # For this example, simulate a response
    print("Simulated API call (uncomment code above for real call):\n")
    simulated_response = {
        "alternative_rankings": {"A1": 0.65, "A2": 0.25, "A3": 0.08, "A4": 0.02},
        "reasoning": "Given the severe 200mm forecast and 4-hour window, immediate evacuation is safest. Historical data shows rapid flooding under these conditions. Barriers may be insufficient for this precipitation level.",
        "confidence": 0.87,
        "key_concerns": [
            "Rapid water rise within 4 hours",
            "Precipitation exceeds historical flood threshold",
            "Limited deployment time for infrastructure",
            "Urban drainage capacity insufficient"
        ]
    }

    # Display results
    print("Assessment Results:")
    print(f"  Top alternative: {max(simulated_response['alternative_rankings'].items(), key=lambda x: x[1])}")
    print(f"  Confidence: {simulated_response['confidence']:.1%}")
    print(f"  Reasoning: {simulated_response['reasoning'][:100]}...")
    print(f"  Key concerns: {len(simulated_response['key_concerns'])} identified\n")


def example_multi_expert_consensus():
    """Example 2: Multi-expert consensus building."""
    print("="*70)
    print("Example 2: Multi-Expert Consensus Building")
    print("="*70 + "\n")

    client = ClaudeClient(api_key="demo_api_key_replace_with_real_key")

    # Simulate assessments from different experts
    meteorologist_assessment = {
        "alternative_rankings": {"A1": 0.65, "A2": 0.25, "A3": 0.08, "A4": 0.02},
        "reasoning": "Weather patterns indicate severe flooding risk...",
        "confidence": 0.87,
        "key_concerns": ["rapid water rise", "precipitation intensity"]
    }

    medical_assessment = {
        "alternative_rankings": {"A1": 0.70, "A2": 0.20, "A3": 0.08, "A4": 0.02},
        "reasoning": "Patient safety requires immediate evacuation...",
        "confidence": 0.92,
        "key_concerns": ["hospital access", "vulnerable populations"]
    }

    logistics_assessment = {
        "alternative_rankings": {"A1": 0.45, "A2": 0.40, "A3": 0.10, "A4": 0.05},
        "reasoning": "Resource constraints suggest combined approach...",
        "confidence": 0.75,
        "key_concerns": ["vehicle availability", "route planning"]
    }

    print("Expert Assessments:")
    print(f"  Meteorologist: A1={meteorologist_assessment['alternative_rankings']['A1']:.2f} (confidence: {meteorologist_assessment['confidence']:.1%})")
    print(f"  Medical:       A1={medical_assessment['alternative_rankings']['A1']:.2f} (confidence: {medical_assessment['confidence']:.1%})")
    print(f"  Logistics:     A1={logistics_assessment['alternative_rankings']['A1']:.2f} (confidence: {logistics_assessment['confidence']:.1%})")

    # Calculate consensus
    all_a1_scores = [
        meteorologist_assessment['alternative_rankings']['A1'],
        medical_assessment['alternative_rankings']['A1'],
        logistics_assessment['alternative_rankings']['A1']
    ]
    avg_a1 = sum(all_a1_scores) / len(all_a1_scores)
    variance = sum((x - avg_a1)**2 for x in all_a1_scores) / len(all_a1_scores)

    print(f"\nConsensus Analysis:")
    print(f"  Average A1 score: {avg_a1:.2f}")
    print(f"  Variance: {variance:.4f}")
    print(f"  Status: {'HIGH CONSENSUS' if variance < 0.01 else 'MODERATE CONSENSUS' if variance < 0.05 else 'LOW CONSENSUS'}\n")


def example_error_handling():
    """Example 3: Error handling."""
    print("="*70)
    print("Example 3: Error Handling")
    print("="*70 + "\n")

    client = ClaudeClient(api_key="demo_api_key_replace_with_real_key")

    # Simulate different error scenarios
    print("Scenario 1: Handling API rate limit")
    print("  - Client will retry up to 3 times with exponential backoff")
    print("  - Delays: 2s, 4s, 8s\n")

    print("Scenario 2: Handling invalid JSON response")
    try:
        invalid_json = "This is not valid JSON at all"
        parsed = client.parse_json_response(invalid_json)
    except Exception as e:
        print(f"  - JSONDecodeError caught: {type(e).__name__}")
        print(f"  - Error would be logged and structured error returned\n")

    print("Scenario 3: Handling missing response fields")
    incomplete_response = {
        "alternative_rankings": {"A1": 0.7, "A2": 0.3},
        "reasoning": "Some reasoning",
        # Missing: confidence, key_concerns
    }
    expected_keys = ['alternative_rankings', 'reasoning', 'confidence', 'key_concerns']
    is_valid = client.validate_response(incomplete_response, expected_keys)
    print(f"  - Response validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  - Missing fields would be flagged in logs\n")


def example_usage_tracking():
    """Example 4: Usage tracking and statistics."""
    print("="*70)
    print("Example 4: Usage Tracking")
    print("="*70 + "\n")

    client = ClaudeClient(api_key="demo_api_key_replace_with_real_key")

    # Simulate some usage
    client.request_count = 15
    client.failed_requests = 2
    client.total_tokens = 25000

    stats = client.get_statistics()

    print("Usage Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Failed requests: {stats['failed_requests']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Model: {stats['model']}")

    # Cost estimation (approximate)
    # Claude Sonnet pricing (example): $3 per million input tokens, $15 per million output
    # Assuming 60/40 split input/output
    input_tokens = stats['total_tokens'] * 0.6
    output_tokens = stats['total_tokens'] * 0.4
    estimated_cost = (input_tokens / 1_000_000 * 3) + (output_tokens / 1_000_000 * 15)

    print(f"\nEstimated Cost:")
    print(f"  Input tokens (~60%): {input_tokens:,.0f}")
    print(f"  Output tokens (~40%): {output_tokens:,.0f}")
    print(f"  Estimated cost: ${estimated_cost:.4f}\n")


def example_prompt_engineering():
    """Example 5: Effective prompt engineering for structured responses."""
    print("="*70)
    print("Example 5: Prompt Engineering Best Practices")
    print("="*70 + "\n")

    print("Best Practices for Crisis Management Assessments:\n")

    print("1. SPECIFY JSON FORMAT EXPLICITLY")
    print("   - Always request JSON response format")
    print("   - Provide exact schema with field names")
    print("   - Include example output\n")

    print("2. PROVIDE COMPLETE CONTEXT")
    print("   - Scenario description (location, threat, timeline)")
    print("   - All alternatives with brief descriptions")
    print("   - Expert role and domain\n")

    print("3. REQUEST SPECIFIC FIELDS")
    print("   - alternative_rankings: numerical scores")
    print("   - reasoning: 2-3 sentence professional analysis")
    print("   - confidence: 0-1 scale")
    print("   - key_concerns: 2-4 domain-specific concerns\n")

    print("4. USE SYSTEM PROMPTS")
    print("   - Define expert role and context")
    print("   - Emphasize professional assessment")
    print("   - Request structured output format\n")

    print("5. HANDLE ERRORS GRACEFULLY")
    print("   - Validate all response fields")
    print("   - Check data types and ranges")
    print("   - Implement retry logic for transient errors\n")

    example_prompt = '''You are a {expert_role} expert assessing crisis alternatives.

SCENARIO:
{scenario_description}

ALTERNATIVES:
{alternatives_list}

Respond with JSON:
{{
    "alternative_rankings": {{"A1": 0.0, ...}},
    "reasoning": "Your analysis...",
    "confidence": 0.0,
    "key_concerns": ["concern1", ...]
}}'''

    print("Example Prompt Template:")
    print("-" * 70)
    print(example_prompt)
    print("-" * 70 + "\n")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("CLAUDE CLIENT - USAGE EXAMPLES")
    print("Crisis Management Multi-Agent System")
    print("="*70 + "\n")

    print("NOTE: To run actual API calls, set ANTHROPIC_API_KEY environment variable")
    print("      and uncomment the API call lines in the examples.\n")

    examples = [
        example_basic_assessment,
        example_multi_expert_consensus,
        example_error_handling,
        example_usage_tracking,
        example_prompt_engineering
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Example Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
