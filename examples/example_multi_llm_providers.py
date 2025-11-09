"""
Multi-LLM Provider Example
Crisis Management Multi-Agent System

Demonstrates how to use Claude, OpenAI, and LM Studio interchangeably
for expert assessments in a crisis management scenario.
"""

from llm_integration.claude_client import ClaudeClient
from llm_integration.openai_client import OpenAIClient
from llm_integration.lmstudio_client import LMStudioClient


def create_assessment_prompt(expert_role: str, alternatives: dict) -> str:
    """
    Create a standardized assessment prompt for any LLM provider.

    Args:
        expert_role: Role of the expert (e.g., "meteorologist", "medical expert")
        alternatives: Dict of alternative IDs and descriptions

    Returns:
        Formatted prompt string
    """
    alternatives_text = "\n".join([
        f"- {alt_id}: {desc}" for alt_id, desc in alternatives.items()
    ])

    prompt = f"""You are a {expert_role} expert assessing flood response alternatives for an urban crisis.

SCENARIO:
- Location: Urban area with 50,000 residents
- Threat: Heavy rainfall forecast (200mm in 48 hours)
- Current flood risk: HIGH
- Available response time: 4 hours

ALTERNATIVES:
{alternatives_text}

Respond with a JSON object containing:
- alternative_rankings: dict with scores for each alternative (A1-A4), values should sum to approximately 1.0
- reasoning: your professional analysis as a {expert_role} (2-3 sentences)
- confidence: your confidence level as a number between 0.0 and 1.0
- key_concerns: list of 2-4 key concerns from your {expert_role} expertise domain

Example format:
{{
    "alternative_rankings": {{"A1": 0.7, "A2": 0.2, "A3": 0.08, "A4": 0.02}},
    "reasoning": "Based on {expert_role} analysis...",
    "confidence": 0.85,
    "key_concerns": ["concern1", "concern2"]
}}

Respond with ONLY the JSON object, no additional text."""

    return prompt


def example_provider_comparison():
    """Example 1: Compare assessments from different providers."""
    print("="*70)
    print("Example 1: Provider Comparison for Same Scenario")
    print("="*70 + "\n")

    # Define alternatives
    alternatives = {
        "A1": "Immediate Mass Evacuation (safest, most expensive, 4h response time)",
        "A2": "Deploy Flood Barriers + Selective Evacuation (moderate, 3h response time)",
        "A3": "Prioritized Rescue Operations (reactive approach, 2h response time)",
        "A4": "Shelter-in-Place with Monitoring (riskiest, cheapest, 1h response time)"
    }

    # Create prompt
    prompt = create_assessment_prompt("meteorologist", alternatives)

    print("Testing prompt with all three providers:\n")

    # Provider 1: Claude (demo - uncomment for real API call)
    print("1. Claude (Anthropic) - Simulated Response:")
    # client_claude = ClaudeClient()
    # result_claude = client_claude.generate_assessment(prompt)
    result_claude = {
        "alternative_rankings": {"A1": 0.65, "A2": 0.25, "A3": 0.08, "A4": 0.02},
        "reasoning": "Given the severe 200mm forecast and 4-hour window, immediate evacuation is safest. Historical data shows rapid flooding under these conditions.",
        "confidence": 0.87,
        "key_concerns": ["rapid water rise", "precipitation intensity", "drainage capacity"]
    }
    print(f"   Top choice: A1 (score: {result_claude['alternative_rankings']['A1']})")
    print(f"   Confidence: {result_claude['confidence']:.1%}")
    print(f"   Concerns: {', '.join(result_claude['key_concerns'][:2])}\n")

    # Provider 2: OpenAI (demo - uncomment for real API call)
    print("2. OpenAI (GPT-4) - Simulated Response:")
    # client_openai = OpenAIClient()
    # result_openai = client_openai.generate_assessment(prompt)
    result_openai = {
        "alternative_rankings": {"A1": 0.70, "A2": 0.20, "A3": 0.08, "A4": 0.02},
        "reasoning": "The 200mm rainfall prediction necessitates immediate evacuation. Flood barriers would be overwhelmed by this volume.",
        "confidence": 0.92,
        "key_concerns": ["extreme precipitation", "infrastructure limitations"]
    }
    print(f"   Top choice: A1 (score: {result_openai['alternative_rankings']['A1']})")
    print(f"   Confidence: {result_openai['confidence']:.1%}")
    print(f"   Concerns: {', '.join(result_openai['key_concerns'])}\n")

    # Provider 3: LM Studio (demo - uncomment for real API call)
    print("3. LM Studio (Local) - Simulated Response:")
    # client_lmstudio = LMStudioClient()
    # result_lmstudio = client_lmstudio.generate_assessment(prompt)
    result_lmstudio = {
        "alternative_rankings": {"A1": 0.60, "A2": 0.28, "A3": 0.10, "A4": 0.02},
        "reasoning": "Evacuation recommended given severe forecast, though barriers could supplement as secondary measure.",
        "confidence": 0.78,
        "key_concerns": ["weather severity", "time constraints", "population safety"]
    }
    print(f"   Top choice: A1 (score: {result_lmstudio['alternative_rankings']['A1']})")
    print(f"   Confidence: {result_lmstudio['confidence']:.1%}")
    print(f"   Concerns: {', '.join(result_lmstudio['key_concerns'][:2])}\n")

    # Consensus analysis
    print("Consensus Analysis:")
    print("  All providers agree: A1 (Evacuation) is the best option")
    print("  Average A1 score: 0.65")
    print("  High confidence across providers (78%-92%)")
    print("  ✓ Strong consensus achieved\n")


def example_provider_selection():
    """Example 2: Selecting the right provider for different scenarios."""
    print("="*70)
    print("Example 2: Provider Selection Strategy")
    print("="*70 + "\n")

    scenarios = [
        {
            "name": "Production Crisis Assessment",
            "requirements": ["highest accuracy", "reliable JSON", "fast response"],
            "recommended": "Claude or OpenAI GPT-4",
            "rationale": "Cloud APIs provide best accuracy and reliability for critical decisions"
        },
        {
            "name": "Development and Testing",
            "requirements": ["no API costs", "fast iteration", "offline access"],
            "recommended": "LM Studio",
            "rationale": "Local model allows unlimited testing without costs or internet"
        },
        {
            "name": "Sensitive Medical Data",
            "requirements": ["data privacy", "HIPAA compliance", "no external transmission"],
            "recommended": "LM Studio",
            "rationale": "All data stays local, meets strict privacy requirements"
        },
        {
            "name": "Multi-Expert Consensus (4+ agents)",
            "requirements": ["many API calls", "parallel requests", "cost efficiency"],
            "recommended": "LM Studio or OpenAI GPT-3.5",
            "rationale": "Lower cost per call, can handle high volume"
        }
    ]

    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"  Requirements: {', '.join(scenario['requirements'])}")
        print(f"  → Recommended: {scenario['recommended']}")
        print(f"  → Rationale: {scenario['rationale']}\n")


def example_unified_interface():
    """Example 3: Using unified interface for provider abstraction."""
    print("="*70)
    print("Example 3: Unified Interface Pattern")
    print("="*70 + "\n")

    print("All clients share the same interface:\n")

    # Initialize clients (with dummy keys for demo)
    clients = {
        "Claude": ClaudeClient(api_key="demo_key"),
        "OpenAI": OpenAIClient(api_key="demo_key"),
        "LM Studio": LMStudioClient()
    }

    print("Common Methods Across All Clients:\n")

    methods = [
        ("generate_assessment()", "Get structured JSON expert assessment"),
        ("parse_json_response()", "Extract JSON from various response formats"),
        ("validate_response()", "Validate response structure and data types"),
        ("get_statistics()", "Get usage metrics and success rate"),
        ("reset_statistics()", "Reset usage counters"),
        ("generate_response()", "Get general non-structured response")
    ]

    for method, description in methods:
        print(f"  • {method}")
        print(f"    {description}")

    print("\n\nExample: Swapping providers without code changes:\n")

    code_example = '''# Choose provider at runtime
provider = "claude"  # or "openai" or "lmstudio"

if provider == "claude":
    client = ClaudeClient()
elif provider == "openai":
    client = OpenAIClient()
else:
    client = LMStudioClient()

# Same code works for all providers
result = client.generate_assessment(prompt)
if not result.get('error'):
    rankings = result['alternative_rankings']
    top_choice = max(rankings.items(), key=lambda x: x[1])
    print(f"Top choice: {top_choice[0]} (score: {top_choice[1]})")'''

    print(code_example)
    print("\n✓ All clients follow the same interface pattern")


def example_multi_expert_system():
    """Example 4: Multi-expert system with different providers."""
    print("="*70)
    print("Example 4: Multi-Expert System with Mixed Providers")
    print("="*70 + "\n")

    print("Scenario: 3 expert agents using different LLM providers\n")

    # Define expert configurations
    experts = [
        {
            "role": "meteorologist",
            "provider": "Claude",
            "rationale": "Best for complex weather analysis"
        },
        {
            "role": "medical expert",
            "provider": "OpenAI GPT-4",
            "rationale": "Strong medical knowledge base"
        },
        {
            "role": "logistics coordinator",
            "provider": "LM Studio",
            "rationale": "Cost-effective for resource planning"
        }
    ]

    print("Expert Configuration:")
    for expert in experts:
        print(f"  • {expert['role'].title()}: {expert['provider']}")
        print(f"    → {expert['rationale']}\n")

    # Simulated assessments
    assessments = {
        "meteorologist": {
            "provider": "Claude",
            "rankings": {"A1": 0.65, "A2": 0.25, "A3": 0.08, "A4": 0.02},
            "confidence": 0.87
        },
        "medical_expert": {
            "provider": "OpenAI",
            "rankings": {"A1": 0.70, "A2": 0.20, "A3": 0.08, "A4": 0.02},
            "confidence": 0.92
        },
        "logistics_coordinator": {
            "provider": "LM Studio",
            "rankings": {"A1": 0.45, "A2": 0.40, "A3": 0.10, "A4": 0.05},
            "confidence": 0.75
        }
    }

    print("Expert Assessments:")
    for expert, assessment in assessments.items():
        top = max(assessment['rankings'].items(), key=lambda x: x[1])
        print(f"  {expert.replace('_', ' ').title()} ({assessment['provider']}):")
        print(f"    Top choice: {top[0]} (score: {top[1]:.2f})")
        print(f"    Confidence: {assessment['confidence']:.1%}\n")

    # Aggregate results
    print("Aggregated Decision:")
    avg_a1 = sum(a['rankings']['A1'] for a in assessments.values()) / 3
    avg_a2 = sum(a['rankings']['A2'] for a in assessments.values()) / 3
    print(f"  A1 (Evacuation) average score: {avg_a1:.2f}")
    print(f"  A2 (Barriers) average score: {avg_a2:.2f}")
    print(f"  → Final recommendation: A1 (Evacuation)")
    print(f"  ✓ Multi-provider consensus achieved\n")


def example_cost_analysis():
    """Example 5: Cost comparison across providers."""
    print("="*70)
    print("Example 5: Cost Analysis")
    print("="*70 + "\n")

    # Assume 10 expert assessments per crisis decision
    assessments_per_decision = 10
    decisions_per_month = 50  # Test/development scenario

    print(f"Scenario: {decisions_per_month} crisis decisions/month")
    print(f"({assessments_per_decision} expert assessments per decision)\n")

    # Estimated costs (approximate, as of 2025)
    claude_cost_per_1k = (3 + 15) / 2  # Average of input/output
    openai_gpt4_cost_per_1k = (30 + 60) / 2  # GPT-4
    openai_gpt35_cost_per_1k = (0.5 + 1.5) / 2  # GPT-3.5
    lmstudio_cost = 0  # Free (local)

    # Assume 2000 tokens per assessment
    tokens_per_assessment = 2000
    total_assessments = assessments_per_decision * decisions_per_month

    providers = [
        ("Claude Sonnet 4", claude_cost_per_1k, "Cloud (Anthropic)"),
        ("OpenAI GPT-4", openai_gpt4_cost_per_1k, "Cloud (OpenAI)"),
        ("OpenAI GPT-3.5", openai_gpt35_cost_per_1k, "Cloud (OpenAI)"),
        ("LM Studio (Local)", lmstudio_cost, "Local (Free)")
    ]

    print("Monthly Cost Comparison:\n")
    for name, cost_per_1k, location in providers:
        monthly_cost = (total_assessments * tokens_per_assessment / 1000) * cost_per_1k
        print(f"  {name} ({location}):")
        print(f"    Cost per 1K tokens: ${cost_per_1k:.2f}")
        print(f"    Monthly cost: ${monthly_cost:.2f}")
        if monthly_cost > 0:
            cost_per_decision = monthly_cost / decisions_per_month
            print(f"    Cost per decision: ${cost_per_decision:.2f}")
        print()

    print("Recommendations:")
    print("  • Production (high volume): LM Studio or GPT-3.5")
    print("  • Critical decisions: Claude or GPT-4")
    print("  • Development/testing: LM Studio")
    print("  • Hybrid approach: LM Studio for initial screening, Claude for final decision\n")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MULTI-LLM PROVIDER EXAMPLES")
    print("Crisis Management Multi-Agent System")
    print("="*70 + "\n")

    print("This demonstrates how to use Claude, OpenAI, and LM Studio")
    print("interchangeably in a crisis management system.\n")

    examples = [
        example_provider_comparison,
        example_provider_selection,
        example_unified_interface,
        example_multi_expert_system,
        example_cost_analysis
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Example Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. All providers use the same interface (generate_assessment)")
    print("2. Claude/OpenAI: Best accuracy, requires API key and internet")
    print("3. LM Studio: Free, local, private - ideal for dev/testing")
    print("4. Can mix providers in multi-expert systems")
    print("5. Provider choice depends on accuracy needs vs cost/privacy\n")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
