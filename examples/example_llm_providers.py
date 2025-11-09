#!/usr/bin/env python3
"""
Example Usage of Multiple LLM Providers
Crisis Management Multi-Agent System

This demonstrates how to use different LLM providers:
- Claude (Anthropic)
- OpenAI (GPT models)
- LM Studio (Local models)

Requirements:
    - For Claude: ANTHROPIC_API_KEY environment variable
    - For OpenAI: OPENAI_API_KEY environment variable
    - For LM Studio: Local LM Studio server running on http://localhost:1234
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_integration.claude_client import ClaudeClient


def check_api_keys():
    """Check which API keys are available."""
    print("="*80)
    print("API KEY STATUS CHECK")
    print("="*80 + "\n")

    claude_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')

    print(f"Claude (Anthropic): {'✓ Available' if claude_key else '✗ Not set'}")
    print(f"OpenAI:             {'✓ Available' if openai_key else '✗ Not set'}")
    print(f"LM Studio:          ℹ Requires local server at http://localhost:1234")
    print()

    return {
        'claude': bool(claude_key),
        'openai': bool(openai_key),
        'lm_studio': True  # Always available if server is running
    }


def example_claude():
    """Example using Claude API."""
    print("="*80)
    print("EXAMPLE 1: Claude (Anthropic)")
    print("="*80 + "\n")

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("⚠ ANTHROPIC_API_KEY not set. Skipping Claude example.")
        print("   To use: export ANTHROPIC_API_KEY='your-api-key-here'\n")
        return

    try:
        # Initialize Claude client
        client = ClaudeClient(api_key=api_key)
        print(f"✓ Claude client initialized")
        print(f"  Model: {client.model}")
        print(f"  Max tokens: {client.max_tokens}")
        print(f"  Temperature: {client.temperature}\n")

        # Create a simple assessment prompt
        prompt = """You are a crisis management expert evaluating response options.

SCENARIO: Urban flood risk requiring immediate decision

ALTERNATIVES:
- A1: Immediate Mass Evacuation
- A2: Deploy Barriers + Selective Evacuation
- A3: Shelter-in-Place with Monitoring

Provide a brief assessment (2-3 sentences) focusing on safety priorities."""

        print("📝 Sending assessment request to Claude...")

        # Note: Actual API call would go here
        # response = client.get_completion(prompt)

        print("\n✓ Example setup complete!")
        print("  (Actual API call commented out to avoid charges)")
        print("\n💡 To make actual API calls:")
        print("   1. Uncomment the response line above")
        print("   2. Add response parsing logic")
        print("   3. Run with valid API key\n")

    except Exception as e:
        print(f"✗ Error with Claude: {e}\n")


def example_openai():
    """Example using OpenAI API."""
    print("="*80)
    print("EXAMPLE 2: OpenAI (GPT Models)")
    print("="*80 + "\n")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠ OPENAI_API_KEY not set. Skipping OpenAI example.")
        print("   To use: export OPENAI_API_KEY='your-api-key-here'\n")
        return

    try:
        # Example OpenAI setup (pseudocode - requires openai package)
        print("✓ OpenAI client setup")
        print("  Model: gpt-4 or gpt-3.5-turbo")
        print("  Max tokens: 1024")
        print("  Temperature: 0.7\n")

        print("📝 Example OpenAI implementation:")
        print("""
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a crisis management expert."},
        {"role": "user", "content": "Assess flood response options..."}
    ],
    max_tokens=1024,
    temperature=0.7
)

assessment = response.choices[0].message.content
""")

        print("\n💡 To use OpenAI:")
        print("   1. Install: pip install openai")
        print("   2. Set API key: export OPENAI_API_KEY='sk-...'")
        print("   3. Implement the code above\n")

    except Exception as e:
        print(f"✗ Error with OpenAI: {e}\n")


def example_lm_studio():
    """Example using LM Studio (local models)."""
    print("="*80)
    print("EXAMPLE 3: LM Studio (Local Models)")
    print("="*80 + "\n")

    print("ℹ LM Studio allows running models locally with OpenAI-compatible API\n")

    print("📋 Setup Instructions:")
    print("   1. Download LM Studio from https://lmstudio.ai/")
    print("   2. Download a model (e.g., Mistral, Llama 2, Phi-2)")
    print("   3. Start local server (default: http://localhost:1234)")
    print("   4. Use OpenAI-compatible endpoint\n")

    print("📝 Example LM Studio implementation:")
    print("""
from openai import OpenAI

# Point to LM Studio local server
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # LM Studio doesn't require API key
)

response = client.chat.completions.create(
    model="local-model",  # Use model loaded in LM Studio
    messages=[
        {"role": "system", "content": "You are a crisis management expert."},
        {"role": "user", "content": "Assess flood response options..."}
    ],
    max_tokens=1024,
    temperature=0.7
)

assessment = response.choices[0].message.content
""")

    print("\n✅ Benefits of LM Studio:")
    print("   • No API costs - runs locally")
    print("   • Data privacy - no external calls")
    print("   • Offline capability")
    print("   • OpenAI-compatible API")
    print("   • Easy model switching\n")

    print("💡 Recommended models for Crisis MAS:")
    print("   • Mistral-7B (balanced performance)")
    print("   • Llama-2-13B (better reasoning)")
    print("   • Phi-2 (lightweight, fast)")
    print("   • OpenHermes-2.5 (instruction-following)\n")


def example_provider_comparison():
    """Compare different LLM providers."""
    print("="*80)
    print("PROVIDER COMPARISON")
    print("="*80 + "\n")

    print("┌────────────┬──────────┬───────────┬───────────┬────────────┐")
    print("│ Provider   │ Cost     │ Speed     │ Quality   │ Privacy    │")
    print("├────────────┼──────────┼───────────┼───────────┼────────────┤")
    print("│ Claude     │ $$       │ Fast      │ Excellent │ External   │")
    print("│ OpenAI     │ $$       │ Fast      │ Excellent │ External   │")
    print("│ LM Studio  │ Free     │ Medium    │ Good      │ Local      │")
    print("└────────────┴──────────┴───────────┴───────────┴────────────┘")
    print()

    print("📊 When to use each:")
    print()
    print("Claude (Anthropic):")
    print("  ✓ Production deployments needing best quality")
    print("  ✓ Complex reasoning and analysis tasks")
    print("  ✓ When cost is not primary concern")
    print("  ✗ Requires API key and internet")
    print()

    print("OpenAI (GPT-4/3.5):")
    print("  ✓ Production deployments with good ecosystem")
    print("  ✓ Wide range of model options (GPT-4, GPT-3.5)")
    print("  ✓ Mature API with extensive documentation")
    print("  ✗ Requires API key and internet")
    print()

    print("LM Studio (Local):")
    print("  ✓ Development and testing without API costs")
    print("  ✓ Privacy-sensitive scenarios (data stays local)")
    print("  ✓ Offline deployment requirements")
    print("  ✓ Educational/research use")
    print("  ✗ Requires local compute resources (GPU recommended)")
    print("  ✗ Generally lower quality than Claude/GPT-4")
    print()


def example_multi_provider_strategy():
    """Example of using multiple providers with fallback."""
    print("="*80)
    print("MULTI-PROVIDER STRATEGY WITH FALLBACK")
    print("="*80 + "\n")

    print("📋 Implementation Strategy:")
    print()
    print("1. Primary Provider (Best Quality)")
    print("   └─> Try Claude or GPT-4 first")
    print()
    print("2. Fallback Provider (Cost-Effective)")
    print("   └─> If primary fails, try GPT-3.5 or local model")
    print()
    print("3. Local Provider (Always Available)")
    print("   └─> LM Studio as last resort")
    print()

    print("📝 Example Implementation:")
    print("""
def get_llm_assessment(prompt, providers=['claude', 'openai', 'lm_studio']):
    '''Try multiple providers with fallback.'''

    for provider in providers:
        try:
            if provider == 'claude' and os.getenv('ANTHROPIC_API_KEY'):
                return get_claude_assessment(prompt)

            elif provider == 'openai' and os.getenv('OPENAI_API_KEY'):
                return get_openai_assessment(prompt)

            elif provider == 'lm_studio':
                # Always try local as last resort
                return get_lm_studio_assessment(prompt)

        except Exception as e:
            print(f"Provider {provider} failed: {e}")
            continue

    raise RuntimeError("All providers failed")

# Usage
assessment = get_llm_assessment(
    prompt="Assess crisis response options...",
    providers=['claude', 'openai', 'lm_studio']  # Priority order
)
""")
    print()

    print("✅ Benefits:")
    print("   • Reliability: Automatic fallback if primary fails")
    print("   • Cost optimization: Use cheaper providers when possible")
    print("   • Flexibility: Easy to add/remove providers")
    print("   • Offline capability: LM Studio as final fallback\n")


def example_crisis_scenario():
    """Full crisis scenario example with multiple providers."""
    print("="*80)
    print("COMPLETE CRISIS SCENARIO EXAMPLE")
    print("="*80 + "\n")

    print("🚨 SCENARIO: Urban Flood Crisis Decision")
    print()
    print("Context:")
    print("  • Location: Metro area, 50,000 residents")
    print("  • Threat: 200mm rainfall in 48h")
    print("  • Time: 4 hours to decide")
    print("  • Stakes: Lives, infrastructure, economy")
    print()

    print("🎯 Decision Alternatives:")
    print()
    print("A1: Immediate Mass Evacuation")
    print("    • Safety: Highest")
    print("    • Cost: $5M")
    print("    • Time: 4h response")
    print("    • Risk: Panic, traffic congestion")
    print()

    print("A2: Deploy Barriers + Selective Evacuation")
    print("    • Safety: Moderate-High")
    print("    • Cost: $2M")
    print("    • Time: 3h response")
    print("    • Risk: Barrier failure")
    print()

    print("A3: Shelter-in-Place + Monitoring")
    print("    • Safety: Moderate")
    print("    • Cost: $500K")
    print("    • Time: 1h response")
    print("    • Risk: Rapid deterioration")
    print()

    print("👥 Expert Perspectives Needed:")
    print("   • Meteorologist: Weather prediction")
    print("   • Structural Engineer: Infrastructure capacity")
    print("   • Emergency Manager: Response coordination")
    print("   • Economist: Cost-benefit analysis")
    print()

    print("🔄 LLM Provider Usage Strategy:")
    print()
    print("Phase 1 - Initial Analysis (Claude/GPT-4):")
    print("   • Use best model for critical initial assessment")
    print("   • Get detailed reasoning and risk analysis")
    print("   • Establish baseline expert opinions")
    print()

    print("Phase 2 - Alternative Perspectives (GPT-3.5/Local):")
    print("   • Generate diverse expert viewpoints")
    print("   • Cost-effective for multiple iterations")
    print("   • Capture range of professional opinions")
    print()

    print("Phase 3 - Consensus Building (Mixed):")
    print("   • Use Claude for final synthesis")
    print("   • Local models for sensitivity analysis")
    print("   • Validate consensus across providers")
    print()


def main():
    """Main example runner."""
    print("\n" + "="*80)
    print("MULTI-PROVIDER LLM EXAMPLES FOR CRISIS MAS")
    print("="*80 + "\n")

    # Check available providers
    available = check_api_keys()

    # Run examples
    if available['claude']:
        example_claude()

    if available['openai']:
        example_openai()

    example_lm_studio()
    example_provider_comparison()
    example_multi_provider_strategy()
    example_crisis_scenario()

    print("="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    print("✓ Demonstrated three LLM provider options")
    print("✓ Showed fallback strategy for reliability")
    print("✓ Explained when to use each provider")
    print("✓ Provided complete implementation examples")
    print()
    print("Next Steps:")
    print("  1. Choose provider(s) based on your needs")
    print("  2. Set up API keys or LM Studio")
    print("  3. Integrate into Crisis MAS agents")
    print("  4. Test with real crisis scenarios")
    print()
    print("For more details, see:")
    print("  • llm_integration/README.md - LLM integration docs")
    print("  • examples/example_multi_llm_providers.py - Original multi-provider example")
    print("  • examples/README.md - All examples overview")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
