"""
Example: Using Prompt Templates with LLM Clients
Crisis Management Multi-Agent System

Demonstrates how to use prompt templates to get expert assessments
from Claude, OpenAI, or LM Studio.
"""

from llm_integration.prompt_templates import PromptTemplates
from llm_integration.claude_client import ClaudeClient
from llm_integration.openai_client import OpenAIClient
from llm_integration.lmstudio_client import LMStudioClient


def example_basic_usage():
    """Example 1: Basic prompt template usage."""
    print("="*70)
    print("Example 1: Basic Prompt Template Usage")
    print("="*70 + "\n")

    # Initialize templates
    templates = PromptTemplates()

    # Define crisis scenario
    scenario = {
        "type": "flood",
        "location": "Urban area with 50,000 residents",
        "severity": 0.85,
        "affected_population": 50000,
        "response_time_hours": 4,
        "weather_forecast": {
            "precipitation_mm": 200,
            "duration_hours": 48
        }
    }

    # Define alternatives
    alternatives = [
        {
            "id": "A1",
            "name": "Immediate Mass Evacuation",
            "safety_score": 0.90,
            "cost_euros": 480000,
            "response_time_hours": 4
        },
        {
            "id": "A2",
            "name": "Deploy Flood Barriers",
            "safety_score": 0.78,
            "cost_euros": 320000,
            "response_time_hours": 3
        }
    ]

    # Generate meteorologist prompt
    met_prompt = templates.generate_meteorologist_prompt(scenario, alternatives)

    print("Generated Meteorologist Prompt (first 500 chars):")
    print("-" * 70)
    print(met_prompt[:500] + "...")
    print("-" * 70)
    print(f"\nTotal length: {len(met_prompt)} characters")
    print(f"Ready to send to any LLM provider\n")


def example_with_claude():
    """Example 2: Using templates with Claude."""
    print("="*70)
    print("Example 2: Prompt Templates with Claude")
    print("="*70 + "\n")

    templates = PromptTemplates()

    scenario = {
        "type": "flood",
        "location": "Downtown area",
        "severity": 0.85,
        "affected_population": 50000
    }

    alternatives = [
        {"id": "A1", "name": "Evacuate", "cost_euros": 480000},
        {"id": "A2", "name": "Deploy Barriers", "cost_euros": 320000}
    ]

    # Generate prompt for meteorologist
    prompt = templates.generate_meteorologist_prompt(scenario, alternatives)
    system_prompt = templates.get_system_prompt("meteorologist")

    print("Using Claude API:")
    print(f"  System prompt: {system_prompt[:60]}...")
    print(f"  User prompt: {len(prompt)} characters")

    # Uncomment to make real API call:
    # client = ClaudeClient()
    # result = client.generate_assessment(
    #     prompt=prompt,
    #     system_prompt=system_prompt
    # )
    # if not result.get('error'):
    #     print(f"\n  Rankings: {result['alternative_rankings']}")
    #     print(f"  Confidence: {result['confidence']}")

    print("\n  (Uncomment code above to make real API call)")
    print("  ✓ Prompt ready for Claude\n")


def example_with_openai():
    """Example 3: Using templates with OpenAI."""
    print("="*70)
    print("Example 3: Prompt Templates with OpenAI")
    print("="*70 + "\n")

    templates = PromptTemplates()

    scenario = {
        "type": "flood",
        "location": "Residential area",
        "severity": 0.78,
        "affected_population": 30000,
        "available_resources": {
            "vehicles": 80,
            "personnel": 400,
            "budget_euros": 800000
        }
    }

    alternatives = [
        {"id": "A1", "name": "Full Evacuation", "cost_euros": 400000},
        {"id": "A2", "name": "Partial Evacuation", "cost_euros": 200000}
    ]

    # Generate prompt for operations director
    prompt = templates.generate_operations_prompt(scenario, alternatives)
    system_prompt = templates.get_system_prompt("operations")

    print("Using OpenAI API (GPT-4):")
    print(f"  System prompt: {system_prompt[:60]}...")
    print(f"  User prompt: {len(prompt)} characters")
    print(f"  Model: gpt-4-turbo-preview")

    # Uncomment to make real API call:
    # client = OpenAIClient()
    # result = client.generate_assessment(
    #     prompt=prompt,
    #     system_prompt=system_prompt
    # )
    # if not result.get('error'):
    #     print(f"\n  Rankings: {result['alternative_rankings']}")
    #     print(f"  Top concern: {result['key_concerns'][0]}")

    print("\n  (Uncomment code above to make real API call)")
    print("  ✓ Prompt ready for OpenAI\n")


def example_with_lmstudio():
    """Example 4: Using templates with LM Studio."""
    print("="*70)
    print("Example 4: Prompt Templates with LM Studio")
    print("="*70 + "\n")

    templates = PromptTemplates()

    scenario = {
        "type": "flood",
        "location": "Hospital district",
        "severity": 0.80,
        "affected_population": 20000,
        "description": "Area includes major hospital with 500 beds"
    }

    alternatives = [
        {"id": "A1", "name": "Evacuate Hospital", "safety_score": 0.90},
        {"id": "A2", "name": "Protect in Place", "safety_score": 0.70},
        {"id": "A3", "name": "Emergency Medical Teams", "safety_score": 0.82}
    ]

    # Generate prompt for medical expert
    prompt = templates.generate_medical_prompt(scenario, alternatives)
    system_prompt = templates.get_system_prompt("medical")

    print("Using LM Studio (Local LLM):")
    print(f"  Endpoint: http://localhost:1234/v1")
    print(f"  System prompt: {system_prompt[:60]}...")
    print(f"  User prompt: {len(prompt)} characters")

    # Uncomment to make real API call (requires LM Studio running):
    # client = LMStudioClient()
    # result = client.generate_assessment(
    #     prompt=prompt,
    #     system_prompt=system_prompt
    # )
    # if not result.get('error'):
    #     print(f"\n  Rankings: {result['alternative_rankings']}")
    #     print(f"  Reasoning: {result['reasoning'][:100]}...")

    print("\n  (Uncomment code above and start LM Studio to make real API call)")
    print("  ✓ Prompt ready for LM Studio\n")


def example_multi_expert_system():
    """Example 5: Multi-expert system using templates."""
    print("="*70)
    print("Example 5: Multi-Expert System with Templates")
    print("="*70 + "\n")

    templates = PromptTemplates()

    # Shared scenario
    scenario = {
        "type": "flood",
        "location": "Metropolitan area",
        "severity": 0.88,
        "affected_population": 75000,
        "response_time_hours": 3,
        "weather_forecast": {
            "precipitation_mm": 250,
            "duration_hours": 36
        },
        "available_resources": {
            "vehicles": 120,
            "personnel": 600,
            "budget_euros": 1200000
        }
    }

    alternatives = [
        {
            "id": "A1",
            "name": "Mass Evacuation",
            "safety_score": 0.92,
            "cost_euros": 500000,
            "response_time_hours": 4
        },
        {
            "id": "A2",
            "name": "Barriers + Selective Evacuation",
            "safety_score": 0.80,
            "cost_euros": 350000,
            "response_time_hours": 3
        },
        {
            "id": "A3",
            "name": "Shelter-in-Place + Monitoring",
            "safety_score": 0.65,
            "cost_euros": 180000,
            "response_time_hours": 2
        }
    ]

    print("Setting up 3-expert system:\n")

    # Expert 1: Meteorologist (Claude)
    print("Expert 1: Meteorologist")
    met_prompt = templates.generate_meteorologist_prompt(scenario, alternatives)
    met_system = templates.get_system_prompt("meteorologist")
    print(f"  Provider: Claude")
    print(f"  Prompt: {len(met_prompt)} chars")
    print(f"  Focus: Weather risks and safety\n")

    # Expert 2: Operations Director (OpenAI)
    print("Expert 2: Operations Director")
    ops_prompt = templates.generate_operations_prompt(scenario, alternatives)
    ops_system = templates.get_system_prompt("operations")
    print(f"  Provider: OpenAI GPT-4")
    print(f"  Prompt: {len(ops_prompt)} chars")
    print(f"  Focus: Resources and logistics\n")

    # Expert 3: Medical Expert (LM Studio)
    print("Expert 3: Medical Expert")
    med_prompt = templates.generate_medical_prompt(scenario, alternatives)
    med_system = templates.get_system_prompt("medical")
    print(f"  Provider: LM Studio (local)")
    print(f"  Prompt: {len(med_prompt)} chars")
    print(f"  Focus: Patient safety and health\n")

    print("Workflow:")
    print("  1. Send meteorologist prompt to Claude")
    print("  2. Send operations prompt to OpenAI")
    print("  3. Send medical prompt to LM Studio")
    print("  4. Aggregate responses using MCDA/Consensus")
    print("  5. Make final recommendation")

    print("\n✓ Multi-expert system configured\n")


def example_custom_criteria():
    """Example 6: Using custom evaluation criteria."""
    print("="*70)
    print("Example 6: Custom Evaluation Criteria")
    print("="*70 + "\n")

    templates = PromptTemplates()

    scenario = {
        "type": "flood",
        "location": "Industrial district",
        "severity": 0.82
    }

    alternatives = [
        {"id": "A1", "name": "Option 1"},
        {"id": "A2", "name": "Option 2"}
    ]

    # Custom criteria for environmental impact
    custom_criteria = [
        "environmental impact (pollution and contamination risks)",
        "infrastructure damage (critical systems protection)",
        "economic disruption (business continuity)",
        "long-term recovery (restoration timeline)"
    ]

    print("Using custom criteria for industrial flood scenario:\n")
    for i, criterion in enumerate(custom_criteria, 1):
        print(f"  {i}. {criterion}")

    prompt = templates.generate_meteorologist_prompt(
        scenario,
        alternatives,
        criteria=custom_criteria
    )

    print(f"\nGenerated prompt with custom criteria")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Contains custom criteria: {all(c in prompt for c in custom_criteria)}")

    print("\n✓ Custom criteria applied successfully\n")


def example_prompt_comparison():
    """Example 7: Compare prompts for different agents."""
    print("="*70)
    print("Example 7: Prompt Comparison Across Agents")
    print("="*70 + "\n")

    templates = PromptTemplates()

    scenario = {"type": "flood", "location": "City", "severity": 0.8}
    alternatives = [{"id": "A1", "name": "Option 1"}, {"id": "A2", "name": "Option 2"}]

    # Generate prompts for all agent types
    prompts = {
        "Meteorologist": templates.generate_meteorologist_prompt(scenario, alternatives),
        "Operations": templates.generate_operations_prompt(scenario, alternatives),
        "Medical": templates.generate_medical_prompt(scenario, alternatives)
    }

    print("Prompt Characteristics:\n")
    print(f"{'Agent Type':<20} {'Length':<10} {'Focus Keywords':<40}")
    print("-" * 70)

    focus_keywords = {
        "Meteorologist": ["Weather patterns", "precipitation", "safety"],
        "Operations": ["Resources", "logistics", "feasibility"],
        "Medical": ["patient safety", "vulnerable", "health"]
    }

    for agent_type, prompt in prompts.items():
        keywords = focus_keywords[agent_type]
        keyword_present = sum(1 for k in keywords if k.lower() in prompt.lower())
        print(f"{agent_type:<20} {len(prompt):<10} {keyword_present}/{len(keywords)} keywords present")

    print("\nKey Differences:")
    print("  • Meteorologist: Technical weather analysis perspective")
    print("  • Operations: Pragmatic resource management perspective")
    print("  • Medical: Healthcare and patient safety perspective")

    print("\n✓ Each agent has specialized focus\n")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PROMPT TEMPLATES USAGE EXAMPLES")
    print("Crisis Management Multi-Agent System")
    print("="*70 + "\n")

    examples = [
        example_basic_usage,
        example_with_claude,
        example_with_openai,
        example_with_lmstudio,
        example_multi_expert_system,
        example_custom_criteria,
        example_prompt_comparison
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
    print("\n1. Templates generate structured prompts for 3 expert types")
    print("2. Each template has specialized focus and evaluation criteria")
    print("3. Works with Claude, OpenAI, and LM Studio")
    print("4. Includes scenario formatting and alternatives formatting")
    print("5. Supports custom evaluation criteria")
    print("6. All prompts request structured JSON responses")
    print("7. Easy to integrate with multi-expert systems")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
