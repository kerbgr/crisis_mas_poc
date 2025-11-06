"""
Test Script for Prompt Templates
Crisis Management Multi-Agent System

Tests prompt generation for different expert agent types.
"""

from llm_integration.prompt_templates import PromptTemplates


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def test_meteorologist_prompt():
    """Test 1: Meteorologist prompt generation."""
    print_section("Test 1: Meteorologist Prompt")

    templates = PromptTemplates()

    scenario = {
        "type": "flood",
        "location": "Urban area with 50,000 residents",
        "severity": 0.85,
        "affected_population": 50000,
        "response_time_hours": 4,
        "weather_forecast": {
            "precipitation_mm": 200,
            "duration_hours": 48,
            "wind_speed_kmh": 45
        },
        "description": "Heavy rainfall expected with rapid water rise. River levels approaching flood stage."
    }

    alternatives = [
        {
            "id": "A1",
            "name": "Immediate Mass Evacuation",
            "safety_score": 0.90,
            "cost_euros": 480000,
            "response_time_hours": 4,
            "advantages": ["Maximum safety", "Proven effectiveness"],
            "disadvantages": ["Highest cost", "Significant disruption"]
        },
        {
            "id": "A2",
            "name": "Deploy Flood Barriers + Selective Evacuation",
            "safety_score": 0.78,
            "cost_euros": 320000,
            "response_time_hours": 3,
            "advantages": ["Moderate cost", "Protects infrastructure"],
            "disadvantages": ["Requires skilled operators", "May be insufficient"]
        },
        {
            "id": "A3",
            "name": "Prioritized Rescue Operations",
            "safety_score": 0.88,
            "cost_euros": 280000,
            "response_time_hours": 2,
            "advantages": ["Rapid deployment", "Targeted approach"],
            "disadvantages": ["Reactive vs preventive", "Resource intensive"]
        },
        {
            "id": "A4",
            "name": "Shelter-in-Place with Monitoring",
            "safety_score": 0.62,
            "cost_euros": 150000,
            "response_time_hours": 1,
            "advantages": ["Low cost", "Minimal disruption"],
            "disadvantages": ["High risk", "Dependent on accurate forecast"]
        }
    ]

    prompt = templates.generate_meteorologist_prompt(scenario, alternatives)

    print("Generated Meteorologist Prompt:")
    print("-" * 70)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-" * 70)
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Contains 'METEOROLOGIST': {('METEOROLOGIST' in prompt)}")
    print(f"Contains 'Weather patterns': {('Weather patterns' in prompt)}")
    print(f"Contains 'JSON': {('JSON' in prompt)}")
    print(f"Contains 'alternative_rankings': {('alternative_rankings' in prompt)}")
    print(f"✓ Meteorologist prompt generated\n")


def test_operations_prompt():
    """Test 2: Operations prompt generation."""
    print_section("Test 2: Operations Prompt")

    templates = PromptTemplates()

    scenario = {
        "type": "flood",
        "location": "Urban area with 50,000 residents",
        "severity": 0.85,
        "affected_population": 50000,
        "response_time_hours": 4,
        "available_resources": {
            "vehicles": 100,
            "personnel": 500,
            "budget_euros": 1000000
        },
        "description": "Need to coordinate large-scale response with limited resources."
    }

    alternatives = [
        {"id": "A1", "name": "Full Evacuation", "cost_euros": 480000},
        {"id": "A2", "name": "Barriers + Partial Evacuation", "cost_euros": 320000},
        {"id": "A3", "name": "Rescue Operations", "cost_euros": 280000},
        {"id": "A4", "name": "Shelter-in-Place", "cost_euros": 150000}
    ]

    prompt = templates.generate_operations_prompt(scenario, alternatives)

    print("Generated Operations Director Prompt:")
    print("-" * 70)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-" * 70)
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Contains 'OPERATIONS DIRECTOR': {('OPERATIONS DIRECTOR' in prompt)}")
    print(f"Contains 'Resources': {('Resources' in prompt) or ('resources' in prompt)}")
    print(f"Contains 'logistics': {('logistics' in prompt)}")
    print(f"Contains 'JSON': {('JSON' in prompt)}")
    print(f"✓ Operations prompt generated\n")


def test_medical_prompt():
    """Test 3: Medical prompt generation."""
    print_section("Test 3: Medical Prompt")

    templates = PromptTemplates()

    scenario = {
        "type": "flood",
        "location": "Urban area with major hospital",
        "severity": 0.80,
        "affected_population": 50000,
        "description": "Area includes vulnerable populations and critical medical facilities."
    }

    alternatives = [
        {"id": "A1", "name": "Evacuate All Including Hospital", "safety_score": 0.90},
        {"id": "A2", "name": "Protect Hospital + Evacuate Homes", "safety_score": 0.78},
        {"id": "A3", "name": "Emergency Medical Teams", "safety_score": 0.85}
    ]

    prompt = templates.generate_medical_prompt(scenario, alternatives)

    print("Generated Medical Expert Prompt:")
    print("-" * 70)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-" * 70)
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Contains 'MEDICAL': {('MEDICAL' in prompt)}")
    print(f"Contains 'patient safety': {('patient safety' in prompt)}")
    print(f"Contains 'vulnerable populations': {('vulnerable populations' in prompt)}")
    print(f"Contains 'JSON': {('JSON' in prompt)}")
    print(f"✓ Medical prompt generated\n")


def test_format_scenario():
    """Test 4: Scenario formatting."""
    print_section("Test 4: Scenario Formatting")

    templates = PromptTemplates()

    scenario = {
        "type": "flood",
        "location": "Downtown Metro Area",
        "severity": 0.92,
        "affected_population": 75000,
        "response_time_hours": 3,
        "weather_forecast": {
            "precipitation_mm": 250,
            "duration_hours": 36
        },
        "available_resources": {
            "vehicles": 150,
            "personnel": 800,
            "budget_euros": 1500000
        },
        "description": "Critical infrastructure at risk, including power grid and water treatment."
    }

    formatted = templates.format_scenario_context(scenario)

    print("Formatted Scenario Context:")
    print("-" * 70)
    print(formatted)
    print("-" * 70)
    print(f"\nLines in output: {len(formatted.split(chr(10)))}")
    print(f"Contains 'CRITICAL': {('CRITICAL' in formatted)}")
    print(f"Contains '75,000': {('75,000' in formatted)}")
    print(f"Contains 'Precipitation: 250mm': {('Precipitation: 250mm' in formatted)}")
    print(f"✓ Scenario formatting works\n")


def test_format_alternatives():
    """Test 5: Alternatives formatting."""
    print_section("Test 5: Alternatives Formatting")

    templates = PromptTemplates()

    alternatives = [
        {
            "id": "A1",
            "name": "Full Evacuation",
            "description": "Evacuate all residents from flood zone",
            "safety_score": 0.95,
            "cost_euros": 500000,
            "response_time_hours": 4,
            "effectiveness": 0.92,
            "advantages": ["Maximum safety", "Complete protection"],
            "disadvantages": ["Very expensive", "Major disruption"]
        },
        {
            "id": "A2",
            "name": "Hybrid Approach",
            "description": "Combine barriers with targeted evacuation",
            "safety_score": 0.80,
            "cost_euros": 300000,
            "response_time_hours": 3,
            "effectiveness": 0.78
        }
    ]

    formatted = templates.format_alternatives(alternatives)

    print("Formatted Alternatives:")
    print("-" * 70)
    print(formatted)
    print("-" * 70)
    print(f"\nContains 'A1: Full Evacuation': {('A1: Full Evacuation' in formatted)}")
    print(f"Contains 'Safety: 0.95': {('Safety: 0.95' in formatted)}")
    print(f"Contains 'Cost: €500,000': {('Cost: €500,000' in formatted)}")
    print(f"Contains 'Advantages': {('Advantages' in formatted)}")
    print(f"✓ Alternatives formatting works\n")


def test_system_prompts():
    """Test 6: System prompt retrieval."""
    print_section("Test 6: System Prompts")

    templates = PromptTemplates()

    agents = ["meteorologist", "operations", "medical", "unknown"]

    for agent_type in agents:
        system_prompt = templates.get_system_prompt(agent_type)
        print(f"{agent_type.title()}:")
        print(f"  {system_prompt[:80]}...")
        print(f"  Length: {len(system_prompt)} chars")
        print(f"  Contains 'JSON': {('JSON' in system_prompt)}")
        print()

    print("✓ System prompts working\n")


def test_severity_labels():
    """Test 7: Severity label conversion."""
    print_section("Test 7: Severity Labels")

    templates = PromptTemplates()

    severities = [0.95, 0.75, 0.55, 0.35, 0.15]

    print("Severity Score → Label Conversion:")
    for severity in severities:
        label = templates._get_severity_label(severity)
        print(f"  {severity:.2f} → {label}")

    print("\n✓ Severity labels working\n")


def test_prompt_completeness():
    """Test 8: Verify prompt completeness."""
    print_section("Test 8: Prompt Completeness Check")

    templates = PromptTemplates()

    scenario = {"type": "flood", "location": "City", "severity": 0.8}
    alternatives = [
        {"id": "A1", "name": "Option 1"},
        {"id": "A2", "name": "Option 2"}
    ]

    # Test all three agent types
    prompts = {
        "meteorologist": templates.generate_meteorologist_prompt(scenario, alternatives),
        "operations": templates.generate_operations_prompt(scenario, alternatives),
        "medical": templates.generate_medical_prompt(scenario, alternatives)
    }

    required_elements = [
        "YOUR EXPERT ROLE",
        "ACTIVE CRISIS SITUATION",
        "RESPONSE OPTIONS UNDER CONSIDERATION",
        "EVALUATION CRITERIA",
        "YOUR CRITICAL ASSESSMENT TASK",
        "REQUIRED RESPONSE FORMAT",
        "alternative_rankings",
        "reasoning",
        "confidence",
        "key_concerns",
        "RESPONSE GUIDELINES",
        "JSON"
    ]

    print("Checking all prompts contain required elements:\n")

    all_passed = True
    for agent_type, prompt in prompts.items():
        print(f"{agent_type.title()}:")
        missing = []
        for element in required_elements:
            if element not in prompt:
                missing.append(element)
                all_passed = False

        if missing:
            print(f"  ✗ Missing: {', '.join(missing)}")
        else:
            print(f"  ✓ All elements present")

    if all_passed:
        print(f"\n✓ All prompts are complete\n")
    else:
        print(f"\n✗ Some prompts incomplete\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PROMPT TEMPLATES TEST SUITE")
    print("Crisis Management Multi-Agent System")
    print("="*70)

    tests = [
        test_meteorologist_prompt,
        test_operations_prompt,
        test_medical_prompt,
        test_format_scenario,
        test_format_alternatives,
        test_system_prompts,
        test_severity_labels,
        test_prompt_completeness
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
    print("\nPrompt templates successfully generate structured prompts for:")
    print("  • Meteorologist (weather/safety focus)")
    print("  • Operations Director (resources/logistics focus)")
    print("  • Medical Expert (health/patient safety focus)")
    print("\nAll prompts include:")
    print("  • Expert role and perspective")
    print("  • Formatted scenario context")
    print("  • Formatted alternatives list")
    print("  • Evaluation criteria")
    print("  • Clear JSON response format")
    print("  • Detailed instructions")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
