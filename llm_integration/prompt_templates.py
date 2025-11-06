"""
Prompt Templates for Crisis Management Expert Agents
Generates structured prompts for different expert agent roles

Designed to work with Claude, OpenAI, and LM Studio LLM providers.
"""

from typing import Dict, Any, List, Optional
import json


class PromptTemplates:
    """
    Generates structured prompts for crisis management expert agents.

    Provides specialized templates for different expert roles (Meteorologist,
    Operations Director, Medical Expert, etc.) with consistent formatting and
    clear JSON response instructions.

    Example:
        >>> templates = PromptTemplates()
        >>> scenario = {"type": "flood", "severity": 0.8, ...}
        >>> alternatives = [{"id": "A1", "name": "Evacuate", ...}]
        >>> prompt = templates.generate_meteorologist_prompt(scenario, alternatives)
    """

    def __init__(self):
        """Initialize prompt templates."""
        pass

    def generate_meteorologist_prompt(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None
    ) -> str:
        """
        Generate prompt for Meteorologist expert agent.

        Focus: Weather risks, safety, environmental factors, prevention
        Perspective: Technical meteorological analysis

        Args:
            scenario: Crisis scenario with weather/environmental data
            alternatives: List of response alternatives to evaluate
            criteria: Optional list of evaluation criteria (uses defaults if None)

        Returns:
            Formatted prompt string for meteorologist assessment

        Example:
            >>> templates = PromptTemplates()
            >>> scenario = {
            ...     "type": "flood",
            ...     "location": "Urban area",
            ...     "severity": 0.85,
            ...     "weather_forecast": {
            ...         "precipitation_mm": 200,
            ...         "duration_hours": 48
            ...     }
            ... }
            >>> prompt = templates.generate_meteorologist_prompt(scenario, alternatives)
        """
        if criteria is None:
            criteria = [
                "safety (public safety and risk to life)",
                "effectiveness (how well it addresses the weather threat)",
                "timing (response speed relative to weather timeline)",
                "preventability (ability to prevent weather-related damage)"
            ]

        scenario_context = self.format_scenario_context(scenario)
        alternatives_text = self.format_alternatives(alternatives)
        criteria_text = "\n".join([f"- {c}" for c in criteria])

        prompt = f"""You are a METEOROLOGIST expert providing a professional assessment for a crisis management decision.

EXPERT ROLE AND PERSPECTIVE:
You are a senior meteorologist with extensive experience in weather-related crisis management. Your analysis should focus on:
- Weather patterns and atmospheric conditions
- Precipitation, flooding, and severe weather risks
- Timeline of weather events and their progression
- Environmental factors affecting public safety
- Historical precedents and meteorological data
- Prevention and early warning capabilities

CRISIS SCENARIO:
{scenario_context}

AVAILABLE RESPONSE ALTERNATIVES:
{alternatives_text}

EVALUATION CRITERIA:
{criteria_text}

YOUR TASK:
Evaluate each alternative from a METEOROLOGIST's perspective. Consider:
1. How does each alternative address the weather-related threats?
2. What is the timing relative to weather event progression?
3. Which alternative best protects people from weather hazards?
4. What are the meteorological risks of each approach?

RESPONSE FORMAT:
Respond with a JSON object containing:

{{
    "alternative_rankings": {{
        "A1": 0.0,
        "A2": 0.0,
        "A3": 0.0,
        "A4": 0.0
    }},
    "reasoning": "Your meteorological analysis in 2-3 sentences explaining why you ranked alternatives this way, focusing on weather risks and safety factors.",
    "confidence": 0.0,
    "key_concerns": [
        "Primary weather-related concern",
        "Secondary meteorological risk",
        "Additional safety consideration"
    ]
}}

IMPORTANT INSTRUCTIONS:
1. alternative_rankings: Assign scores between 0 and 1 to each alternative. Higher score = better option from meteorological perspective. Scores should sum to approximately 1.0.
2. reasoning: Provide your professional meteorological analysis (2-3 sentences). Focus on weather patterns, safety risks, and timing.
3. confidence: Your confidence in this assessment (0.0 to 1.0). Consider data quality and forecast certainty.
4. key_concerns: List 2-4 specific meteorological or weather-safety concerns that influenced your assessment.

Respond ONLY with the JSON object. No additional text before or after."""

        return prompt

    def generate_operations_prompt(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None
    ) -> str:
        """
        Generate prompt for Operations Director expert agent.

        Focus: Resources, logistics, cost-effectiveness, feasibility
        Perspective: Pragmatic operational management

        Args:
            scenario: Crisis scenario with operational constraints
            alternatives: List of response alternatives to evaluate
            criteria: Optional list of evaluation criteria (uses defaults if None)

        Returns:
            Formatted prompt string for operations director assessment

        Example:
            >>> templates = PromptTemplates()
            >>> scenario = {
            ...     "type": "flood",
            ...     "affected_population": 50000,
            ...     "available_resources": {
            ...         "vehicles": 100,
            ...         "personnel": 500,
            ...         "budget_euros": 1000000
            ...     }
            ... }
            >>> prompt = templates.generate_operations_prompt(scenario, alternatives)
        """
        if criteria is None:
            criteria = [
                "feasibility (operational practicality and resource availability)",
                "cost-effectiveness (resource efficiency and budget impact)",
                "logistics (coordination complexity and execution challenges)",
                "scalability (ability to handle the affected population size)"
            ]

        scenario_context = self.format_scenario_context(scenario)
        alternatives_text = self.format_alternatives(alternatives)
        criteria_text = "\n".join([f"- {c}" for c in criteria])

        prompt = f"""You are an OPERATIONS DIRECTOR expert providing a professional assessment for a crisis management decision.

EXPERT ROLE AND PERSPECTIVE:
You are an experienced Operations Director with a track record of managing large-scale crisis responses. Your analysis should focus on:
- Resource availability and allocation
- Logistical feasibility and execution challenges
- Cost-effectiveness and budget constraints
- Personnel and equipment requirements
- Coordination complexity and bottlenecks
- Practical implementation timelines
- Scalability to the affected population

CRISIS SCENARIO:
{scenario_context}

AVAILABLE RESPONSE ALTERNATIVES:
{alternatives_text}

EVALUATION CRITERIA:
{criteria_text}

YOUR TASK:
Evaluate each alternative from an OPERATIONS DIRECTOR's perspective. Consider:
1. Can we realistically execute this with available resources?
2. What are the logistical challenges and bottlenecks?
3. Which alternative offers the best resource efficiency?
4. What are the operational risks and coordination requirements?

RESPONSE FORMAT:
Respond with a JSON object containing:

{{
    "alternative_rankings": {{
        "A1": 0.0,
        "A2": 0.0,
        "A3": 0.0,
        "A4": 0.0
    }},
    "reasoning": "Your operational analysis in 2-3 sentences explaining why you ranked alternatives this way, focusing on feasibility, resources, and logistics.",
    "confidence": 0.0,
    "key_concerns": [
        "Primary operational constraint",
        "Key logistical challenge",
        "Resource or coordination issue"
    ]
}}

IMPORTANT INSTRUCTIONS:
1. alternative_rankings: Assign scores between 0 and 1 to each alternative. Higher score = more operationally feasible and cost-effective. Scores should sum to approximately 1.0.
2. reasoning: Provide your professional operational analysis (2-3 sentences). Focus on resources, logistics, and feasibility.
3. confidence: Your confidence in this assessment (0.0 to 1.0). Consider resource certainty and operational experience.
4. key_concerns: List 2-4 specific operational constraints or logistical challenges that influenced your assessment.

Respond ONLY with the JSON object. No additional text before or after."""

        return prompt

    def generate_medical_prompt(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None
    ) -> str:
        """
        Generate prompt for Medical/Health expert agent.

        Focus: Public health, patient safety, medical infrastructure
        Perspective: Healthcare and medical emergency management

        Args:
            scenario: Crisis scenario with health-related impacts
            alternatives: List of response alternatives to evaluate
            criteria: Optional list of evaluation criteria

        Returns:
            Formatted prompt string for medical expert assessment
        """
        if criteria is None:
            criteria = [
                "patient safety (protection of vulnerable populations)",
                "medical access (maintaining healthcare services)",
                "health risks (disease, injury, and contamination)",
                "capacity (hospital and medical resource adequacy)"
            ]

        scenario_context = self.format_scenario_context(scenario)
        alternatives_text = self.format_alternatives(alternatives)
        criteria_text = "\n".join([f"- {c}" for c in criteria])

        prompt = f"""You are a MEDICAL/HEALTH expert providing a professional assessment for a crisis management decision.

EXPERT ROLE AND PERSPECTIVE:
You are a senior medical professional with experience in emergency health management. Your analysis should focus on:
- Public health and patient safety
- Vulnerable populations (elderly, disabled, chronically ill)
- Hospital and medical facility access
- Disease prevention and contamination risks
- Medical resource capacity and surge capability
- Emergency medical services coordination
- Health outcomes and mortality risk

CRISIS SCENARIO:
{scenario_context}

AVAILABLE RESPONSE ALTERNATIVES:
{alternatives_text}

EVALUATION CRITERIA:
{criteria_text}

YOUR TASK:
Evaluate each alternative from a MEDICAL/HEALTH perspective. Consider:
1. Which alternative best protects vulnerable populations?
2. How does each maintain access to medical care?
3. What are the health risks of each approach?
4. Which alternative minimizes injury and mortality?

RESPONSE FORMAT:
Respond with a JSON object containing:

{{
    "alternative_rankings": {{
        "A1": 0.0,
        "A2": 0.0,
        "A3": 0.0,
        "A4": 0.0
    }},
    "reasoning": "Your medical analysis in 2-3 sentences explaining why you ranked alternatives this way, focusing on patient safety and health outcomes.",
    "confidence": 0.0,
    "key_concerns": [
        "Primary patient safety concern",
        "Medical access or capacity issue",
        "Public health risk"
    ]
}}

IMPORTANT INSTRUCTIONS:
1. alternative_rankings: Assign scores between 0 and 1 to each alternative. Higher score = better health outcomes and patient safety. Scores should sum to approximately 1.0.
2. reasoning: Provide your professional medical analysis (2-3 sentences). Focus on health risks and patient safety.
3. confidence: Your confidence in this assessment (0.0 to 1.0).
4. key_concerns: List 2-4 specific health-related concerns that influenced your assessment.

Respond ONLY with the JSON object. No additional text before or after."""

        return prompt

    def format_scenario_context(self, scenario: Dict[str, Any]) -> str:
        """
        Format scenario information into readable text.

        Args:
            scenario: Scenario dictionary with crisis information

        Returns:
            Formatted scenario description string

        Example:
            >>> scenario = {
            ...     "type": "flood",
            ...     "location": "Urban area",
            ...     "severity": 0.85,
            ...     "affected_population": 50000
            ... }
            >>> context = templates.format_scenario_context(scenario)
        """
        lines = []

        # Type and location
        crisis_type = scenario.get('type', 'Unknown Crisis').title()
        location = scenario.get('location', 'Unknown Location')
        lines.append(f"Crisis Type: {crisis_type}")
        lines.append(f"Location: {location}")

        # Severity
        severity = scenario.get('severity', 0.5)
        severity_label = self._get_severity_label(severity)
        lines.append(f"Severity: {severity_label} ({severity:.2f})")

        # Affected population
        if 'affected_population' in scenario:
            pop = scenario['affected_population']
            lines.append(f"Affected Population: {pop:,} people")

        # Time constraints
        if 'response_time_hours' in scenario:
            time = scenario['response_time_hours']
            lines.append(f"Available Response Time: {time} hours")

        # Weather forecast (if present)
        if 'weather_forecast' in scenario:
            forecast = scenario['weather_forecast']
            lines.append("\nWeather Forecast:")
            if 'precipitation_mm' in forecast:
                lines.append(f"  - Precipitation: {forecast['precipitation_mm']}mm")
            if 'duration_hours' in forecast:
                lines.append(f"  - Duration: {forecast['duration_hours']} hours")
            if 'wind_speed_kmh' in forecast:
                lines.append(f"  - Wind Speed: {forecast['wind_speed_kmh']} km/h")

        # Available resources (if present)
        if 'available_resources' in scenario:
            resources = scenario['available_resources']
            lines.append("\nAvailable Resources:")
            if 'vehicles' in resources:
                lines.append(f"  - Vehicles: {resources['vehicles']}")
            if 'personnel' in resources:
                lines.append(f"  - Personnel: {resources['personnel']}")
            if 'budget_euros' in resources:
                budget = resources['budget_euros']
                lines.append(f"  - Budget: €{budget:,}")

        # Additional context
        if 'description' in scenario:
            lines.append(f"\nAdditional Context:")
            lines.append(f"{scenario['description']}")

        return "\n".join(lines)

    def format_alternatives(self, alternatives: List[Dict[str, Any]]) -> str:
        """
        Format alternatives into readable text.

        Args:
            alternatives: List of alternative action dictionaries

        Returns:
            Formatted alternatives description string

        Example:
            >>> alternatives = [
            ...     {"id": "A1", "name": "Evacuate", "safety_score": 0.9},
            ...     {"id": "A2", "name": "Deploy Barriers", "safety_score": 0.7}
            ... ]
            >>> text = templates.format_alternatives(alternatives)
        """
        lines = []

        for alt in alternatives:
            alt_id = alt.get('id', 'Unknown')
            name = alt.get('name', 'Unknown Alternative')

            # Start with ID and name
            line = f"{alt_id}: {name}"
            lines.append(line)

            # Add description if available
            if 'description' in alt:
                lines.append(f"    Description: {alt['description']}")

            # Add key metrics
            metrics = []
            if 'safety_score' in alt:
                metrics.append(f"Safety: {alt['safety_score']:.2f}")
            if 'cost_euros' in alt:
                metrics.append(f"Cost: €{alt['cost_euros']:,}")
            if 'response_time_hours' in alt:
                metrics.append(f"Response Time: {alt['response_time_hours']}h")
            if 'effectiveness' in alt:
                metrics.append(f"Effectiveness: {alt['effectiveness']:.2f}")

            if metrics:
                lines.append(f"    Metrics: {', '.join(metrics)}")

            # Add advantages/disadvantages if available
            if 'advantages' in alt and alt['advantages']:
                lines.append(f"    Advantages: {', '.join(alt['advantages'])}")
            if 'disadvantages' in alt and alt['disadvantages']:
                lines.append(f"    Disadvantages: {', '.join(alt['disadvantages'])}")

            lines.append("")  # Blank line between alternatives

        return "\n".join(lines)

    def _get_severity_label(self, severity: float) -> str:
        """
        Convert severity score to human-readable label.

        Args:
            severity: Severity score (0.0 to 1.0)

        Returns:
            Severity label string
        """
        if severity >= 0.8:
            return "CRITICAL"
        elif severity >= 0.6:
            return "HIGH"
        elif severity >= 0.4:
            return "MODERATE"
        elif severity >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"

    def get_system_prompt(self, agent_type: str) -> str:
        """
        Get system prompt for specific agent type.

        System prompts define the overall behavior and tone for the LLM.

        Args:
            agent_type: Type of agent ("meteorologist", "operations", "medical")

        Returns:
            System prompt string

        Example:
            >>> templates = PromptTemplates()
            >>> sys_prompt = templates.get_system_prompt("meteorologist")
        """
        system_prompts = {
            "meteorologist": (
                "You are a senior meteorologist expert with extensive experience in "
                "weather-related crisis management. You provide professional assessments "
                "focusing on weather risks, safety, and environmental factors. "
                "Always respond with valid JSON format as specified."
            ),
            "operations": (
                "You are an experienced Operations Director specializing in crisis "
                "management logistics and resource coordination. You provide pragmatic "
                "assessments focusing on feasibility, cost-effectiveness, and operational "
                "constraints. Always respond with valid JSON format as specified."
            ),
            "medical": (
                "You are a senior medical professional with expertise in emergency health "
                "management and public health crises. You provide assessments focusing on "
                "patient safety, vulnerable populations, and health outcomes. "
                "Always respond with valid JSON format as specified."
            )
        }

        return system_prompts.get(
            agent_type.lower(),
            "You are an expert providing structured assessments for crisis management. "
            "Always respond with valid JSON format as specified."
        )

    def __repr__(self) -> str:
        """String representation."""
        return "PromptTemplates(agents=['meteorologist', 'operations', 'medical'])"
