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

        prompt = f"""You are a SENIOR METEOROLOGIST providing a critical expert assessment for an active crisis response decision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR EXPERT ROLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are a senior meteorologist with 15+ years of experience in weather-related crisis management. Lives depend on the accuracy of your assessment. Your expertise includes:

• Advanced weather pattern analysis and atmospheric dynamics
• Severe weather forecasting (floods, storms, extreme precipitation)
• Risk assessment for weather-driven emergencies
• Historical event analysis and precedent evaluation
• Early warning system design and implementation
• Public safety impact prediction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ACTIVE CRISIS SITUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{scenario_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE OPTIONS UNDER CONSIDERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{alternatives_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{criteria_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR CRITICAL ASSESSMENT TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

As the meteorological expert on this crisis response team, evaluate each response alternative through the lens of weather science and public safety:

1. **Weather Threat Analysis**: How effectively does each option address the specific meteorological threats we're facing?

2. **Timing & Window of Action**: Given the weather event progression timeline, which alternatives align with our critical decision windows?

3. **Public Safety Impact**: From a meteorological perspective, which option provides the best protection against weather-related harm?

4. **Risk Assessment**: What are the meteorological risks or failure modes of each approach?

Time is critical. Decision-makers need your expert meteorological judgment NOW.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 REQUIRED RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide your expert meteorological assessment as a JSON object:

{{
    "alternative_rankings": {{
        "A1": 0.0,
        "A2": 0.0,
        "A3": 0.0,
        "A4": 0.0
    }},
    "reasoning": "Your professional meteorological analysis explaining your rankings. Be specific about weather threats, timing, and safety implications. 2-3 compelling sentences.",
    "confidence": 0.0,
    "key_concerns": [
        "Most critical weather-related concern",
        "Secondary meteorological risk factor",
        "Additional safety or timing consideration"
    ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ RESPONSE GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**alternative_rankings**: Assign scores 0.0-1.0 to each option based on meteorological merit. Higher scores indicate better alignment with weather safety and threat mitigation. Scores should sum to ~1.0.

**reasoning**: Write 2-3 concise sentences that convey the meteorological logic behind your rankings. Focus on specific weather threats, critical time windows, and safety outcomes. Decision-makers will use this to understand your expert perspective.

**confidence**: Rate your confidence 0.0-1.0 based on forecast certainty, data quality, and the clarity of weather patterns. Be honest—acknowledging uncertainty in crisis situations is professional and necessary.

**key_concerns**: List 2-4 specific meteorological factors that most influenced your assessment. Think: precipitation intensity, timing of peak impact, historical precedents, or vulnerable exposure periods.

⚠️ CRITICAL: Respond ONLY with the JSON object. No preamble, no explanation before or after. Your JSON assessment will be directly integrated into the crisis decision system."""

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

        prompt = f"""You are an OPERATIONS DIRECTOR providing a critical resource and logistics assessment for an active crisis response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR EXPERT ROLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are an experienced Operations Director with a proven track record of executing complex, large-scale crisis responses. Your decisions directly impact whether response plans succeed or fail on the ground. Your expertise includes:

• Strategic resource allocation under extreme time pressure
• Large-scale logistics coordination (personnel, vehicles, equipment)
• Budget management and cost-benefit analysis in emergencies
• Identifying operational bottlenecks before they become critical failures
• Multi-agency coordination and command structure optimization
• Real-world implementation feasibility assessment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ACTIVE CRISIS SITUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{scenario_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE OPTIONS UNDER CONSIDERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{alternatives_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{criteria_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR CRITICAL ASSESSMENT TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

As the operations expert on this crisis response team, evaluate each alternative through the hard lens of operational reality—what can actually be executed, with the resources we have, in the time available:

1. **Execution Feasibility**: Can we realistically pull this off with our current resources, personnel, and infrastructure? What's the implementation risk?

2. **Resource Efficiency**: Which option delivers the best outcome per euro spent and per resource deployed? Where do we get maximum impact?

3. **Logistical Complexity**: What are the coordination challenges, bottlenecks, and failure points? Which operations can we execute smoothly vs. which will strain our capabilities?

4. **Scalability & Coordination**: Can we scale this to the affected population size? How many moving parts need to work in sync?

The team needs your operational reality check. Which options are executable and which are logistical nightmares?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 REQUIRED RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide your expert operational assessment as a JSON object:

{{
    "alternative_rankings": {{
        "A1": 0.0,
        "A2": 0.0,
        "A3": 0.0,
        "A4": 0.0
    }},
    "reasoning": "Your operational reality check in 2-3 sentences. Address feasibility, resource constraints, and execution risks. Be direct about what's achievable vs. aspirational.",
    "confidence": 0.0,
    "key_concerns": [
        "Primary operational bottleneck or constraint",
        "Critical logistical challenge",
        "Resource availability or coordination risk"
    ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ RESPONSE GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**alternative_rankings**: Score each option 0.0-1.0 based on operational feasibility and resource efficiency. Higher scores = more executable with available resources. Think about what you can actually deliver on the ground. Scores should sum to ~1.0.

**reasoning**: Give decision-makers 2-3 sentences of operational truth. What's realistically achievable? What are the resource gaps? Which options play to our strengths vs. expose our weaknesses? Ground this in real operational constraints.

**confidence**: Rate your confidence 0.0-1.0 based on resource certainty, complexity of coordination required, and your operational experience with similar scenarios. If you're uncertain about resource availability, say so.

**key_concerns**: List 2-4 operational challenges that most influenced your assessment. Focus on: resource bottlenecks, coordination complexity, personnel limitations, equipment gaps, timeline feasibility, or budget constraints.

⚠️ CRITICAL: Respond ONLY with the JSON object. No preamble, no explanation before or after. Your operational assessment will be directly integrated into the crisis decision system."""

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

        prompt = f"""You are a SENIOR MEDICAL DIRECTOR providing a critical health impact assessment for an active crisis response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR EXPERT ROLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are a senior medical professional with extensive experience in emergency health management and crisis medicine. Your assessment will directly influence decisions that affect patient outcomes and population health. Your expertise includes:

• Emergency medical response planning and triage protocols
• Protecting vulnerable populations (elderly, disabled, chronically ill, pediatric)
• Hospital and healthcare facility surge capacity management
• Public health risk assessment during crises
• Disease prevention and contamination control
• Emergency medical services (EMS) coordination
• Health outcome prediction and mortality risk mitigation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ACTIVE CRISIS SITUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{scenario_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE OPTIONS UNDER CONSIDERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{alternatives_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{criteria_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR CRITICAL ASSESSMENT TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

As the medical expert on this crisis response team, evaluate each alternative through the lens of patient safety, health outcomes, and medical system capacity:

1. **Vulnerable Population Protection**: Which option provides the best protection for our most at-risk community members—elderly, chronically ill, disabled, children?

2. **Healthcare Access & Continuity**: How does each alternative affect people's ability to access critical medical care? What happens to ongoing treatments, dialysis, oxygen therapy, medications?

3. **Health Risk Assessment**: What are the direct and indirect health risks of each approach? Consider injury risk, disease transmission, contamination, mental health impacts, and cascade effects.

4. **Mortality & Morbidity Impact**: Which option minimizes preventable deaths and serious injuries? Where do we have the best chance of keeping people safe and healthy?

Lives are at stake. The team needs your medical expertise to evaluate which response options will result in the best health outcomes for the affected population.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 REQUIRED RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide your expert medical assessment as a JSON object:

{{
    "alternative_rankings": {{
        "A1": 0.0,
        "A2": 0.0,
        "A3": 0.0,
        "A4": 0.0
    }},
    "reasoning": "Your medical judgment in 2-3 sentences. Focus on health outcomes, patient safety, and vulnerable populations. Be clear about mortality/morbidity implications.",
    "confidence": 0.0,
    "key_concerns": [
        "Most critical patient safety or health risk",
        "Secondary health concern or vulnerable population issue",
        "Medical access or capacity challenge"
    ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ RESPONSE GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**alternative_rankings**: Score each option 0.0-1.0 based on health outcomes and patient safety. Higher scores = better protection of life and health. Consider both immediate risks and downstream health impacts. Scores should sum to ~1.0.

**reasoning**: Provide 2-3 sentences of medical perspective that decision-makers will rely on. Which option saves the most lives? Protects the vulnerable? Maintains healthcare access? Your clinical judgment matters—be direct about health trade-offs and mortality risks.

**confidence**: Rate your confidence 0.0-1.0 based on the medical evidence, clarity of health impacts, and your clinical experience. If health outcomes are uncertain, acknowledge it—we need honest medical assessment, not false certainty.

**key_concerns**: List 2-4 health factors that most influenced your rankings. Think: vulnerable populations at risk, medical access disruption, injury/mortality likelihood, disease transmission, chronic condition management, mental health impacts, or healthcare system strain.

⚠️ CRITICAL: Respond ONLY with the JSON object. No preamble, no explanation before or after. Your medical assessment will be directly integrated into the crisis decision system."""

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
                "You are a senior meteorologist with 15+ years of experience in weather-related "
                "crisis management. Lives depend on the accuracy of your weather assessments. "
                "You provide expert analysis of meteorological threats, safety implications, and "
                "critical time windows for crisis response decisions. Your role is to give "
                "decision-makers the weather science perspective they need to protect the public. "
                "Always respond with valid JSON format as specified in the prompt."
            ),
            "operations": (
                "You are an experienced Operations Director with a proven track record of "
                "executing complex crisis responses under pressure. Your operational reality "
                "checks prevent well-meaning plans from failing due to resource constraints or "
                "logistical impossibilities. You assess what can actually be delivered on the "
                "ground with available resources, personnel, and time. Your role is to ensure "
                "chosen responses are executable, not just aspirational. "
                "Always respond with valid JSON format as specified in the prompt."
            ),
            "medical": (
                "You are a senior medical professional with extensive experience in emergency "
                "health management and crisis medicine. Your clinical judgment directly influences "
                "decisions that affect patient outcomes, population health, and mortality rates. "
                "You evaluate health risks, assess impacts on vulnerable populations, and determine "
                "which response options will save the most lives and minimize suffering. Your role "
                "is to ensure the medical and public health perspective guides crisis decisions. "
                "Always respond with valid JSON format as specified in the prompt."
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
