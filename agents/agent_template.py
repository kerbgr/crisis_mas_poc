"""
Agent Template
==============

This template demonstrates how to create a custom agent for the Crisis Management MAS.
Use this as a starting point for implementing new agent types with specific behaviors.

USAGE INSTRUCTIONS:
------------------
1. Copy this file to a new filename (e.g., 'my_custom_agent.py')
2. Rename the class (e.g., 'MyCustomAgent')
3. Implement the required abstract methods:
   - evaluate_scenario()
   - propose_action()
4. Add custom logic and methods as needed
5. Create agent profile in agents/agent_profiles.json
6. Test your agent with the system

REQUIRED METHODS:
-----------------
- evaluate_scenario(): Analyze a crisis scenario and return assessment
- propose_action(): Suggest an action based on scenario and criteria

OPTIONAL CUSTOMIZATION:
-----------------------
- Add domain-specific validation
- Implement custom reasoning logic
- Override confidence update mechanisms
- Add specialized analysis methods
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Import base agent class (required)
from agents.base_agent import BaseAgent

# Import LLM clients (optional - choose based on your needs)
from llm_integration.claude_client import ClaudeClient
from llm_integration.openai_client import OpenAIClient
from llm_integration.lmstudio_client import LMStudioClient
from llm_integration.prompt_templates import PromptTemplates


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomAgentTemplate(BaseAgent):
    """
    Template for creating custom crisis management agents.

    This template demonstrates:
    - How to inherit from BaseAgent
    - How to use LLM clients for reasoning
    - How to structure evaluate_scenario() and propose_action()
    - How to add custom agent-specific functionality

    IMPLEMENTATION STEPS:
    --------------------
    1. Rename this class to match your agent type (e.g., EconomicAgent, LegalAgent)
    2. Update docstring with agent-specific description
    3. Implement evaluate_scenario() - analyze crisis scenarios
    4. Implement propose_action() - suggest actions based on criteria
    5. Add any custom methods your agent needs

    AGENT PROFILE:
    -------------
    Before using this agent, create a profile in agents/agent_profiles.json:

    {
      "agent_id": "agent_custom_001",
      "name": "Custom Expert",
      "role": "Custom Specialist",
      "expertise": "custom_domain",
      "experience_years": 10,
      "risk_tolerance": 0.5,
      "weight_preferences": {
        "effectiveness": 0.30,
        "safety": 0.25,
        "speed": 0.20,
        "cost": 0.15,
        "public_acceptance": 0.10
      },
      "confidence_level": 0.85,
      "description": "Expert in custom domain analysis",
      "expertise_tags": ["custom", "specialist", "domain_expert"]
    }

    EXAMPLE USAGE:
    -------------
    >>> from llm_integration import ClaudeClient
    >>> llm_client = ClaudeClient()
    >>>
    >>> # Initialize agent
    >>> agent = CustomAgentTemplate(
    ...     agent_id="agent_custom_001",
    ...     llm_client=llm_client  # Optional
    ... )
    >>>
    >>> # Evaluate a scenario
    >>> scenario = {
    ...     'type': 'flood',
    ...     'severity': 0.85,
    ...     'location': 'Urban Area',
    ...     'affected_population': 50000,
    ...     'description': 'Rapid flooding threatening residential areas'
    ... }
    >>>
    >>> alternatives = [
    ...     {'id': 'A1', 'name': 'Immediate Evacuation', 'description': '...'},
    ...     {'id': 'A2', 'name': 'Deploy Barriers', 'description': '...'},
    ...     {'id': 'A3', 'name': 'Shelter in Place', 'description': '...'}
    ... ]
    >>>
    >>> assessment = agent.evaluate_scenario(scenario, alternatives)
    >>> print(f"Top recommendation: {assessment['recommended_alternative']}")
    >>> print(f"Confidence: {assessment['confidence']:.2%}")
    """

    def __init__(
        self,
        agent_id: str,
        llm_client: Optional[Union[ClaudeClient, OpenAIClient, LMStudioClient]] = None,
        profile_path: str = "agents/agent_profiles.json",
        use_llm: bool = True
    ):
        """
        Initialize custom agent.

        Args:
            agent_id: Unique identifier matching profile in agent_profiles.json
            llm_client: Optional LLM client for enhanced reasoning
                       (ClaudeClient, OpenAIClient, or LMStudioClient)
            profile_path: Path to agent profiles JSON file
            use_llm: Whether to use LLM for reasoning (requires llm_client)

        Raises:
            ValueError: If use_llm is True but llm_client is None
            FileNotFoundError: If profile file doesn't exist
            ValueError: If agent_id not found in profiles

        Example:
            >>> # With LLM (recommended for complex reasoning)
            >>> client = ClaudeClient(api_key="your-key")
            >>> agent = CustomAgentTemplate("agent_custom_001", llm_client=client)
            >>>
            >>> # Without LLM (rule-based only)
            >>> agent = CustomAgentTemplate("agent_custom_001", use_llm=False)
        """
        # Initialize base agent (loads profile from JSON)
        super().__init__(agent_id, profile_path)

        # Store LLM configuration
        self.use_llm = use_llm
        self.llm_client = llm_client

        # Validate LLM configuration
        if self.use_llm and self.llm_client is None:
            logger.warning(
                f"Agent '{agent_id}' configured to use LLM but no client provided. "
                f"Falling back to rule-based reasoning."
            )
            self.use_llm = False

        # Initialize prompt templates if using LLM
        if self.use_llm:
            self.prompt_templates = PromptTemplates()
            logger.info(
                f"{self.name} initialized with {type(self.llm_client).__name__}"
            )
        else:
            self.prompt_templates = None
            logger.info(f"{self.name} initialized in rule-based mode")

        # Custom agent state variables
        # Add any agent-specific state tracking here
        self.assessment_count = 0
        self.last_scenario_type = None
        self.domain_specific_data = {}

        logger.info(
            f"CustomAgentTemplate created: {self.name} ({self.role}, {self.expertise})"
        )

    def evaluate_scenario(
        self,
        scenario: Dict[str, Any],
        alternatives: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a crisis scenario and provide assessment.

        IMPLEMENTATION REQUIRED: This is an abstract method that must be implemented.

        This method should:
        1. Analyze the scenario using agent's expertise
        2. Evaluate each alternative (if provided)
        3. Generate belief distribution over alternatives
        4. Provide reasoning and confidence
        5. Return structured assessment

        Args:
            scenario: Dictionary containing crisis scenario data:
                - type (str): Crisis type (e.g., 'flood', 'earthquake')
                - severity (float): Severity level 0-1
                - location (str): Location description
                - affected_population (int): Number affected
                - description (str): Detailed description
                - additional domain-specific fields

            alternatives: Optional list of action alternatives to evaluate:
                Each alternative is a dict with:
                - id (str): Alternative identifier (e.g., 'A1', 'A2')
                - name (str): Alternative name
                - description (str): Detailed description
                - criteria_scores (dict): Optional pre-computed scores

            **kwargs: Additional parameters for custom behavior

        Returns:
            Dictionary containing assessment with structure:
            {
                'agent_id': str,              # Agent identifier
                'agent_name': str,            # Agent name
                'expertise': str,             # Agent expertise domain
                'scenario_type': str,         # Type of scenario evaluated

                # Main outputs
                'belief_distribution': {      # Probability distribution over alternatives
                    'A1': 0.65,              # Must sum to ~1.0
                    'A2': 0.25,
                    'A3': 0.10
                },
                'recommended_alternative': str,  # Top choice
                'confidence': float,          # Confidence level 0-1
                'reasoning': str,             # Explanation of assessment

                # Optional detailed analysis
                'criteria_scores': {          # Scores per criterion per alternative
                    'A1': {'safety': 0.9, 'cost': 0.4, ...},
                    'A2': {'safety': 0.7, 'cost': 0.8, ...}
                },
                'key_concerns': List[str],    # Major concerns identified
                'assumptions': List[str],     # Assumptions made
                'risks': List[str],           # Identified risks

                # Metadata
                'timestamp': str,             # ISO format timestamp
                'llm_used': bool,             # Whether LLM was used
                'llm_metadata': dict          # LLM response metadata (if used)
            }

        Raises:
            ValueError: If scenario is invalid or missing required fields
            RuntimeError: If evaluation fails

        Example Implementation:
            >>> def evaluate_scenario(self, scenario, alternatives=None, **kwargs):
            ...     # Validate inputs
            ...     if not scenario or 'type' not in scenario:
            ...         raise ValueError("Invalid scenario: missing 'type'")
            ...
            ...     # Use LLM for assessment if available
            ...     if self.use_llm and alternatives:
            ...         return self._evaluate_with_llm(scenario, alternatives)
            ...     else:
            ...         return self._evaluate_rule_based(scenario, alternatives)
        """
        # INCREMENT ASSESSMENT COUNTER
        self.assessment_count += 1
        self.last_scenario_type = scenario.get('type', 'unknown')

        logger.info(
            f"{self.name} evaluating scenario #{self.assessment_count}: "
            f"{self.last_scenario_type}"
        )

        # VALIDATE INPUTS
        if not scenario:
            raise ValueError("Scenario cannot be None or empty")

        if 'type' not in scenario:
            raise ValueError("Scenario must include 'type' field")

        # ROUTE TO APPROPRIATE EVALUATION METHOD
        if self.use_llm and self.llm_client and alternatives:
            # Use LLM-based reasoning
            assessment = self._evaluate_with_llm(scenario, alternatives)
        else:
            # Use rule-based reasoning
            assessment = self._evaluate_rule_based(scenario, alternatives)

        # LOG DECISION
        self.log_decision(assessment)

        return assessment

    def propose_action(
        self,
        scenario: Dict[str, Any],
        criteria: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Propose an action based on scenario and decision criteria.

        IMPLEMENTATION REQUIRED: This is an abstract method that must be implemented.

        This method should:
        1. Analyze scenario and criteria weights
        2. Generate or select appropriate action
        3. Justify the proposal based on agent expertise
        4. Provide implementation details

        Args:
            scenario: Crisis scenario data (same as evaluate_scenario)
            criteria: Decision criteria and weights:
                {
                    'effectiveness': 0.30,
                    'safety': 0.25,
                    'speed': 0.20,
                    'cost': 0.15,
                    'public_acceptance': 0.10
                }
            **kwargs: Additional parameters

        Returns:
            Dictionary containing proposed action:
            {
                'agent_id': str,
                'proposed_action': {
                    'name': str,              # Action name
                    'description': str,        # Detailed description
                    'type': str,              # Action type/category
                    'priority': str,          # 'high', 'medium', 'low'

                    'estimated_impact': {      # Expected outcomes
                        'effectiveness': float,
                        'safety': float,
                        'speed': float,
                        'cost': float,
                        'public_acceptance': float
                    },

                    'implementation_steps': List[str],  # Steps to execute
                    'required_resources': List[str],    # Needed resources
                    'timeline': str,                    # Expected duration
                    'risks': List[str],                 # Associated risks
                    'dependencies': List[str]           # Prerequisite conditions
                },

                'justification': str,         # Why this action
                'alternative_considered': List[str],  # Other options considered
                'confidence': float,          # 0-1
                'timestamp': str
            }

        Raises:
            ValueError: If scenario or criteria are invalid

        Example Implementation:
            >>> def propose_action(self, scenario, criteria, **kwargs):
            ...     action = {
            ...         'name': 'Immediate Evacuation',
            ...         'description': 'Evacuate affected population to safe zones',
            ...         'priority': 'high',
            ...         'estimated_impact': {
            ...             'effectiveness': 0.9,
            ...             'safety': 0.95,
            ...             'speed': 0.7
            ...         }
            ...     }
            ...     return {
            ...         'agent_id': self.agent_id,
            ...         'proposed_action': action,
            ...         'justification': 'Life safety is paramount...',
            ...         'confidence': self.confidence_level
            ...     }
        """
        logger.info(f"{self.name} proposing action for {scenario.get('type', 'unknown')}")

        # VALIDATE INPUTS
        if not scenario or not criteria:
            raise ValueError("Both scenario and criteria are required")

        # GENERATE ACTION PROPOSAL
        if self.use_llm and self.llm_client:
            proposal = self._propose_with_llm(scenario, criteria)
        else:
            proposal = self._propose_rule_based(scenario, criteria)

        # LOG DECISION
        self.log_decision(proposal)

        return proposal

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================
    # Implement these methods to support the main interface methods above

    def _evaluate_with_llm(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate scenario using LLM reasoning.

        This is a helper method that demonstrates how to:
        1. Generate appropriate prompts
        2. Call LLM API
        3. Parse and validate response
        4. Convert to standardized format

        You can customize this for your agent's specific needs.
        """
        try:
            # STEP 1: Generate role-specific prompt
            prompt = self._generate_evaluation_prompt(scenario, alternatives)
            system_prompt = self._generate_system_prompt()

            logger.debug(f"Generated prompt: {len(prompt)} characters")

            # STEP 2: Call LLM API
            llm_response = self.llm_client.generate_assessment(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.7
            )

            # STEP 3: Check for errors
            if llm_response.get('error'):
                logger.error(f"LLM call failed: {llm_response['error_message']}")
                # Fall back to rule-based
                return self._evaluate_rule_based(scenario, alternatives)

            # STEP 4: Parse response
            assessment = self._parse_llm_assessment(llm_response, scenario, alternatives)
            assessment['llm_used'] = True
            assessment['llm_metadata'] = llm_response.get('_metadata', {})

            return assessment

        except Exception as e:
            logger.error(f"LLM evaluation failed: {str(e)}")
            logger.info("Falling back to rule-based evaluation")
            return self._evaluate_rule_based(scenario, alternatives)

    def _evaluate_rule_based(
        self,
        scenario: Dict[str, Any],
        alternatives: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate scenario using rule-based logic (no LLM).

        CUSTOMIZE THIS for your agent's domain-specific logic.

        This example shows a simple heuristic approach:
        - Higher severity → prioritize safety and speed
        - Lower severity → balance all criteria
        - Agent's risk tolerance affects recommendations
        """
        severity = scenario.get('severity', 0.5)
        scenario_type = scenario.get('type', 'unknown')

        # Simple heuristic: create belief distribution
        if alternatives:
            # Distribute beliefs based on severity and risk tolerance
            belief_dist = {}
            for i, alt in enumerate(alternatives):
                alt_id = alt.get('id', f'A{i+1}')

                # Simple scoring logic (CUSTOMIZE THIS)
                if severity > 0.7 and self.risk_tolerance < 0.5:
                    # High severity + low risk tolerance → prefer safe options
                    score = 0.8 if i == 0 else 0.15 / (len(alternatives) - 1)
                else:
                    # Balanced distribution
                    score = 1.0 / len(alternatives)

                belief_dist[alt_id] = score

            # Normalize to sum to 1.0
            total = sum(belief_dist.values())
            belief_dist = {k: v/total for k, v in belief_dist.items()}

            recommended = max(belief_dist.items(), key=lambda x: x[1])[0]
        else:
            belief_dist = {}
            recommended = None

        # Construct assessment
        assessment = {
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'expertise': self.expertise,
            'scenario_type': scenario_type,

            'belief_distribution': belief_dist,
            'recommended_alternative': recommended,
            'confidence': self.confidence_level * 0.8,  # Lower confidence without LLM
            'reasoning': (
                f"Based on {self.expertise} expertise and scenario severity of {severity:.2f}, "
                f"I recommend a cautious approach considering risk tolerance of {self.risk_tolerance:.2f}."
            ),

            'key_concerns': self._identify_concerns(scenario),
            'risks': self._identify_risks(scenario),
            'assumptions': [
                "Standard operating procedures apply",
                "Resources are available as needed",
                "Communication channels are operational"
            ],

            'timestamp': datetime.now().isoformat(),
            'llm_used': False,
            'llm_metadata': None
        }

        return assessment

    def _propose_with_llm(
        self,
        scenario: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propose action using LLM reasoning.

        Similar pattern to _evaluate_with_llm but for action proposal.
        CUSTOMIZE the prompt generation for your needs.
        """
        try:
            # Generate prompt for action proposal
            prompt = self._generate_proposal_prompt(scenario, criteria)
            system_prompt = self._generate_system_prompt()

            # Call LLM
            llm_response = self.llm_client.generate_assessment(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.7
            )

            if llm_response.get('error'):
                return self._propose_rule_based(scenario, criteria)

            # Parse into action proposal format
            proposal = self._parse_llm_proposal(llm_response, scenario, criteria)
            return proposal

        except Exception as e:
            logger.error(f"LLM proposal failed: {str(e)}")
            return self._propose_rule_based(scenario, criteria)

    def _propose_rule_based(
        self,
        scenario: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propose action using rule-based logic.

        CUSTOMIZE THIS with domain-specific action generation logic.
        """
        scenario_type = scenario.get('type', 'unknown')
        severity = scenario.get('severity', 0.5)

        # Simple action proposal (CUSTOMIZE THIS)
        if severity > 0.7:
            action_name = f"Emergency {scenario_type.title()} Response"
            priority = 'high'
            timeline = '0-2 hours'
        else:
            action_name = f"Standard {scenario_type.title()} Response"
            priority = 'medium'
            timeline = '2-6 hours'

        proposal = {
            'agent_id': self.agent_id,
            'proposed_action': {
                'name': action_name,
                'description': f"Implement {self.expertise}-based response to {scenario_type}",
                'type': 'coordinated_response',
                'priority': priority,

                'estimated_impact': {
                    criterion: 0.6 + severity * 0.3  # Simple heuristic
                    for criterion in criteria.keys()
                },

                'implementation_steps': [
                    "Assess immediate threats",
                    "Mobilize resources",
                    "Execute response plan",
                    "Monitor and adjust"
                ],

                'required_resources': [
                    "Personnel",
                    "Equipment",
                    "Communication systems"
                ],

                'timeline': timeline,
                'risks': self._identify_risks(scenario),
                'dependencies': ["Clear communication", "Available resources"]
            },

            'justification': (
                f"As a {self.expertise} expert with {self.experience_years} years experience, "
                f"this action addresses the {severity:.0%} severity {scenario_type} effectively."
            ),
            'alternatives_considered': ["Wait and monitor", "Partial response"],
            'confidence': self.confidence_level * 0.8,
            'timestamp': datetime.now().isoformat()
        }

        return proposal

    # =========================================================================
    # CUSTOM HELPER METHODS
    # =========================================================================
    # Add domain-specific helper methods here

    def _generate_evaluation_prompt(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]]
    ) -> str:
        """
        Generate LLM prompt for scenario evaluation.

        CUSTOMIZE THIS to match your agent's expertise and communication style.
        """
        # Extract scenario details
        scenario_type = scenario.get('type', 'unknown')
        severity = scenario.get('severity', 'unknown')
        description = scenario.get('description', 'No description provided')

        # Format alternatives
        alt_text = "\n".join([
            f"- {alt.get('id', f'A{i+1}')}: {alt.get('name', 'Unnamed')} - "
            f"{alt.get('description', 'No description')}"
            for i, alt in enumerate(alternatives)
        ])

        # Build prompt
        prompt = f"""You are {self.name}, a {self.role} with {self.experience_years} years of experience in {self.expertise}.

SCENARIO:
Type: {scenario_type}
Severity: {severity}
Description: {description}

ALTERNATIVES TO EVALUATE:
{alt_text}

Based on your expertise in {self.expertise}, please evaluate these alternatives and provide:

1. A probability distribution over alternatives (must sum to 1.0)
2. Your reasoning as a {self.expertise} expert
3. Your confidence level (0.0 to 1.0)
4. Key concerns from your domain perspective
5. Any critical assumptions you're making

Respond with JSON format:
{{
    "alternative_rankings": {{"A1": 0.x, "A2": 0.y, ...}},
    "reasoning": "Your expert analysis...",
    "confidence": 0.x,
    "key_concerns": ["concern1", "concern2", ...]
}}"""

        return prompt

    def _generate_proposal_prompt(
        self,
        scenario: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> str:
        """Generate LLM prompt for action proposal."""
        # CUSTOMIZE THIS
        return f"""As {self.name} ({self.expertise}), propose an action for: {scenario.get('description', '')}

Consider these criteria: {', '.join(criteria.keys())}

Provide detailed action proposal with implementation steps."""

    def _generate_system_prompt(self) -> str:
        """Generate system prompt for LLM."""
        return (
            f"You are {self.name}, an expert {self.role} specializing in {self.expertise}. "
            f"You have {self.experience_years} years of experience and a risk tolerance of {self.risk_tolerance:.2f}. "
            f"Provide professional, expert-level analysis in JSON format."
        )

    def _parse_llm_assessment(
        self,
        llm_response: Dict[str, Any],
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse LLM response into standardized assessment format.

        CUSTOMIZE THIS to handle your LLM response structure.
        """
        # Extract key fields from LLM response
        belief_dist = llm_response.get('alternative_rankings', {})
        reasoning = llm_response.get('reasoning', 'No reasoning provided')
        confidence = llm_response.get('confidence', self.confidence_level)
        concerns = llm_response.get('key_concerns', [])

        # Find top recommendation
        if belief_dist:
            recommended = max(belief_dist.items(), key=lambda x: x[1])[0]
        else:
            recommended = None

        # Construct standardized assessment
        assessment = {
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'expertise': self.expertise,
            'scenario_type': scenario.get('type', 'unknown'),

            'belief_distribution': belief_dist,
            'recommended_alternative': recommended,
            'confidence': confidence,
            'reasoning': reasoning,

            'key_concerns': concerns,
            'risks': self._identify_risks(scenario),
            'assumptions': llm_response.get('assumptions', []),

            'timestamp': datetime.now().isoformat(),
            'llm_used': True
        }

        return assessment

    def _parse_llm_proposal(
        self,
        llm_response: Dict[str, Any],
        scenario: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse LLM response into action proposal format."""
        # CUSTOMIZE THIS based on your LLM response structure
        return {
            'agent_id': self.agent_id,
            'proposed_action': llm_response.get('proposed_action', {}),
            'justification': llm_response.get('reasoning', ''),
            'confidence': llm_response.get('confidence', self.confidence_level),
            'timestamp': datetime.now().isoformat()
        }

    def _identify_concerns(self, scenario: Dict[str, Any]) -> List[str]:
        """
        Identify concerns based on scenario.

        CUSTOMIZE THIS with domain-specific concern identification logic.
        """
        concerns = []

        severity = scenario.get('severity', 0)
        if severity > 0.7:
            concerns.append("High severity situation")

        if scenario.get('affected_population', 0) > 10000:
            concerns.append("Large affected population")

        # Add domain-specific concerns
        if self.expertise == 'medical' and severity > 0.5:
            concerns.append("Potential casualties")
        elif self.expertise == 'logistics' and scenario.get('type') == 'flood':
            concerns.append("Supply chain disruption")

        return concerns

    def _identify_risks(self, scenario: Dict[str, Any]) -> List[str]:
        """
        Identify risks based on scenario.

        CUSTOMIZE THIS with domain-specific risk identification logic.
        """
        risks = []

        if scenario.get('severity', 0) > 0.8:
            risks.append("Escalation risk")

        if scenario.get('type') in ['flood', 'earthquake']:
            risks.append("Infrastructure damage")

        # Add domain-specific risks
        risks.append(f"{self.expertise.title()}-related complications")

        return risks

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"CustomAgentTemplate("
            f"id='{self.agent_id}', "
            f"name='{self.name}', "
            f"expertise='{self.expertise}', "
            f"llm={'Yes' if self.use_llm else 'No'})"
        )


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Example of how to use the custom agent template.

    Run this file directly to see a demonstration:
        python agents/agent_template.py
    """

    print("="*70)
    print("CRISIS MAS - Custom Agent Template Demonstration")
    print("="*70)
    print()

    # Example 1: Create agent without LLM (rule-based)
    print("Example 1: Creating agent without LLM")
    print("-" * 70)
    try:
        agent_rule_based = CustomAgentTemplate(
            agent_id="agent_logistics_coord",  # Must exist in agent_profiles.json
            use_llm=False
        )
        print(f"✓ Created: {agent_rule_based}")
        print(f"  Expertise: {agent_rule_based.expertise}")
        print(f"  Risk Tolerance: {agent_rule_based.risk_tolerance:.2f}")
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Note: Ensure agent_id exists in agents/agent_profiles.json")
        print()

    # Example 2: Evaluate scenario (rule-based)
    print("Example 2: Evaluating scenario (rule-based)")
    print("-" * 70)

    test_scenario = {
        'type': 'flood',
        'severity': 0.85,
        'location': 'Urban Downtown Area',
        'affected_population': 50000,
        'description': 'Rapid flooding threatening residential and commercial areas'
    }

    test_alternatives = [
        {
            'id': 'A1',
            'name': 'Immediate Mass Evacuation',
            'description': 'Evacuate all affected areas immediately'
        },
        {
            'id': 'A2',
            'name': 'Deploy Flood Barriers',
            'description': 'Deploy temporary barriers and selective evacuation'
        },
        {
            'id': 'A3',
            'name': 'Shelter in Place',
            'description': 'Advise residents to shelter in upper floors'
        }
    ]

    try:
        # This will use rule-based evaluation since no LLM client
        assessment = agent_rule_based.evaluate_scenario(
            scenario=test_scenario,
            alternatives=test_alternatives
        )

        print(f"Scenario Type: {assessment['scenario_type']}")
        print(f"Recommended: {assessment['recommended_alternative']}")
        print(f"Confidence: {assessment['confidence']:.2%}")
        print(f"Reasoning: {assessment['reasoning'][:100]}...")
        print(f"\nBelief Distribution:")
        for alt_id, prob in assessment['belief_distribution'].items():
            print(f"  {alt_id}: {prob:.2%}")
        print()
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        print()

    # Example 3: With LLM (requires API key)
    print("Example 3: Creating agent with LLM")
    print("-" * 70)
    print("To use LLM-enhanced reasoning:")
    print()
    print("from llm_integration import ClaudeClient, OpenAIClient, LMStudioClient")
    print()
    print("# Option 1: Claude")
    print("llm_client = ClaudeClient(api_key='your-key')")
    print()
    print("# Option 2: OpenAI")
    print("llm_client = OpenAIClient(api_key='your-key')")
    print()
    print("# Option 3: LM Studio (local, free)")
    print("llm_client = LMStudioClient()")
    print()
    print("agent = CustomAgentTemplate(")
    print("    agent_id='agent_logistics_coord',")
    print("    llm_client=llm_client")
    print(")")
    print()

    print("="*70)
    print("Template demonstration complete!")
    print()
    print("NEXT STEPS:")
    print("1. Copy this file to create your custom agent")
    print("2. Implement domain-specific logic in the helper methods")
    print("3. Add your agent profile to agents/agent_profiles.json")
    print("4. Test with real scenarios")
    print("="*70)
