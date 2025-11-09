"""
Expert Agent Class
Represents a domain expert (meteorologist, operations director, medical) using LLM reasoning

This agent leverages LLM capabilities to simulate expert decision-making in crisis scenarios.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from agents.base_agent import BaseAgent
from llm_integration.claude_client import ClaudeClient
from llm_integration.openai_client import OpenAIClient
from llm_integration.lmstudio_client import LMStudioClient
from llm_integration.prompt_templates import PromptTemplates


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertAgent(BaseAgent):
    """
    Expert agent that uses LLM reasoning to evaluate crisis scenarios.

    Represents a human expert (meteorologist, operations director, medical expert)
    by using LLM to generate assessments based on agent profile and expertise.

    Example:
        >>> from llm_integration.claude_client import ClaudeClient
        >>> client = ClaudeClient()
        >>> agent = ExpertAgent("agent_meteorologist", claude_client=client)
        >>> assessment = agent.evaluate_scenario(scenario, alternatives)
        >>> print(assessment['belief_distribution'])
        {'A1': 0.7, 'A2': 0.2, 'A3': 0.08, 'A4': 0.02}
    """

    def __init__(
        self,
        agent_id: str,
        llm_client: Union[ClaudeClient, OpenAIClient, LMStudioClient],
        profile_path: str = "agents/agent_profiles.json"
    ):
        """
        Initialize Expert Agent with LLM client.

        Args:
            agent_id: Unique identifier for the agent
            llm_client: LLM client instance (Claude, OpenAI, or LM Studio)
            profile_path: Path to agent profiles JSON file

        Raises:
            ValueError: If llm_client is None or invalid type
        """
        # Initialize base agent (loads profile)
        super().__init__(agent_id, profile_path)

        # Validate and store LLM client
        if llm_client is None:
            raise ValueError("llm_client cannot be None")

        valid_types = (ClaudeClient, OpenAIClient, LMStudioClient)
        if not isinstance(llm_client, valid_types):
            raise ValueError(
                f"llm_client must be instance of ClaudeClient, OpenAIClient, or LMStudioClient. "
                f"Got: {type(llm_client)}"
            )

        self.llm_client = llm_client
        self.prompt_templates = PromptTemplates()

        # Agent state
        self.last_assessment: Optional[Dict[str, Any]] = None
        self.assessment_count: int = 0

        logger.info(
            f"ExpertAgent initialized: {self.name} ({self.role}) "
            f"using {type(llm_client).__name__}"
        )

    def evaluate_scenario(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a crisis scenario and return structured assessment.

        This is the main method that:
        1. Generates role-specific prompt using PromptTemplates
        2. Calls LLM API with the prompt
        3. Validates and parses the response
        4. Converts to standardized belief distribution format
        5. Returns comprehensive assessment

        Args:
            scenario: Crisis scenario data (type, severity, location, etc.)
            alternatives: List of response alternatives to evaluate
            criteria: Optional custom evaluation criteria

        Returns:
            Dictionary containing:
            - agent_id: Agent identifier
            - belief_distribution: Alternative rankings as probability distribution
            - criteria_scores: Individual criterion scores per alternative
            - reasoning: Expert's reasoning for the assessment
            - confidence: Confidence level (0.0 to 1.0)
            - timestamp: ISO format timestamp
            - llm_metadata: LLM response metadata (tokens, model, etc.)

        Raises:
            ValueError: If scenario or alternatives are invalid
            RuntimeError: If LLM call fails after retries

        Example:
            >>> scenario = {"type": "flood", "severity": 0.85, "affected_population": 50000}
            >>> alternatives = [{"id": "A1", "name": "Evacuate"}, {"id": "A2", "name": "Barriers"}]
            >>> assessment = agent.evaluate_scenario(scenario, alternatives)
            >>> print(f"Recommended: {max(assessment['belief_distribution'].items(), key=lambda x: x[1])}")
        """
        logger.info(
            f"{self.name} evaluating scenario: {scenario.get('type', 'Unknown')} "
            f"(severity: {scenario.get('severity', 'N/A')})"
        )

        # Validate inputs
        self._validate_evaluation_inputs(scenario, alternatives)

        try:
            # Step 1: Generate role-specific prompt
            prompt, system_prompt = self._generate_prompt(scenario, alternatives, criteria)

            logger.debug(
                f"Generated prompt for {self.role}: {len(prompt)} chars, "
                f"system prompt: {len(system_prompt)} chars"
            )

            # Step 2: Call LLM API
            llm_response = self._call_llm(prompt, system_prompt)

            # Step 3: Check for errors
            if llm_response.get('error'):
                error_msg = llm_response.get('error_message', 'Unknown error')
                logger.error(f"LLM call failed: {error_msg}")
                raise RuntimeError(f"LLM assessment failed: {error_msg}")

            # Step 4: Validate response structure
            self._validate_llm_response(llm_response)

            # Step 5: Generate belief distribution
            belief_distribution = self.generate_belief_distribution(llm_response)

            # Step 6: Generate criteria scores (if weight preferences defined)
            criteria_scores = self.get_criteria_scores(alternatives, llm_response)

            # Step 7: Build comprehensive assessment
            assessment = self._build_assessment(
                belief_distribution=belief_distribution,
                criteria_scores=criteria_scores,
                llm_response=llm_response,
                scenario=scenario
            )

            # Update agent state
            self.last_assessment = assessment
            self.assessment_count += 1
            self.decision_history.append({
                'timestamp': assessment['timestamp'],
                'scenario_type': scenario.get('type'),
                'top_choice': max(belief_distribution.items(), key=lambda x: x[1])[0],
                'confidence': assessment['confidence']
            })

            logger.info(
                f"{self.name} assessment complete. "
                f"Top choice: {max(belief_distribution.items(), key=lambda x: x[1])[0]} "
                f"(confidence: {assessment['confidence']:.2f})"
            )

            return assessment

        except Exception as e:
            logger.error(f"Error during scenario evaluation: {str(e)}", exc_info=True)
            raise

    def generate_belief_distribution(
        self,
        llm_response: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Convert LLM alternative rankings to normalized belief distribution.

        Takes the raw alternative_rankings from LLM and normalizes them to
        ensure they sum to 1.0, creating a proper probability distribution.

        Args:
            llm_response: Response from LLM containing alternative_rankings

        Returns:
            Dictionary mapping alternative IDs to belief values (summing to 1.0)

        Example:
            >>> llm_response = {"alternative_rankings": {"A1": 0.7, "A2": 0.25, "A3": 0.05}}
            >>> belief_dist = agent.generate_belief_distribution(llm_response)
            >>> print(belief_dist)
            {'A1': 0.7, 'A2': 0.25, 'A3': 0.05}
            >>> print(sum(belief_dist.values()))
            1.0
        """
        rankings = llm_response.get('alternative_rankings', {})

        if not rankings:
            logger.warning("No alternative_rankings in LLM response, returning empty distribution")
            return {}

        # Calculate sum for normalization
        total = sum(rankings.values())

        if total == 0:
            logger.warning("Sum of rankings is zero, using uniform distribution")
            n = len(rankings)
            return {alt_id: 1.0 / n for alt_id in rankings.keys()}

        # Normalize to sum to 1.0
        belief_distribution = {
            alt_id: score / total
            for alt_id, score in rankings.items()
        }

        logger.debug(
            f"Generated belief distribution: "
            f"{list(belief_distribution.items())} (sum: {sum(belief_distribution.values()):.4f})"
        )

        return belief_distribution

    def get_criteria_scores(
        self,
        alternatives: List[Dict[str, Any]],
        llm_response: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract or generate criteria scores for each alternative.

        If the agent has weight_preferences defined, creates criteria scores
        based on alternative metrics and rankings. Otherwise returns empty dict.

        Args:
            alternatives: List of alternative dictionaries
            llm_response: LLM response with rankings

        Returns:
            Dictionary mapping criteria IDs to alternative scores:
            {
                "C1_safety": {"A1": 0.95, "A2": 0.80, "A3": 0.65},
                "C2_cost": {"A1": 0.60, "A2": 0.85, "A3": 0.90}
            }

        Example:
            >>> alternatives = [
            ...     {"id": "A1", "safety_score": 0.9, "cost_euros": 500000},
            ...     {"id": "A2", "safety_score": 0.8, "cost_euros": 300000}
            ... ]
            >>> scores = agent.get_criteria_scores(alternatives, llm_response)
        """
        criteria_scores = {}

        # If no weight preferences, we can't generate meaningful criteria scores
        if not self.weight_preferences:
            logger.debug("No weight preferences defined, returning empty criteria scores")
            return criteria_scores

        # Get rankings for reference
        rankings = llm_response.get('alternative_rankings', {})

        # For each criterion in weight preferences
        for criterion_name in self.weight_preferences.keys():
            criterion_id = f"C_{criterion_name}"
            criterion_scores = {}

            # Generate scores for each alternative
            for alt in alternatives:
                alt_id = alt.get('id', alt.get('name', 'unknown'))

                # Try to find metric matching criterion name
                score = None

                # PRIORITY 1: Check for criteria_scores object (from scenario JSON)
                if 'criteria_scores' in alt and criterion_name in alt['criteria_scores']:
                    score = alt['criteria_scores'][criterion_name]
                    logger.debug(f"Found score for {alt_id} on {criterion_name}: {score}")

                # PRIORITY 2: Check for direct metric (e.g., "safety" -> "safety_score")
                elif f"{criterion_name}_score" in alt:
                    metric_key = f"{criterion_name}_score"
                    score = alt[metric_key]

                # Check for cost (inverted - lower is better)
                elif criterion_name == "cost" and "cost_euros" in alt:
                    # Normalize cost to 0-1 scale (lower cost = higher score)
                    all_costs = [a.get('cost_euros', 0) for a in alternatives]
                    max_cost = max(all_costs) if all_costs else 1
                    min_cost = min(all_costs) if all_costs else 0
                    if max_cost > min_cost:
                        # Invert: lower cost = higher score
                        score = 1.0 - (alt['cost_euros'] - min_cost) / (max_cost - min_cost)
                    else:
                        score = 0.5

                # Fall back to using ranking as proxy for all criteria
                if score is None and alt_id in rankings:
                    score = rankings[alt_id]

                # Default if no score found
                if score is None:
                    score = 0.5
                    logger.warning(
                        f"No score found for {alt_id} on {criterion_name}, using default 0.5"
                    )

                criterion_scores[alt_id] = float(score)

            criteria_scores[criterion_id] = criterion_scores

        logger.debug(f"Generated {len(criteria_scores)} criterion score sets")

        return criteria_scores

    def explain_assessment(self) -> str:
        """
        Generate human-readable explanation of the last assessment.

        Returns:
            Formatted string explaining the assessment, or message if no assessment yet

        Example:
            >>> explanation = agent.explain_assessment()
            >>> print(explanation)
            Agent: Dr. Sarah Chen (Meteorologist)
            Assessment confidence: 87.0%
            Recommended action: A1 (score: 0.65)
            Reasoning: Given the severe 200mm forecast...
        """
        if not self.last_assessment:
            return f"Agent {self.name} has not made any assessments yet."

        assessment = self.last_assessment
        belief_dist = assessment['belief_distribution']

        # Find top choice
        top_choice = max(belief_dist.items(), key=lambda x: x[1])

        explanation = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERT ASSESSMENT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent: {self.name} ({self.role})
Expertise: {self.expertise}
Experience: {self.experience_years} years
Assessment Time: {assessment['timestamp']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Top Choice: {top_choice[0]} (belief: {top_choice[1]:.3f})
Confidence Level: {assessment['confidence']:.1%}

Alternative Rankings:
"""
        # Sort alternatives by score
        for alt_id, score in sorted(belief_dist.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(score * 40)
            bar = "█" * bar_length
            explanation += f"  {alt_id}: {score:.3f} {bar}\n"

        explanation += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERT REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{assessment['reasoning']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY CONCERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for i, concern in enumerate(assessment['key_concerns'], 1):
            explanation += f"  {i}. {concern}\n"

        return explanation

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helper Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_prompt(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        criteria: Optional[List[str]]
    ) -> tuple[str, str]:
        """
        Generate domain-specific prompts tailored to agent's expertise and role.

        OBJECTIVE:
        Creates specialized LLM prompts that leverage agent's expertise by using
        domain-specific language, evaluation criteria, and reasoning patterns.
        Ensures LLM acts as the appropriate expert (medical, logistics, meteorologist, etc.).

        WHY THIS EXISTS:
        Generic prompts produce generic responses. By mapping agent roles to specialized
        prompt templates, we get more nuanced, expert-level assessments that consider
        domain-specific factors. For example:
        - Medical expert focuses on casualties, triage, medical resources
        - Logistics expert focuses on supply chains, transportation, resource allocation
        - Meteorologist focuses on weather patterns, forecasting, environmental conditions

        INPUTS:
        - scenario: Crisis scenario containing type, severity, context
        - alternatives: Response alternatives to evaluate
        - criteria: Optional custom evaluation criteria (if None, uses role defaults)

        OUTPUTS:
        Tuple of (user_prompt, system_prompt):
        - user_prompt: Main prompt with scenario details and alternatives
        - system_prompt: System message defining agent role and behavior

        ROLE MAPPING LOGIC:
        Matches agent expertise/role keywords to prompt templates:
        - "meteorolog" or "weather" → meteorologist_prompt
        - "logistic", "operation", "supply_chain" → operations_prompt
        - "medical" or "health" → medical_prompt
        - Default: operations_prompt (with warning logged)

        PROMPT STRUCTURE:
        Each prompt template includes:
        - Role definition and expertise area
        - Scenario context and severity
        - List of alternatives with details
        - Evaluation criteria (domain-specific)
        - Output format requirements (JSON structure)
        - Examples of reasoning patterns

        EXAMPLE:
        Medical expert for earthquake scenario:
        - system_prompt: "You are an emergency medicine specialist..."
        - user_prompt: "Evaluate earthquake response focusing on casualties,
                       medical resources, triage procedures..."
        """
        # Map agent expertise/role to prompt generation method
        expertise_lower = self.expertise.lower()
        role_lower = self.role.lower()

        # Determine which template to use
        if 'meteorolog' in expertise_lower or 'meteorolog' in role_lower or 'weather' in expertise_lower:
            prompt = self.prompt_templates.generate_meteorologist_prompt(
                scenario, alternatives, criteria
            )
            system_prompt = self.prompt_templates.get_system_prompt("meteorologist")

        elif ('operation' in role_lower or 'logistic' in role_lower or
              'logistic' in expertise_lower or 'supply_chain' in expertise_lower or
              'operation' in expertise_lower):
            prompt = self.prompt_templates.generate_operations_prompt(
                scenario, alternatives, criteria
            )
            system_prompt = self.prompt_templates.get_system_prompt("operations")

        elif 'medical' in expertise_lower or 'health' in expertise_lower or 'medical' in role_lower:
            prompt = self.prompt_templates.generate_medical_prompt(
                scenario, alternatives, criteria
            )
            system_prompt = self.prompt_templates.get_system_prompt("medical")

        else:
            # Default to operations if role unclear
            logger.warning(
                f"Agent role '{self.role}' / expertise '{self.expertise}' not clearly mapped. "
                f"Using operations template as default."
            )
            prompt = self.prompt_templates.generate_operations_prompt(
                scenario, alternatives, criteria
            )
            system_prompt = self.prompt_templates.get_system_prompt("operations")

        return prompt, system_prompt

    def _call_llm(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """
        Execute LLM API call with automatic retry logic and comprehensive error handling.

        OBJECTIVE:
        Provides robust interface to LLM provider (Claude, OpenAI, LM Studio) that
        handles transient failures, rate limits, and malformed responses gracefully.
        Ensures system continues operating even when LLM calls fail.

        WHY THIS EXISTS:
        LLM API calls are network operations that can fail for many reasons:
        - Network timeouts or connection errors
        - Rate limiting from provider
        - Malformed JSON responses
        - Provider outages or maintenance
        This wrapper ensures resilience through retry logic (built into llm_client)
        and structured error responses that don't crash the agent.

        INPUTS:
        - prompt: User message containing scenario, alternatives, and evaluation request
        - system_prompt: System message defining agent role and behavior constraints

        OUTPUTS:
        Dictionary containing either:
        SUCCESS CASE:
        - alternative_rankings: {alt_id: score} dictionary
        - reasoning: Expert's analysis and justification
        - confidence: Self-reported confidence (0.0-1.0)
        - key_concerns: List of critical issues identified
        - _metadata: LLM provider metadata (tokens, model, latency)

        ERROR CASE:
        - error: True
        - error_message: Human-readable error description
        - error_type: Exception class name for debugging

        LLM CLIENT CONFIGURATION:
        - max_tokens: 2000 (sufficient for detailed reasoning)
        - temperature: 0.7 (balanced creativity/consistency)
        - Retry logic: Automatic exponential backoff (in llm_client)
        - Timeout: Provider-specific defaults

        ERROR HANDLING STRATEGY:
        Does NOT raise exceptions - always returns dictionary
        Upstream code checks for 'error' field to handle failures
        Logs full exception with stack trace for debugging
        """
        try:
            response = self.llm_client.generate_assessment(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.7
            )
            return response

        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}", exc_info=True)
            return {
                'error': True,
                'error_message': str(e),
                'error_type': type(e).__name__
            }

    def _validate_evaluation_inputs(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]]
    ):
        """
        Validate that scenario and alternatives meet minimum requirements before evaluation.

        OBJECTIVE:
        Performs early validation to catch malformed inputs before expensive LLM calls,
        providing clear error messages for debugging and preventing system failures.

        WHY THIS EXISTS:
        Fail-fast principle - detect invalid inputs immediately rather than during LLM
        processing. Saves API costs and time by rejecting bad data early. Provides
        actionable error messages to upstream callers.

        VALIDATION RULES:
        1. Scenario must be non-empty dictionary
           - Prevents null/None scenarios from reaching LLM
           - Ensures minimum context available

        2. Alternatives list must be non-empty
           - At least one alternative required for evaluation
           - Prevents meaningless assessment requests

        3. Each alternative must have 'id' or 'name' field
           - Ensures alternatives can be referenced in rankings
           - Prevents KeyError when building belief distributions
           - Reports specific index of invalid alternative

        ERROR CASES:
        - Empty scenario: "Scenario cannot be empty"
        - Empty alternatives: "Alternatives list cannot be empty"
        - Missing ID: "Alternative at index {i} must have either 'id' or 'name' field"

        VALIDATION FLOW:
        Called at start of evaluate_scenario() before prompt generation
        Raises ValueError immediately on first validation failure
        Does NOT return - either succeeds silently or raises exception
        """
        if not scenario:
            raise ValueError("Scenario cannot be empty")

        if not alternatives or len(alternatives) == 0:
            raise ValueError("Alternatives list cannot be empty")

        # Check that alternatives have IDs
        for i, alt in enumerate(alternatives):
            if 'id' not in alt and 'name' not in alt:
                raise ValueError(
                    f"Alternative at index {i} must have either 'id' or 'name' field"
                )

    def _validate_llm_response(self, llm_response: Dict[str, Any]):
        """
        Validate LLM response structure and content to ensure processability.

        OBJECTIVE:
        Verifies that LLM returned all required fields in expected formats before
        downstream processing. Catches LLM hallucinations, format errors, and
        incomplete responses early.

        WHY THIS EXISTS:
        LLMs are probabilistic and may:
        - Omit required fields
        - Return malformed JSON
        - Provide out-of-range values (confidence > 1.0)
        - Return wrong data types (string instead of dict)
        This validation catches these issues before they cause crashes in belief
        distribution generation or consensus calculation.

        VALIDATION CHECKS:
        1. Required fields present (logs warning if missing):
           - alternative_rankings: Agent's preference scores
           - reasoning: Justification text
           - confidence: Self-reported certainty
           - key_concerns: Critical issues identified

        2. alternative_rankings validation:
           - Must be non-empty dictionary
           - Keys = alternative IDs, Values = numeric scores
           - Raises ValueError if missing or wrong type

        3. Confidence value validation:
           - Must be numeric (int or float)
           - Must be in range [0, 1]
           - Logs warning if out of range (doesn't raise)

        VALIDATION STRATEGY:
        - Required fields: Log warnings but continue (degraded functionality)
        - alternative_rankings: Raise ValueError (critical for operation)
        - confidence: Log warning if invalid (use default 0.5)

        WHY NOT RAISE ON MISSING FIELDS:
        System should be resilient. Missing reasoning or key_concerns reduces
        quality but doesn't prevent decision-making. Only critical structural
        issues (no rankings) cause failures.

        EXAMPLE ERROR MESSAGES:
        - "LLM response missing field: reasoning"
        - "alternative_rankings must be non-empty dictionary"
        - "Confidence value 1.5 is not in valid range [0, 1]"
        """
        required_fields = ['alternative_rankings', 'reasoning', 'confidence', 'key_concerns']

        for field in required_fields:
            if field not in llm_response:
                logger.warning(f"LLM response missing field: {field}")

        # Validate specific fields
        if 'alternative_rankings' in llm_response:
            rankings = llm_response['alternative_rankings']
            if not isinstance(rankings, dict) or len(rankings) == 0:
                raise ValueError("alternative_rankings must be non-empty dictionary")

        if 'confidence' in llm_response:
            confidence = llm_response['confidence']
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                logger.warning(
                    f"Confidence value {confidence} is not in valid range [0, 1]"
                )

    def _build_assessment(
        self,
        belief_distribution: Dict[str, float],
        criteria_scores: Dict[str, Dict[str, float]],
        llm_response: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Construct complete agent assessment dictionary with all required fields.

        OBJECTIVE:
        Assembles final assessment output by combining processed LLM response with
        agent metadata and scenario context. Creates standardized format expected
        by CoordinatorAgent for belief aggregation and consensus building.

        WHY THIS EXISTS:
        Centralized construction ensures consistent assessment format across all
        expert agents. Combines multiple data sources (agent profile, LLM output,
        computed distributions) into single structure. Enables easy extension by
        adding new fields in one place.

        INPUTS:
        - belief_distribution: Normalized probability distribution over alternatives
          Format: {alt_id: probability} where sum(probabilities) ≈ 1.0
          Created by generate_belief_distribution() from LLM rankings

        - criteria_scores: Per-criterion evaluation scores
          Format: {criterion_id: {alt_id: score}}
          Created by get_criteria_scores() from scenario data + LLM reasoning

        - llm_response: Raw LLM output containing reasoning, confidence, concerns
          Used to extract qualitative assessment details

        - scenario: Original scenario data for context and metadata

        OUTPUTS:
        Dictionary containing:
        AGENT IDENTIFICATION:
        - agent_id: Unique agent identifier
        - agent_name: Human-readable name
        - agent_role: Domain role (e.g., "Emergency Medicine Specialist")
        - expertise: Domain expertise area

        QUANTITATIVE ASSESSMENT:
        - belief_distribution: Probability distribution over alternatives
        - criteria_scores: Detailed scores per criterion per alternative

        QUALITATIVE ASSESSMENT:
        - reasoning: LLM's detailed analysis and justification
        - confidence: Agent's self-reported confidence (0.0-1.0)
        - key_concerns: List of critical issues identified

        METADATA:
        - timestamp: ISO format assessment time
        - scenario_type: Type of crisis (flood, earthquake, etc.)
        - llm_metadata: LLM provider stats (tokens, model, latency)
        - assessment_number: Sequential counter for this agent

        USAGE IN SYSTEM:
        1. ExpertAgent returns this from evaluate_scenario()
        2. CoordinatorAgent collects from all agents
        3. ER/GAT uses belief_distribution for aggregation
        4. MCDA uses criteria_scores for ranking
        5. Consensus uses belief_distribution for similarity
        6. Evaluation uses criteria_scores for quality metrics
        """
        return {
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'agent_role': self.role,
            'expertise': self.expertise,
            'belief_distribution': belief_distribution,
            'criteria_scores': criteria_scores,
            'reasoning': llm_response.get('reasoning', ''),
            'confidence': llm_response.get('confidence', self.confidence_level),
            'key_concerns': llm_response.get('key_concerns', []),
            'timestamp': datetime.now().isoformat(),
            'scenario_type': scenario.get('type', 'unknown'),
            'llm_metadata': llm_response.get('_metadata', {}),
            'assessment_number': self.assessment_count + 1
        }

    def propose_action(
        self,
        scenario: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propose an action based on scenario and criteria.

        This method is required by BaseAgent. It evaluates the scenario with
        alternatives extracted from the criteria, then returns the top-ranked
        alternative as the proposed action.

        Args:
            scenario: Crisis scenario data
            criteria: Dictionary containing 'alternatives' key with list of alternatives

        Returns:
            Dictionary containing:
                - proposed_action: The recommended alternative
                - action_id: ID of the proposed alternative
                - confidence: Agent's confidence in this action
                - justification: Reasoning for this action
                - full_assessment: Complete assessment data

        Example:
            >>> criteria = {
            ...     'alternatives': [
            ...         {'id': 'A1', 'name': 'Evacuate', ...},
            ...         {'id': 'A2', 'name': 'Deploy Barriers', ...}
            ...     ]
            ... }
            >>> action = agent.propose_action(scenario, criteria)
            >>> print(action['proposed_action'])
            'A1'
        """
        logger.info(
            f"Agent {self.agent_id} proposing action for scenario {scenario.get('scenario_id', 'unknown')}"
        )

        # Extract alternatives from criteria
        alternatives = criteria.get('alternatives', [])
        if not alternatives:
            raise ValueError("criteria must contain 'alternatives' key with list of alternatives")

        # Extract criteria names if provided
        criteria_names = criteria.get('criteria', None)

        # Evaluate scenario to get assessments
        assessment = self.evaluate_scenario(scenario, alternatives, criteria_names)

        # Find top-ranked alternative
        belief_dist = assessment['belief_distribution']
        top_alternative_id = max(belief_dist.items(), key=lambda x: x[1])[0]
        top_score = belief_dist[top_alternative_id]

        # Find full alternative details
        top_alternative = None
        for alt in alternatives:
            alt_id = alt.get('id', alt.get('name'))
            if alt_id == top_alternative_id:
                top_alternative = alt
                break

        # Build action proposal
        action_proposal = {
            'proposed_action': top_alternative_id,
            'action_name': top_alternative.get('name', top_alternative_id) if top_alternative else top_alternative_id,
            'action_details': top_alternative,
            'confidence': assessment['confidence'],
            'belief_score': top_score,
            'justification': assessment['reasoning'],
            'key_concerns': assessment['key_concerns'],
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'agent_role': self.role,
            'timestamp': datetime.now().isoformat(),
            'full_assessment': assessment
        }

        logger.info(
            f"Agent {self.agent_id} proposes action '{top_alternative_id}' "
            f"with confidence {assessment['confidence']:.2f}"
        )

        return action_proposal

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExpertAgent(id={self.agent_id}, name='{self.name}', role='{self.role}', "
            f"expertise='{self.expertise}', llm={type(self.llm_client).__name__}, "
            f"assessments={self.assessment_count})"
        )
