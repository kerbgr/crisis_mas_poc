"""
Coordinator Agent Class
Orchestrates multi-agent decision-making process for crisis management

This agent coordinates expert agents, aggregates their assessments,
detects consensus, and generates final decisions with explanations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.expert_agent import ExpertAgent
from decision_framework.evidential_reasoning import EvidentialReasoning
from decision_framework.mcda_engine import MCDAEngine
from decision_framework.consensus_model import ConsensusModel
from decision_framework.gat_aggregator import GATAggregator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    Coordinator agent that orchestrates multi-agent decision-making.

    This agent:
    - Manages multiple expert agents
    - Collects and aggregates assessments
    - Detects consensus and conflicts
    - Generates final decisions with explanations

    Example:
        >>> experts = [meteorologist_agent, operations_agent, medical_agent]
        >>> coordinator = CoordinatorAgent(
        ...     expert_agents=experts,
        ...     er_engine=er_engine,
        ...     mcda_engine=mcda_engine,
        ...     consensus_model=consensus_model
        ... )
        >>> decision = coordinator.make_final_decision(scenario, alternatives)
        >>> print(decision['recommended_alternative'])
        'A1'
    """

    def __init__(
        self,
        expert_agents: List[ExpertAgent],
        er_engine: EvidentialReasoning,
        mcda_engine: MCDAEngine,
        consensus_model: ConsensusModel,
        agent_weights: Optional[Dict[str, float]] = None,
        consensus_threshold: float = 0.75,
        parallel_assessment: bool = True,
        gat_aggregator: Optional[GATAggregator] = None,
        aggregation_method: str = "ER"
    ):
        """
        Initialize the Coordinator Agent.

        Args:
            expert_agents: List of ExpertAgent instances to coordinate
            er_engine: Evidential Reasoning engine for belief aggregation
            mcda_engine: MCDA engine for multi-criteria scoring
            consensus_model: Consensus model for conflict detection
            agent_weights: Optional custom weights for agents (will use equal if not provided)
            consensus_threshold: Minimum consensus level to avoid conflict resolution (0-1)
            parallel_assessment: Whether to collect assessments in parallel (default: True)
            gat_aggregator: Optional Graph Attention Network aggregator (alternative to ER)
            aggregation_method: Aggregation method to use: "ER" or "GAT" (default: "ER")

        Raises:
            ValueError: If expert_agents list is empty or contains invalid agents
        """
        # Validate expert agents
        if not expert_agents:
            raise ValueError("Must provide at least one expert agent")

        for agent in expert_agents:
            if not isinstance(agent, ExpertAgent):
                raise ValueError(
                    f"All agents must be ExpertAgent instances, got {type(agent)}"
                )

        self.expert_agents = expert_agents
        self.er_engine = er_engine
        self.mcda_engine = mcda_engine
        self.consensus_model = consensus_model
        self.consensus_threshold = consensus_threshold
        self.parallel_assessment = parallel_assessment

        # GAT aggregation support
        self.gat_aggregator = gat_aggregator
        self.aggregation_method = aggregation_method.upper()

        # Validate aggregation method
        if self.aggregation_method not in ["ER", "GAT"]:
            logger.warning(
                f"Invalid aggregation method '{aggregation_method}', defaulting to 'ER'"
            )
            self.aggregation_method = "ER"

        # Create GAT aggregator if GAT method selected but not provided
        if self.aggregation_method == "GAT" and self.gat_aggregator is None:
            logger.info("GAT method selected, creating default GATAggregator")
            self.gat_aggregator = GATAggregator(
                num_attention_heads=4,
                use_multi_head=True
            )

        # Set up agent weights (equal weights if not provided)
        if agent_weights is None:
            n = len(expert_agents)
            self.agent_weights = {agent.agent_id: 1.0 / n for agent in expert_agents}
        else:
            self.agent_weights = agent_weights

        # Normalize weights
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            self.agent_weights = {
                agent_id: weight / total_weight
                for agent_id, weight in self.agent_weights.items()
            }

        # State tracking
        self.decision_history: List[Dict[str, Any]] = []
        self.last_decision: Optional[Dict[str, Any]] = None
        self.decision_count: int = 0

        logger.info(
            f"CoordinatorAgent initialized with {len(expert_agents)} expert agents "
            f"(aggregation={self.aggregation_method}, parallel={parallel_assessment}, "
            f"consensus_threshold={consensus_threshold:.2f})"
        )

    def collect_assessments(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Collect assessments from all expert agents.

        Collects assessments either sequentially or in parallel based on
        the parallel_assessment setting.

        Args:
            scenario: Crisis scenario data
            alternatives: List of response alternatives
            criteria: Optional list of criteria names

        Returns:
            Dictionary containing:
                - assessments: Dict[agent_id, assessment]
                - collection_time_seconds: float
                - agents_responded: int
                - agents_failed: List[agent_id]

        Example:
            >>> results = coordinator.collect_assessments(scenario, alternatives)
            >>> for agent_id, assessment in results['assessments'].items():
            ...     print(f"{agent_id}: {assessment['belief_distribution']}")
        """
        logger.info(
            f"Collecting assessments from {len(self.expert_agents)} agents "
            f"(parallel={self.parallel_assessment})"
        )

        start_time = datetime.now()
        assessments = {}
        failed_agents = []

        if self.parallel_assessment:
            # Parallel collection using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(self.expert_agents)) as executor:
                # Submit all assessment tasks
                future_to_agent = {
                    executor.submit(
                        agent.evaluate_scenario,
                        scenario,
                        alternatives,
                        criteria
                    ): agent
                    for agent in self.expert_agents
                }

                # Collect results as they complete
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        assessment = future.result(timeout=60)  # 60 second timeout
                        assessments[agent.agent_id] = assessment
                        logger.info(
                            f"Received assessment from {agent.agent_id} "
                            f"(confidence: {assessment.get('confidence', 0):.2f})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Agent {agent.agent_id} failed to provide assessment: {e}"
                        )
                        failed_agents.append(agent.agent_id)

        else:
            # Sequential collection
            for agent in self.expert_agents:
                try:
                    assessment = agent.evaluate_scenario(scenario, alternatives, criteria)
                    assessments[agent.agent_id] = assessment
                    logger.info(
                        f"Received assessment from {agent.agent_id} "
                        f"(confidence: {assessment.get('confidence', 0):.2f})"
                    )
                except Exception as e:
                    logger.error(
                        f"Agent {agent.agent_id} failed to provide assessment: {e}"
                    )
                    failed_agents.append(agent.agent_id)

        end_time = datetime.now()
        collection_time = (end_time - start_time).total_seconds()

        logger.info(
            f"Assessment collection completed: {len(assessments)}/{len(self.expert_agents)} "
            f"agents responded in {collection_time:.2f}s"
        )

        return {
            'assessments': assessments,
            'collection_time_seconds': collection_time,
            'agents_responded': len(assessments),
            'agents_failed': failed_agents,
            'timestamp': datetime.now().isoformat()
        }

    def aggregate_beliefs(
        self,
        agent_assessments: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate belief distributions using configured method (ER or GAT).

        Combines individual agent beliefs into a single aggregated belief
        distribution using either Evidential Reasoning or Graph Attention Network.

        Args:
            agent_assessments: Dictionary of assessments from collect_assessments
            scenario: Optional scenario (required for GAT method)

        Returns:
            Dictionary containing:
                - aggregated_beliefs: Dict[alternative_id, belief_score]
                - uncertainty: float
                - confidence: float
                - method: str (ER or GAT)
                - method_details: Full aggregation engine output

        Example:
            >>> results = coordinator.collect_assessments(scenario, alternatives)
            >>> aggregated = coordinator.aggregate_beliefs(results['assessments'], scenario)
            >>> print(aggregated['aggregated_beliefs'])
            {'A1': 0.65, 'A2': 0.25, 'A3': 0.08, 'A4': 0.02}
        """
        if self.aggregation_method == "GAT":
            return self._aggregate_with_gat(agent_assessments, scenario)
        else:
            return self._aggregate_with_er(agent_assessments)

    def _aggregate_with_er(
        self,
        agent_assessments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate agent beliefs using Evidential Reasoning (Dempster-Shafer Theory).

        OBJECTIVE:
        Combines belief distributions from multiple agents into a single aggregated
        distribution using classical Evidential Reasoning. This method applies
        Dempster's rule of combination with confidence-based discounting to handle
        uncertainty and conflicting evidence.

        WHY THIS EXISTS:
        Provides a classical, mathematically rigorous approach to belief aggregation
        that explicitly handles uncertainty and conflicting opinions. ER is particularly
        useful when agents have varying confidence levels and may provide partially
        conflicting assessments.

        INPUTS:
        - agent_assessments: Dictionary mapping agent_id to assessment data
          Each assessment must contain 'belief_distribution' field with alternative probabilities

        OUTPUTS:
        Dictionary containing:
        - aggregated_beliefs: Combined probability distribution over alternatives
        - uncertainty: Measure of remaining uncertainty after combination (0-1)
        - confidence: Overall confidence in aggregated result (0-1)
        - method: 'ER' identifier
        - method_details: Full ER engine output with discounting and combination steps

        PROCESS:
        1. Extract belief distributions from all agent assessments
        2. Filter agent weights to match agents who provided assessments
        3. Apply Dempster-Shafer combination via ER engine
        4. Handle conflicts using confidence-weighted discounting
        5. Return aggregated distribution with uncertainty quantification
        """
        logger.info("Aggregating beliefs using Evidential Reasoning")

        # Extract belief distributions from assessments
        agent_beliefs = {}
        for agent_id, assessment in agent_assessments.items():
            if 'belief_distribution' in assessment:
                agent_beliefs[agent_id] = assessment['belief_distribution']

        if not agent_beliefs:
            logger.warning("No belief distributions found in assessments")
            return {
                'aggregated_beliefs': {},
                'uncertainty': 1.0,
                'confidence': 0.0,
                'method': 'ER',
                'method_details': None
            }

        # Filter agent weights to only include agents that provided assessments
        filtered_weights = {
            agent_id: self.agent_weights.get(agent_id, 1.0 / len(agent_beliefs))
            for agent_id in agent_beliefs.keys()
        }

        # Use ER engine to combine beliefs
        try:
            er_result = self.er_engine.combine_beliefs(agent_beliefs, filtered_weights)

            logger.info(
                f"Beliefs aggregated: uncertainty={er_result.get('uncertainty', 0):.3f}, "
                f"confidence={er_result.get('confidence', 0):.3f}"
            )

            return {
                'aggregated_beliefs': er_result.get('combined_beliefs', {}),
                'uncertainty': er_result.get('uncertainty', 1.0),
                'confidence': er_result.get('confidence', 0.0),
                'method': 'ER',
                'method_details': er_result
            }

        except Exception as e:
            logger.error(f"Error aggregating beliefs: {e}")
            return {
                'aggregated_beliefs': {},
                'uncertainty': 1.0,
                'confidence': 0.0,
                'method': 'ER',
                'method_details': None,
                'error': str(e)
            }

    def _aggregate_with_gat(
        self,
        agent_assessments: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate agent beliefs using Graph Attention Network (GAT) with dynamic weighting.

        OBJECTIVE:
        Combines belief distributions using neural attention mechanism that learns
        optimal agent weights based on 9-dimensional feature vectors including
        confidence, expertise relevance, historical reliability, and belief certainty.
        Produces data-driven weights rather than uniform or predefined weights.

        WHY THIS EXISTS:
        GAT provides adaptive, context-aware aggregation that automatically identifies
        which agents are most relevant for each specific scenario. Unlike ER which uses
        fixed or confidence-based weights, GAT dynamically adjusts agent influence based
        on expertise match, past performance, and scenario characteristics.

        INPUTS:
        - agent_assessments: Dictionary mapping agent_id to assessment data
          Each assessment should contain belief_distribution, confidence, and agent metadata
        - scenario: Optional scenario data for context-aware weighting
          Used to calculate expertise relevance and scenario-specific features

        OUTPUTS:
        Dictionary containing:
        - aggregated_beliefs: Weighted combination of agent belief distributions
        - uncertainty: Measure of uncertainty in aggregated result (0-1)
        - confidence: Overall confidence based on attention weights and agent confidence
        - method: 'GAT' identifier
        - method_details: Full GAT output including features and attention computation
        - attention_weights: Learned weights showing each agent's influence (interpretability)

        PROCESS:
        1. Extract agent features (9 dimensions): confidence, belief_certainty, expertise,
           risk_tolerance, severity_awareness, top_choice_strength, num_concerns,
           reasoning_quality, historical_reliability
        2. Build graph with agents as nodes and beliefs as node features
        3. Apply multi-head attention (4 heads) to compute attention weights
        4. Aggregate beliefs using learned attention weights
        5. Return aggregated distribution with attention weights for transparency

        KEY DIFFERENCES FROM ER:
        - ER: Fixed/confidence-based weights, mathematically rigorous uncertainty handling
        - GAT: Learned weights, context-aware, incorporates historical performance
        """
        logger.info("Aggregating beliefs using Graph Attention Network (GAT)")

        if self.gat_aggregator is None:
            logger.error("GAT aggregator not initialized")
            return {
                'aggregated_beliefs': {},
                'uncertainty': 1.0,
                'confidence': 0.0,
                'method': 'GAT',
                'method_details': None,
                'error': 'GAT aggregator not initialized'
            }

        if scenario is None:
            logger.warning("Scenario not provided for GAT, using empty scenario")
            scenario = {}

        try:
            gat_result = self.gat_aggregator.aggregate_beliefs_with_gat(
                agent_assessments,
                scenario
            )

            logger.info(
                f"GAT aggregation complete: confidence={gat_result.get('confidence', 0):.3f}, "
                f"uncertainty={gat_result.get('uncertainty', 0):.3f}"
            )

            return {
                'aggregated_beliefs': gat_result.get('aggregated_beliefs', {}),
                'uncertainty': gat_result.get('uncertainty', 1.0),
                'confidence': gat_result.get('confidence', 0.0),
                'method': 'GAT',
                'method_details': gat_result,
                'attention_weights': gat_result.get('attention_weights', {})
            }

        except Exception as e:
            logger.error(f"Error in GAT aggregation: {e}")
            return {
                'aggregated_beliefs': {},
                'uncertainty': 1.0,
                'confidence': 0.0,
                'method': 'GAT',
                'method_details': None,
                'error': str(e)
            }

    def check_consensus(
        self,
        agent_assessments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check consensus level and detect conflicts.

        Uses the ConsensusModel to calculate agreement between agents
        and identify any significant conflicts.

        Args:
            agent_assessments: Dictionary of agent assessments

        Returns:
            Dictionary containing:
                - consensus_level: float (0-1)
                - consensus_reached: bool
                - conflicts: List of detected conflicts
                - pairwise_agreements: Dict of agent pair similarities

        Example:
            >>> consensus = coordinator.check_consensus(assessments)
            >>> if consensus['consensus_reached']:
            ...     print("Agents are in agreement!")
            >>> else:
            ...     print(f"Conflicts detected: {consensus['conflicts']}")
        """
        logger.info("Checking consensus level")

        # Extract belief distributions
        agent_beliefs = {}
        for agent_id, assessment in agent_assessments.items():
            if 'belief_distribution' in assessment:
                agent_beliefs[agent_id] = assessment['belief_distribution']

        if len(agent_beliefs) < 2:
            logger.warning("Need at least 2 agents for consensus check")
            return {
                'consensus_level': 1.0,
                'consensus_reached': True,
                'conflicts': [],
                'pairwise_agreements': {},
                'message': 'Insufficient agents for consensus calculation'
            }

        try:
            # Calculate consensus level
            consensus_level = self.consensus_model.calculate_consensus_level(agent_beliefs)

            # Check if consensus is reached
            consensus_reached = self.consensus_model.is_consensus_reached(consensus_level)

            # Detect conflicts
            conflicts = []
            if not consensus_reached:
                conflicts = self.consensus_model.detect_conflicts(agent_beliefs)

            # Get detailed analysis
            analysis = self.consensus_model.analyze_consensus(agent_beliefs)

            logger.info(
                f"Consensus level: {consensus_level:.3f} "
                f"(threshold: {self.consensus_threshold:.3f}, "
                f"reached: {consensus_reached})"
            )

            if conflicts:
                logger.warning(f"Detected {len(conflicts)} conflicts")

            return {
                'consensus_level': consensus_level,
                'consensus_reached': consensus_reached,
                'conflicts': conflicts,
                'pairwise_agreements': analysis.get('pairwise_similarities', {}),
                'analysis_details': analysis
            }

        except Exception as e:
            logger.error(f"Error checking consensus: {e}")
            return {
                'consensus_level': 0.0,
                'consensus_reached': False,
                'conflicts': [],
                'pairwise_agreements': {},
                'error': str(e)
            }

    def resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        agent_assessments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest conflict resolution strategies.

        Uses the ConsensusModel to suggest how to resolve disagreements
        between agents.

        Args:
            conflicts: List of conflicts from check_consensus
            agent_assessments: Dictionary of agent assessments

        Returns:
            Dictionary containing:
                - resolution_strategy: str
                - suggested_actions: List[str]
                - compromise_alternatives: List[str]
                - rationale: str

        Example:
            >>> if not consensus['consensus_reached']:
            ...     resolution = coordinator.resolve_conflicts(
            ...         consensus['conflicts'],
            ...         assessments
            ...     )
            ...     print(resolution['resolution_strategy'])
        """
        logger.info(f"Resolving {len(conflicts)} conflicts")

        if not conflicts:
            return {
                'resolution_strategy': 'no_action_needed',
                'suggested_actions': [],
                'compromise_alternatives': [],
                'rationale': 'No conflicts detected - consensus reached'
            }

        # Extract belief distributions
        agent_beliefs = {}
        for agent_id, assessment in agent_assessments.items():
            if 'belief_distribution' in assessment:
                agent_beliefs[agent_id] = assessment['belief_distribution']

        try:
            # Get resolution suggestions from consensus model
            resolution = self.consensus_model.suggest_resolution(conflicts, agent_beliefs)

            logger.info(f"Resolution strategy: {resolution.get('strategy', 'unknown')}")

            return {
                'resolution_strategy': resolution.get('strategy', 'weighted_aggregation'),
                'suggested_actions': resolution.get('suggestions', []),
                'compromise_alternatives': resolution.get('compromise_alternatives', []),
                'rationale': resolution.get('rationale', ''),
                'full_resolution': resolution
            }

        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")
            return {
                'resolution_strategy': 'weighted_aggregation',
                'suggested_actions': [
                    'Use weighted aggregation to combine agent opinions',
                    'Rely on agent expertise weights for final decision'
                ],
                'compromise_alternatives': [],
                'rationale': f'Default resolution due to error: {e}',
                'error': str(e)
            }

    def make_final_decision(
        self,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute complete decision-making workflow and generate final decision.

        This is the main coordination method that:
        1. Collects assessments from all experts
        2. Aggregates beliefs using ER
        3. Scores alternatives using MCDA
        4. Checks consensus
        5. Resolves conflicts if needed
        6. Generates final decision with explanation

        Args:
            scenario: Crisis scenario data
            alternatives: List of response alternatives
            criteria: Optional list of criteria names

        Returns:
            Dictionary containing complete decision with:
                - recommended_alternative: str
                - confidence: float
                - consensus_level: float
                - final_scores: Dict[alternative_id, score]
                - agent_opinions: Dict[agent_id, opinion]
                - consensus_reached: bool
                - conflicts: List
                - explanation: str
                - timestamp: str

        Example:
            >>> decision = coordinator.make_final_decision(scenario, alternatives)
            >>> print(f"Recommended: {decision['recommended_alternative']}")
            >>> print(f"Consensus: {decision['consensus_level']:.1%}")
            >>> print(decision['explanation'])
        """
        logger.info(
            f"Making final decision for scenario {scenario.get('scenario_id', 'unknown')}"
        )

        start_time = datetime.now()

        # Step 1: Collect assessments from all expert agents
        logger.info("Step 1/6: Collecting expert assessments")
        collection_results = self.collect_assessments(scenario, alternatives, criteria)
        agent_assessments = collection_results['assessments']

        if not agent_assessments:
            logger.error("No assessments collected - cannot make decision")
            return self._create_error_decision(
                "No expert assessments available",
                scenario,
                alternatives
            )

        # Step 2: Aggregate beliefs using configured method (ER or GAT)
        logger.info(f"Step 2/6: Aggregating beliefs with {self.aggregation_method}")
        aggregated = self.aggregate_beliefs(agent_assessments, scenario)

        # Step 3: Score alternatives using MCDA
        logger.info("Step 3/6: Scoring alternatives with MCDA")
        mcda_rankings = self.mcda_engine.rank_alternatives(alternatives)

        # Convert MCDA rankings to scores dict
        mcda_scores = {alt_id: score for alt_id, score, _ in mcda_rankings}

        # Step 4: Check consensus level
        logger.info("Step 4/6: Checking consensus")
        consensus_info = self.check_consensus(agent_assessments)

        # Step 5: Resolve conflicts if consensus not reached
        resolution = None
        if not consensus_info['consensus_reached']:
            logger.info("Step 5/6: Resolving conflicts")
            resolution = self.resolve_conflicts(
                consensus_info['conflicts'],
                agent_assessments
            )
        else:
            logger.info("Step 5/6: No conflicts to resolve")
            resolution = {
                'resolution_strategy': 'consensus_reached',
                'suggested_actions': [],
                'compromise_alternatives': [],
                'rationale': 'Strong consensus among agents'
            }

        # Step 6: Generate final decision
        logger.info("Step 6/6: Generating final decision")

        # Combine ER beliefs and MCDA scores (60/40 weighting)
        combined_scores = {}
        er_beliefs = aggregated['aggregated_beliefs']

        for alt in alternatives:
            alt_id = alt.get('id', alt.get('name'))
            er_score = er_beliefs.get(alt_id, 0.0)
            mcda_score = mcda_scores.get(alt_id, 0.0)
            # Weighted combination: 60% ER beliefs, 40% MCDA scores
            combined_scores[alt_id] = 0.6 * er_score + 0.4 * mcda_score

        # Find recommended alternative
        if combined_scores:
            recommended_alt = max(combined_scores.items(), key=lambda x: x[1])[0]
            recommended_score = combined_scores[recommended_alt]
        else:
            recommended_alt = None
            recommended_score = 0.0

        # Extract agent opinions
        agent_opinions = {}
        agent_confidences = []
        for agent_id, assessment in agent_assessments.items():
            belief_dist = assessment.get('belief_distribution', {})
            if belief_dist:
                top_choice = max(belief_dist.items(), key=lambda x: x[1])
                agent_conf = assessment.get('confidence', 0.0)
                agent_confidences.append(agent_conf)
                agent_opinions[agent_id] = {
                    'agent_name': assessment.get('agent_name', agent_id),
                    'preference': top_choice[0],
                    'confidence': agent_conf,
                    'belief_score': top_choice[1],
                    'reasoning': assessment.get('reasoning', '')[:200] + '...'  # Truncate
                }

        # Calculate overall confidence based on consensus and agent confidences
        # Confidence = combination of consensus level and average agent confidence
        avg_agent_confidence = sum(agent_confidences) / len(agent_confidences) if agent_confidences else 0.0
        # Weight: 60% consensus level, 40% average agent confidence
        overall_confidence = 0.6 * consensus_info['consensus_level'] + 0.4 * avg_agent_confidence

        # Build final decision
        decision = {
            'recommended_alternative': recommended_alt,
            'confidence': overall_confidence,  # Overall confidence in the decision
            'decision_quality_score': recommended_score,  # Quality score based on ER+MCDA
            'consensus_level': consensus_info['consensus_level'],
            'final_scores': combined_scores,
            'er_scores': er_beliefs,
            'mcda_scores': mcda_scores,
            'agent_opinions': agent_opinions,
            'consensus_reached': consensus_info['consensus_reached'],
            'conflicts': consensus_info.get('conflicts', []),
            'resolution': resolution,
            'timestamp': datetime.now().isoformat(),
            'scenario_id': scenario.get('scenario_id', 'unknown'),
            'decision_time_seconds': (datetime.now() - start_time).total_seconds(),
            'agents_participated': len(agent_assessments),
            'collection_info': collection_results
        }

        # Generate explanation
        decision['explanation'] = self.generate_explanation(decision)

        # Update state
        self.decision_count += 1
        self.last_decision = decision
        self.decision_history.append({
            'decision_number': self.decision_count,
            'decision': decision,
            'timestamp': decision['timestamp']
        })

        logger.info(
            f"Final decision: {recommended_alt} (quality_score: {recommended_score:.2f}, "
            f"confidence: {overall_confidence:.2f}, consensus: {consensus_info['consensus_level']:.2f})"
        )

        return decision

    def generate_explanation(self, decision: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of the decision.

        Creates a detailed, multi-paragraph explanation that includes:
        - Recommended alternative and rationale
        - Consensus level interpretation
        - Individual agent opinions
        - Conflict resolution (if applicable)
        - Confidence analysis

        Args:
            decision: Decision dictionary from make_final_decision

        Returns:
            Formatted explanation string

        Example:
            >>> explanation = coordinator.generate_explanation(decision)
            >>> print(explanation)
        """
        recommended = decision.get('recommended_alternative', 'Unknown')
        confidence = decision.get('confidence', 0.0)
        consensus_level = decision.get('consensus_level', 0.0)
        consensus_reached = decision.get('consensus_reached', False)
        agent_opinions = decision.get('agent_opinions', {})
        conflicts = decision.get('conflicts', [])

        # Build explanation sections
        explanation_parts = []

        # Header
        explanation_parts.append("━" * 80)
        explanation_parts.append("COORDINATED DECISION ANALYSIS")
        explanation_parts.append("━" * 80)
        explanation_parts.append("")

        # Recommendation section
        explanation_parts.append("RECOMMENDATION")
        explanation_parts.append("-" * 80)
        explanation_parts.append(
            f"Recommended Alternative: {recommended} (Confidence: {confidence:.1%})"
        )
        explanation_parts.append("")

        # Consensus section
        explanation_parts.append("CONSENSUS ANALYSIS")
        explanation_parts.append("-" * 80)

        if consensus_level >= 0.85:
            consensus_desc = "STRONG CONSENSUS - Agents are in strong agreement"
        elif consensus_level >= 0.75:
            consensus_desc = "MODERATE CONSENSUS - Reasonable agreement among agents"
        elif consensus_level >= 0.60:
            consensus_desc = "WEAK CONSENSUS - Some disagreement present"
        else:
            consensus_desc = "LOW CONSENSUS - Significant disagreement among agents"

        explanation_parts.append(f"Consensus Level: {consensus_level:.1%} - {consensus_desc}")
        explanation_parts.append(f"Consensus Reached: {'Yes' if consensus_reached else 'No'}")
        explanation_parts.append("")

        # Agent opinions section
        explanation_parts.append("EXPERT AGENT OPINIONS")
        explanation_parts.append("-" * 80)

        # Group agents by preference
        preference_groups = {}
        for agent_id, opinion in agent_opinions.items():
            pref = opinion['preference']
            if pref not in preference_groups:
                preference_groups[pref] = []
            preference_groups[pref].append(opinion)

        for alt_id, agents in sorted(preference_groups.items(), key=lambda x: len(x[1]), reverse=True):
            agent_names = [a['agent_name'] for a in agents]
            avg_confidence = sum(a['confidence'] for a in agents) / len(agents)
            explanation_parts.append(
                f"  Alternative {alt_id}: Supported by {len(agents)} agent(s) "
                f"({', '.join(agent_names)}) - Avg Confidence: {avg_confidence:.1%}"
            )

        explanation_parts.append("")

        # Detailed agent reasoning
        explanation_parts.append("INDIVIDUAL AGENT REASONING")
        explanation_parts.append("-" * 80)
        for agent_id, opinion in agent_opinions.items():
            explanation_parts.append(
                f"  • {opinion['agent_name']}: Prefers {opinion['preference']} "
                f"(confidence: {opinion['confidence']:.1%})"
            )
            if opinion.get('reasoning'):
                explanation_parts.append(f"    Reasoning: {opinion['reasoning']}")

        explanation_parts.append("")

        # Conflict resolution section (if applicable)
        if conflicts:
            explanation_parts.append("CONFLICT RESOLUTION")
            explanation_parts.append("-" * 80)
            explanation_parts.append(f"Conflicts Detected: {len(conflicts)}")

            resolution = decision.get('resolution', {})
            strategy = resolution.get('resolution_strategy', 'weighted_aggregation')
            rationale = resolution.get('rationale', '')

            explanation_parts.append(f"Resolution Strategy: {strategy}")
            if rationale:
                explanation_parts.append(f"Rationale: {rationale}")

            if resolution.get('suggested_actions'):
                explanation_parts.append("Suggested Actions:")
                for action in resolution['suggested_actions']:
                    explanation_parts.append(f"  • {action}")

            explanation_parts.append("")

        # Decision methodology section
        explanation_parts.append("DECISION METHODOLOGY")
        explanation_parts.append("-" * 80)
        explanation_parts.append(
            "This decision was generated using a multi-stage process:"
        )
        explanation_parts.append(
            "1. Expert assessments collected from all domain specialists"
        )
        explanation_parts.append(
            "2. Belief distributions aggregated using Evidential Reasoning (ER)"
        )
        explanation_parts.append(
            "3. Alternatives scored using Multi-Criteria Decision Analysis (MCDA)"
        )
        explanation_parts.append(
            "4. Final scores computed as weighted combination: 60% ER + 40% MCDA"
        )
        explanation_parts.append(
            "5. Consensus level calculated using cosine similarity"
        )
        explanation_parts.append("")

        # Confidence analysis
        explanation_parts.append("CONFIDENCE ANALYSIS")
        explanation_parts.append("-" * 80)

        if confidence >= 0.80 and consensus_reached:
            confidence_msg = "HIGH CONFIDENCE: Strong recommendation with agent consensus"
        elif confidence >= 0.70:
            confidence_msg = "MODERATE CONFIDENCE: Good recommendation but monitor situation"
        elif confidence >= 0.60:
            confidence_msg = "ACCEPTABLE CONFIDENCE: Valid but consider alternatives"
        else:
            confidence_msg = "LOW CONFIDENCE: Recommendation uncertain, consider gathering more information"

        explanation_parts.append(confidence_msg)
        explanation_parts.append("")

        # Footer
        explanation_parts.append("━" * 80)
        explanation_parts.append(
            f"Decision generated at: {decision.get('timestamp', 'unknown')}"
        )
        explanation_parts.append(
            f"Agents participated: {decision.get('agents_participated', 0)}"
        )
        explanation_parts.append(
            f"Processing time: {decision.get('decision_time_seconds', 0):.2f} seconds"
        )
        explanation_parts.append("━" * 80)

        return "\n".join(explanation_parts)

    def _create_error_decision(
        self,
        error_message: str,
        scenario: Dict[str, Any],
        alternatives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a structured error decision when normal decision-making process fails.

        OBJECTIVE:
        Provides graceful failure handling by creating a valid decision structure
        that clearly indicates an error occurred, allowing upstream systems to
        handle the failure appropriately without crashing.

        WHY THIS EXISTS:
        In production systems, complete failures must be handled gracefully. Rather
        than raising exceptions that crash the system, this method creates a
        structured error response that:
        1. Maintains the expected decision dictionary format
        2. Clearly indicates no valid decision was made
        3. Provides error context for debugging
        4. Allows the system to continue processing other scenarios

        INPUTS:
        - error_message: Human-readable description of what went wrong
          Examples: "No agent assessments available", "MCDA engine failed",
                   "Aggregation timeout exceeded"
        - scenario: Original scenario data (preserved for traceability)
        - alternatives: List of alternatives that were being considered

        OUTPUTS:
        Dictionary with decision structure containing:
        - recommended_alternative: None (no valid recommendation)
        - confidence: 0.0 (no confidence in error state)
        - consensus_level: 0.0 (no consensus achieved)
        - final_scores: {} (no scores calculated)
        - agent_opinions: {} (no opinions processed)
        - consensus_reached: False
        - conflicts: [] (no conflicts detected)
        - resolution: None
        - timestamp: When error occurred (ISO format)
        - scenario_id: Original scenario identifier for tracing
        - error: Error message for logging/debugging
        - explanation: User-friendly error description

        USE CASES:
        - Agent collection fails (timeout, no responses)
        - Belief aggregation fails (ER/GAT errors)
        - MCDA engine crashes
        - Consensus building times out
        - Invalid input data detected

        EXAMPLE ERROR FLOW:
        try:
            decision = coordinator.make_final_decision(...)
        except Exception as e:
            error_decision = coordinator._create_error_decision(
                str(e), scenario, alternatives
            )
            # Log error_decision['error']
            # Return error_decision to client
            # Client sees None recommendation and displays error_decision['explanation']
        """
        return {
            'recommended_alternative': None,
            'confidence': 0.0,
            'consensus_level': 0.0,
            'final_scores': {},
            'agent_opinions': {},
            'consensus_reached': False,
            'conflicts': [],
            'resolution': None,
            'timestamp': datetime.now().isoformat(),
            'scenario_id': scenario.get('scenario_id', 'unknown'),
            'error': error_message,
            'explanation': f"ERROR: {error_message}\n\nUnable to generate decision."
        }

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """
        Get complete decision history.

        Returns:
            List of all decisions made by this coordinator
        """
        return self.decision_history.copy()

    def get_last_decision(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent decision.

        Returns:
            Last decision dictionary or None
        """
        return self.last_decision

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CoordinatorAgent(experts={len(self.expert_agents)}, "
            f"consensus_threshold={self.consensus_threshold:.2f}, "
            f"decisions_made={self.decision_count})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        agent_names = [agent.name for agent in self.expert_agents]
        return (
            f"Coordinator Agent\n"
            f"  Expert Agents: {len(self.expert_agents)} ({', '.join(agent_names)})\n"
            f"  Consensus Threshold: {self.consensus_threshold:.2f}\n"
            f"  Decisions Made: {self.decision_count}\n"
            f"  Parallel Assessment: {self.parallel_assessment}"
        )
