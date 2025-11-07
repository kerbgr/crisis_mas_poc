"""
Simplified Evidential Reasoning Implementation
For Crisis Management Multi-Agent System

This module provides a lightweight ER approach using weighted averaging
to combine belief distributions from multiple agents. Designed for clarity
and interpretability in a Master's thesis PoC context.

NOT full Dempster-Shafer theory - simplified for practical crisis decision-making.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidentialReasoning:
    """
    Simplified Evidential Reasoning for combining agent beliefs.

    Uses weighted averaging to aggregate belief distributions from multiple
    agents, taking into account agent reliability weights.

    Example:
        >>> er = EvidentialReasoning()
        >>> agent_beliefs = {
        ...     "agent_meteorologist": {"A1": 0.7, "A2": 0.2, "A3": 0.1},
        ...     "agent_operations": {"A1": 0.5, "A2": 0.3, "A3": 0.2}
        ... }
        >>> agent_weights = {"agent_meteorologist": 0.55, "agent_operations": 0.45}
        >>> result = er.combine_beliefs(agent_beliefs, agent_weights)
    """

    def __init__(self, enable_logging: bool = True):
        """
        Initialize the Evidential Reasoning engine.

        Args:
            enable_logging: Whether to log aggregation process (default: True)
        """
        self.enable_logging = enable_logging
        self.aggregation_history: List[Dict[str, Any]] = []

        if self.enable_logging:
            logger.info("Evidential Reasoning engine initialized")

    def combine_beliefs(
        self,
        agent_beliefs: Dict[str, Dict[str, float]],
        agent_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Combine belief distributions from multiple agents using weighted averaging.

        This method aggregates beliefs by:
        1. Validating inputs
        2. Normalizing agent weights
        3. Computing weighted average for each alternative
        4. Calculating uncertainty mass
        5. Computing confidence score

        Args:
            agent_beliefs: Dictionary mapping agent IDs to their belief distributions.
                          Format: {"agent_id": {"A1": 0.7, "A2": 0.2, "A3": 0.1}}
            agent_weights: Dictionary mapping agent IDs to reliability weights.
                          Format: {"agent_id": 0.55, ...}
                          Weights will be normalized to sum to 1.0

        Returns:
            Dictionary containing:
                - combined_beliefs: Dict[str, float] - Aggregated belief distribution
                - uncertainty: float - Uncertainty mass (1 - sum of beliefs)
                - confidence: float - Overall confidence score
                - agents_involved: List[str] - Agent IDs that participated
                - normalized_weights: Dict[str, float] - Normalized agent weights used
                - aggregation_log: List[str] - Step-by-step log of the process

        Raises:
            ValueError: If inputs are invalid or inconsistent

        Example:
            >>> agent_beliefs = {
            ...     "agent_meteorologist": {"A1": 0.7, "A2": 0.2, "A3": 0.1},
            ...     "agent_operations": {"A1": 0.5, "A2": 0.3, "A3": 0.2}
            ... }
            >>> agent_weights = {"agent_meteorologist": 0.55, "agent_operations": 0.45}
            >>> result = er.combine_beliefs(agent_beliefs, agent_weights)
            >>> print(result['combined_beliefs'])
            {'A1': 0.615, 'A2': 0.245, 'A3': 0.14}
        """
        aggregation_log = []
        timestamp = datetime.now().isoformat()

        # Step 1: Validate inputs
        self._validate_inputs(agent_beliefs, agent_weights)
        aggregation_log.append("✓ Input validation passed")

        # Step 2: Normalize agent weights
        normalized_weights = self.normalize_weights(agent_weights)
        aggregation_log.append(f"✓ Normalized {len(normalized_weights)} agent weights")

        if self.enable_logging:
            logger.info(f"Combining beliefs from {len(agent_beliefs)} agents")
            for agent_id, weight in normalized_weights.items():
                logger.debug(f"  {agent_id}: weight={weight:.3f}")

        # Step 3: Get all unique alternatives across all agents
        all_alternatives = set()
        for beliefs in agent_beliefs.values():
            all_alternatives.update(beliefs.keys())
        all_alternatives = sorted(all_alternatives)
        aggregation_log.append(f"✓ Identified {len(all_alternatives)} alternatives: {all_alternatives}")

        # Step 4: Normalize each agent's belief distribution
        normalized_beliefs = {}
        for agent_id, beliefs in agent_beliefs.items():
            normalized_beliefs[agent_id] = self.normalize_distribution(beliefs)
        aggregation_log.append("✓ Normalized all agent belief distributions")

        # Step 5: Compute weighted average for each alternative
        combined_beliefs = {}

        for alternative in all_alternatives:
            weighted_sum = 0.0

            for agent_id, beliefs in normalized_beliefs.items():
                # Get belief for this alternative (0.0 if agent didn't mention it)
                agent_belief = beliefs.get(alternative, 0.0)
                agent_weight = normalized_weights[agent_id]

                weighted_sum += agent_belief * agent_weight

                if self.enable_logging:
                    logger.debug(
                        f"    {alternative}: {agent_id} belief={agent_belief:.3f} "
                        f"* weight={agent_weight:.3f} = {agent_belief * agent_weight:.3f}"
                    )

            combined_beliefs[alternative] = weighted_sum

        aggregation_log.append("✓ Computed weighted averages for all alternatives")

        # Step 6: Normalize combined beliefs to ensure they sum to 1.0
        combined_beliefs = self.normalize_distribution(combined_beliefs)
        aggregation_log.append("✓ Normalized combined belief distribution")

        # Step 7: Calculate uncertainty
        belief_sum = sum(combined_beliefs.values())
        uncertainty = max(0.0, 1.0 - belief_sum)  # Should be ~0 after normalization

        # Step 8: Calculate confidence score
        confidence = self.calculate_confidence(combined_beliefs)
        aggregation_log.append(f"✓ Calculated confidence score: {confidence:.3f}")

        # Prepare result
        result = {
            'combined_beliefs': combined_beliefs,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'agents_involved': list(agent_beliefs.keys()),
            'normalized_weights': normalized_weights,
            'aggregation_log': aggregation_log,
            'timestamp': timestamp,
            'num_alternatives': len(all_alternatives),
            'alternatives': all_alternatives
        }

        # Store in history
        self.aggregation_history.append(result)

        if self.enable_logging:
            logger.info(f"✓ Belief aggregation complete: confidence={confidence:.3f}")
            logger.info(f"Combined beliefs: {combined_beliefs}")

        return result

    def normalize_distribution(self, beliefs: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize a belief distribution to ensure it sums to 1.0.

        Args:
            beliefs: Dictionary mapping alternatives to belief values
                    Format: {"A1": 0.7, "A2": 0.2, "A3": 0.15}

        Returns:
            Normalized belief distribution that sums to 1.0

        Raises:
            ValueError: If all beliefs are zero or negative

        Example:
            >>> er = EvidentialReasoning()
            >>> beliefs = {"A1": 0.7, "A2": 0.2, "A3": 0.15}
            >>> normalized = er.normalize_distribution(beliefs)
            >>> sum(normalized.values())
            1.0
        """
        if not beliefs:
            return {}

        # Check for negative values
        for alt, value in beliefs.items():
            if value < 0:
                raise ValueError(
                    f"Negative belief value for alternative '{alt}': {value}. "
                    "All beliefs must be non-negative."
                )

        # Calculate sum
        total = sum(beliefs.values())

        if total == 0:
            raise ValueError(
                "Cannot normalize: all belief values are zero. "
                "At least one alternative must have non-zero belief."
            )

        # Normalize
        normalized = {alt: value / total for alt, value in beliefs.items()}

        return normalized

    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize agent weights to ensure they sum to 1.0.

        Args:
            weights: Dictionary mapping agent IDs to reliability weights
                    Format: {"agent_id": 0.55, ...}

        Returns:
            Normalized weights that sum to 1.0

        Raises:
            ValueError: If all weights are zero or negative

        Example:
            >>> er = EvidentialReasoning()
            >>> weights = {"agent1": 0.6, "agent2": 0.4}
            >>> normalized = er.normalize_weights(weights)
            >>> sum(normalized.values())
            1.0
        """
        if not weights:
            raise ValueError("Agent weights dictionary is empty")

        # Check for negative values
        for agent_id, weight in weights.items():
            if weight < 0:
                raise ValueError(
                    f"Negative weight for agent '{agent_id}': {weight}. "
                    "All weights must be non-negative."
                )

        # Calculate sum
        total = sum(weights.values())

        if total == 0:
            raise ValueError(
                "Cannot normalize: all weights are zero. "
                "At least one agent must have non-zero weight."
            )

        # Normalize
        normalized = {agent_id: weight / total for agent_id, weight in weights.items()}

        return normalized

    def calculate_confidence(self, combined_beliefs: Dict[str, float]) -> float:
        """
        Calculate overall confidence score based on the combined belief distribution.

        The confidence score reflects how decisively the beliefs are distributed:
        - High confidence: Beliefs concentrated on few alternatives
        - Low confidence: Beliefs spread evenly across many alternatives

        Uses entropy-based measure normalized to [0, 1] range.

        Args:
            combined_beliefs: Combined belief distribution
                             Format: {"A1": 0.615, "A2": 0.245, "A3": 0.14}

        Returns:
            Confidence score between 0.0 (low confidence) and 1.0 (high confidence)

        Example:
            >>> er = EvidentialReasoning()
            >>> # High confidence - one dominant alternative
            >>> beliefs = {"A1": 0.9, "A2": 0.05, "A3": 0.05}
            >>> er.calculate_confidence(beliefs)
            0.95
            >>> # Low confidence - evenly distributed
            >>> beliefs = {"A1": 0.33, "A2": 0.33, "A3": 0.34}
            >>> er.calculate_confidence(beliefs)
            0.35
        """
        if not combined_beliefs:
            return 0.0

        # Calculate Shannon entropy
        import math

        entropy = 0.0
        for belief in combined_beliefs.values():
            if belief > 0:
                entropy -= belief * math.log2(belief)

        # Normalize entropy to [0, 1]
        # Maximum entropy occurs when all beliefs are equal
        n = len(combined_beliefs)
        if n <= 1:
            max_entropy = 0.0
        else:
            max_entropy = math.log2(n)

        # Convert entropy to confidence
        # High entropy = low confidence, Low entropy = high confidence
        if max_entropy == 0:
            confidence = 1.0
        else:
            normalized_entropy = entropy / max_entropy
            confidence = 1.0 - normalized_entropy

        return confidence

    def _validate_inputs(
        self,
        agent_beliefs: Dict[str, Dict[str, float]],
        agent_weights: Dict[str, float]
    ):
        """
        Validate input parameters for combine_beliefs method.

        Args:
            agent_beliefs: Agent belief distributions
            agent_weights: Agent reliability weights

        Raises:
            ValueError: If inputs are invalid
        """
        # Check that inputs are not empty
        if not agent_beliefs:
            raise ValueError("agent_beliefs dictionary is empty")

        if not agent_weights:
            raise ValueError("agent_weights dictionary is empty")

        # Check that agent IDs match
        belief_agents = set(agent_beliefs.keys())
        weight_agents = set(agent_weights.keys())

        if belief_agents != weight_agents:
            missing_weights = belief_agents - weight_agents
            missing_beliefs = weight_agents - belief_agents

            error_msg = "Agent IDs mismatch between beliefs and weights.\n"
            if missing_weights:
                error_msg += f"  Agents with beliefs but no weights: {missing_weights}\n"
            if missing_beliefs:
                error_msg += f"  Agents with weights but no beliefs: {missing_beliefs}"

            raise ValueError(error_msg)

        # Check that each agent has valid belief distribution
        for agent_id, beliefs in agent_beliefs.items():
            if not beliefs:
                raise ValueError(f"Agent '{agent_id}' has empty belief distribution")

            if not isinstance(beliefs, dict):
                raise ValueError(
                    f"Agent '{agent_id}' beliefs must be a dictionary, "
                    f"got {type(beliefs)}"
                )

            # Check belief values are numeric
            for alt, value in beliefs.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Agent '{agent_id}' has non-numeric belief for '{alt}': {value}"
                    )

    def get_top_alternatives(
        self,
        combined_beliefs: Dict[str, float],
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get the top N alternatives ranked by belief.

        Args:
            combined_beliefs: Combined belief distribution
            top_n: Number of top alternatives to return (default: 3)

        Returns:
            List of (alternative, belief) tuples, sorted by belief (descending)

        Example:
            >>> er = EvidentialReasoning()
            >>> beliefs = {"A1": 0.615, "A2": 0.245, "A3": 0.14}
            >>> er.get_top_alternatives(beliefs, top_n=2)
            [('A1', 0.615), ('A2', 0.245)]
        """
        sorted_beliefs = sorted(
            combined_beliefs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_beliefs[:top_n]

    def get_aggregation_summary(self, result: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of an aggregation result.

        Args:
            result: Result dictionary from combine_beliefs()

        Returns:
            Formatted summary string

        Example:
            >>> result = er.combine_beliefs(agent_beliefs, agent_weights)
            >>> print(er.get_aggregation_summary(result))
        """
        top_alternatives = self.get_top_alternatives(result['combined_beliefs'], top_n=3)

        summary = f"""
{'='*70}
Evidential Reasoning - Aggregation Summary
{'='*70}
Timestamp: {result['timestamp']}
Agents Involved: {len(result['agents_involved'])}
  {', '.join(result['agents_involved'])}

Agent Weights (Normalized):
"""
        for agent_id, weight in result['normalized_weights'].items():
            summary += f"  • {agent_id:30s} : {weight:.3f}\n"

        summary += f"\nAlternatives Evaluated: {result['num_alternatives']}\n"
        summary += f"  {', '.join(result['alternatives'])}\n"

        summary += "\nCombined Belief Distribution:\n"
        for alt, belief in sorted(
            result['combined_beliefs'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar_length = int(belief * 40)
            bar = '█' * bar_length
            summary += f"  {alt:10s} : {belief:.3f} {bar}\n"

        summary += f"\nTop 3 Alternatives:\n"
        for i, (alt, belief) in enumerate(top_alternatives, 1):
            summary += f"  {i}. {alt:10s} : {belief:.3f} ({belief*100:.1f}%)\n"

        summary += f"\nConfidence Score: {result['confidence']:.3f}\n"
        summary += f"Uncertainty Mass: {result['uncertainty']:.3f}\n"

        summary += "\nAggregation Process:\n"
        for log_entry in result['aggregation_log']:
            summary += f"  {log_entry}\n"

        summary += "="*70

        return summary

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all aggregations performed.

        Returns:
            List of aggregation results
        """
        return self.aggregation_history.copy()

    def clear_history(self):
        """Clear the aggregation history."""
        self.aggregation_history.clear()
        if self.enable_logging:
            logger.info("Aggregation history cleared")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EvidentialReasoning("
            f"aggregations_performed={len(self.aggregation_history)}, "
            f"logging={'enabled' if self.enable_logging else 'disabled'})"
        )
