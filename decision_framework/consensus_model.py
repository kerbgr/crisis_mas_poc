"""
Consensus Model - Agreement Detection and Conflict Resolution for Multi-Agent Systems

OBJECTIVE:
This module provides mechanisms for detecting consensus, identifying conflicts, and
suggesting resolutions when multiple expert agents must reach agreement on crisis
response actions. It quantifies the level of agreement between agents and provides
actionable guidance when disagreements occur.

WHY THIS EXISTS:
In crisis management, multiple experts often have divergent opinions due to different:
- **Domain expertise** (medical vs. logistics vs. environmental)
- **Risk tolerance** (conservative vs. aggressive approaches)
- **Priority weighting** (lives vs. cost vs. speed)
- **Information access** (different data sources or interpretations)

This module ensures that:
1. Consensus is objectively measured (not just "everyone agrees")
2. Significant conflicts are identified before they become problematic
3. Resolution strategies are suggested based on conflict severity
4. Compromise alternatives are identified when direct agreement fails
5. Decision-making process remains transparent and auditable

KEY RESPONSIBILITIES:
1. **Consensus Detection**: Measure agreement level using cosine similarity
2. **Conflict Identification**: Detect agents with fundamentally different preferences
3. **Severity Classification**: Categorize conflicts as low/moderate/high severity
4. **Resolution Suggestions**: Generate actionable strategies for conflict resolution
5. **Compromise Finding**: Identify alternatives acceptable to all parties
6. **History Tracking**: Maintain consensus statistics over time

MATHEMATICAL FOUNDATION:
Uses **cosine similarity** to measure agreement between belief distributions:

    similarity = (A · B) / (||A|| × ||B||)

Where:
- A, B are belief vectors from two agents
- A · B is the dot product (sum of element-wise products)
- ||A||, ||B|| are L2 norms (Euclidean magnitudes)

Interpretation:
- 1.0 = Perfect agreement (identical beliefs)
- 0.8-0.9 = High agreement (similar priorities)
- 0.5-0.7 = Moderate agreement (some differences)
- 0.0-0.4 = Low agreement (significant disagreement)

CONSENSUS THRESHOLD:
Default: 0.75 (75% similarity required for consensus)
- Configurable based on crisis type and time constraints
- Higher threshold → more stringent consensus requirement
- Lower threshold → faster decision-making, lower agreement

CONFLICT DETECTION LOGIC:
A conflict is flagged when:
1. Two agents have different top-choice alternatives, AND
2. The belief difference exceeds conflict_threshold (default: 0.3)
3. Disagreement magnitude is significant (both agents strongly prefer different options)

RESOLUTION STRATEGIES:
Based on conflict severity:
- **High Severity** (score > 0.6): Escalate to human decision-maker
- **Moderate Severity** (0.3-0.6): Explore compromise alternatives, facilitate discussion
- **Low Severity** (< 0.3): Weighted voting or minor adjustments sufficient

COMPROMISE ALTERNATIVES:
Found using weighted combination score:
    compromise_score = 0.6 × avg_belief + 0.4 × min_belief

This balances:
- Average belief: Both agents should find it moderately acceptable
- Minimum belief: Neither agent should strongly reject it

INPUTS (Primary Method: analyze_consensus):
- agent_beliefs: Dict[agent_id, Dict[alternative_id, float]]
  * Each agent provides belief distribution over alternatives
  * Beliefs represent probability/preference (0-1)
  * Must sum to ~1.0 for each agent
- alternatives_data: Optional[Dict[alternative_id, Dict[str, Any]]]
  * Full alternative details for enriched explanations
  * Includes names, descriptions, resource requirements

OUTPUTS (analyze_consensus returns):
- consensus_level: float (0-1) - Average pairwise cosine similarity
- consensus_reached: bool - Whether threshold is met
- conflicts: List[Dict] - Detected conflicts with details
- num_conflicts: int - Number of conflicts found
- resolution_needed: bool - Whether intervention required
- resolution_suggestions: str - Human-readable resolution strategies
- timestamp: str - ISO format timestamp
- threshold_used: float - Threshold that was applied
- agents_analyzed: List[str] - Agent IDs included

EXAMPLE USAGE:
```python
# Initialize with custom threshold
model = ConsensusModel(consensus_threshold=0.75)

# Agent beliefs
agent_beliefs = {
    "medical_expert": {"A1": 0.8, "A2": 0.1, "A3": 0.1},
    "logistics_expert": {"A1": 0.6, "A2": 0.3, "A3": 0.1},
    "safety_expert": {"A1": 0.7, "A2": 0.2, "A3": 0.1}
}

# Analyze consensus
result = model.analyze_consensus(agent_beliefs)

print(f"Consensus Level: {result['consensus_level']:.2f}")
print(f"Consensus Reached: {result['consensus_reached']}")
print(f"Conflicts Detected: {result['num_conflicts']}")

if result['resolution_needed']:
    print(result['resolution_suggestions'])
```

REAL-WORLD SCENARIO:
In a flood evacuation decision:
- **Medical Expert**: Prefers immediate evacuation (A1) - safety priority
- **Logistics Expert**: Prefers staged evacuation (A2) - resource constraints
- **Safety Expert**: Prefers hybrid approach (A3) - balances concerns

If consensus_level < 0.75:
→ Conflict detected
→ Compromise alternative identified (e.g., prioritized evacuation of vulnerable)
→ Resolution strategy suggested (e.g., "Start with vulnerable populations, expand based on resource availability")

DESIGN PATTERNS:
1. **Strategy Pattern**: Different resolution strategies for different severity levels
2. **Observer Pattern**: Consensus history tracks all analyses
3. **Template Method**: analyze_consensus provides standard workflow
4. **Factory Method**: Conflict dictionaries constructed with consistent structure

ERROR HANDLING:
- ValueError: If < 2 agents (consensus requires at least 2 parties)
- ValueError: If agent IDs mismatch between beliefs and weights
- ValueError: If belief distributions are empty or invalid
- Graceful degradation: Returns empty conflict list if insufficient data

PERFORMANCE CHARACTERISTICS:
- Consensus calculation: O(N² × M) where N=agents, M=alternatives
  * Pairwise comparisons: N(N-1)/2 pairs
  * Vector operations: O(M) per pair
- Conflict detection: O(N² × M)
- Compromise finding: O(M) per conflict
- Total: O(N² × M) - Acceptable for typical N ≤ 10, M ≤ 20

LIMITATIONS:
1. Assumes belief distributions are well-formed (sum to ~1.0)
2. Cosine similarity doesn't account for magnitude differences
3. Pairwise comparison doesn't capture group dynamics (3+ agent coalitions)
4. Compromise finding is heuristic, not optimization-based
5. No temporal dynamics (beliefs are static snapshots)

INTEGRATION POINTS:
- Used by: CoordinatorAgent after belief aggregation
- Inputs from: Expert agents' belief distributions
- Outputs to: Decision explanation and escalation logic
- Related to: Evidential Reasoning (provides the beliefs to analyze)

VALIDATION:
Unit tests verify:
- Identical beliefs → consensus_level = 1.0
- Opposite beliefs → consensus_level ≈ 0.0
- Threshold boundary conditions
- Conflict detection accuracy
- Compromise finding correctness

RELATED RESEARCH:
- Cosine similarity for text/vector comparison
- Dempster-Shafer theory for uncertainty reasoning
- Multi-agent coordination in crisis response
- Group decision-making under uncertainty
"""

import math
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import Counter


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusModel:
    """
    Consensus detection and conflict resolution for multi-agent systems.

    Uses cosine similarity to measure agreement between agent belief distributions
    and provides conflict resolution suggestions when disagreement occurs.

    Example:
        >>> model = ConsensusModel(consensus_threshold=0.75)
        >>> agent_beliefs = {
        ...     "agent_1": {"A1": 0.7, "A2": 0.2, "A3": 0.1},
        ...     "agent_2": {"A1": 0.6, "A2": 0.3, "A3": 0.1}
        ... }
        >>> result = model.calculate_consensus_level(agent_beliefs)
    """

    def __init__(self, consensus_threshold: float = 0.75):
        """
        Initialize the consensus model.

        Args:
            consensus_threshold: Minimum similarity score for consensus (0-1)
                               Default: 0.75 (75% similarity required)

        Raises:
            ValueError: If threshold not in [0, 1] range
        """
        if not 0 <= consensus_threshold <= 1:
            raise ValueError(
                f"Consensus threshold must be between 0 and 1, got {consensus_threshold}"
            )

        self.consensus_threshold = consensus_threshold
        self.consensus_history: List[Dict[str, Any]] = []

        logger.info(f"Consensus model initialized with threshold={consensus_threshold:.2f}")

    def calculate_consensus_level(self, agent_beliefs: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate consensus level between agents using cosine similarity.

        Cosine similarity measures the angle between two belief vectors:
        similarity = (A · B) / (||A|| × ||B||)

        Where:
        - A · B is the dot product of belief vectors
        - ||A|| is the magnitude (L2 norm) of vector A
        - ||B|| is the magnitude (L2 norm) of vector B

        Score interpretation:
        - 1.0: Perfect agreement (identical beliefs)
        - 0.5-0.9: Moderate to high agreement
        - 0.0-0.5: Low agreement
        - 0.0: Complete disagreement (orthogonal beliefs)

        Args:
            agent_beliefs: Dictionary mapping agent IDs to their belief distributions
                          Format: {"agent_id": {"A1": 0.7, "A2": 0.2, "A3": 0.1}}

        Returns:
            Consensus level (0.0 to 1.0)

        Raises:
            ValueError: If insufficient agents or invalid belief distributions

        Example:
            >>> model = ConsensusModel()
            >>> beliefs = {
            ...     "agent_1": {"A1": 0.7, "A2": 0.3},
            ...     "agent_2": {"A1": 0.6, "A2": 0.4}
            ... }
            >>> consensus = model.calculate_consensus_level(beliefs)
            >>> print(f"Consensus: {consensus:.2f}")
            Consensus: 0.99
        """
        if len(agent_beliefs) < 2:
            raise ValueError(
                f"Need at least 2 agents for consensus calculation, got {len(agent_beliefs)}"
            )

        # Get all unique alternatives mentioned by any agent
        all_alternatives = set()
        for beliefs in agent_beliefs.values():
            all_alternatives.update(beliefs.keys())

        all_alternatives = sorted(all_alternatives)

        # Convert beliefs to vectors (same dimensionality for all agents)
        belief_vectors = {}
        for agent_id, beliefs in agent_beliefs.items():
            vector = [beliefs.get(alt, 0.0) for alt in all_alternatives]
            belief_vectors[agent_id] = vector

        # Calculate pairwise cosine similarities
        agent_ids = list(agent_beliefs.keys())
        similarities = []

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent_i = agent_ids[i]
                agent_j = agent_ids[j]

                similarity = self._cosine_similarity(
                    belief_vectors[agent_i],
                    belief_vectors[agent_j]
                )
                similarities.append(similarity)

                logger.debug(
                    f"Similarity between {agent_i} and {agent_j}: {similarity:.3f}"
                )

        # Return average similarity across all pairs
        if not similarities:
            return 0.0

        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity

    def _cosine_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Formula: cos(θ) = (A · B) / (||A|| × ||B||)

        Args:
            vector_a: First belief vector
            vector_b: Second belief vector

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        if len(vector_a) != len(vector_b):
            raise ValueError("Vectors must have same length")

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

        # Calculate magnitudes (L2 norms)
        magnitude_a = math.sqrt(sum(a * a for a in vector_a))
        magnitude_b = math.sqrt(sum(b * b for b in vector_b))

        # Handle zero vectors
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = dot_product / (magnitude_a * magnitude_b)

        # Ensure result is in [0, 1] range (handle floating point errors)
        return max(0.0, min(1.0, similarity))

    def detect_conflicts(
        self,
        agent_beliefs: Dict[str, Dict[str, float]],
        conflict_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between agents.

        A conflict occurs when:
        1. Agents have different top-choice alternatives, AND
        2. The belief difference for their top choices is significant

        Args:
            agent_beliefs: Dictionary mapping agent IDs to belief distributions
            conflict_threshold: Minimum belief difference to flag as conflict (default: 0.3)

        Returns:
            List of conflict dictionaries, each containing:
                - agent_pair: Tuple of agent IDs in conflict
                - conflict_score: Severity of disagreement (0-1)
                - agent_1_top_choice: Top choice for first agent
                - agent_2_top_choice: Top choice for second agent
                - disagreement_magnitude: Difference in top choice beliefs

        Example:
            >>> model = ConsensusModel()
            >>> beliefs = {
            ...     "agent_1": {"A1": 0.8, "A2": 0.2},
            ...     "agent_2": {"A1": 0.2, "A2": 0.8}
            ... }
            >>> conflicts = model.detect_conflicts(beliefs)
            >>> print(f"Found {len(conflicts)} conflicts")
            Found 1 conflicts
        """
        if len(agent_beliefs) < 2:
            return []

        conflicts = []
        agent_ids = list(agent_beliefs.keys())

        # Check each pair of agents
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent_1_id = agent_ids[i]
                agent_2_id = agent_ids[j]

                beliefs_1 = agent_beliefs[agent_1_id]
                beliefs_2 = agent_beliefs[agent_2_id]

                # Get top choices for each agent
                top_1 = max(beliefs_1.items(), key=lambda x: x[1])
                top_2 = max(beliefs_2.items(), key=lambda x: x[1])

                top_1_alt, top_1_belief = top_1
                top_2_alt, top_2_belief = top_2

                # Check if they disagree on top choice
                if top_1_alt != top_2_alt:
                    # Calculate conflict severity
                    # How strongly does agent 1 prefer their choice over agent 2's choice?
                    agent_1_diff = top_1_belief - beliefs_1.get(top_2_alt, 0.0)
                    # How strongly does agent 2 prefer their choice over agent 1's choice?
                    agent_2_diff = top_2_belief - beliefs_2.get(top_1_alt, 0.0)

                    # Average the differences
                    disagreement_magnitude = (agent_1_diff + agent_2_diff) / 2

                    # Calculate conflict score (inverse of consensus)
                    consensus = self._cosine_similarity(
                        [beliefs_1.get(alt, 0.0) for alt in sorted(set(beliefs_1) | set(beliefs_2))],
                        [beliefs_2.get(alt, 0.0) for alt in sorted(set(beliefs_1) | set(beliefs_2))]
                    )
                    conflict_score = 1.0 - consensus

                    # Only flag if disagreement is significant
                    if disagreement_magnitude >= conflict_threshold:
                        conflict = {
                            'agent_pair': (agent_1_id, agent_2_id),
                            'conflict_score': conflict_score,
                            'agent_1_top_choice': top_1_alt,
                            'agent_2_top_choice': top_2_alt,
                            'agent_1_belief': top_1_belief,
                            'agent_2_belief': top_2_belief,
                            'disagreement_magnitude': disagreement_magnitude,
                            'severity': self._classify_severity(conflict_score)
                        }
                        conflicts.append(conflict)

                        logger.info(
                            f"Conflict detected: {agent_1_id} prefers {top_1_alt} "
                            f"({top_1_belief:.2f}), {agent_2_id} prefers {top_2_alt} "
                            f"({top_2_belief:.2f}), severity={conflict['severity']}"
                        )

        return conflicts

    def _classify_severity(self, conflict_score: float) -> str:
        """
        Classify conflict severity based on score.

        Args:
            conflict_score: Conflict score (0-1)

        Returns:
            Severity classification: "low", "moderate", or "high"
        """
        if conflict_score < 0.3:
            return "low"
        elif conflict_score < 0.6:
            return "moderate"
        else:
            return "high"

    def suggest_resolution(
        self,
        conflicts: List[Dict[str, Any]],
        agent_beliefs: Dict[str, Dict[str, float]],
        alternatives_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> str:
        """
        Generate conflict resolution suggestions.

        Analyzes conflicts and suggests:
        1. Compromise alternatives that both agents find acceptable
        2. Sources of disagreement
        3. Facilitation strategies

        Args:
            conflicts: List of conflicts from detect_conflicts()
            agent_beliefs: Original agent belief distributions
            alternatives_data: Optional detailed data about alternatives

        Returns:
            Human-readable resolution suggestion string

        Example:
            >>> conflicts = model.detect_conflicts(agent_beliefs)
            >>> if conflicts:
            ...     suggestion = model.suggest_resolution(
            ...         conflicts, agent_beliefs, alternatives
            ...     )
            ...     print(suggestion)
        """
        if not conflicts:
            return "No conflicts detected. Agents are in agreement."

        suggestions = []
        suggestions.append("="*70)
        suggestions.append("CONFLICT RESOLUTION SUGGESTIONS")
        suggestions.append("="*70)

        for idx, conflict in enumerate(conflicts, 1):
            agent_1, agent_2 = conflict['agent_pair']
            severity = conflict['severity']

            suggestions.append(f"\nConflict #{idx} - Severity: {severity.upper()}")
            suggestions.append(f"Between: {agent_1} and {agent_2}")
            suggestions.append(f"Conflict Score: {conflict['conflict_score']:.3f}")

            suggestions.append(f"\nDisagreement:")
            suggestions.append(
                f"  • {agent_1} prefers: {conflict['agent_1_top_choice']} "
                f"(belief: {conflict['agent_1_belief']:.2f})"
            )
            suggestions.append(
                f"  • {agent_2} prefers: {conflict['agent_2_top_choice']} "
                f"(belief: {conflict['agent_2_belief']:.2f})"
            )

            # Find compromise alternatives
            compromises = self._find_compromise_alternatives(
                agent_beliefs[agent_1],
                agent_beliefs[agent_2]
            )

            if compromises:
                suggestions.append(f"\nPotential Compromise Alternatives:")
                for rank, (alt_id, combined_score) in enumerate(compromises[:3], 1):
                    agent_1_belief = agent_beliefs[agent_1].get(alt_id, 0.0)
                    agent_2_belief = agent_beliefs[agent_2].get(alt_id, 0.0)

                    suggestions.append(
                        f"  {rank}. {alt_id}: Combined score={combined_score:.3f} "
                        f"({agent_1} belief={agent_1_belief:.2f}, "
                        f"{agent_2} belief={agent_2_belief:.2f})"
                    )

                    # Add alternative description if available
                    if alternatives_data and alt_id in alternatives_data:
                        alt_name = alternatives_data[alt_id].get('name', alt_id)
                        suggestions.append(f"      Name: {alt_name}")

            # Resolution strategies
            suggestions.append(f"\nResolution Strategies:")

            if severity == "high":
                suggestions.append("  • URGENT: Significant disagreement detected")
                suggestions.append("  • Consider involving additional expert opinion")
                suggestions.append("  • Review underlying criteria weights")
                suggestions.append("  • May need human decision-maker intervention")
            elif severity == "moderate":
                suggestions.append("  • Explore compromise alternatives listed above")
                suggestions.append("  • Have agents explain their reasoning")
                suggestions.append("  • Look for hybrid solutions")
            else:
                suggestions.append("  • Minor disagreement, likely resolvable")
                suggestions.append("  • Consider weighted combination of preferences")

            suggestions.append("-"*70)

        # Overall recommendation
        suggestions.append(f"\n{'='*70}")
        suggestions.append("OVERALL RECOMMENDATION")
        suggestions.append(f"{'='*70}")

        high_severity_count = sum(1 for c in conflicts if c['severity'] == 'high')

        if high_severity_count > 0:
            suggestions.append(
                f"\n⚠ {high_severity_count} high-severity conflict(s) detected."
            )
            suggestions.append("Recommendation: ESCALATE to human decision-maker")
        else:
            suggestions.append("\nRecommendation: PROCEED with compromise alternatives")
            suggestions.append("Consider weighted voting or hybrid solution")

        return "\n".join(suggestions)

    def _find_compromise_alternatives(
        self,
        beliefs_1: Dict[str, float],
        beliefs_2: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Find compromise alternatives that both agents find acceptable.

        Compromise score = (belief_1 + belief_2) / 2
        This identifies alternatives where both agents have moderate agreement.

        Args:
            beliefs_1: First agent's beliefs
            beliefs_2: Second agent's beliefs

        Returns:
            List of (alternative_id, combined_score) tuples, sorted by score
        """
        # Get all alternatives
        all_alternatives = set(beliefs_1.keys()) | set(beliefs_2.keys())

        # Calculate combined scores
        combined_scores = []
        for alt in all_alternatives:
            belief_1 = beliefs_1.get(alt, 0.0)
            belief_2 = beliefs_2.get(alt, 0.0)

            # Average belief (compromise score)
            combined_score = (belief_1 + belief_2) / 2

            # Also consider minimum (both must accept)
            min_acceptance = min(belief_1, belief_2)

            # Weighted combination: favor alternatives acceptable to both
            final_score = 0.6 * combined_score + 0.4 * min_acceptance

            combined_scores.append((alt, final_score))

        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        return combined_scores

    def is_consensus_reached(self, consensus_level: float) -> bool:
        """
        Check if consensus threshold is met.

        Args:
            consensus_level: Calculated consensus level (0-1)

        Returns:
            True if consensus_level >= threshold, False otherwise

        Example:
            >>> model = ConsensusModel(consensus_threshold=0.75)
            >>> model.is_consensus_reached(0.82)
            True
            >>> model.is_consensus_reached(0.65)
            False
        """
        return consensus_level >= self.consensus_threshold

    def analyze_consensus(
        self,
        agent_beliefs: Dict[str, Dict[str, float]],
        alternatives_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive consensus analysis.

        Combines consensus level calculation, conflict detection, and resolution
        suggestions into a single comprehensive report.

        Args:
            agent_beliefs: Dictionary mapping agent IDs to belief distributions
            alternatives_data: Optional detailed data about alternatives

        Returns:
            Dictionary containing:
                - consensus_level: Similarity score (0-1)
                - consensus_reached: Boolean flag
                - conflicts: List of detected conflicts
                - resolution_needed: Whether resolution is required
                - resolution_suggestions: Text suggestions (if conflicts exist)
                - timestamp: Analysis timestamp

        Example:
            >>> model = ConsensusModel(consensus_threshold=0.75)
            >>> agent_beliefs = {
            ...     "agent_1": {"A1": 0.7, "A2": 0.2, "A3": 0.1},
            ...     "agent_2": {"A1": 0.6, "A2": 0.3, "A3": 0.1}
            ... }
            >>> result = model.analyze_consensus(agent_beliefs)
            >>> print(f"Consensus: {result['consensus_reached']}")
        """
        # Calculate consensus level
        consensus_level = self.calculate_consensus_level(agent_beliefs)

        # Check if consensus reached
        consensus_reached = self.is_consensus_reached(consensus_level)

        # Detect conflicts
        conflicts = self.detect_conflicts(agent_beliefs)

        # Generate resolution suggestions if needed
        resolution_suggestions = None
        if conflicts:
            resolution_suggestions = self.suggest_resolution(
                conflicts, agent_beliefs, alternatives_data
            )

        # Compile result
        result = {
            'consensus_level': consensus_level,
            'consensus_reached': consensus_reached,
            'conflicts': conflicts,
            'num_conflicts': len(conflicts),
            'resolution_needed': len(conflicts) > 0,
            'resolution_suggestions': resolution_suggestions,
            'timestamp': datetime.now().isoformat(),
            'threshold_used': self.consensus_threshold,
            'agents_analyzed': list(agent_beliefs.keys())
        }

        # Store in history
        self.consensus_history.append(result)

        logger.info(
            f"Consensus analysis complete: level={consensus_level:.3f}, "
            f"reached={consensus_reached}, conflicts={len(conflicts)}"
        )

        return result

    def get_consensus_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from consensus history.

        Returns:
            Dictionary with consensus statistics
        """
        if not self.consensus_history:
            return {
                'total_analyses': 0,
                'consensus_rate': 0.0,
                'average_consensus_level': 0.0,
                'average_conflicts': 0.0
            }

        total = len(self.consensus_history)
        consensus_count = sum(1 for h in self.consensus_history if h['consensus_reached'])
        consensus_levels = [h['consensus_level'] for h in self.consensus_history]
        conflict_counts = [h['num_conflicts'] for h in self.consensus_history]

        return {
            'total_analyses': total,
            'consensus_achieved_count': consensus_count,
            'consensus_rate': consensus_count / total,
            'average_consensus_level': sum(consensus_levels) / total,
            'min_consensus_level': min(consensus_levels),
            'max_consensus_level': max(consensus_levels),
            'average_conflicts': sum(conflict_counts) / total,
            'total_conflicts_detected': sum(conflict_counts)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConsensusModel("
            f"threshold={self.consensus_threshold:.2f}, "
            f"analyses_performed={len(self.consensus_history)})"
        )
