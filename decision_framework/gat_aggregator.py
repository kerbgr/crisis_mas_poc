"""
Graph Attention Network (GAT) Aggregator for Crisis Management MAS
Master's Thesis - Advanced Decision Aggregation using Graph Neural Networks

This module implements a Graph Attention Network to aggregate expert agent beliefs
by modeling the multi-agent system as a graph where:
- Nodes = Expert agents
- Edges = Trust/similarity relationships
- Attention = Dynamic importance weighting based on context

Unlike simple weighted averaging, GAT learns to weight agents based on:
1. Agent expertise relevance to the scenario
2. Confidence in their assessments
3. Agreement with other agents
4. Trust relationships in the network

Reference:
Veličković et al. (2018) "Graph Attention Networks" ICLR 2018
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class GraphAttentionLayer:
    """
    Single Graph Attention Layer for agent network.

    Computes attention coefficients α_ij indicating how much agent i should
    attend to agent j when making decisions.

    Attention mechanism:
        α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))

    Where:
        h_i = feature vector for agent i
        W = learnable weight matrix (here: hand-crafted rules)
        a = attention weight vector
        || = concatenation
    """

    def __init__(
        self,
        feature_dim: int = 9,
        attention_heads: int = 4,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.1
    ):
        """
        Initialize Graph Attention Layer.

        Args:
            feature_dim: Dimension of agent feature vectors
            attention_heads: Number of parallel attention heads
            leaky_relu_slope: Negative slope for LeakyReLU activation
            dropout: Dropout probability (for training, not used in inference)
        """
        self.feature_dim = feature_dim
        self.attention_heads = attention_heads
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = dropout

        logger.info(
            f"GraphAttentionLayer initialized: "
            f"features={feature_dim}, heads={attention_heads}"
        )

    def extract_agent_features(
        self,
        agent_id: str,
        assessment: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract feature vector for an agent.

        Features include:
        1. Confidence score
        2. Belief certainty (inverse entropy)
        3. Expertise relevance (how well expertise matches scenario)
        4. Risk tolerance
        5. Severity awareness
        6. Top choice strength
        7. Thoroughness (number of concerns)
        8. Reasoning quality
        9. Historical reliability

        Args:
            agent_id: Agent identifier
            assessment: Agent's assessment dictionary
            scenario: Current scenario

        Returns:
            Feature vector (numpy array of size feature_dim)
        """
        features = []

        # Feature 1: Confidence
        confidence = assessment.get('confidence', 0.5)
        features.append(confidence)

        # Feature 2: Belief certainty (inverse entropy)
        beliefs = assessment.get('belief_distribution', {})
        if beliefs:
            probs = np.array(list(beliefs.values()))
            probs = probs[probs > 0]  # Filter zeros
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(beliefs))
            certainty = 1.0 - (entropy / (max_entropy + 1e-10))
        else:
            certainty = 0.0
        features.append(certainty)

        # Feature 3: Expertise relevance
        expertise = assessment.get('expertise', '').lower()
        scenario_type = scenario.get('type', '').lower()
        scenario_tags = [t.lower() for t in scenario.get('tags', [])]

        # Check if expertise matches scenario
        relevance = 0.5  # Default
        if expertise in scenario_type or scenario_type in expertise:
            relevance = 1.0
        elif any(tag in expertise for tag in scenario_tags):
            relevance = 0.8
        features.append(relevance)

        # Feature 4: Risk tolerance (higher = more aggressive)
        risk_tolerance = assessment.get('risk_tolerance', 0.5)
        features.append(risk_tolerance)

        # Feature 5: Severity awareness (does assessment mention severity?)
        severity = scenario.get('severity', 0.5)
        features.append(severity)

        # Feature 6: Top choice strength (difference between top and second)
        if beliefs and len(beliefs) >= 2:
            sorted_beliefs = sorted(beliefs.values(), reverse=True)
            top_choice_strength = sorted_beliefs[0] - sorted_beliefs[1]
        else:
            top_choice_strength = 0.0
        features.append(top_choice_strength)

        # Feature 7: Number of key concerns (indicator of thoroughness)
        num_concerns = len(assessment.get('key_concerns', []))
        concern_score = min(num_concerns / 5.0, 1.0)  # Normalize to [0,1]
        features.append(concern_score)

        # Feature 8: Reasoning quality (length as proxy)
        reasoning = assessment.get('reasoning', '')
        reasoning_quality = min(len(reasoning) / 500.0, 1.0)  # Normalize
        features.append(reasoning_quality)

        # Feature 9: Historical reliability score
        reliability_score = assessment.get('reliability_score', 0.8)  # Default 0.8
        features.append(reliability_score)

        # Ensure we have exactly feature_dim features
        features = features[:self.feature_dim]
        while len(features) < self.feature_dim:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def compute_attention_coefficients(
        self,
        features: Dict[str, np.ndarray],
        adjacency: np.ndarray
    ) -> np.ndarray:
        """
        Compute attention coefficients between all agent pairs.

        For each edge (i, j), compute how much agent i should attend to agent j.

        Args:
            features: Dictionary mapping agent_id to feature vector
            adjacency: Adjacency matrix (1 if connected, 0 otherwise)

        Returns:
            Attention matrix (N x N) where α[i,j] = attention from i to j
        """
        agent_ids = list(features.keys())
        n_agents = len(agent_ids)

        if n_agents == 0:
            return np.zeros((0, 0))

        # Stack features into matrix
        feature_matrix = np.stack([features[aid] for aid in agent_ids])

        # Compute pairwise attention scores
        attention_logits = np.zeros((n_agents, n_agents))

        for i in range(n_agents):
            for j in range(n_agents):
                if adjacency[i, j] == 0:
                    attention_logits[i, j] = -np.inf
                    continue

                # Concatenate features
                f_i = feature_matrix[i]
                f_j = feature_matrix[j]

                # Simple attention: weighted dot product
                # Higher confidence + relevance → higher attention
                confidence_weight = f_j[0]  # Confidence
                relevance_weight = f_j[2]   # Expertise relevance
                certainty_weight = f_j[1]   # Certainty

                # Attention score combines multiple factors
                score = (
                    0.4 * confidence_weight +
                    0.3 * relevance_weight +
                    0.3 * certainty_weight
                )

                # Add similarity bonus (cosine similarity of features)
                similarity = np.dot(f_i, f_j) / (
                    np.linalg.norm(f_i) * np.linalg.norm(f_j) + 1e-10
                )
                score += 0.2 * max(similarity, 0)  # Bonus for agreement

                # LeakyReLU activation
                if score < 0:
                    score = self.leaky_relu_slope * score

                attention_logits[i, j] = score

        # Softmax normalization per row (each agent's attention sums to 1)
        attention_weights = np.zeros_like(attention_logits)
        for i in range(n_agents):
            row = attention_logits[i]
            if np.all(np.isinf(row)):
                # No neighbors, attend only to self
                attention_weights[i, i] = 1.0
            else:
                # Softmax over valid neighbors
                row_exp = np.exp(row - np.max(row[~np.isinf(row)]))
                row_exp[np.isinf(row)] = 0
                row_sum = np.sum(row_exp) + 1e-10
                attention_weights[i] = row_exp / row_sum

        return attention_weights


class GATAggregator:
    """
    Graph Attention Network-based aggregator for multi-agent decisions.

    Uses GAT to learn dynamic importance weights for agents based on:
    - Agent features (expertise, confidence, etc.)
    - Network structure (trust relationships)
    - Scenario context (severity, type, etc.)

    Benefits over simple weighted averaging:
    1. Context-aware: Weights adapt to scenario
    2. Dynamic: Different agents important for different scenarios
    3. Interpretable: Attention scores show reasoning
    4. Network-aware: Considers trust relationships
    """

    def __init__(
        self,
        num_attention_heads: int = 4,
        use_multi_head: bool = True,
        default_trust: float = 0.8
    ):
        """
        Initialize GAT Aggregator.

        Args:
            num_attention_heads: Number of parallel attention mechanisms
            use_multi_head: Whether to use multi-head attention
            default_trust: Default trust value for unspecified relationships
        """
        self.num_heads = num_attention_heads
        self.use_multi_head = use_multi_head
        self.default_trust = default_trust

        # Create attention layers
        self.attention_layers = [
            GraphAttentionLayer(
                feature_dim=9,  # Updated to include historical reliability
                attention_heads=num_attention_heads
            )
            for _ in range(num_attention_heads if use_multi_head else 1)
        ]

        self.aggregation_history: List[Dict[str, Any]] = []

        logger.info(
            f"GATAggregator initialized: "
            f"heads={num_attention_heads}, multi_head={use_multi_head}"
        )

    def build_adjacency_matrix(
        self,
        agent_ids: List[str],
        trust_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ) -> np.ndarray:
        """
        Build adjacency matrix for agent network.

        Args:
            agent_ids: List of agent identifiers
            trust_matrix: Optional trust relationships (agent_i -> agent_j -> trust_value)

        Returns:
            Adjacency matrix (N x N) with values in [0, 1] representing trust
        """
        n = len(agent_ids)
        adjacency = np.ones((n, n)) * self.default_trust  # Default: trust all

        # Set self-loops to 1.0
        np.fill_diagonal(adjacency, 1.0)

        # Apply custom trust values if provided
        if trust_matrix:
            for i, agent_i in enumerate(agent_ids):
                if agent_i in trust_matrix:
                    for j, agent_j in enumerate(agent_ids):
                        if agent_j in trust_matrix[agent_i]:
                            adjacency[i, j] = trust_matrix[agent_i][agent_j]

        return adjacency

    def aggregate_beliefs_with_gat(
        self,
        agent_assessments: Dict[str, Dict[str, Any]],
        scenario: Dict[str, Any],
        trust_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate agent beliefs using Graph Attention Network.

        Args:
            agent_assessments: Dictionary of agent assessments
            scenario: Current scenario dictionary
            trust_matrix: Optional trust relationships between agents

        Returns:
            Aggregation result containing:
                - aggregated_beliefs: Combined belief distribution
                - attention_weights: Learned attention coefficients
                - confidence: Overall confidence score
                - explanation: Human-readable explanation
        """
        start_time = datetime.now()

        try:
            if not agent_assessments:
                logger.error("No agent assessments provided to GAT aggregator")
                return self._create_empty_result()

            agent_ids = list(agent_assessments.keys())
            n_agents = len(agent_ids)

            logger.info(f"GAT aggregating beliefs from {n_agents} agents")

            # Step 1: Extract features for each agent
            features = {}
            for agent_id, assessment in agent_assessments.items():
                features[agent_id] = self.attention_layers[0].extract_agent_features(
                    agent_id,
                    assessment,
                    scenario
                )

            # Step 2: Build adjacency matrix
            adjacency = self.build_adjacency_matrix(agent_ids, trust_matrix)

            # Step 3: Compute attention coefficients
            if self.use_multi_head:
                # Multi-head attention: average across heads
                all_attention = []
                for layer in self.attention_layers:
                    att = layer.compute_attention_coefficients(features, adjacency)
                    all_attention.append(att)

                # Average attention across heads
                attention_weights = np.mean(all_attention, axis=0)
            else:
                # Single head
                attention_weights = self.attention_layers[0].compute_attention_coefficients(
                    features, adjacency
                )

            # Step 4: Aggregate beliefs using attention weights
            # For each alternative, compute weighted sum of beliefs
            all_alternatives = set()
            for assessment in agent_assessments.values():
                beliefs = assessment.get('belief_distribution', {})
                all_alternatives.update(beliefs.keys())

            aggregated_beliefs = {}
            for alt in all_alternatives:
                weighted_sum = 0.0
                total_weight = 0.0

                for i, agent_id in enumerate(agent_ids):
                    assessment = agent_assessments[agent_id]
                    beliefs = assessment.get('belief_distribution', {})
                    belief_value = beliefs.get(alt, 0.0)

                    # Use attention weight from self (diagonal)
                    agent_weight = attention_weights[i, i]

                    weighted_sum += agent_weight * belief_value
                    total_weight += agent_weight

                if total_weight > 0:
                    aggregated_beliefs[alt] = weighted_sum / total_weight
                else:
                    aggregated_beliefs[alt] = 0.0

            # Normalize to sum to 1.0
            total = sum(aggregated_beliefs.values())
            if total > 0:
                aggregated_beliefs = {
                    k: v / total for k, v in aggregated_beliefs.items()
                }

            # Step 5: Compute overall confidence
            # Weight agent confidences by attention
            overall_confidence = 0.0
            for i, agent_id in enumerate(agent_ids):
                agent_confidence = agent_assessments[agent_id].get('confidence', 0.5)
                agent_weight = attention_weights[i, i]
                overall_confidence += agent_weight * agent_confidence

            # Step 6: Compute uncertainty (entropy of aggregated beliefs)
            probs = np.array(list(aggregated_beliefs.values()))
            probs = probs[probs > 0]
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(aggregated_beliefs))
                uncertainty = entropy / (max_entropy + 1e-10)
            else:
                uncertainty = 1.0

            # Step 7: Generate explanation
            explanation = self._generate_explanation(
                agent_ids,
                attention_weights,
                aggregated_beliefs,
                agent_assessments
            )

            # Build result
            result = {
                'aggregated_beliefs': aggregated_beliefs,
                'attention_weights': {
                    agent_ids[i]: {
                        agent_ids[j]: float(attention_weights[i, j])
                        for j in range(n_agents)
                    }
                    for i in range(n_agents)
                },
                'confidence': overall_confidence,
                'uncertainty': uncertainty,
                'method': 'GAT',
                'num_agents': n_agents,
                'num_heads': self.num_heads if self.use_multi_head else 1,
                'explanation': explanation,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }

            # Store in history
            self.aggregation_history.append(result)

            logger.info(
                f"GAT aggregation completed: confidence={overall_confidence:.3f}, "
                f"uncertainty={uncertainty:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in GAT aggregation: {e}", exc_info=True)
            return self._create_empty_result()

    def _generate_explanation(
        self,
        agent_ids: List[str],
        attention_weights: np.ndarray,
        aggregated_beliefs: Dict[str, float],
        agent_assessments: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable explanation of aggregation.

        Args:
            agent_ids: List of agent IDs
            attention_weights: Attention weight matrix
            aggregated_beliefs: Aggregated belief distribution
            agent_assessments: Original agent assessments

        Returns:
            Explanation string
        """
        lines = ["GAT Aggregation Summary:", ""]

        # Top recommendation
        if aggregated_beliefs:
            top_alt = max(aggregated_beliefs.items(), key=lambda x: x[1])
            lines.append(f"Top Recommendation: {top_alt[0]} (belief: {top_alt[1]:.3f})")
            lines.append("")

        # Agent importance (self-attention weights)
        lines.append("Agent Importance (Self-Attention):")
        agent_importance = [(agent_ids[i], attention_weights[i, i])
                           for i in range(len(agent_ids))]
        agent_importance.sort(key=lambda x: x[1], reverse=True)

        for agent_id, weight in agent_importance:
            agent_name = agent_assessments[agent_id].get('agent_name', agent_id)
            lines.append(f"  - {agent_name}: {weight:.3f}")

        lines.append("")
        lines.append("The attention mechanism dynamically weighted agents based on:")
        lines.append("  • Expertise relevance to the scenario")
        lines.append("  • Confidence in their assessments")
        lines.append("  • Certainty of their beliefs")
        lines.append("  • Agreement with other agents")

        return "\n".join(lines)

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when aggregation fails."""
        return {
            'aggregated_beliefs': {},
            'attention_weights': {},
            'confidence': 0.0,
            'uncertainty': 1.0,
            'method': 'GAT',
            'error': True,
            'explanation': 'GAT aggregation failed',
            'timestamp': datetime.now().isoformat()
        }

    def get_attention_summary(
        self,
        attention_weights: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Get summary of attention weights (self-attention only).

        Args:
            attention_weights: Full attention weight dictionary

        Returns:
            Dictionary mapping agent_id to self-attention weight
        """
        return {
            agent_i: weights.get(agent_i, 0.0)
            for agent_i, weights in attention_weights.items()
        }

    def __repr__(self) -> str:
        return (
            f"GATAggregator(heads={self.num_heads}, "
            f"multi_head={self.use_multi_head}, "
            f"aggregations={len(self.aggregation_history)})"
        )
