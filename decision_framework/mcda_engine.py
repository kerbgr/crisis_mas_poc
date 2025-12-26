"""
Multi-Criteria Decision Analysis (MCDA) Engine - Alternative Ranking and Evaluation

OBJECTIVE:
This module implements a Multi-Criteria Decision Analysis engine for evaluating and
ranking crisis response alternatives across multiple competing criteria. It provides
a systematic, transparent method for comparing complex alternatives when no single
criterion dominates (e.g., safety vs. cost vs. speed trade-offs).

WHY THIS EXISTS:
Crisis management decisions involve inherent trade-offs:
- **Safety vs. Cost**: Safest option may be prohibitively expensive
- **Speed vs. Quality**: Fastest response may compromise effectiveness
- **Short-term vs. Long-term**: Immediate relief vs. sustainable solutions
- **Lives vs. Livelihoods**: Public health vs. economic impact

Decision-makers need a structured way to:
1. Compare alternatives across multiple dimensions simultaneously
2. Apply consistent evaluation criteria across different scenarios
3. Weight criteria according to context (e.g., safety weighted higher in severe crises)
4. Explain and justify decisions to stakeholders
5. Perform sensitivity analysis (how rankings change if weights change)

This module provides that structured, auditable decision framework.

METHODOLOGICAL CLASSIFICATION:
- **Approach**: Weighted sum with vector normalization
- **Family**: TOPSIS-inspired (Technique for Order of Preference by Similarity to Ideal Solution)
- **Related Methods**: MAUT (Multi-Attribute Utility Theory), SAW (Simple Additive Weighting)
- **Research Context**: Serves as "παρόμοιες προσεγγίσεις" (similar approaches) to
  Group UTA mentioned in thesis abstract, providing multi-criteria group decision-making
  through weighted aggregation of heterogeneous expert preferences

MATHEMATICAL FOUNDATION:
Weighted Sum with Normalization:

For each alternative A:
    score(A) = Σ [w_i × normalized(v_i)]

Where:
- w_i = weight for criterion i (from criteria_weights.json)
- v_i = raw value for criterion i
- normalized(v_i) = (v_i - min) / (max - min) for benefit criteria
- normalized(v_i) = (max - v_i) / (max - min) for cost criteria

Normalization ensures:
1. All criteria on same scale (0-1)
2. Scale independence (euros vs. hours vs. lives)
3. Benefit criteria: higher is better → higher normalized score
4. Cost criteria: lower is better → higher normalized score

CRITERION TYPES:
1. **Benefit Criteria** (maximize):
   - Safety score: Higher = better
   - Effectiveness: Higher = better
   - Social acceptance: Higher = better
   - Public confidence: Higher = better

2. **Cost Criteria** (minimize):
   - Financial cost: Lower = better
   - Response time: Lower = better
   - Resource consumption: Lower = better
   - Environmental impact: Lower = better

INPUTS (rank_alternatives method):
- alternatives: List[Dict[str, Any]]
  * Each alternative contains:
    - id: str (unique identifier, e.g., "A1", "action_evacuation")
    - name: str (human-readable name)
    - criteria_scores: Dict[criterion_id, float] (0-1 scores, pre-normalized)
      OR
    - estimated_metrics: Dict[metric_name, value] (raw values to be normalized)

- custom_weights: Optional[Dict[criterion_id, float]]
  * Override default weights from criteria_weights.json
  * Useful for sensitivity analysis or expert-specific weighting

Example Alternative:
```json
{
  "id": "A1",
  "name": "Immediate Evacuation",
  "criteria_scores": {
    "effectiveness": 0.85,
    "safety": 0.90,
    "speed": 0.75,
    "cost": 0.40,
    "public_acceptance": 0.70
  }
}
```

OUTPUTS (rank_alternatives returns):
List[Tuple[alt_id, weighted_score, normalized_scores]]
- Sorted by weighted_score (descending, best first)
- alt_id: Alternative identifier
- weighted_score: Overall score (0-1)
- normalized_scores: Dict[criterion_id, float] - individual criterion scores

Example Output:
```python
[
  ("A1", 0.782, {"safety": 0.90, "cost": 0.40, ...}),  # Rank 1
  ("A3", 0.645, {"safety": 0.85, "cost": 0.45, ...}),  # Rank 2
  ("A2", 0.523, {"safety": 0.70, "cost": 0.70, ...})   # Rank 3
]
```

KEY FEATURES:
1. **Dual Input Support**:
   - Direct criteria_scores (pre-normalized, from scenarios)
   - Raw estimated_metrics (auto-normalized across alternatives)

2. **Automatic Normalization**:
   - Handles different scales (euros, hours, percentages)
   - Respects criterion type (benefit vs. cost)

3. **Sensitivity Analysis**:
   - Test how rankings change with different weights
   - Identify "robust" winners (stable across weight profiles)
   - Find weight thresholds where winners change

4. **Weight Profile Comparison**:
   - Compare rankings under different stakeholder perspectives
   - Medical-focused vs. cost-focused vs. speed-focused

5. **Explanation Generation**:
   - Human-readable ranking explanations
   - Visual bars showing criterion contributions
   - Comparison tables across alternatives

TYPICAL WORKFLOW:
```python
# 1. Initialize with criteria configuration
mcda = MCDAEngine("scenarios/criteria_weights.json")

# 2. Get alternatives from scenario
alternatives = scenario['available_actions']

# 3. Rank alternatives
ranked = mcda.rank_alternatives(alternatives)

# 4. Get top recommendation
winner = ranked[0]
print(f"Winner: {winner[0]} with score {winner[1]:.3f}")

# 5. Generate explanation
explanation = mcda.explain_ranking(ranked, alternatives_data)
print(explanation)

# 6. Sensitivity analysis (optional)
sensitivity = mcda.sensitivity_analysis(
    alternatives,
    criterion_to_vary='safety',
    weight_range=(0.1, 0.6)
)
```

CRITERIA CONFIGURATION (criteria_weights.json):
```json
{
  "decision_criteria": {
    "effectiveness": {
      "name": "Effectiveness",
      "weight": 0.30,
      "type": "benefit",
      "description": "How well the action addresses the crisis"
    },
    "safety": {
      "name": "Safety",
      "weight": 0.25,
      "type": "benefit",
      "description": "Safety for responders and affected population"
    },
    "speed": {
      "name": "Response Speed",
      "weight": 0.20,
      "type": "benefit",
      "description": "How quickly the action can be executed"
    },
    "cost": {
      "name": "Cost",
      "weight": 0.15,
      "type": "cost",
      "description": "Financial and resource cost"
    },
    "public_acceptance": {
      "name": "Public Acceptance",
      "weight": 0.10,
      "type": "benefit",
      "description": "Level of public support"
    }
  }
}
```

DESIGN DECISIONS:
1. **Weighted Sum vs. Full TOPSIS**:
   - Chose weighted sum for simplicity and interpretability
   - Full TOPSIS adds ideal/anti-ideal distance calculations (more complex)
   - Weighted sum sufficient for crisis decision-making

2. **Vector Normalization**:
   - Ensures scale independence
   - Alternative: Z-score normalization (less interpretable)

3. **Equal Weights Fallback**:
   - If criterion not in config → use 1/N equal weight
   - Handles scenario-specific criteria gracefully

4. **History Tracking**:
   - Maintains analysis_history for auditing
   - Useful for learning and improvement

ERROR HANDLING:
- FileNotFoundError: Criteria weights file missing
- json.JSONDecodeError: Malformed criteria configuration
- ValueError: Missing required fields (name, weight, type)
- ValueError: Invalid criterion type (must be 'benefit' or 'cost')
- Graceful defaults: Missing metrics → 0.5 (neutral)

PERFORMANCE:
- Time Complexity: O(A × C)
  * A = number of alternatives (typically 3-10)
  * C = number of criteria (typically 5-8)
- Space Complexity: O(A × C)
- Typical Runtime: < 1ms for A=5, C=7
- Bottleneck: Sensitivity analysis (O(A × C × S) where S=steps)

INTEGRATION POINTS:
- Called by: CoordinatorAgent after belief aggregation
- Inputs from: Scenario JSON (criteria_scores) or agent assessments (estimated_metrics)
- Outputs to: Decision quality calculation in evaluation/metrics.py
- Related: Evidential Reasoning produces the "recommended" alternative to score

SENSITIVITY ANALYSIS USE CASES:
1. **Robustness Testing**: Does winner change if safety weight varies ±10%?
2. **Stakeholder Alignment**: Do all stakeholder weight profiles agree on winner?
3. **Critical Weight Identification**: At what weight threshold does ranking flip?
4. **Consensus Building**: Find weights where all stakeholders accept result

LIMITATIONS:
1. Assumes criteria independence (no interaction effects)
2. Linear aggregation (doesn't capture threshold effects)
3. Static weights (no dynamic adaptation to scenario)
4. No uncertainty propagation (scores treated as certain)
5. Ordinal rankings only (score differences may not be meaningful)

VALIDATION:
Unit tests verify:
- Benefit criterion normalization (higher raw → higher score)
- Cost criterion normalization (lower raw → higher score)
- Weight application correctness
- Ranking stability
- Sensitivity analysis accuracy

RELATED RESEARCH:
- TOPSIS: Hwang & Yoon (1981)
- MAUT: Keeney & Raiffa (1976)
- SAW: MacCrimon (1968)
- Multi-criteria decision analysis in emergency management

COMPARISON WITH OTHER MCDA METHODS:
| Method | Complexity | Interpretability | Best For |
|--------|-----------|------------------|----------|
| Weighted Sum (This) | Low | High | Crisis decisions, transparent process |
| Full TOPSIS | Medium | Medium | Engineering design, optimization |
| AHP | High | Medium | Hierarchical criteria, pairwise comparisons |
| ELECTRE | High | Low | Outranking problems, incomparability |
| PROMETHEE | Medium | Medium | Preference modeling, partial rankings |
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCDAEngine:
    """
    Multi-Criteria Decision Analysis engine for crisis management decisions.

    Evaluates and ranks alternatives based on multiple criteria with different
    importance weights. Supports both benefit and cost criteria.

    Example:
        >>> mcda = MCDAEngine("scenarios/criteria_weights.json")
        >>> alternatives = [
        ...     {
        ...         "id": "A1",
        ...         "name": "Immediate Evacuation",
        ...         "estimated_metrics": {
        ...             "safety_score": 0.95,
        ...             "cost_euros": 450000,
        ...             "response_time_hours": 2,
        ...             "social_disruption": 0.8
        ...         }
        ...     }
        ... ]
        >>> ranked = mcda.rank_alternatives(alternatives)
    """

    def __init__(self, criteria_weights_path: str = "scenarios/criteria_weights.json"):
        """
        Initialize the MCDA engine.

        Args:
            criteria_weights_path: Path to JSON file containing criteria weights
                                  and configuration

        Raises:
            FileNotFoundError: If criteria weights file doesn't exist
        """
        self.criteria_weights_path = criteria_weights_path
        self.criteria_config = self.load_criteria_weights()
        self.analysis_history: List[Dict[str, Any]] = []

        logger.info(f"MCDA Engine initialized with {len(self.criteria_config)} criteria")

    def load_criteria_weights(self) -> Dict[str, Any]:
        """
        Load criteria weights and configuration from JSON file.

        Returns:
            Dictionary containing criteria configuration with weights, types, and descriptions

        Raises:
            FileNotFoundError: If the criteria file doesn't exist
            json.JSONDecodeError: If the JSON is malformed
            ValueError: If required fields are missing

        Example:
            >>> mcda = MCDAEngine()
            >>> config = mcda.load_criteria_weights()
            >>> print(config['safety'])
            {'name': 'Safety', 'weight': 0.30, 'type': 'benefit', ...}
        """
        path = Path(self.criteria_weights_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Criteria weights file not found: {path}\n"
                f"Please ensure the file exists at the specified location."
            )

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            criteria = data.get('decision_criteria', {})

            if not criteria:
                raise ValueError("No decision criteria found in configuration file")

            # Validate criteria structure
            for criterion_id, config in criteria.items():
                required_fields = ['name', 'weight', 'type']
                for field in required_fields:
                    if field not in config:
                        raise ValueError(
                            f"Criterion '{criterion_id}' missing required field: {field}"
                        )

                if config['type'] not in ['benefit', 'cost']:
                    raise ValueError(
                        f"Criterion '{criterion_id}' has invalid type: {config['type']}. "
                        f"Must be 'benefit' or 'cost'."
                    )

            logger.info(f"Loaded {len(criteria)} criteria from {path}")
            return criteria

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Malformed JSON in criteria file: {path}",
                e.doc,
                e.pos
            )

    def normalize_score(
        self,
        value: float,
        min_val: float,
        max_val: float,
        criterion_type: str
    ) -> float:
        """
        Normalize a criterion value to [0, 1] range.

        For benefit criteria: higher values are better (normalized to higher scores)
        For cost criteria: lower values are better (normalized to higher scores)

        Args:
            value: The value to normalize
            min_val: Minimum value in the dataset
            max_val: Maximum value in the dataset
            criterion_type: Either 'benefit' or 'cost'

        Returns:
            Normalized score between 0.0 and 1.0

        Example:
            >>> mcda = MCDAEngine()
            >>> # Benefit criterion (higher is better)
            >>> mcda.normalize_score(8, 5, 10, 'benefit')
            0.6
            >>> # Cost criterion (lower is better)
            >>> mcda.normalize_score(8, 5, 10, 'cost')
            0.4
        """
        # Handle edge case where all values are the same
        if max_val == min_val:
            return 1.0

        if criterion_type == 'benefit':
            # Higher is better: normalize directly
            normalized = (value - min_val) / (max_val - min_val)
        elif criterion_type == 'cost':
            # Lower is better: invert the normalization
            normalized = (max_val - value) / (max_val - min_val)
        else:
            raise ValueError(f"Invalid criterion type: {criterion_type}. Must be 'benefit' or 'cost'.")

        # Ensure value is in [0, 1] range
        return max(0.0, min(1.0, normalized))

    def score_alternative(
        self,
        alternative_data: Dict[str, Any],
        all_alternatives: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate normalized scores for an alternative across all criteria.

        Args:
            alternative_data: Dictionary containing alternative metrics
            all_alternatives: List of all alternatives (needed for normalization)

        Returns:
            Dictionary mapping criterion names to normalized scores

        Example:
            >>> alternative = {
            ...     "id": "A1",
            ...     "estimated_metrics": {
            ...         "safety_score": 0.95,
            ...         "cost_euros": 450000
            ...     }
            ... }
            >>> scores = mcda.score_alternative(alternative, all_alternatives)
            >>> print(scores)
            {'safety': 0.95, 'cost': 0.85}
        """
        scores = {}

        # Check for criteria_scores first (from scenario JSON), then estimated_metrics
        if 'criteria_scores' in alternative_data:
            # Direct use of criteria_scores from scenario JSON
            # These are already normalized 0-1 scores, all benefit-oriented (higher is better)
            criteria_scores = alternative_data['criteria_scores']

            # Use scores directly - scenario criteria are already properly scored
            for criterion_name, score in criteria_scores.items():
                scores[criterion_name] = float(score)

            logger.debug(f"Using criteria_scores directly: {scores}")
            return scores

        # Fall back to estimated_metrics format (original behavior)
        metrics = alternative_data.get('estimated_metrics', {})

        # Map metric names to criterion IDs
        metric_mapping = {
            'safety_score': 'safety',
            'cost_euros': 'cost',
            'response_time_hours': 'response_time',
            'social_disruption': 'social_acceptance'  # Lower disruption = higher acceptance
        }

        for metric_name, criterion_id in metric_mapping.items():
            if criterion_id not in self.criteria_config:
                continue

            if metric_name not in metrics:
                # Use default mid-point if metric missing
                scores[criterion_id] = 0.5
                continue

            # Get criterion configuration
            criterion_config = self.criteria_config[criterion_id]
            criterion_type = criterion_config['type']

            # Collect all values for this metric across alternatives
            all_values = []
            for alt in all_alternatives:
                alt_metrics = alt.get('estimated_metrics', {})
                if metric_name in alt_metrics:
                    all_values.append(alt_metrics[metric_name])

            if not all_values:
                scores[criterion_id] = 0.5
                continue

            min_val = min(all_values)
            max_val = max(all_values)
            current_val = metrics[metric_name]

            # Special handling for social_disruption (inverse of acceptance)
            if metric_name == 'social_disruption':
                # Lower disruption = higher acceptance, so treat as cost
                normalized = self.normalize_score(current_val, min_val, max_val, 'cost')
            else:
                normalized = self.normalize_score(current_val, min_val, max_val, criterion_type)

            scores[criterion_id] = normalized

        return scores

    def calculate_weighted_score(
        self,
        normalized_scores: Dict[str, float],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted overall score for an alternative.

        Args:
            normalized_scores: Dictionary of normalized scores per criterion
            custom_weights: Optional custom weights (overrides config weights)

        Returns:
            Weighted score between 0.0 and 1.0

        Example:
            >>> scores = {'safety': 0.9, 'cost': 0.7, 'response_time': 0.8}
            >>> weighted = mcda.calculate_weighted_score(scores)
            0.82
        """
        # Use custom weights if provided, otherwise use config weights
        if custom_weights:
            weights = custom_weights
        else:
            weights = {
                cid: config['weight']
                for cid, config in self.criteria_config.items()
            }

        weighted_sum = 0.0
        total_weight = 0.0

        # For criteria not in weights (e.g., from scenario criteria_scores),
        # use equal weight
        num_criteria = len(normalized_scores)
        default_weight = 1.0 / num_criteria if num_criteria > 0 else 0.0

        for criterion_id, score in normalized_scores.items():
            if criterion_id in weights:
                weight = weights[criterion_id]
            else:
                # Use default equal weight for criteria not in config
                weight = default_weight
                logger.debug(f"Using default weight {weight:.3f} for criterion '{criterion_id}'")

            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def rank_alternatives(
        self,
        alternatives: List[Dict[str, Any]],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Rank alternatives based on MCDA scoring.

        Args:
            alternatives: List of alternative dictionaries with id and estimated_metrics
            custom_weights: Optional custom criterion weights

        Returns:
            List of tuples (alternative_id, weighted_score, normalized_scores),
            sorted by score descending

        Example:
            >>> alternatives = [
            ...     {"id": "A1", "estimated_metrics": {...}},
            ...     {"id": "A2", "estimated_metrics": {...}}
            ... ]
            >>> ranked = mcda.rank_alternatives(alternatives)
            >>> print(f"Winner: {ranked[0][0]} with score {ranked[0][1]:.3f}")
        """
        if not alternatives:
            return []

        results = []

        for alternative in alternatives:
            alt_id = alternative.get('id', 'unknown')

            # Calculate normalized scores
            normalized_scores = self.score_alternative(alternative, alternatives)

            # Calculate weighted score
            weighted_score = self.calculate_weighted_score(normalized_scores, custom_weights)

            results.append((alt_id, weighted_score, normalized_scores))

        # Sort by weighted score (descending)
        ranked = sorted(results, key=lambda x: x[1], reverse=True)

        # Log to history
        self.analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'num_alternatives': len(alternatives),
            'winner': ranked[0][0] if ranked else None,
            'winner_score': ranked[0][1] if ranked else 0.0,
            'method': 'weighted_sum'
        })

        logger.info(
            f"Ranked {len(alternatives)} alternatives. "
            f"Winner: {ranked[0][0]} with score {ranked[0][1]:.3f}"
        )

        return ranked

    def explain_ranking(
        self,
        ranked_alternatives: List[Tuple[str, float, Dict[str, float]]],
        alternatives_data: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable explanation of the ranking.

        Args:
            ranked_alternatives: Output from rank_alternatives()
            alternatives_data: Dictionary mapping alternative IDs to full data

        Returns:
            Formatted explanation string

        Example:
            >>> ranked = mcda.rank_alternatives(alternatives)
            >>> explanation = mcda.explain_ranking(ranked, alt_data)
            >>> print(explanation)
        """
        if not ranked_alternatives:
            return "No alternatives to explain."

        explanation = []
        explanation.append("="*70)
        explanation.append("MCDA RANKING EXPLANATION")
        explanation.append("="*70)

        # Criteria weights
        explanation.append("\nCriteria Weights:")
        for criterion_id, config in self.criteria_config.items():
            explanation.append(
                f"  • {config['name']:25s} : {config['weight']:.2f} ({config['type']})"
            )

        explanation.append(f"\nTotal Alternatives Evaluated: {len(ranked_alternatives)}")
        explanation.append("\n" + "="*70)

        # Detailed ranking
        for rank, (alt_id, weighted_score, normalized_scores) in enumerate(ranked_alternatives, 1):
            alt_data = alternatives_data.get(alt_id, {})
            alt_name = alt_data.get('name', alt_id)

            explanation.append(f"\nRANK #{rank}: {alt_name} (ID: {alt_id})")
            explanation.append(f"Overall Score: {weighted_score:.3f}")

            # Show contribution of each criterion
            explanation.append("\nCriterion Contributions:")
            for criterion_id, config in self.criteria_config.items():
                if criterion_id in normalized_scores:
                    norm_score = normalized_scores[criterion_id]
                    weight = config['weight']
                    contribution = norm_score * weight

                    # Get raw metric if available
                    raw_value = ""
                    metrics = alt_data.get('estimated_metrics', {})
                    metric_mapping_reverse = {
                        'safety': 'safety_score',
                        'cost': 'cost_euros',
                        'response_time': 'response_time_hours',
                        'social_acceptance': 'social_disruption'
                    }
                    metric_name = metric_mapping_reverse.get(criterion_id)
                    if metric_name and metric_name in metrics:
                        raw_value = f" (raw: {metrics[metric_name]})"

                    bar_length = int(norm_score * 30)
                    bar = '█' * bar_length

                    explanation.append(
                        f"  • {config['name']:25s}: "
                        f"norm={norm_score:.3f}, "
                        f"weight={weight:.2f}, "
                        f"contrib={contribution:.3f} {bar}{raw_value}"
                    )

            explanation.append("-"*70)

        # Summary comparison
        explanation.append("\n" + "="*70)
        explanation.append("SUMMARY COMPARISON")
        explanation.append("="*70)

        for criterion_id, config in self.criteria_config.items():
            explanation.append(f"\n{config['name']}:")

            # Show scores for all alternatives
            for rank, (alt_id, _, normalized_scores) in enumerate(ranked_alternatives, 1):
                if criterion_id in normalized_scores:
                    score = normalized_scores[criterion_id]
                    alt_name = alternatives_data.get(alt_id, {}).get('name', alt_id)
                    bar_length = int(score * 40)
                    bar = '█' * bar_length
                    explanation.append(f"  {rank}. {alt_name:30s}: {score:.3f} {bar}")

        return "\n".join(explanation)

    def sensitivity_analysis(
        self,
        alternatives: List[Dict[str, Any]],
        criterion_to_vary: str,
        weight_range: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 11
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis by varying one criterion's weight.

        Args:
            alternatives: List of alternatives to analyze
            criterion_to_vary: ID of criterion to vary
            weight_range: Range of weights to test (min, max)
            num_steps: Number of steps in the range

        Returns:
            Dictionary with sensitivity analysis results including:
                - weight_values: List of weight values tested
                - rankings: Rankings at each weight value
                - winner_changes: Points where the winner changes
                - stability_score: Measure of ranking stability

        Example:
            >>> sensitivity = mcda.sensitivity_analysis(
            ...     alternatives, 'safety', weight_range=(0.1, 0.6)
            ... )
            >>> print(f"Stability: {sensitivity['stability_score']:.2f}")
        """
        if criterion_to_vary not in self.criteria_config:
            raise ValueError(
                f"Criterion '{criterion_to_vary}' not found in configuration. "
                f"Available: {list(self.criteria_config.keys())}"
            )

        import numpy as np

        weight_values = np.linspace(weight_range[0], weight_range[1], num_steps)
        results = []
        winners = []

        original_weight = self.criteria_config[criterion_to_vary]['weight']

        for weight in weight_values:
            # Create custom weights with varied criterion
            custom_weights = {
                cid: config['weight']
                for cid, config in self.criteria_config.items()
            }
            custom_weights[criterion_to_vary] = weight

            # Normalize other weights proportionally
            other_criteria_sum = sum(
                w for cid, w in custom_weights.items() if cid != criterion_to_vary
            )

            if other_criteria_sum > 0:
                remaining_weight = 1.0 - weight
                for cid in custom_weights:
                    if cid != criterion_to_vary:
                        custom_weights[cid] *= (remaining_weight / other_criteria_sum)

            # Rank with these weights
            ranked = self.rank_alternatives(alternatives, custom_weights)

            winner = ranked[0][0] if ranked else None
            winners.append(winner)

            results.append({
                'weight': float(weight),
                'winner': winner,
                'winner_score': ranked[0][1] if ranked else 0.0,
                'full_ranking': ranked
            })

        # Analyze winner changes
        winner_changes = []
        for i in range(1, len(winners)):
            if winners[i] != winners[i-1]:
                winner_changes.append({
                    'weight_threshold': (weight_values[i-1] + weight_values[i]) / 2,
                    'previous_winner': winners[i-1],
                    'new_winner': winners[i]
                })

        # Calculate stability score
        unique_winners = len(set(winners))
        stability_score = 1.0 - (unique_winners - 1) / len(winners) if len(winners) > 1 else 1.0

        return {
            'criterion_varied': criterion_to_vary,
            'criterion_name': self.criteria_config[criterion_to_vary]['name'],
            'original_weight': original_weight,
            'weight_values': [float(w) for w in weight_values],
            'results': results,
            'winner_changes': winner_changes,
            'stability_score': stability_score,
            'num_winner_changes': len(winner_changes),
            'unique_winners': list(set(winners))
        }

    def compare_weight_profiles(
        self,
        alternatives: List[Dict[str, Any]],
        profiles: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compare rankings under different weight profiles.

        Args:
            alternatives: List of alternatives to analyze
            profiles: Dictionary of weight profiles
                     (e.g., {"medical_focused": {...}, "cost_focused": {...}})

        Returns:
            Dictionary with comparison results

        Example:
            >>> profiles = {
            ...     "medical": {"safety": 0.5, "cost": 0.2, ...},
            ...     "cost": {"safety": 0.2, "cost": 0.5, ...}
            ... }
            >>> comparison = mcda.compare_weight_profiles(alternatives, profiles)
        """
        results = {}

        for profile_name, weights in profiles.items():
            ranked = self.rank_alternatives(alternatives, weights)
            results[profile_name] = {
                'ranking': ranked,
                'winner': ranked[0][0] if ranked else None,
                'winner_score': ranked[0][1] if ranked else 0.0
            }

        # Check if all profiles agree on winner
        winners = [r['winner'] for r in results.values()]
        all_agree = len(set(winners)) == 1

        return {
            'profiles_compared': list(profiles.keys()),
            'results': results,
            'all_agree': all_agree,
            'consensus_winner': winners[0] if all_agree else None,
            'winner_distribution': {w: winners.count(w) for w in set(winners)}
        }

    def get_criteria_info(self) -> Dict[str, Any]:
        """
        Get detailed information about configured criteria.

        Returns:
            Dictionary with criteria information
        """
        return {
            criterion_id: {
                'name': config['name'],
                'weight': config['weight'],
                'type': config['type'],
                'description': config.get('description', ''),
                'unit': config.get('unit', ''),
                'scale': config.get('scale', '')
            }
            for criterion_id, config in self.criteria_config.items()
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MCDAEngine("
            f"criteria={len(self.criteria_config)}, "
            f"analyses_performed={len(self.analysis_history)})"
        )
