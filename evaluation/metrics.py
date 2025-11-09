"""
Performance Metrics Module
Evaluates MAS system performance compared to single-agent baseline

Metrics:
1. Decision Quality Score (DQS) - Weighted criteria satisfaction
2. Consensus Level (CL) - Agent agreement percentage
3. Time to Consensus (TtC) - Iterations and API calls required
4. Confidence Score (CS) - Average confidence and uncertainty
5. Expert Contribution Balance (ECB) - Participation balance
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import scipy.stats as stats


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Comprehensive metrics evaluator for multi-agent crisis management system.

    Calculates performance metrics and compares multi-agent vs single-agent approaches.

    Example:
        >>> evaluator = MetricsEvaluator()
        >>> quality = evaluator.calculate_decision_quality(decision, ground_truth)
        >>> consensus = evaluator.calculate_consensus_metrics(agent_assessments)
        >>> comparison = evaluator.compare_to_baseline(multi_results, single_results)
    """

    def __init__(self):
        """Initialize the metrics evaluator."""
        self.evaluation_history: List[Dict[str, Any]] = []
        logger.info("MetricsEvaluator initialized")

    def calculate_decision_quality(
        self,
        decision: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
        criteria_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Decision Quality Score (DQS).

        Measures how well the decision satisfies weighted criteria.

        Args:
            decision: Decision from CoordinatorAgent.make_final_decision()
            ground_truth: Optional ground truth for comparison
            criteria_weights: Optional custom criteria weights

        Returns:
            Dictionary with DQS metrics:
                - weighted_score: Overall quality score (0-1)
                - criteria_satisfaction: Per-criterion scores
                - ground_truth_match: Match with ground truth if available

        Example:
            >>> dqs = evaluator.calculate_decision_quality(decision)
            >>> print(f"Quality: {dqs['weighted_score']:.2f}")
        """
        logger.info("Calculating decision quality score")

        # Extract scores from decision
        final_scores = decision.get('final_scores', {})
        recommended = decision.get('recommended_alternative')
        confidence = decision.get('confidence', 0.0)

        if not final_scores or not recommended:
            logger.warning("Decision missing required fields for quality calculation")
            return {
                'weighted_score': 0.0,
                'criteria_satisfaction': {},
                'error': 'Insufficient decision data'
            }

        # Calculate weighted score based on criteria satisfaction
        # Option 1: Use criteria_scores if available (single-agent or detailed multi-agent)
        criteria_scores = decision.get('criteria_scores', {})

        # Option 2: Use MCDA scores (multi-agent)
        mcda_scores = decision.get('mcda_scores', {})

        # Option 3: Use final_scores for the recommended alternative
        weighted_score = 0.0
        criteria_satisfaction = {}

        # Try to calculate quality score from available data
        quality_calculated = False

        if criteria_scores and isinstance(criteria_scores, dict) and criteria_scores:
            # Single-agent or detailed criteria scoring
            # criteria_scores can be in two formats:
            # Format 1: {criterion_id: {alt_id: score}} (from expert agent)
            # Format 2: {criterion_name: score} (flat format for one alternative)

            # Check if this is nested format (Format 1)
            first_value = next(iter(criteria_scores.values()), None)
            if isinstance(first_value, dict):
                # Format 1: Extract scores for the recommended alternative
                alt_criteria_scores = {}
                for criterion, alt_scores in criteria_scores.items():
                    if recommended in alt_scores:
                        alt_criteria_scores[criterion] = alt_scores[recommended]

                if alt_criteria_scores:
                    # Calculate weighted average
                    if criteria_weights:
                        total_weight = sum(criteria_weights.values())
                        weighted_score = sum(
                            alt_criteria_scores.get(criterion, 0.0) * weight
                            for criterion, weight in criteria_weights.items()
                        ) / total_weight if total_weight > 0 else 0.0
                    else:
                        # Equal weights for all criteria
                        scores = list(alt_criteria_scores.values())
                        weighted_score = sum(scores) / len(scores) if scores else 0.0

                    criteria_satisfaction = alt_criteria_scores
                    quality_calculated = True
            else:
                # Format 2: Flat format (already scores for one alternative)
                if criteria_weights:
                    total_weight = sum(criteria_weights.values())
                    weighted_score = sum(
                        criteria_scores.get(criterion, 0.0) * weight
                        for criterion, weight in criteria_weights.items()
                    ) / total_weight if total_weight > 0 else 0.0
                else:
                    # Equal weights for all criteria
                    scores = list(criteria_scores.values())
                    weighted_score = sum(scores) / len(scores) if scores else 0.0

                criteria_satisfaction = criteria_scores.copy()
                quality_calculated = True

        if not quality_calculated and mcda_scores and recommended in mcda_scores:
            # Multi-agent MCDA scoring
            # MCDA score for the recommended alternative
            weighted_score = mcda_scores[recommended]
            criteria_satisfaction = {
                'overall_mcda': mcda_scores[recommended]
            }
            quality_calculated = True

        if not quality_calculated and final_scores and recommended in final_scores:
            # Fallback: use the final score for the recommended alternative
            weighted_score = final_scores[recommended]
            criteria_satisfaction = {
                'final_score': final_scores[recommended]
            }
            quality_calculated = True

        if not quality_calculated:
            logger.warning(f"No quality metrics found for {recommended}")
            weighted_score = 0.0

        # Check ground truth match if available
        ground_truth_match = None
        if ground_truth and 'correct_alternative' in ground_truth:
            ground_truth_match = {
                'match': recommended == ground_truth['correct_alternative'],
                'recommended': recommended,
                'correct': ground_truth['correct_alternative']
            }

            # Boost quality score if ground truth matches
            if ground_truth_match['match']:
                weighted_score = max(weighted_score, 0.9)

            logger.info(
                f"Ground truth comparison: {ground_truth_match['match']} "
                f"(recommended={recommended}, correct={ground_truth['correct_alternative']})"
            )

        result = {
            'weighted_score': weighted_score,
            'confidence': confidence,
            'criteria_satisfaction': criteria_satisfaction,
            'recommended_alternative': recommended,
            'final_scores': final_scores,
            'ground_truth_match': ground_truth_match
        }

        logger.info(f"Decision quality score: {weighted_score:.3f}")
        return result

    def calculate_consensus_metrics(
        self,
        agent_assessments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate Consensus Level (CL) metrics.

        Measures agreement between agents using distribution similarity.

        Args:
            agent_assessments: Agent assessments from CoordinatorAgent.collect_assessments()

        Returns:
            Dictionary with consensus metrics:
                - consensus_level: Overall agreement (0-1)
                - pairwise_similarities: Agent pair similarities
                - preference_distribution: Alternative preferences
                - agreement_percentage: Percentage agreeing on top choice

        Example:
            >>> consensus = evaluator.calculate_consensus_metrics(assessments)
            >>> print(f"Consensus: {consensus['consensus_level']:.1%}")
        """
        logger.info("Calculating consensus metrics")

        # Extract belief distributions
        agent_beliefs = {}
        agent_preferences = {}

        for agent_id, assessment in agent_assessments.items():
            if 'belief_distribution' in assessment:
                agent_beliefs[agent_id] = assessment['belief_distribution']
                # Get top preference
                top_alt = max(
                    assessment['belief_distribution'].items(),
                    key=lambda x: x[1]
                )[0]
                agent_preferences[agent_id] = top_alt

        if len(agent_beliefs) < 2:
            logger.warning("Need at least 2 agents for consensus calculation")
            return {
                'consensus_level': 1.0,
                'pairwise_similarities': {},
                'preference_distribution': agent_preferences,
                'agreement_percentage': 1.0,
                'message': 'Insufficient agents for consensus'
            }

        # Calculate pairwise cosine similarities
        pairwise_sims = {}
        agent_ids = list(agent_beliefs.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent_i = agent_ids[i]
                agent_j = agent_ids[j]

                # Get belief vectors
                beliefs_i = agent_beliefs[agent_i]
                beliefs_j = agent_beliefs[agent_j]

                # Ensure same alternatives
                all_alts = set(beliefs_i.keys()) | set(beliefs_j.keys())
                vec_i = [beliefs_i.get(alt, 0.0) for alt in sorted(all_alts)]
                vec_j = [beliefs_j.get(alt, 0.0) for alt in sorted(all_alts)]

                # Cosine similarity
                similarity = self._cosine_similarity(vec_i, vec_j)
                pair_key = f"{agent_i}_{agent_j}"
                pairwise_sims[pair_key] = similarity

        # Overall consensus level (average pairwise similarity)
        consensus_level = np.mean(list(pairwise_sims.values()))

        # Calculate agreement percentage (agents agreeing on top choice)
        preference_counts = Counter(agent_preferences.values())
        most_common = preference_counts.most_common(1)[0]
        agreement_percentage = most_common[1] / len(agent_preferences)

        result = {
            'consensus_level': float(consensus_level),
            'pairwise_similarities': pairwise_sims,
            'preference_distribution': agent_preferences,
            'preference_counts': dict(preference_counts),
            'agreement_percentage': float(agreement_percentage),
            'top_preference': most_common[0],
            'num_agents': len(agent_beliefs)
        }

        logger.info(
            f"Consensus level: {consensus_level:.3f}, "
            f"Agreement: {agreement_percentage:.1%} on {most_common[0]}"
        )

        return result

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec_a: First vector
            vec_b: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)

        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def calculate_confidence_metrics(
        self,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate Confidence Score (CS) metrics.

        Measures confidence and uncertainty in the decision.

        Args:
            decision: Decision from CoordinatorAgent

        Returns:
            Dictionary with confidence metrics:
                - average_confidence: Mean confidence across agents
                - decision_confidence: Overall decision confidence
                - uncertainty: Uncertainty measure
                - confidence_variance: Variance in agent confidence

        Example:
            >>> conf = evaluator.calculate_confidence_metrics(decision)
            >>> print(f"Confidence: {conf['decision_confidence']:.1%}")
        """
        logger.info("Calculating confidence metrics")

        agent_opinions = decision.get('agent_opinions', {})
        decision_confidence = decision.get('confidence', 0.0)

        # Extract individual agent confidences
        agent_confidences = [
            opinion['confidence']
            for opinion in agent_opinions.values()
            if 'confidence' in opinion
        ]

        if not agent_confidences:
            logger.warning("No agent confidences found")
            return {
                'average_confidence': decision_confidence,
                'decision_confidence': decision_confidence,
                'uncertainty': 1.0 - decision_confidence,
                'confidence_variance': 0.0,
                'num_agents': 0
            }

        # Calculate metrics
        avg_confidence = np.mean(agent_confidences)
        conf_variance = np.var(agent_confidences)
        uncertainty = 1.0 - decision_confidence

        result = {
            'average_confidence': float(avg_confidence),
            'decision_confidence': float(decision_confidence),
            'uncertainty': float(uncertainty),
            'confidence_variance': float(conf_variance),
            'confidence_std': float(np.std(agent_confidences)),
            'min_confidence': float(np.min(agent_confidences)),
            'max_confidence': float(np.max(agent_confidences)),
            'num_agents': len(agent_confidences),
            'agent_confidences': agent_confidences
        }

        logger.info(
            f"Confidence: decision={decision_confidence:.3f}, "
            f"average={avg_confidence:.3f}, uncertainty={uncertainty:.3f}"
        )

        return result

    def calculate_efficiency_metrics(
        self,
        execution_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate Time to Consensus (TtC) and efficiency metrics.

        Measures iterations, API calls, and processing time.

        Args:
            execution_log: List of execution events/logs

        Returns:
            Dictionary with efficiency metrics:
                - time_to_consensus: Number of iterations
                - api_calls_used: Total API calls
                - processing_time: Total time in seconds
                - efficiency_score: Overall efficiency (0-1)

        Example:
            >>> eff = evaluator.calculate_efficiency_metrics(logs)
            >>> print(f"API calls: {eff['api_calls_used']}")
        """
        logger.info("Calculating efficiency metrics")

        # Initialize counters
        iterations = 0
        api_calls = 0
        total_time = 0.0
        start_time = None
        end_time = None

        # Parse execution log
        for event in execution_log:
            event_type = event.get('type', '')

            if event_type == 'iteration':
                iterations += 1
            elif event_type == 'api_call':
                api_calls += 1
            elif event_type == 'llm_call':
                api_calls += 1
            elif event_type == 'start':
                start_time = event.get('timestamp')
            elif event_type == 'end':
                end_time = event.get('timestamp')

            # Accumulate time if available
            if 'duration' in event:
                total_time += event['duration']

        # Calculate processing time from timestamps
        if start_time and end_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
            total_time = (end_time - start_time).total_seconds()

        # Calculate efficiency score (lower is better, normalized to 0-1)
        # Baseline: 1 iteration, 3 API calls, 5 seconds
        iteration_efficiency = 1.0 / (1.0 + iterations) if iterations > 0 else 1.0
        api_efficiency = 1.0 / (1.0 + api_calls / 3.0) if api_calls > 0 else 1.0
        time_efficiency = 1.0 / (1.0 + total_time / 5.0) if total_time > 0 else 1.0

        efficiency_score = (iteration_efficiency + api_efficiency + time_efficiency) / 3.0

        result = {
            'time_to_consensus': iterations,
            'api_calls_used': api_calls,
            'processing_time_seconds': float(total_time),
            'efficiency_score': float(efficiency_score),
            'iteration_efficiency': float(iteration_efficiency),
            'api_efficiency': float(api_efficiency),
            'time_efficiency': float(time_efficiency)
        }

        logger.info(
            f"Efficiency: {iterations} iterations, {api_calls} API calls, "
            f"{total_time:.2f}s, score={efficiency_score:.3f}"
        )

        return result

    def calculate_expert_contribution_balance(
        self,
        agent_assessments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate Expert Contribution Balance (ECB).

        Measures how balanced participation is across agents.

        Args:
            agent_assessments: Agent assessments dictionary

        Returns:
            Dictionary with ECB metrics:
                - balance_score: Overall balance (0-1, 1=perfectly balanced)
                - participation_distribution: Contribution per agent
                - diversity_score: Diversity of perspectives

        Example:
            >>> ecb = evaluator.calculate_expert_contribution_balance(assessments)
            >>> print(f"Balance: {ecb['balance_score']:.1%}")
        """
        logger.info("Calculating expert contribution balance")

        if not agent_assessments:
            return {
                'balance_score': 0.0,
                'participation_distribution': {},
                'diversity_score': 0.0,
                'message': 'No assessments provided'
            }

        # Calculate participation metrics
        num_agents = len(agent_assessments)
        agent_contributions = {}

        for agent_id, assessment in agent_assessments.items():
            # Contribution factors
            confidence = assessment.get('confidence', 0.0)
            belief_dist = assessment.get('belief_distribution', {})

            # Calculate contribution score based on confidence and belief entropy
            if belief_dist:
                # Entropy measures information content
                probs = list(belief_dist.values())
                entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
                max_entropy = np.log2(len(belief_dist)) if len(belief_dist) > 1 else 1.0
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                normalized_entropy = 0.0

            # Contribution = confidence + diversity of opinion
            contribution = (confidence + normalized_entropy) / 2.0
            agent_contributions[agent_id] = float(contribution)

        # Calculate balance score using Gini coefficient
        # Lower Gini = more balanced (we want 1 - Gini for balance score)
        contributions = list(agent_contributions.values())
        gini = self._gini_coefficient(contributions)
        balance_score = 1.0 - gini  # Higher is more balanced

        # Calculate diversity score (variance in preferences)
        preferences = []
        for assessment in agent_assessments.values():
            belief_dist = assessment.get('belief_distribution', {})
            if belief_dist:
                top_pref = max(belief_dist.items(), key=lambda x: x[1])[0]
                preferences.append(top_pref)

        unique_preferences = len(set(preferences))
        diversity_score = unique_preferences / num_agents if num_agents > 0 else 0.0

        result = {
            'balance_score': float(balance_score),
            'participation_distribution': agent_contributions,
            'diversity_score': float(diversity_score),
            'gini_coefficient': float(gini),
            'unique_preferences': unique_preferences,
            'num_agents': num_agents
        }

        logger.info(
            f"Expert balance: {balance_score:.3f}, "
            f"diversity: {diversity_score:.3f}, "
            f"Gini: {gini:.3f}"
        )

        return result

    def _gini_coefficient(self, values: List[float]) -> float:
        """
        Calculate Gini coefficient for balance measurement.

        Args:
            values: List of contribution values

        Returns:
            Gini coefficient (0=perfectly equal, 1=perfectly unequal)
        """
        if not values or len(values) == 1:
            return 0.0

        values = np.array(sorted(values))
        n = len(values)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n

        return float(gini)

    def compare_to_baseline(
        self,
        multi_agent_results: Dict[str, Any],
        single_agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare multi-agent system to single-agent baseline.

        Args:
            multi_agent_results: Results from multi-agent system
            single_agent_results: Results from single-agent baseline

        Returns:
            Dictionary with comparison metrics:
                - decision_quality: Quality comparison
                - confidence: Confidence comparison
                - efficiency: Efficiency comparison
                - improvement: Overall improvement percentage

        Example:
            >>> comparison = evaluator.compare_to_baseline(multi_results, single_results)
            >>> print(f"Improvement: {comparison['improvement']:.1%}")
        """
        logger.info("Comparing multi-agent to single-agent baseline")

        comparison = {
            'decision_quality': {},
            'confidence': {},
            'efficiency': {},
            'consensus': {},
            'improvement': 0.0
        }

        # Compare decision quality
        multi_quality = multi_agent_results.get('decision_quality', {})
        single_quality = single_agent_results.get('decision_quality', {})

        if multi_quality and single_quality:
            multi_score = multi_quality.get('weighted_score', 0.0)
            single_score = single_quality.get('weighted_score', 0.0)

            comparison['decision_quality'] = {
                'multi_agent': multi_score,
                'single_agent': single_score,
                'improvement': multi_score - single_score,
                'improvement_percentage': ((multi_score - single_score) / single_score * 100)
                    if single_score > 0 else 0.0
            }

        # Compare confidence
        multi_conf = multi_agent_results.get('confidence', {})
        single_conf = single_agent_results.get('confidence', {})

        if multi_conf and single_conf:
            multi_score = multi_conf.get('decision_confidence', 0.0)
            single_score = single_conf.get('decision_confidence', 0.0)

            comparison['confidence'] = {
                'multi_agent': multi_score,
                'single_agent': single_score,
                'improvement': multi_score - single_score,
                'improvement_percentage': ((multi_score - single_score) / single_score * 100)
                    if single_score > 0 else 0.0
            }

        # Compare efficiency
        multi_eff = multi_agent_results.get('efficiency', {})
        single_eff = single_agent_results.get('efficiency', {})

        if multi_eff and single_eff:
            multi_calls = multi_eff.get('api_calls_used', 0)
            single_calls = single_eff.get('api_calls_used', 0)

            comparison['efficiency'] = {
                'multi_agent_api_calls': multi_calls,
                'single_agent_api_calls': single_calls,
                'additional_calls': multi_calls - single_calls,
                'efficiency_ratio': single_calls / multi_calls if multi_calls > 0 else 0.0
            }

        # Consensus (only multi-agent has this)
        if 'consensus' in multi_agent_results:
            comparison['consensus'] = {
                'multi_agent': multi_agent_results['consensus'].get('consensus_level', 0.0),
                'single_agent': 1.0,  # Single agent always has "consensus" with itself
                'note': 'Consensus only applicable to multi-agent'
            }

        # Calculate overall improvement
        improvements = []
        if comparison['decision_quality']:
            improvements.append(comparison['decision_quality']['improvement'])
        if comparison['confidence']:
            improvements.append(comparison['confidence']['improvement'])

        if improvements:
            comparison['improvement'] = float(np.mean(improvements))
            comparison['improvement_percentage'] = comparison['improvement'] * 100

        logger.info(
            f"Baseline comparison: improvement={comparison['improvement']:.3f} "
            f"({comparison.get('improvement_percentage', 0):.1f}%)"
        )

        return comparison

    def calculate_statistical_significance(
        self,
        multi_agent_scores: List[float],
        single_agent_scores: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance of multi-agent vs single-agent.

        Uses t-test and Cohen's d effect size.

        Args:
            multi_agent_scores: List of scores from multi-agent runs
            single_agent_scores: List of scores from single-agent runs
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with statistical test results:
                - t_statistic: T-test statistic
                - p_value: P-value
                - significant: Whether difference is significant
                - cohens_d: Cohen's d effect size
                - interpretation: Human-readable interpretation

        Example:
            >>> sig = evaluator.calculate_statistical_significance(
            ...     multi_scores, single_scores
            ... )
            >>> print(f"Significant: {sig['significant']}")
        """
        logger.info("Calculating statistical significance")

        if len(multi_agent_scores) < 2 or len(single_agent_scores) < 2:
            logger.warning("Need at least 2 samples for each group")
            return {
                'error': 'Insufficient samples',
                'significant': False
            }

        # Perform independent t-test
        t_statistic, p_value = stats.ttest_ind(
            multi_agent_scores,
            single_agent_scores
        )

        # Calculate Cohen's d effect size
        cohens_d = self._cohens_d(multi_agent_scores, single_agent_scores)

        # Determine significance
        significant = p_value < alpha

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        result = {
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'significant': significant,
            'alpha': alpha,
            'cohens_d': float(cohens_d),
            'effect_size': effect_interpretation,
            'multi_agent_mean': float(np.mean(multi_agent_scores)),
            'single_agent_mean': float(np.mean(single_agent_scores)),
            'multi_agent_std': float(np.std(multi_agent_scores)),
            'single_agent_std': float(np.std(single_agent_scores)),
            'n_multi': len(multi_agent_scores),
            'n_single': len(single_agent_scores)
        }

        logger.info(
            f"Statistical significance: p={p_value:.4f}, "
            f"significant={significant}, Cohen's d={cohens_d:.3f} ({effect_interpretation})"
        )

        return result

    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group of scores
            group2: Second group of scores

        Returns:
            Cohen's d effect size
        """
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)
        n1 = len(group1)
        n2 = len(group2)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        cohens_d = (mean1 - mean2) / pooled_std
        return float(cohens_d)

    def generate_report(
        self,
        all_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive human-readable metrics report.

        Args:
            all_metrics: Dictionary containing all calculated metrics

        Returns:
            Formatted report string

        Example:
            >>> report = evaluator.generate_report(metrics)
            >>> print(report)
        """
        logger.info("Generating metrics report")

        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("MULTI-AGENT SYSTEM PERFORMANCE METRICS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Decision Quality
        if 'decision_quality' in all_metrics:
            lines.append("DECISION QUALITY SCORE (DQS)")
            lines.append("-" * 80)
            dq = all_metrics['decision_quality']
            lines.append(f"  Weighted Score: {dq.get('weighted_score', 0):.3f}")
            lines.append(f"  Confidence: {dq.get('confidence', 0):.3f}")
            lines.append(f"  Recommended: {dq.get('recommended_alternative', 'N/A')}")

            if dq.get('ground_truth_match'):
                gt = dq['ground_truth_match']
                lines.append(f"  Ground Truth Match: {'✓' if gt['match'] else '✗'}")
            lines.append("")

        # Consensus Level
        if 'consensus' in all_metrics:
            lines.append("CONSENSUS LEVEL (CL)")
            lines.append("-" * 80)
            cl = all_metrics['consensus']
            lines.append(f"  Consensus Level: {cl.get('consensus_level', 0):.1%}")
            lines.append(f"  Agreement Percentage: {cl.get('agreement_percentage', 0):.1%}")
            lines.append(f"  Top Preference: {cl.get('top_preference', 'N/A')}")
            lines.append(f"  Number of Agents: {cl.get('num_agents', 0)}")
            lines.append("")

        # Confidence Score
        if 'confidence' in all_metrics:
            lines.append("CONFIDENCE SCORE (CS)")
            lines.append("-" * 80)
            cs = all_metrics['confidence']
            lines.append(f"  Decision Confidence: {cs.get('decision_confidence', 0):.3f}")
            lines.append(f"  Average Agent Confidence: {cs.get('average_confidence', 0):.3f}")
            lines.append(f"  Uncertainty: {cs.get('uncertainty', 0):.3f}")
            lines.append(f"  Confidence Variance: {cs.get('confidence_variance', 0):.4f}")
            lines.append("")

        # Efficiency
        if 'efficiency' in all_metrics:
            lines.append("TIME TO CONSENSUS (TtC)")
            lines.append("-" * 80)
            eff = all_metrics['efficiency']
            lines.append(f"  Iterations: {eff.get('time_to_consensus', 0)}")
            lines.append(f"  API Calls: {eff.get('api_calls_used', 0)}")
            lines.append(f"  Processing Time: {eff.get('processing_time_seconds', 0):.2f}s")
            lines.append(f"  Efficiency Score: {eff.get('efficiency_score', 0):.3f}")
            lines.append("")

        # Expert Contribution Balance
        if 'expert_contribution_balance' in all_metrics:
            lines.append("EXPERT CONTRIBUTION BALANCE (ECB)")
            lines.append("-" * 80)
            ecb = all_metrics['expert_contribution_balance']
            lines.append(f"  Balance Score: {ecb.get('balance_score', 0):.3f}")
            lines.append(f"  Diversity Score: {ecb.get('diversity_score', 0):.3f}")
            lines.append(f"  Gini Coefficient: {ecb.get('gini_coefficient', 0):.3f}")
            lines.append("")

        # Baseline Comparison
        if 'baseline_comparison' in all_metrics:
            lines.append("MULTI-AGENT VS SINGLE-AGENT COMPARISON")
            lines.append("-" * 80)
            comp = all_metrics['baseline_comparison']

            if 'decision_quality' in comp:
                dq_comp = comp['decision_quality']
                lines.append(f"  Decision Quality:")
                lines.append(f"    Multi-Agent: {dq_comp.get('multi_agent', 0):.3f}")
                lines.append(f"    Single-Agent: {dq_comp.get('single_agent', 0):.3f}")
                lines.append(f"    Improvement: {dq_comp.get('improvement_percentage', 0):.1f}%")

            if 'confidence' in comp:
                conf_comp = comp['confidence']
                lines.append(f"  Confidence:")
                lines.append(f"    Multi-Agent: {conf_comp.get('multi_agent', 0):.3f}")
                lines.append(f"    Single-Agent: {conf_comp.get('single_agent', 0):.3f}")
                lines.append(f"    Improvement: {conf_comp.get('improvement_percentage', 0):.1f}%")

            lines.append("")

        # Statistical Significance
        if 'statistical_significance' in all_metrics:
            lines.append("STATISTICAL SIGNIFICANCE")
            lines.append("-" * 80)
            sig = all_metrics['statistical_significance']
            lines.append(f"  P-value: {sig.get('p_value', 0):.4f}")
            lines.append(f"  Significant: {'Yes' if sig.get('significant') else 'No'}")
            lines.append(f"  Cohen's d: {sig.get('cohens_d', 0):.3f} ({sig.get('effect_size', 'unknown')})")
            lines.append("")

        # Footer
        lines.append("=" * 80)
        lines.append(f"Report generated: {datetime.now().isoformat()}")
        lines.append("=" * 80)

        report = "\n".join(lines)

        logger.info("Metrics report generated")
        return report

    def save_evaluation(
        self,
        metrics: Dict[str, Any],
        filename: str
    ):
        """
        Save evaluation metrics to history.

        Args:
            metrics: Metrics dictionary
            filename: Optional filename for saving
        """
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'filename': filename
        }

        self.evaluation_history.append(evaluation_record)
        logger.info(f"Evaluation saved: {filename}")

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Get complete evaluation history.

        Returns:
            List of all evaluation records
        """
        return self.evaluation_history.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"MetricsEvaluator(evaluations={len(self.evaluation_history)})"
