"""
Historical Reliability Tracker for Crisis Management Agents
Tracks agent performance over time to support dynamic weighting models

This module addresses the revised abstract requirement:
"Τα μοντέλα αυτά θα μπορούσαν να λαμβάνουν υπόψη παράγοντες όπως...
η αξιοπιστία και συνέπεια των προηγούμενων αξιολογήσεών του"

"These models could take into account factors such as...
the reliability and consistency of their previous assessments"
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ReliabilityMetrics:
    """Container for agent reliability metrics."""

    def __init__(self):
        self.overall_reliability: float = 0.8  # Initial reliability score
        self.recent_reliability: float = 0.8  # Recent performance (last 10 assessments)
        self.consistency_score: float = 0.8  # Variance in assessment quality
        self.domain_reliability: Dict[str, float] = {}  # Per crisis type
        self.total_assessments: int = 0
        self.accurate_assessments: int = 0
        self.last_updated: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'overall_reliability': self.overall_reliability,
            'recent_reliability': self.recent_reliability,
            'consistency_score': self.consistency_score,
            'domain_reliability': self.domain_reliability.copy(),
            'total_assessments': self.total_assessments,
            'accurate_assessments': self.accurate_assessments,
            'accuracy_rate': (self.accurate_assessments / self.total_assessments
                            if self.total_assessments > 0 else 0.0),
            'last_updated': self.last_updated.isoformat()
        }


class AssessmentRecord:
    """Record of a single agent assessment with outcome."""

    def __init__(
        self,
        assessment_id: str,
        scenario_type: str,
        agent_prediction: Dict[str, Any],
        timestamp: datetime,
        confidence: float = 0.5
    ):
        self.assessment_id = assessment_id
        self.scenario_type = scenario_type
        self.agent_prediction = agent_prediction
        self.timestamp = timestamp
        self.confidence = confidence
        self.actual_outcome: Optional[Dict[str, Any]] = None
        self.accuracy_score: Optional[float] = None
        self.evaluation_timestamp: Optional[datetime] = None

    def record_outcome(self, actual_outcome: Dict[str, Any], accuracy_score: float):
        """Record the actual outcome and calculated accuracy."""
        self.actual_outcome = actual_outcome
        self.accuracy_score = accuracy_score
        self.evaluation_timestamp = datetime.now()

    def is_evaluated(self) -> bool:
        """Check if this assessment has been evaluated."""
        return self.actual_outcome is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            'assessment_id': self.assessment_id,
            'scenario_type': self.scenario_type,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'predicted': self.agent_prediction,
            'actual': self.actual_outcome,
            'accuracy_score': self.accuracy_score,
            'evaluated': self.is_evaluated(),
            'evaluation_timestamp': (self.evaluation_timestamp.isoformat()
                                    if self.evaluation_timestamp else None)
        }


class ReliabilityTracker:
    """
    Tracks agent reliability over time based on historical performance.

    Features:
    - Overall reliability score (lifetime)
    - Recent reliability (sliding window)
    - Consistency score (variance analysis)
    - Domain-specific reliability (per crisis type)
    - Temporal decay (older assessments count less)
    - Confidence-weighted accuracy
    """

    def __init__(
        self,
        agent_id: str,
        window_size: int = 10,
        decay_factor: float = 0.95,
        min_assessments_for_reliability: int = 3
    ):
        """
        Initialize reliability tracker.

        Args:
            agent_id: Agent identifier
            window_size: Number of recent assessments for recent reliability
            decay_factor: Temporal decay factor (0-1, higher = slower decay)
            min_assessments_for_reliability: Minimum assessments before tracking
        """
        self.agent_id = agent_id
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.min_assessments = min_assessments_for_reliability

        # Storage
        self.assessment_history: List[AssessmentRecord] = []
        self.recent_assessments: deque = deque(maxlen=window_size)

        # Metrics
        self.metrics = ReliabilityMetrics()

        logger.info(
            f"ReliabilityTracker initialized for agent '{agent_id}' "
            f"(window={window_size}, decay={decay_factor})"
        )

    def record_assessment(
        self,
        assessment_id: str,
        scenario_type: str,
        agent_prediction: Dict[str, Any],
        confidence: float = 0.5
    ) -> AssessmentRecord:
        """
        Record a new assessment made by the agent.

        Args:
            assessment_id: Unique identifier for this assessment
            scenario_type: Type of crisis scenario
            agent_prediction: Agent's prediction/assessment
            confidence: Agent's confidence in this assessment (0-1)

        Returns:
            AssessmentRecord object
        """
        record = AssessmentRecord(
            assessment_id=assessment_id,
            scenario_type=scenario_type,
            agent_prediction=agent_prediction,
            timestamp=datetime.now(),
            confidence=confidence
        )

        self.assessment_history.append(record)
        self.recent_assessments.append(record)

        logger.debug(
            f"Recorded assessment '{assessment_id}' for agent '{self.agent_id}' "
            f"(type={scenario_type}, confidence={confidence:.2f})"
        )

        return record

    def update_assessment_outcome(
        self,
        assessment_id: str,
        actual_outcome: Dict[str, Any],
        accuracy_score: Optional[float] = None
    ):
        """
        Update an assessment with its actual outcome and accuracy.

        Args:
            assessment_id: ID of the assessment to update
            actual_outcome: The actual outcome that occurred
            accuracy_score: Pre-calculated accuracy score (0-1), or None to calculate

        Raises:
            ValueError: If assessment_id not found
        """
        # Find the assessment
        record = None
        for r in self.assessment_history:
            if r.assessment_id == assessment_id:
                record = r
                break

        if record is None:
            raise ValueError(
                f"Assessment '{assessment_id}' not found for agent '{self.agent_id}'"
            )

        # Calculate accuracy if not provided
        if accuracy_score is None:
            accuracy_score = self._calculate_accuracy(
                record.agent_prediction,
                actual_outcome
            )

        # Record the outcome
        record.record_outcome(actual_outcome, accuracy_score)

        # Update reliability metrics
        self._update_reliability_metrics()

        logger.info(
            f"Updated assessment '{assessment_id}' for agent '{self.agent_id}' "
            f"(accuracy={accuracy_score:.3f})"
        )

    def _calculate_accuracy(
        self,
        prediction: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> float:
        """
        Calculate accuracy score by comparing prediction with actual outcome.

        For belief distributions, uses:
        - Probability score (predicted belief for actual outcome)
        - Rank accuracy (was top choice correct?)
        - Distribution similarity (KL divergence)

        Args:
            prediction: Agent's prediction (with belief_distribution)
            actual: Actual outcome (with selected_alternative)

        Returns:
            Accuracy score (0-1)
        """
        # Extract belief distribution
        predicted_beliefs = prediction.get('belief_distribution', {})
        if not predicted_beliefs:
            return 0.5  # Neutral score if no beliefs

        # Get actual selected alternative
        actual_alternative = actual.get('selected_alternative')
        if actual_alternative is None:
            return 0.5  # Neutral if no actual alternative

        # Method 1: Probability score (belief assigned to actual outcome)
        probability_score = predicted_beliefs.get(actual_alternative, 0.0)

        # Method 2: Rank accuracy (was prediction correct?)
        predicted_top = max(predicted_beliefs.items(), key=lambda x: x[1])[0]
        rank_accuracy = 1.0 if predicted_top == actual_alternative else 0.0

        # Method 3: Margin of correctness
        # If agent was confident in correct answer: high score
        # If agent was uncertain but correct: medium score
        # If agent was confident but wrong: low score
        confidence = prediction.get('confidence', 0.5)
        if predicted_top == actual_alternative:
            # Correct prediction - reward confidence
            margin_score = 0.5 + 0.5 * confidence
        else:
            # Wrong prediction - penalize confidence
            margin_score = 0.5 - 0.5 * confidence

        # Combined accuracy (weighted average)
        accuracy = (
            0.4 * probability_score +
            0.3 * rank_accuracy +
            0.3 * margin_score
        )

        return float(np.clip(accuracy, 0.0, 1.0))

    def _update_reliability_metrics(self):
        """Recalculate all reliability metrics based on current history."""
        evaluated_assessments = [r for r in self.assessment_history if r.is_evaluated()]

        if len(evaluated_assessments) < self.min_assessments:
            # Not enough data yet
            return

        # Update counts
        self.metrics.total_assessments = len(evaluated_assessments)

        # Calculate overall reliability with temporal decay
        weighted_sum = 0.0
        weight_total = 0.0
        current_time = datetime.now()

        for record in evaluated_assessments:
            # Calculate temporal weight (older assessments count less)
            age_days = (current_time - record.timestamp).days
            temporal_weight = self.decay_factor ** age_days

            # Confidence weighting (more confident assessments count more)
            confidence_weight = 0.5 + 0.5 * record.confidence

            # Combined weight
            weight = temporal_weight * confidence_weight

            weighted_sum += weight * record.accuracy_score
            weight_total += weight

        if weight_total > 0:
            self.metrics.overall_reliability = weighted_sum / weight_total

        # Calculate recent reliability (last N assessments)
        recent_evaluated = [r for r in self.recent_assessments if r.is_evaluated()]
        if recent_evaluated:
            recent_scores = [r.accuracy_score for r in recent_evaluated]
            self.metrics.recent_reliability = float(np.mean(recent_scores))

        # Calculate consistency score (inverse of variance)
        if len(recent_evaluated) >= 3:
            scores = [r.accuracy_score for r in recent_evaluated]
            variance = float(np.var(scores))
            # Convert to consistency: low variance = high consistency
            self.metrics.consistency_score = 1.0 / (1.0 + variance)
        else:
            self.metrics.consistency_score = 0.8  # Default

        # Calculate domain-specific reliability
        domain_scores: Dict[str, List[float]] = {}
        for record in evaluated_assessments:
            scenario_type = record.scenario_type
            if scenario_type not in domain_scores:
                domain_scores[scenario_type] = []
            domain_scores[scenario_type].append(record.accuracy_score)

        self.metrics.domain_reliability = {
            domain: float(np.mean(scores))
            for domain, scores in domain_scores.items()
        }

        # Count accurate assessments (accuracy >= 0.7)
        self.metrics.accurate_assessments = sum(
            1 for r in evaluated_assessments if r.accuracy_score >= 0.7
        )

        self.metrics.last_updated = datetime.now()

    def get_reliability_score(
        self,
        scenario_type: Optional[str] = None,
        mode: str = 'overall'
    ) -> float:
        """
        Get reliability score for the agent.

        Args:
            scenario_type: Optional crisis type for domain-specific reliability
            mode: 'overall', 'recent', or 'consistent'

        Returns:
            Reliability score (0-1)
        """
        # Domain-specific reliability if requested and available
        if scenario_type and scenario_type in self.metrics.domain_reliability:
            return self.metrics.domain_reliability[scenario_type]

        # Mode-based reliability
        if mode == 'recent':
            return self.metrics.recent_reliability
        elif mode == 'consistent':
            return self.metrics.consistency_score
        else:  # overall
            return self.metrics.overall_reliability

    def get_metrics(self) -> ReliabilityMetrics:
        """Get current reliability metrics."""
        return self.metrics

    def get_assessment_history(
        self,
        limit: Optional[int] = None,
        scenario_type: Optional[str] = None,
        evaluated_only: bool = False
    ) -> List[AssessmentRecord]:
        """
        Get assessment history with optional filtering.

        Args:
            limit: Maximum number of records to return (most recent)
            scenario_type: Filter by scenario type
            evaluated_only: Only return evaluated assessments

        Returns:
            List of AssessmentRecord objects
        """
        history = self.assessment_history

        # Filter by scenario type
        if scenario_type:
            history = [r for r in history if r.scenario_type == scenario_type]

        # Filter evaluated only
        if evaluated_only:
            history = [r for r in history if r.is_evaluated()]

        # Limit to most recent
        if limit:
            history = history[-limit:]

        return history

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.

        Returns:
            Dictionary with performance statistics
        """
        evaluated = [r for r in self.assessment_history if r.is_evaluated()]

        if not evaluated:
            return {
                'agent_id': self.agent_id,
                'total_assessments': 0,
                'message': 'No evaluated assessments yet'
            }

        accuracy_scores = [r.accuracy_score for r in evaluated]

        return {
            'agent_id': self.agent_id,
            'total_assessments': len(self.assessment_history),
            'evaluated_assessments': len(evaluated),
            'pending_assessments': len(self.assessment_history) - len(evaluated),
            'reliability_metrics': self.metrics.to_dict(),
            'performance_stats': {
                'mean_accuracy': float(np.mean(accuracy_scores)),
                'median_accuracy': float(np.median(accuracy_scores)),
                'std_accuracy': float(np.std(accuracy_scores)),
                'min_accuracy': float(np.min(accuracy_scores)),
                'max_accuracy': float(np.max(accuracy_scores)),
                'above_threshold_70': sum(1 for s in accuracy_scores if s >= 0.7),
                'above_threshold_80': sum(1 for s in accuracy_scores if s >= 0.8),
                'above_threshold_90': sum(1 for s in accuracy_scores if s >= 0.9)
            },
            'domain_breakdown': self._get_domain_breakdown()
        }

    def _get_domain_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by domain."""
        evaluated = [r for r in self.assessment_history if r.is_evaluated()]

        breakdown: Dict[str, Dict[str, Any]] = {}
        for record in evaluated:
            domain = record.scenario_type
            if domain not in breakdown:
                breakdown[domain] = {
                    'count': 0,
                    'accuracy_scores': []
                }
            breakdown[domain]['count'] += 1
            breakdown[domain]['accuracy_scores'].append(record.accuracy_score)

        # Calculate statistics per domain
        for domain, data in breakdown.items():
            scores = data['accuracy_scores']
            breakdown[domain] = {
                'count': data['count'],
                'mean_accuracy': float(np.mean(scores)),
                'std_accuracy': float(np.std(scores)) if len(scores) > 1 else 0.0,
                'reliability_score': self.metrics.domain_reliability.get(domain, 0.8)
            }

        return breakdown

    def export_history(self) -> List[Dict[str, Any]]:
        """Export full assessment history as list of dictionaries."""
        return [record.to_dict() for record in self.assessment_history]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ReliabilityTracker(agent='{self.agent_id}', "
            f"assessments={len(self.assessment_history)}, "
            f"reliability={self.metrics.overall_reliability:.3f})"
        )
