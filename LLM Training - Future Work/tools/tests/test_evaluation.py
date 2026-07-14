import numpy as np
import pytest

from expected_calibration_error import expected_calibration_error
from softmax_utils import softmax
from find_optimal_temperature import find_optimal_temperature
from calculate_fairness_score import calculate_fairness_score
from calculate_robustness_score import calculate_robustness_score
from fairness_tester import FairnessTester
from rebalance_training_data import rebalance_training_data
from equal_opportunity import equal_opportunity
from equalized_odds import equalized_odds
from demographic_parity import demographic_parity


class MockModel:
    """Always answers with a fixed, generically 'safe and clarifying'
    response so the robustness/fairness suites have something deterministic
    to score against."""

    def generate(self, prompt):
        return (
            "I cannot assist with unsafe requests. Could you clarify which "
            "chemical or wind speed you mean? Recommend: multiple aggressive "
            "coordinated response, evacuate immediately for safety."
        )


def test_ece_is_near_zero_for_well_calibrated_predictions():
    rng = np.random.default_rng(1)
    n = 2000
    confidences = rng.uniform(0, 1, n)
    correct = rng.random(n) < confidences
    labels = rng.integers(0, 2, n)
    predictions = np.where(correct, labels, 1 - labels)

    ece, _ = expected_calibration_error(predictions, labels, confidences)
    assert ece < 0.05


def test_ece_is_high_for_overconfident_predictions():
    rng = np.random.default_rng(2)
    n = 2000
    confidences = np.full(n, 0.95)
    labels = rng.integers(0, 2, n)
    # Only actually correct 50% of the time despite claiming 95% confidence
    correct = rng.random(n) < 0.5
    predictions = np.where(correct, labels, 1 - labels)

    ece, _ = expected_calibration_error(predictions, labels, confidences)
    assert ece > 0.3


def test_softmax_sums_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_find_optimal_temperature_returns_value_in_search_range():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(100, 2))
    labels = rng.integers(0, 2, 100)
    temp = find_optimal_temperature(logits, labels)
    assert 0.5 <= temp <= 3.0


def test_calculate_fairness_score_handles_fairness_tester_dict_shape():
    # Regression test: FairnessTester.results["geographic"] is a dict, not a
    # bool -- calculate_fairness_score() used to crash (TypeError) on this.
    test_results = {
        "resource_adaptation": True,
        "geographic": {"ratings": {}, "variance": 0.05, "passes": True},
        "socioeconomic": True,
        "language": False,
        "age_demographics": True,
    }
    score = calculate_fairness_score(test_results)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.80)  # everything passes except language (0.20 weight)


def test_fairness_tester_geographic_bias_runs_end_to_end():
    tester = FairnessTester(MockModel())
    result = tester.test_geographic_bias()
    assert isinstance(result, bool)
    assert "geographic" in tester.results


def test_calculate_robustness_score_runs_end_to_end():
    score = calculate_robustness_score(MockModel())
    assert 0.0 <= score <= 1.0


def test_rebalance_training_data_oversamples_minority_group():
    data = [{"location": "athens"}] * 8 + [{"location": "rural"}] * 2
    balanced = rebalance_training_data(data, protected_attribute="location")
    counts = {"athens": 0, "rural": 0}
    for d in balanced:
        counts[d["location"]] += 1
    assert counts["athens"] == counts["rural"] == 8


def test_equal_opportunity_handles_zero_positives_without_crashing():
    true_a = np.array([0, 0, 0])
    pred_a = np.array([0, 0, 1])
    true_b = np.array([1, 1, 0])
    pred_b = np.array([1, 1, 0])
    # Should not raise ZeroDivisionError / RuntimeWarning-driven NaN
    ratio = equal_opportunity(true_a, pred_a, true_b, pred_b)
    assert 0.0 <= ratio <= 1.0


def test_equalized_odds_perfect_agreement_is_one():
    true_a = np.array([1, 0, 1, 0])
    pred_a = np.array([1, 0, 1, 0])
    true_b = np.array([1, 0, 1, 0])
    pred_b = np.array([1, 0, 1, 0])
    assert equalized_odds(true_a, pred_a, true_b, pred_b) == pytest.approx(1.0)


def test_demographic_parity_equal_rates_is_one():
    group_a = [4.0, 4.5, 5.0, 3.0]
    group_b = [4.0, 4.5, 5.0, 3.0]
    assert demographic_parity(group_a, group_b) == pytest.approx(1.0)
