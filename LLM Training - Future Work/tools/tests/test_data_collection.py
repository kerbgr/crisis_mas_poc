import pytest

from inter_rater_reliability import calculate_cohens_kappa, interpret_kappa
from calculate_fleiss_kappa import calculate_fleiss_kappa
from calculate_icc import calculate_icc
from data_versioning import DatasetVersion
from resolve_with_confidence_weighting import resolve_with_confidence_weighting
from resolve_with_context import resolve_with_context
from resolve_with_tiebreaker import resolve_with_tiebreaker
from resolve_disagreement import resolve_disagreement, is_factual_question, is_tactical_question


def test_cohens_kappa_perfect_agreement():
    assert calculate_cohens_kappa([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_cohens_kappa_matches_sklearn_semantics():
    # Disagreement on every item with only two categories used symmetrically
    # should score at or below chance (kappa <= 0), not "almost perfect".
    kappa = calculate_cohens_kappa([1, 1, 1, 1], [2, 2, 2, 2])
    assert kappa <= 0


def test_interpret_kappa_buckets():
    assert interpret_kappa(-0.1).startswith("Poor")
    assert interpret_kappa(0.9).startswith("Almost Perfect")


def test_fleiss_kappa_perfect_agreement_is_one():
    ratings = [[5, 5, 5], [3, 3, 3], [1, 1, 1]]
    assert calculate_fleiss_kappa(ratings) == pytest.approx(1.0)


def test_calculate_icc_runs_and_returns_float(capsys):
    import pandas as pd

    # pingouin's underlying ANOVA requires at least 5 non-missing rows.
    df = pd.DataFrame({
        'Example': ['EX1', 'EX1', 'EX2', 'EX2', 'EX3', 'EX3'],
        'Expert': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Rating': [0.9, 0.9, 0.5, 0.5, 0.7, 0.6],
    })
    icc = calculate_icc(df)
    assert isinstance(icc, float)


def test_dataset_version_hash_changes_when_file_changes(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"q": "a"}\n{"q": "b"}\n')

    dv = DatasetVersion(str(data_path))
    original_hash = dv.metadata["dataset_hash"]
    assert dv.metadata["num_examples"] == 2
    assert dv.verify_integrity() is True

    data_path.write_text('{"q": "a"}\n{"q": "b"}\n{"q": "c"}\n')
    dv2 = DatasetVersion(str(data_path))
    assert dv2.metadata["dataset_hash"] != original_hash


def test_dataset_version_splits_partition_all_examples(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("\n".join(f'{{"q": {i}}}' for i in range(20)) + "\n")

    dv = DatasetVersion(str(data_path))
    dv.create_splits(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    sizes = dv.metadata["splits"]
    assert sizes["train"]["size"] + sizes["val"]["size"] + sizes["test"]["size"] == 20


def test_confidence_weighting_normalizes_by_winning_answer_votes():
    # Regression test for the bug where confidence was divided by the total
    # number of raters instead of the number who picked the winning answer.
    result = resolve_with_confidence_weighting([
        ("Evacuate", 0.95),
        ("Defend", 0.50),
        ("Evacuate", 0.80),
    ])
    assert result["answer"] == "Evacuate"
    assert result["confidence"] == pytest.approx((0.95 + 0.80) / 2)


def test_resolve_with_context_creates_one_example_per_context():
    examples = resolve_with_context("Evacuate or defend?", ["Evacuate", "Defend"])
    assert len(examples) == 2
    assert examples[0]["answer"] == "Evacuate"
    assert examples[1]["answer"] == "Defend"


def test_resolve_with_tiebreaker_majority_vote():
    result = resolve_with_tiebreaker(
        question="Evacuate or defend?",
        answer_a="Evacuate",
        answer_b="Defend",
        confidence_a=0.9,
        confidence_b=0.7,
        get_third_expert_answer=lambda q: "Evacuate",
    )
    assert result["answer"] == "Evacuate"
    assert result["metadata"]["resolution_method"] == "tie-breaker"


def test_is_factual_vs_tactical_question():
    assert is_factual_question("What is the IDLH for ammonia?")
    assert not is_factual_question("Should we evacuate or defend?")
    assert is_tactical_question("Should we evacuate or defend?")


def test_resolve_disagreement_routes_to_confidence_weighting_on_large_gap():
    ratings = [
        {"expert": "A", "answer": "Evacuate", "confidence": 0.95, "quality": 5},
        {"expert": "B", "answer": "Defend", "confidence": 0.40, "quality": 3},
    ]
    result = resolve_disagreement("Should we evacuate or defend?", ratings)
    assert result["answer"] == "Evacuate"


def test_resolve_disagreement_unanimous_short_circuits():
    ratings = [
        {"expert": "A", "answer": "Evacuate", "confidence": 0.9, "quality": 5},
        {"expert": "B", "answer": "Evacuate", "confidence": 0.8, "quality": 4},
    ]
    result = resolve_disagreement("Should we evacuate or defend?", ratings)
    assert result["metadata"]["agreement"] == "unanimous"
