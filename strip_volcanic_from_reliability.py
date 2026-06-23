"""
Strip volcanic_seismic entries from reliability JSON files and recompute metrics.
Pure stdlib — no project imports, no numpy dependency.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import deque

RELIABILITY_DIR = Path("results/reliability")
STRIP_TYPE = "volcanic_seismic"
DECAY_FACTOR = 0.95
WINDOW_SIZE = 10
MIN_ASSESSMENTS = 3


def mean(values):
    return sum(values) / len(values) if values else 0.0


def variance(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def recompute_metrics(history: list, decay_factor: float, window_size: int) -> dict:
    evaluated = [h for h in history if h.get("evaluated") and h.get("accuracy_score") is not None]

    if len(evaluated) < MIN_ASSESSMENTS:
        return {
            "overall_reliability": 0.8,
            "recent_reliability": 0.8,
            "consistency_score": 0.8,
            "domain_reliability": {},
            "total_assessments": len(evaluated),
            "accurate_assessments": 0,
            "accuracy_rate": 0.0,
            "last_updated": datetime.now().isoformat(),
        }

    now = datetime.now()

    # Overall reliability with temporal decay + confidence weighting
    weighted_sum = 0.0
    weight_total = 0.0
    for h in evaluated:
        ts = datetime.fromisoformat(h["timestamp"])
        age_days = (now - ts).days
        temporal_weight = decay_factor ** age_days
        confidence = h.get("confidence", 0.5)
        confidence_weight = 0.5 + 0.5 * confidence
        weight = temporal_weight * confidence_weight
        weighted_sum += weight * h["accuracy_score"]
        weight_total += weight

    overall = weighted_sum / weight_total if weight_total > 0 else 0.8

    # Recent reliability (last window_size evaluated)
    recent = evaluated[-window_size:]
    recent_scores = [h["accuracy_score"] for h in recent]
    recent_reliability = mean(recent_scores) if recent_scores else 0.8

    # Consistency (inverse variance of recent scores)
    if len(recent_scores) >= 3:
        var = variance(recent_scores)
        consistency = 1.0 / (1.0 + var)
    else:
        consistency = 0.8

    # Domain reliability
    domain_scores: dict = {}
    for h in evaluated:
        domain = h.get("scenario_type", "unknown")
        domain_scores.setdefault(domain, []).append(h["accuracy_score"])
    domain_reliability = {d: mean(scores) for d, scores in domain_scores.items()}

    accurate = sum(1 for h in evaluated if h["accuracy_score"] >= 0.7)

    return {
        "overall_reliability": clip(overall, 0.0, 1.0),
        "recent_reliability": clip(recent_reliability, 0.0, 1.0),
        "consistency_score": clip(consistency, 0.0, 1.0),
        "domain_reliability": domain_reliability,
        "total_assessments": len(evaluated),
        "accurate_assessments": accurate,
        "accuracy_rate": accurate / len(evaluated) if evaluated else 0.0,
        "last_updated": datetime.now().isoformat(),
    }


def strip_and_recompute(filepath: Path) -> dict:
    with open(filepath) as f:
        data = json.load(f)

    agent_id = data["agent_id"]
    history = data.get("assessment_history", [])

    kept = [h for h in history if h.get("scenario_type") != STRIP_TYPE]
    removed = len(history) - len(kept)

    new_metrics = recompute_metrics(kept, DECAY_FACTOR, WINDOW_SIZE)

    data["assessment_history"] = kept
    data["metrics"] = new_metrics

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return {
        "agent": agent_id,
        "before": len(history),
        "removed": removed,
        "after": len(kept),
        "overall": new_metrics["overall_reliability"],
        "domains": list(new_metrics["domain_reliability"].keys()),
    }


def main():
    files = sorted(RELIABILITY_DIR.glob("*_reliability.json"))
    print(f"Processing {len(files)} reliability files — stripping '{STRIP_TYPE}' entries\n")
    print(f"{'Agent':<40} {'Before':>7} {'Removed':>8} {'After':>7} {'Overall':>9}  Domains")
    print("-" * 95)

    total_removed = 0
    for filepath in files:
        r = strip_and_recompute(filepath)
        total_removed += r["removed"]
        print(
            f"{r['agent']:<40} {r['before']:>7} {r['removed']:>8} "
            f"{r['after']:>7} {r['overall']:>9.3f}  {r['domains']}"
        )

    print(f"\nTotal volcanic_seismic records removed: {total_removed}")
    print("Reliability files now reflect training-only state (flood + wildfire + hazmat).")
    print(f"Backup preserved in: results/reliability_backup_*/")


if __name__ == "__main__":
    main()
