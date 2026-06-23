"""
Frozen-weight volcanic test runner.

Runs santorini_volcanic_seismic N times using training-phase reliability weights
(frozen - never updated between runs). After each run, captures the new
volcanic_seismic accuracy scores recorded during that run, then restores the
reliability files to their pre-run state.

Zero changes to coordinator, reliability_tracker, or main.py.

Usage:
    python run_frozen_volcanic_test.py [--runs 5] [--provider lmstudio]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RELIABILITY_DIR = Path("results/reliability")
TEST_OUTPUT_DIR = Path("results/reliability_test_volcanic")
SNAPSHOT_DIR = Path("/tmp/crisismas_reliability_snapshot")


def snapshot(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def restore(src: Path, dst: Path):
    shutil.rmtree(dst)
    shutil.copytree(src, dst)


def extract_new_volcanic_records(snapshot_dir: Path, current_dir: Path) -> dict:
    """Return new volcanic_seismic entries added since snapshot, keyed by agent_id."""
    new_records = {}
    for current_file in sorted(current_dir.glob("*_reliability.json")):
        snapshot_file = snapshot_dir / current_file.name
        if not snapshot_file.exists():
            continue
        with open(snapshot_file) as f:
            before = {h["assessment_id"] for h in json.load(f).get("assessment_history", [])}
        with open(current_file) as f:
            current_data = json.load(f)
        after_history = current_data.get("assessment_history", [])
        new = [h for h in after_history
               if h["assessment_id"] not in before
               and h.get("scenario_type") == "volcanic_seismic"]
        if new:
            new_records[current_data["agent_id"]] = new
    return new_records


def compute_run_stats(records: dict) -> dict:
    """Compute per-agent accuracy for this run."""
    stats = {}
    for agent_id, entries in records.items():
        evaluated = [e for e in entries if e.get("evaluated") and e.get("accuracy_score") is not None]
        if evaluated:
            scores = [e["accuracy_score"] for e in evaluated]
            stats[agent_id] = {
                "n": len(scores),
                "mean_accuracy": sum(scores) / len(scores),
                "scores": scores,
            }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--provider", default="lmstudio")
    args = parser.parse_args()

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Frozen-weight volcanic test: {args.runs} runs, provider={args.provider}")
    print(f"Training weights frozen from: {RELIABILITY_DIR}")
    print(f"Results will be saved to: {TEST_OUTPUT_DIR}\n")

    all_run_stats = []
    all_new_records = {}

    for run_num in range(1, args.runs + 1):
        print(f"--- Run {run_num}/{args.runs} ---")

        # Step 1: Snapshot training weights
        snapshot(RELIABILITY_DIR, SNAPSHOT_DIR)
        print(f"  Snapshot saved to {SNAPSHOT_DIR}")

        # Step 2: Run the scenario
        cmd = [
            sys.executable, "main.py",
            "--scenario", "santorini_volcanic_seismic",
            "--llm-provider", args.provider,
            "--expert-selection", "auto",
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            print(f"  WARNING: main.py exited with code {result.returncode} on run {run_num}")

        # Step 3: Extract new volcanic records added by this run
        new_records = extract_new_volcanic_records(SNAPSHOT_DIR, RELIABILITY_DIR)
        run_stats = compute_run_stats(new_records)
        all_run_stats.append(run_stats)

        agents_captured = len(run_stats)
        mean_acc = (
            sum(v["mean_accuracy"] for v in run_stats.values()) / agents_captured
            if agents_captured else 0.0
        )
        print(f"  Captured {agents_captured} agents, mean accuracy={mean_acc:.3f}")

        # Accumulate across runs
        for agent_id, records in new_records.items():
            all_new_records.setdefault(agent_id, []).extend(records)

        # Step 4: Restore training weights (undo reliability update)
        restore(SNAPSHOT_DIR, RELIABILITY_DIR)
        print(f"  Reliability files restored to training state\n")

    # Aggregate results across all runs
    print("=" * 60)
    print(f"FROZEN-WEIGHT TEST RESULTS ({args.runs} runs)")
    print("=" * 60)

    agent_test_scores = {}
    for agent_id, records in all_new_records.items():
        evaluated = [r for r in records if r.get("evaluated") and r.get("accuracy_score") is not None]
        if evaluated:
            scores = [r["accuracy_score"] for r in evaluated]
            agent_test_scores[agent_id] = {
                "n": len(scores),
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
            }

    gold, silver = [], []
    print(f"\n{'Agent':<40} {'n':>4} {'Mean':>7} {'Min':>7} {'Max':>7}  Level")
    print("-" * 75)
    for agent_id, s in sorted(agent_test_scores.items(), key=lambda x: -x[1]["mean"]):
        level = "GOLD" if "gold" in agent_id else "SILVER"
        print(f"{agent_id:<40} {s['n']:>4} {s['mean']:>7.3f} {s['min']:>7.3f} {s['max']:>7.3f}  {level}")
        if level == "GOLD":
            gold.append(s["mean"])
        else:
            silver.append(s["mean"])

    all_scores = [s["mean"] for s in agent_test_scores.values()]
    if gold and silver:
        print(f"\nGOLD avg:   {sum(gold)/len(gold):.3f}")
        print(f"SILVER avg: {sum(silver)/len(silver):.3f}")
        print(f"Gap:        +{(sum(gold)/len(gold)-sum(silver)/len(silver))*100:.1f} pp")
    if all_scores:
        print(f"Range:      {min(all_scores):.3f} - {max(all_scores):.3f}")
    total_records = sum(s["n"] for s in agent_test_scores.values())
    print(f"Total test records: {total_records}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "runs": args.runs,
        "provider": args.provider,
        "mode": "frozen_weights",
        "training_records": 1211,
        "agent_scores": agent_test_scores,
        "per_run_stats": [
            {agent: v for agent, v in run.items()}
            for run in all_run_stats
        ],
        "all_records": all_new_records,
    }
    outfile = TEST_OUTPUT_DIR / f"frozen_test_{args.runs}runs_{args.provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {outfile}")
    print("Training reliability weights unchanged.")


if __name__ == "__main__":
    main()
