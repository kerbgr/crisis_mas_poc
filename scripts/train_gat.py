#!/usr/bin/env python3
"""
GAT Attention-Weight Training Script

Trains the 4 GAT attention coefficients [w_conf, w_rel, w_cert, w_sim] on
historical run results from the three training scenarios (flood, forest_fire,
hazmat). Saves the learned weights to models/gat_weights/gat_trained_weights.json
for use by the coordinator's GAT_TRAINED aggregation method.

Usage
-----
    python scripts/train_gat.py
    python scripts/train_gat.py --results-dir results --output models/gat_weights/gat_trained_weights.json
    python scripts/train_gat.py --max-iter 500 --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from evaluation.visualizations import SystemVisualizer
from training.gat_training_data import GATTrainingDataExtractor, print_corpus_stats
from training.gat_trainer import GATTrainer, save_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GAT attention weights from historical Crisis MAS runs"
    )
    p.add_argument(
        "--results-dir", default="results",
        help="Root results directory (default: results)"
    )
    p.add_argument(
        "--scenarios-dir", default="scenarios",
        help="Scenarios directory for loading metadata (default: scenarios)"
    )
    p.add_argument(
        "--output", default="models/gat_weights/gat_trained_weights.json",
        help="Output path for trained weights JSON"
    )
    p.add_argument(
        "--max-iter", type=int, default=300,
        help="Maximum L-BFGS-B iterations (default: 300)"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    p.add_argument(
        "--plot-dir", default="models/gat_weights",
        help="Directory to save the training result plot (default: models/gat_weights)"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    print("=" * 60)
    print("  Crisis MAS — GAT Attention Weight Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build training corpus
    # ------------------------------------------------------------------
    print("\n[1/4] Building training corpus ...")
    extractor = GATTrainingDataExtractor(scenarios_dir=args.scenarios_dir)
    corpus = extractor.build_corpus(results_dir=args.results_dir)

    if not corpus:
        print("ERROR: No training samples found. Check --results-dir.")
        return 1

    print_corpus_stats(corpus)

    # ------------------------------------------------------------------
    # 2. Baseline evaluation (prior weights)
    # ------------------------------------------------------------------
    print("\n[2/4] Evaluating prior (hand-crafted) weights ...")
    trainer = GATTrainer(corpus)
    prior_metrics = trainer.evaluate(weights=GATTrainer.PRIOR_WEIGHTS)
    _print_metrics("Prior [0.40, 0.30, 0.30, 0.20]", prior_metrics)

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print(f"\n[3/4] Optimising attention weights (max_iter={args.max_iter}) ...")
    learned_weights = trainer.train(max_iter=args.max_iter, verbose=True)

    print(f"\n  Learned weights:")
    labels = ["w_confidence", "w_relevance ", "w_certainty ", "w_similarity"]
    prior  = GATTrainer.PRIOR_WEIGHTS
    for label, lw, pw in zip(labels, learned_weights, prior):
        delta = lw - pw
        print(f"    {label}: {lw:.4f}  (prior {pw:.4f}, delta {delta:+.4f})")

    # ------------------------------------------------------------------
    # 4. Evaluate trained weights
    # ------------------------------------------------------------------
    print("\n[4/4] Evaluating learned weights ...")
    trained_metrics = trainer.evaluate(weights=learned_weights)
    _print_metrics(f"Trained {learned_weights.round(3).tolist()}", trained_metrics)

    delta_top1 = trained_metrics["top1_accuracy"] - prior_metrics["top1_accuracy"]
    delta_rank = prior_metrics["mean_rank"] - trained_metrics["mean_rank"]
    print(f"\n  Improvement vs prior:")
    print(f"    Top-1 accuracy: {delta_top1:+.1%}")
    print(f"    Mean rank:      {delta_rank:+.2f} (lower is better)")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    save_weights(
        weights=learned_weights,
        corpus=corpus,
        metrics={
            "prior":   prior_metrics,
            "trained": trained_metrics,
        },
        path=args.output,
    )
    print(f"\n  Weights saved -> {args.output}")

    # ------------------------------------------------------------------
    # 6. Generate training result plot
    # ------------------------------------------------------------------
    print(f"\n[6/6] Generating training result plot ...")
    try:
        viz = SystemVisualizer(output_dir=args.plot_dir, dpi=150)
        plot_path = viz.plot_gat_training_result(
            weights_path=args.output,
            save_path="gat_training_result.png",
        )
        if plot_path:
            print(f"  Plot saved -> {plot_path}")
    except Exception as exc:
        print(f"  WARNING: could not save plot: {exc}")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("  Run 4-way comparison:")
    print("    python main.py --scenario flood_scenario \\")
    print("      --compare-methods --agents all --llm-provider claude")
    print("=" * 60)

    return 0


def _print_metrics(label: str, m: dict) -> None:
    print(f"  {label}")
    print(f"    Top-1 accuracy:    {m.get('top1_accuracy', 0):.1%}")
    print(f"    Mean rank:         {m.get('mean_rank', 0):.2f}")
    print(f"    Rank percentile:   {m.get('mean_rank_percentile', 0):.1%}")
    print(f"    Samples evaluated: {m.get('n_samples', 0)}")


if __name__ == "__main__":
    sys.exit(main())
