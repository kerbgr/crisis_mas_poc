"""
GAT Attention-Weight Trainer

Learns the 4 attention coefficients [w_conf, w_rel, w_cert, w_sim] that
maximise the probability of recovering the consensus ground-truth alternative
on the training corpus (flood + forest_fire + hazmat runs).

Algorithm
---------
* Forward pass:  re-simulate the GAT aggregation using stored agent features
* Loss:          cross-entropy  -mean(log softmax(T * DQS)[ground_truth_idx])
                 + L2 regularisation toward hand-crafted prior [0.4, 0.3, 0.3, 0.2]
* Optimiser:     scipy.optimize.minimize with L-BFGS-B and box bounds [0, 1]
* Warm start:    prior weights [0.4, 0.3, 0.3, 0.2]

Constraint enforcement (soft, inside forward):
  w_conf + w_rel + w_cert are L1-normalised at each forward call, so the
  optimiser searches unconstrained in [0, 1] and the decoder ensures the
  ratio is what matters.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from decision_framework.gat_aggregator import GraphAttentionLayer

logger = logging.getLogger(__name__)

# Shared feature extractor (stateless, reused across all forward calls)
_FEATURE_EXTRACTOR = GraphAttentionLayer(feature_dim=9)

# Default weights path
DEFAULT_WEIGHTS_PATH = "models/gat_weights/gat_trained_weights.json"


class GATTrainer:
    """Optimise GAT attention weights on the training corpus."""

    PRIOR_WEIGHTS = np.array([0.4, 0.3, 0.3, 0.2], dtype=np.float64)
    L2_LAMBDA = 0.1        # Regularisation strength toward prior
    SOFTMAX_TEMP = 10.0    # Temperature scaling for DQS cross-entropy loss

    def __init__(self, corpus: List[Dict]):
        if not corpus:
            raise ValueError("Training corpus is empty.")
        self.corpus = corpus
        self._loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, max_iter: int = 300, verbose: bool = True) -> np.ndarray:
        """Optimise attention weights; returns the learned weight vector."""
        logger.info(
            f"Starting GAT weight optimisation: "
            f"{len(self.corpus)} samples, max_iter={max_iter}"
        )
        t0 = time.time()

        initial_loss = self._loss(self.PRIOR_WEIGHTS)
        if verbose:
            print(f"  Initial loss (prior weights): {initial_loss:.4f}")

        result = minimize(
            fun=self._loss,
            x0=self.PRIOR_WEIGHTS.copy(),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * 4,
            options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-6},
        )

        learned = result.x.astype(np.float32)
        final_loss = result.fun
        elapsed = time.time() - t0

        if verbose:
            print(f"  Final loss (learned weights): {final_loss:.4f}")
            print(f"  Converged: {result.success}  ({result.message})")
            print(f"  Iterations: {result.nit}  |  Time: {elapsed:.1f}s")

        logger.info(
            f"Training complete: loss {initial_loss:.4f} -> {final_loss:.4f}, "
            f"converged={result.success}, iters={result.nit}, "
            f"weights={learned.tolist()}"
        )
        return learned

    def evaluate(
        self,
        test_corpus: Optional[List[Dict]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute accuracy metrics on corpus (or test_corpus if provided).

        Returns dict with top1_accuracy, mean_rank, mean_rank_percentile.
        """
        corpus = test_corpus if test_corpus is not None else self.corpus
        w = weights if weights is not None else self.PRIOR_WEIGHTS

        ranks, hits = [], []
        for sample in corpus:
            try:
                dqs = self._forward(w, sample)
                if not dqs:
                    continue
                sorted_alts = sorted(dqs, key=dqs.get, reverse=True)
                gt = sample["ground_truth"]
                rank = sorted_alts.index(gt) + 1 if gt in sorted_alts else len(sorted_alts) + 1
                ranks.append(rank)
                hits.append(1 if rank == 1 else 0)
            except Exception as exc:
                logger.debug(f"Eval forward failed: {exc}")

        if not ranks:
            return {"top1_accuracy": 0.0, "mean_rank": 0.0, "mean_rank_percentile": 0.0}

        n_alts_avg = np.mean([len(s["mcda_scores"]) for s in corpus])
        return {
            "top1_accuracy": float(np.mean(hits)),
            "mean_rank": float(np.mean(ranks)),
            "mean_rank_percentile": float(np.mean([(r - 1) / max(n_alts_avg - 1, 1) for r in ranks])),
            "n_samples": len(ranks),
        }

    # ------------------------------------------------------------------
    # Internal: forward pass
    # ------------------------------------------------------------------

    def _forward(self, w_raw: np.ndarray, sample: Dict) -> Dict[str, float]:
        """Re-simulate the GAT DQS computation for a single training sample.

        Normalises w_raw internally so the optimiser works unconstrained.
        """
        assessments = sample["agent_assessments"]
        mcda_scores = sample["mcda_scores"]
        scenario    = sample["scenario"]

        agent_ids = list(assessments.keys())
        n = len(agent_ids)
        if n == 0:
            return {}

        # Decode weights: first 3 sum to 1, 4th is non-negative similarity bonus
        w = _decode_weights(w_raw)

        # Extract 9-dimensional feature vectors
        features = {
            aid: _FEATURE_EXTRACTOR.extract_agent_features(aid, asmt, scenario)
            for aid, asmt in assessments.items()
        }
        feat_matrix = np.stack([features[aid] for aid in agent_ids])  # (N, 9)

        # Compute attention logits (N × N)
        logits = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                f_i, f_j = feat_matrix[i], feat_matrix[j]
                score = (
                    w[0] * f_j[0] +   # confidence
                    w[1] * f_j[2] +   # expertise relevance
                    w[2] * f_j[1]     # belief certainty
                )
                sim = (np.dot(f_i, f_j)
                       / (np.linalg.norm(f_i) * np.linalg.norm(f_j) + 1e-10))
                score += w[3] * max(float(sim), 0.0)
                # LeakyReLU (slope=0.2)
                logits[i, j] = score if score >= 0 else 0.2 * score

        # Row-wise softmax → attention weights
        attention = _row_softmax(logits)

        # Aggregate beliefs via self-attention (diagonal)
        all_alts: set = set()
        for asmt in assessments.values():
            all_alts.update(asmt.get("belief_distribution", {}).keys())

        aggregated: Dict[str, float] = {}
        for alt in all_alts:
            ws = tw = 0.0
            for i, aid in enumerate(agent_ids):
                b = assessments[aid].get("belief_distribution", {}).get(alt, 0.0)
                aw = float(attention[i, i])
                ws += aw * b
                tw += aw
            aggregated[alt] = ws / tw if tw > 1e-12 else 0.0

        # L1 normalise aggregated beliefs
        tot = sum(aggregated.values())
        if tot > 1e-12:
            aggregated = {k: v / tot for k, v in aggregated.items()}

        # L1 normalise MCDA scores
        mcda_tot = sum(mcda_scores.values())
        mcda_norm = (
            {k: v / mcda_tot for k, v in mcda_scores.items()}
            if mcda_tot > 1e-12 else mcda_scores
        )

        # DQS = 0.6 * beliefs + 0.4 * MCDA  (only alts present in mcda_scores)
        dqs: Dict[str, float] = {}
        for alt in mcda_scores:
            dqs[alt] = (0.6 * aggregated.get(alt, 0.0)
                        + 0.4 * mcda_norm.get(alt, 0.0))
        return dqs

    # ------------------------------------------------------------------
    # Internal: loss
    # ------------------------------------------------------------------

    def _loss(self, w_raw: np.ndarray) -> float:
        """Cross-entropy loss + L2 regularisation."""
        total_ce = 0.0
        n_valid  = 0

        for sample in self.corpus:
            try:
                dqs = self._forward(w_raw, sample)
                gt  = sample["ground_truth"]
                if not dqs or gt not in dqs:
                    continue

                # Softmax over DQS scores (temperature-scaled)
                alts   = list(dqs.keys())
                scores = np.array([dqs[a] for a in alts])
                scores_scaled = scores * self.SOFTMAX_TEMP
                scores_scaled -= scores_scaled.max()          # numerical stability
                exp_s  = np.exp(scores_scaled)
                probs  = exp_s / (exp_s.sum() + 1e-12)

                gt_idx = alts.index(gt)
                ce = -np.log(probs[gt_idx] + 1e-12)
                total_ce += ce
                n_valid  += 1

            except Exception as exc:
                logger.debug(f"Forward pass failed in loss: {exc}")

        if n_valid == 0:
            return 1e6

        mean_ce = total_ce / n_valid

        # L2 regularisation toward hand-crafted prior
        w = _decode_weights(w_raw)
        reg = self.L2_LAMBDA * float(np.sum((w - self.PRIOR_WEIGHTS) ** 2))

        loss = mean_ce + reg
        self._loss_history.append(loss)
        return loss


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _decode_weights(w_raw: np.ndarray) -> np.ndarray:
    """Normalise w_raw so first 3 sum to 1 and 4th is non-negative."""
    w = np.abs(w_raw).astype(np.float64)
    s = w[0] + w[1] + w[2] + 1e-12
    return np.array([w[0] / s, w[1] / s, w[2] / s, w[3]], dtype=np.float64)


def _row_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l   = np.exp(shifted)
    return exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-12)


def save_weights(
    weights: np.ndarray,
    corpus: List[Dict],
    metrics: Dict,
    path: str = DEFAULT_WEIGHTS_PATH,
) -> None:
    """Persist trained weights to JSON alongside training metadata."""
    prior = GATTrainer.PRIOR_WEIGHTS
    payload = {
        "weights":        weights.tolist(),
        "labels":         ["w_confidence", "w_relevance", "w_certainty", "w_similarity"],
        "trained":        True,
        "prior_weights":  prior.tolist(),
        "weight_delta":   (weights - prior).tolist(),
        "training_metrics": metrics,
        "training_scenarios": list({s["scenario_type"] for s in corpus}),
        "n_training_samples": len(corpus),
        "timestamp": datetime.now().isoformat(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Trained weights saved to {path}")
