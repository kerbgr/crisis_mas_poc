"""
GAT Training Data Extractor

Reconstructs agent feature vectors and ground-truth labels from historical
run results so the GAT attention weights can be trained offline.

Each training sample contains:
  - agent_assessments: dict[agent_id -> assessment dict] with the fields
    that GATAggregator.extract_agent_features() needs
  - mcda_scores:       dict[alt_id -> float]  (raw TOPSIS, L1-norm at DQS time)
  - ground_truth:      str  (recommended_alternative from the run)
  - scenario:          dict with 'type', 'severity', 'tags'
  - run_dir:           str  (for debugging)
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Map scenario directory names -> scenario JSON file stems
_SCENARIO_DIR_TO_JSON = {
    "flood_scenario":        "flood_scenario",
    "forest_fire_evia":      "forest_fire_evia",
    "ammonia_leak_elefsina": "ammonia_leak_elefsina",
}

# Held-out scenario: NEVER included in training corpus
_HELD_OUT_SCENARIO_TYPES = {"volcanic_seismic", "volcanic", "santorini"}


class GATTrainingDataExtractor:
    """Extract training samples from historical Crisis MAS run results."""

    TRAINING_SCENARIOS = list(_SCENARIO_DIR_TO_JSON.keys())

    def __init__(self, scenarios_dir: str = "scenarios"):
        self.scenarios_dir = Path(scenarios_dir)
        self._scenario_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_corpus(self, results_dir: str = "results") -> List[Dict[str, Any]]:
        """Build full training corpus from all training scenarios.

        Returns list of sample dicts. Skips incomplete or held-out runs.
        """
        corpus: List[Dict[str, Any]] = []
        results_path = Path(results_dir)

        for scenario_dir_name in self.TRAINING_SCENARIOS:
            scenario_path = results_path / scenario_dir_name
            if not scenario_path.exists():
                logger.warning(f"Scenario results dir not found: {scenario_path}")
                continue

            run_dirs = sorted(
                d for d in scenario_path.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            )

            for run_dir in run_dirs:
                sample = self.extract_sample_from_run(run_dir, scenario_dir_name)
                if sample is not None:
                    corpus.append(sample)
                else:
                    logger.debug(f"Skipped run: {run_dir}")

        logger.info(
            f"Built training corpus: {len(corpus)} samples from "
            f"{len(self.TRAINING_SCENARIOS)} scenarios"
        )
        return corpus

    def extract_sample_from_run(
        self,
        run_dir: Path,
        scenario_dir_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Extract one training sample from a run directory.

        Prefers top-level results.json; falls back to er/results.json for
        comparative runs.
        """
        run_dir = Path(run_dir)

        results_file = run_dir / "results.json"
        if not results_file.exists():
            results_file = run_dir / "er" / "results.json"
        if not results_file.exists():
            return None

        try:
            data = json.loads(results_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Cannot read {results_file}: {exc}")
            return None

        decision = data.get("decision", {})

        # Guard: skip held-out scenarios
        scenario_type = self._infer_scenario_type(decision, scenario_dir_name)
        if scenario_type in _HELD_OUT_SCENARIO_TYPES:
            logger.info(f"Skipping held-out scenario run: {run_dir}")
            return None

        ground_truth = decision.get("recommended_alternative")
        if not ground_truth:
            return None

        mcda_scores: Dict[str, float] = decision.get("mcda_scores", {})
        if not mcda_scores:
            return None

        # Parse per-agent assessments from collection_info
        raw_assessments: Dict[str, Any] = (
            decision.get("collection_info", {}).get("assessments", {})
        )
        if not raw_assessments:
            return None

        agent_assessments = self._parse_assessments(raw_assessments)
        if not agent_assessments:
            return None

        # Guard: ground_truth alternative must appear in mcda_scores
        if ground_truth not in mcda_scores:
            return None

        scenario_meta = self._load_scenario_meta(scenario_dir_name or scenario_type)

        return {
            "agent_assessments": agent_assessments,
            "mcda_scores": mcda_scores,
            "ground_truth": ground_truth,
            "scenario": scenario_meta,
            "scenario_type": scenario_type,
            "run_dir": str(run_dir),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_scenario_type(
        self,
        decision: Dict[str, Any],
        scenario_dir_name: Optional[str]
    ) -> str:
        """Best-effort inference of scenario type for held-out guard."""
        if scenario_dir_name:
            if "flood" in scenario_dir_name:
                return "flood"
            if "fire" in scenario_dir_name or "wildfire" in scenario_dir_name:
                return "wildfire"
            if "hazmat" in scenario_dir_name or "ammonia" in scenario_dir_name:
                return "hazmat"
            if "santorini" in scenario_dir_name or "volcanic" in scenario_dir_name:
                return "volcanic_seismic"
        return "unknown"

    def _load_scenario_meta(self, scenario_key: str) -> Dict[str, Any]:
        """Load severity and tags from the scenario JSON file."""
        if scenario_key in self._scenario_cache:
            return self._scenario_cache[scenario_key]

        stem = _SCENARIO_DIR_TO_JSON.get(scenario_key, scenario_key)
        scenario_file = self.scenarios_dir / f"{stem}.json"

        if not scenario_file.exists():
            # Fallback: minimal defaults keyed by type
            defaults = {
                "flood":    {"type": "flood",    "severity": 0.8, "tags": ["flood", "urban"]},
                "wildfire": {"type": "wildfire", "severity": 0.9, "tags": ["wildfire", "forest"]},
                "hazmat":   {"type": "hazmat",   "severity": 0.75, "tags": ["hazmat", "industrial"]},
            }
            meta = defaults.get(scenario_key, {"type": scenario_key, "severity": 0.7, "tags": []})
        else:
            raw = json.loads(scenario_file.read_text())
            meta = {
                "type":     raw.get("type", scenario_key),
                "severity": raw.get("severity", 0.7),
                "tags":     raw.get("tags", []),
            }

        self._scenario_cache[scenario_key] = meta
        return meta

    def _parse_assessments(
        self,
        raw: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Convert raw assessment values (str or dict) into plain dicts."""
        result: Dict[str, Dict[str, Any]] = {}
        for agent_id, value in raw.items():
            if isinstance(value, dict):
                parsed = value
            elif isinstance(value, str):
                parsed = _parse_assessment_string(value)
            else:
                continue

            if parsed:
                result[agent_id] = parsed

        return result


# ------------------------------------------------------------------
# Assessment string parser
# ------------------------------------------------------------------

def _parse_assessment_string(s: str) -> Optional[Dict[str, Any]]:
    """Parse a Pydantic __repr__-style assessment string into a plain dict.

    Extracts the subset of fields needed for GAT feature extraction:
      belief_distribution, confidence, expertise, reliability_score,
      key_concerns, reasoning
    """
    try:
        result: Dict[str, Any] = {}

        # confidence (simple float after 'confidence=')
        m = re.search(r"\bconfidence=(\d+(?:\.\d+)?)", s)
        result["confidence"] = float(m.group(1)) if m else 0.5

        # reliability_score
        m = re.search(r"\breliability_score=(\d+(?:\.\d+)?)", s)
        result["reliability_score"] = float(m.group(1)) if m else 0.8

        # expertise from metadata dict
        m = re.search(r"'expertise':\s*'([^']*)'", s)
        result["expertise"] = m.group(1) if m else ""

        # belief_distribution — extract the inner dict literal
        m = re.search(r"BeliefDistribution\(beliefs=(\{[^}]+\})", s)
        if m:
            result["belief_distribution"] = ast.literal_eval(m.group(1))
        else:
            result["belief_distribution"] = {}

        # key_concerns — extract list literal (may span multiple lines)
        m = re.search(r"key_concerns=(\[.*?\])(?=\s+recommended_actions)", s, re.DOTALL)
        if m:
            try:
                result["key_concerns"] = ast.literal_eval(m.group(1))
            except Exception:
                result["key_concerns"] = []
        else:
            result["key_concerns"] = []

        # reasoning — text between reasoning=' and ' key_concerns=
        m = re.search(r"reasoning='(.*?)'\s+key_concerns=", s, re.DOTALL)
        result["reasoning"] = m.group(1) if m else ""

        # agent_name (for logging / explanations)
        m = re.search(r"agent_name='([^']*)'", s)
        result["agent_name"] = m.group(1) if m else ""

        return result if result.get("belief_distribution") else None

    except Exception as exc:
        logger.debug(f"Failed to parse assessment string: {exc}")
        return None


# ------------------------------------------------------------------
# Quick corpus stats helper
# ------------------------------------------------------------------

def print_corpus_stats(corpus: List[Dict[str, Any]]) -> None:
    from collections import Counter
    by_type: Counter = Counter(s["scenario_type"] for s in corpus)
    print(f"Training corpus: {len(corpus)} samples")
    for stype, count in sorted(by_type.items()):
        print(f"  {stype:20s}: {count}")
    if corpus:
        n_agents = [len(s["agent_assessments"]) for s in corpus]
        n_alts   = [len(s["mcda_scores"]) for s in corpus]
        print(f"  agents per run : {min(n_agents)}-{max(n_agents)} (avg {sum(n_agents)/len(n_agents):.1f})")
        print(f"  alternatives   : {min(n_alts)}-{max(n_alts)} (avg {sum(n_alts)/len(n_alts):.1f})")
