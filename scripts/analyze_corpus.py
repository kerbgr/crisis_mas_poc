#!/usr/bin/env python3
"""
Corpus analysis — regenerates the journal paper's result tables from results/.

Reads the 45-run experimental corpus (3 scenarios x 3 providers x 5 replicates,
each with er/ and gat/ result files), the reliability training corpus, and the
frozen-weight volcanic holdout summary. Emits all paper tables (II-V, §4.5
train/holdout) as markdown plus a machine-readable JSON.

Metric definitions (kept consistent with the paper):
  DQS  = raw TOPSIS closeness coefficient of the recommended alternative
         (mcda_scores_raw[recommended]) — criterion-satisfaction quality only.
  CS   = combined score of the recommended alternative
         (final_scores[recommended] = 0.6*belief + 0.4*L1-normalised TOPSIS).
  CL   = consensus_level (mean pairwise cosine similarity of agent beliefs).
  DC   = decision confidence (0.6*CL + 0.4*mean agent confidence).
  ECB  = CS(collective) - max over agents of CS_i, where
         CS_i = 0.6*belief_i(top_i) + 0.4*mcda_norm[top_i] is the combined
         score agent i's own top choice would achieve as a solo decision-maker.

Usage:
    python scripts/analyze_corpus.py [--output-dir results/analysis]
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

try:
    from scipy import stats as sps
except ImportError:
    sps = None

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = {
    "flood_scenario": "Karditsa Flood",
    "forest_fire_evia": "Evia Wildfire",
    "ammonia_leak_elefsina": "Elefsina HAZMAT",
}
PROVIDERS = ["lmstudio", "openai", "claude"]
METHODS = {"er": "ER", "gat": "RBGA"}
HOLDOUT_DIR = ROOT / "results" / "reliability_test_volcanic"
RELIABILITY_DIR = ROOT / "results" / "reliability"


def mean_sd(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


# ---------------------------------------------------------------------------
# Load the 45-run corpus
# ---------------------------------------------------------------------------

def load_runs():
    runs = []
    problems = []
    for scen_dir, scen_name in SCENARIOS.items():
        base = ROOT / "results" / scen_dir
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            provider = run_dir.name.split("_")[-1]
            if provider not in PROVIDERS:
                problems.append(f"unrecognised provider dir: {run_dir}")
                continue
            rec = {"scenario": scen_dir, "scenario_name": scen_name,
                   "run": run_dir.name, "provider": provider, "methods": {}}
            for mdir, mname in METHODS.items():
                f = run_dir / mdir / "results.json"
                if not f.exists():
                    problems.append(f"missing {f}")
                    continue
                d = json.load(open(f))
                dec = d["decision"]
                rec_alt = dec["recommended_alternative"]
                raw = dec.get("mcda_scores_raw", {})
                rec["methods"][mname] = {
                    "recommended": rec_alt,
                    "dqs": raw.get(rec_alt),
                    "cs": dec.get("final_scores", {}).get(rec_alt),
                    "consensus": dec.get("consensus_level"),
                    "confidence": dec.get("confidence"),
                    "time_s": dec.get("decision_time_seconds"),
                    "agents": dec.get("agents_participated"),
                    "excluded_by_terrain": len(dec.get("agents_excluded_by_terrain", []) or []),
                    "mcda_norm": dec.get("mcda_scores", {}),
                    "mcda_raw": dec.get("mcda_scores_raw", {}),
                }
            # individual-agent data for ECB (same assessments both paths; use ER file)
            top = run_dir / "results.json"
            if top.exists():
                d = json.load(open(top))
                ic = d.get("metrics", {}).get("individual_comparisons", {})
                rec["individual_agents"] = ic.get("individual_agents", [])
            runs.append(rec)
    return runs, problems


# ---------------------------------------------------------------------------
# Table computations
# ---------------------------------------------------------------------------

def table2(runs):
    """System performance by scenario (both methods pooled, as in the paper)."""
    rows = []
    overall = {"dqs": [], "cl": [], "dc": [], "t": []}
    for scen, name in SCENARIOS.items():
        dqs, cl, dc, t, recs_er = [], [], [], [], []
        for r in runs:
            if r["scenario"] != scen:
                continue
            for m in r["methods"].values():
                dqs.append(m["dqs"]); cl.append(m["consensus"])
                dc.append(m["confidence"])
            # time: ER path only — in --compare-methods mode the RBGA
            # coordinator reuses the ER collection, so its own timer is ~0
            if "ER" in r["methods"]:
                recs_er.append(r["methods"]["ER"]["recommended"])
                t.append(r["methods"]["ER"]["time_s"])
        modal = max(set(recs_er), key=recs_er.count) if recs_er else None
        consist = sum(1 for x in recs_er if x == modal)
        rows.append({"scenario": name, "dqs": mean_sd(dqs), "cl": mean_sd(cl),
                     "dc": mean_sd(dc), "time": statistics.mean(t),
                     "n_runs": len(recs_er), "modal": modal,
                     "consistency": f"{consist}/{len(recs_er)}"})
        overall["dqs"] += dqs; overall["cl"] += cl; overall["dc"] += dc; overall["t"] += t
    rows.append({"scenario": "Overall", "dqs": mean_sd(overall["dqs"]),
                 "cl": mean_sd(overall["cl"]), "dc": mean_sd(overall["dc"]),
                 "time": statistics.mean(overall["t"]), "n_runs": None,
                 "modal": None, "consistency": None})
    return rows


def table3(runs):
    """ER vs RBGA overall + paired significance tests."""
    out = {}
    for mname in ("ER", "RBGA"):
        for key in ("dqs", "consensus", "confidence"):
            out.setdefault(mname, {})[key] = [r["methods"][mname][key]
                                              for r in runs if mname in r["methods"]]
    agree = [r["methods"]["ER"]["recommended"] == r["methods"]["RBGA"]["recommended"]
             for r in runs if len(r["methods"]) == 2]
    disagreements = [(r["scenario_name"], r["run"],
                      r["methods"]["ER"]["recommended"], r["methods"]["RBGA"]["recommended"])
                     for r in runs if len(r["methods"]) == 2
                     and r["methods"]["ER"]["recommended"] != r["methods"]["RBGA"]["recommended"]]
    tests = {}
    if sps is not None:
        for key in ("dqs", "consensus", "confidence"):
            er = out["ER"][key]; rb = out["RBGA"][key]
            diffs = [a - b for a, b in zip(er, rb)]
            if any(abs(d) > 1e-12 for d in diffs):
                w = sps.wilcoxon(er, rb)
                tests[key] = {"wilcoxon_stat": float(w.statistic), "p": float(w.pvalue)}
            else:
                tests[key] = {"wilcoxon_stat": None, "p": 1.0, "note": "all paired diffs zero"}
    return out, agree, disagreements, tests


def table4(runs):
    """Scenario-level ER vs RBGA breakdown."""
    rows = []
    for scen, name in SCENARIOS.items():
        for mname in ("ER", "RBGA"):
            vals = [r["methods"][mname] for r in runs
                    if r["scenario"] == scen and mname in r["methods"]]
            cl_ok = sum(1 for v in vals if v["consensus"] >= 0.75)
            rows.append({
                "scenario": name, "method": mname,
                "dqs": mean_sd([v["dqs"] for v in vals]),
                "cl": mean_sd([v["consensus"] for v in vals]),
                "dc": mean_sd([v["confidence"] for v in vals]),
                "time": mean_sd([v["time_s"] for v in vals]),
                "cl_ok": f"{cl_ok}/{len(vals)}",
            })
    return rows


def table5(runs):
    """Provider comparison on the ER path (single-collection assessments)."""
    rows = {}
    for p in PROVIDERS:
        vals = [r["methods"]["ER"] for r in runs
                if r["provider"] == p and "ER" in r["methods"]]
        rows[p] = {
            "n": len(vals),
            "cs": mean_sd([v["cs"] for v in vals]),
            "dqs": mean_sd([v["dqs"] for v in vals]),
            "cl": mean_sd([v["consensus"] for v in vals]),
            "dc": mean_sd([v["confidence"] for v in vals]),
            "time": mean_sd([v["time_s"] for v in vals]),
        }
    tests = {}
    if sps is not None:
        for key, getter in (("cs", "cs"), ("consensus", "consensus"),
                            ("confidence", "confidence"), ("time", "time_s")):
            groups = [[r["methods"]["ER"][getter] for r in runs
                       if r["provider"] == p and "ER" in r["methods"]]
                      for p in PROVIDERS]
            h = sps.kruskal(*groups)
            tests[key] = {"H": float(h.statistic), "p": float(h.pvalue)}
    return rows, tests


def ecb(runs):
    """Collective vs individual choice quality, per scenario (DQS scale).

    Both sides are scored on the SAME scale: the raw TOPSIS closeness
    coefficient of the alternative each decision-maker selects. Collective =
    mcda_raw[system recommendation]; individual agent i = mcda_raw[agent i's
    own top-belief alternative]. This isolates the question "does aggregation
    choose alternatives with better criterion satisfaction than solo experts?"
    without mixing belief-mass and TOPSIS scales. Best-individual is
    identified post-hoc (optimistic upper bound), matching the paper's caveat;
    agreement_rate is the fraction of agents whose solo choice equals the
    system recommendation.
    """
    rows = []
    for scen, name in SCENARIOS.items():
        coll, best_ind, mean_ind, agree_frac = [], [], [], []
        for r in runs:
            if r["scenario"] != scen or "ER" not in r["methods"]:
                continue
            m = r["methods"]["ER"]
            raw = m["mcda_raw"]
            ind_scores, agrees = [], 0
            for a in r.get("individual_agents", []):
                beliefs = (a.get("decision_quality", {}).get("final_scores", {})
                           .get("beliefs", {}))
                if not beliefs:
                    continue
                top_alt = max(beliefs, key=beliefs.get)
                ind_scores.append(raw.get(top_alt, 0.0))
                if top_alt == m["recommended"]:
                    agrees += 1
            if not ind_scores:
                continue
            coll.append(raw.get(m["recommended"], 0.0))
            best_ind.append(max(ind_scores))
            mean_ind.append(statistics.mean(ind_scores))
            agree_frac.append(agrees / len(ind_scores))
        if coll:
            rows.append({
                "scenario": name, "n": len(coll),
                "collective": statistics.mean(coll),
                "best_individual": statistics.mean(best_ind),
                "mean_individual": statistics.mean(mean_ind),
                "margin_vs_best_pp": (statistics.mean(coll) - statistics.mean(best_ind)) * 100,
                "margin_vs_mean_pp": (statistics.mean(coll) - statistics.mean(mean_ind)) * 100,
                "solo_agreement_rate": statistics.mean(agree_frac),
            })
    return rows


def training_reliability():
    agents = {}
    total = 0
    for f in sorted(RELIABILITY_DIR.glob("*_reliability.json")):
        d = json.load(open(f))
        hist = [h for h in d.get("assessment_history", [])
                if h.get("evaluated") and h.get("accuracy_score") is not None]
        total += len(d.get("assessment_history", []))
        if not hist:
            continue
        by_type = {}
        for h in hist:
            by_type.setdefault(h.get("scenario_type", "?"), []).append(h["accuracy_score"])
        agents[d["agent_id"]] = {
            "n": len(hist),
            "overall": statistics.mean(h["accuracy_score"] for h in hist),
            "by_type": {k: statistics.mean(v) for k, v in by_type.items()},
        }
    gold = [v["overall"] for k, v in agents.items() if "gold" in k]
    silver = [v["overall"] for k, v in agents.items() if "gold" not in k]
    return {"agents": agents, "total_records": total,
            "gold_avg": statistics.mean(gold) if gold else None,
            "silver_avg": statistics.mean(silver) if silver else None}


def holdout():
    files = sorted(HOLDOUT_DIR.glob("frozen_test_*.json"))
    if not files:
        return None
    d = json.load(open(files[-1]))
    scores = d["agent_scores"]
    gold = [v["mean"] for k, v in scores.items() if "gold" in k]
    silver = [v["mean"] for k, v in scores.items() if "gold" not in k]
    return {"file": files[-1].name, "provider": d["provider"], "runs": d["runs"],
            "training_records_at_freeze": d.get("training_records"),
            "git_commit": d.get("git_commit"),
            "n_records": sum(v["n"] for v in scores.values()),
            "agents": scores,
            "gold_avg": statistics.mean(gold), "silver_avg": statistics.mean(silver)}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render(runs, problems, out_json):
    L = []
    L.append(f"# Corpus Analysis — generated {datetime.now().isoformat(timespec='seconds')}")
    L.append("")
    n_by = {}
    for r in runs:
        n_by.setdefault((r["scenario_name"], r["provider"]), 0)
        n_by[(r["scenario_name"], r["provider"])] += 1
    L.append(f"Runs found: {len(runs)} "
             f"({', '.join(f'{s}/{p}: {n}' for (s, p), n in sorted(n_by.items()))})")
    agents_seen = sorted({r["methods"]["ER"]["agents"] for r in runs if "ER" in r["methods"]})
    L.append(f"Agents participated per run (distinct values): {agents_seen}")
    excl = sorted({(r['scenario_name'], r['methods']['ER']['excluded_by_terrain'])
                   for r in runs if 'ER' in r['methods']})
    L.append(f"Terrain exclusions (scenario, n_excluded): {excl}")
    if problems:
        L.append(f"\n**PROBLEMS ({len(problems)}):**")
        L += [f"- {p}" for p in problems]
    out_json["run_inventory"] = {f"{s}/{p}": n for (s, p), n in sorted(n_by.items())}

    # Table II
    L.append("\n## Table II — System Performance by Scenario (both methods pooled)")
    L.append("| Metric | " + " | ".join(r["scenario"] for r in out_json["table2"]) + " |")
    L.append("|---" * (len(out_json["table2"]) + 1) + "|")
    t2 = out_json["table2"]
    L.append("| DQS (raw TOPSIS of rec.) | " + " | ".join(fmt(*r["dqs"]) for r in t2) + " |")
    L.append("| Consensus CL | " + " | ".join(fmt(*r["cl"]) for r in t2) + " |")
    L.append("| Confidence DC | " + " | ".join(fmt(*r["dc"]) for r in t2) + " |")
    L.append("| Run consistency (ER modal) | " +
             " | ".join(str(r["consistency"] or "—") for r in t2) + " |")
    L.append("| Mean time (s) | " + " | ".join(f"{r['time']:.1f}" for r in t2) + " |")
    L.append("| Dominant rec. (ER) | " + " | ".join(str(r["modal"] or "—") for r in t2) + " |")

    # Table III
    t3, agree, disagreements, tests = out_json["table3"]
    L.append("\n## Table III — ER vs RBGA (paired, Wilcoxon signed-rank)")
    L.append("| Metric | ER | RBGA | Wilcoxon p |")
    L.append("|---|---|---|---|")
    for key, label in (("dqs", "DQS"), ("consensus", "Consensus CL"), ("confidence", "Confidence DC")):
        p = tests.get(key, {}).get("p")
        L.append(f"| {label} | {fmt(*mean_sd(t3['ER'][key]))} | "
                 f"{fmt(*mean_sd(t3['RBGA'][key]))} | "
                 f"{'n/a' if p is None else f'{p:.3f}'} |")
    L.append(f"| Recommendation agreement | — | — | "
             f"{sum(agree)}/{len(agree)} ({100*sum(agree)/len(agree):.1f} %) |")
    if disagreements:
        L.append("\nDisagreements:")
        for s, run, er, rb in disagreements:
            L.append(f"- {s} / {run}: ER={er} vs RBGA={rb}")

    # Table IV
    L.append("\n## Table IV — Scenario-Level ER vs RBGA Breakdown")
    L.append("| Scenario | Method | DQS | Consensus | Confidence | Time (s) | CL ≥ 0.75 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in out_json["table4"]:
        L.append(f"| {r['scenario']} | {r['method']} | {fmt(*r['dqs'])} | "
                 f"{fmt(*r['cl'])} | {fmt(*r['dc'])} | "
                 f"{r['time'][0]:.1f} ± {r['time'][1]:.1f} | {r['cl_ok']} |")
    L.append("\n*RBGA rows show ~0 s because in --compare-methods mode the RBGA "
             "coordinator reuses the ER collection's assessments; its time is "
             "aggregation-only. End-to-end run time is the ER-path figure.*")

    # Table V
    t5, t5tests = out_json["table5"]
    L.append("\n## Table V — LLM Provider Comparison (ER path, n per provider below)")
    L.append("| Provider | n | Combined Score | DQS | Consensus | Confidence | Time (s) |")
    L.append("|---|---|---|---|---|---|---|")
    for p in PROVIDERS:
        r = t5[p]
        L.append(f"| {p} | {r['n']} | {fmt(*r['cs'])} | {fmt(*r['dqs'])} | "
                 f"{fmt(*r['cl'])} | {fmt(*r['dc'])} | {r['time'][0]:.1f} ± {r['time'][1]:.1f} |")
    if t5tests:
        L.append("\nKruskal-Wallis across providers: " + "; ".join(
            f"{k}: H={v['H']:.2f}, p={v['p']:.4f}" for k, v in t5tests.items()))

    # ECB
    L.append("\n## Collective vs Individual choice quality (raw TOPSIS of chosen alternative)")
    L.append("| Scenario | n | Collective | Best individual | Mean individual | "
             "vs best (pp) | vs mean (pp) | Solo-agrees-with-system |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in out_json["ecb"]:
        L.append(f"| {r['scenario']} | {r['n']} | {r['collective']:.3f} | "
                 f"{r['best_individual']:.3f} | {r['mean_individual']:.3f} | "
                 f"{r['margin_vs_best_pp']:+.1f} | {r['margin_vs_mean_pp']:+.1f} | "
                 f"{100*r['solo_agreement_rate']:.1f} % |")

    # Training reliability
    tr = out_json["training_reliability"]
    L.append(f"\n## §4.5 Training Reliability ({tr['total_records']} records)")
    L.append("| Agent | n | Overall | flood | wildfire | hazmat |")
    L.append("|---|---|---|---|---|---|")
    for aid, v in sorted(tr["agents"].items(), key=lambda x: -x[1]["overall"]):
        bt = v["by_type"]
        L.append(f"| {aid} | {v['n']} | {v['overall']:.3f} | "
                 f"{bt.get('flood', float('nan')):.3f} | {bt.get('wildfire', float('nan')):.3f} | "
                 f"{bt.get('hazmat', float('nan')):.3f} |")
    L.append(f"\nGOLD avg: {tr['gold_avg']:.3f} | SILVER avg: {tr['silver_avg']:.3f} | "
             f"gap: {100*(tr['gold_avg']-tr['silver_avg']):+.1f} pp")

    # Holdout
    ho = out_json["holdout"]
    if ho:
        L.append(f"\n## §4.5 Frozen-Weight Holdout ({ho['file']}, {ho['n_records']} records, "
                 f"{ho['runs']} runs, provider {ho['provider']}, "
                 f"training corpus at freeze: {ho['training_records_at_freeze']})")
        L.append("| Agent | n | Mean | Min | Max |")
        L.append("|---|---|---|---|---|")
        for aid, v in sorted(ho["agents"].items(), key=lambda x: -x[1]["mean"]):
            L.append(f"| {aid} | {v['n']} | {v['mean']:.3f} | {v['min']:.3f} | {v['max']:.3f} |")
        L.append(f"\nGOLD avg: {ho['gold_avg']:.3f} | SILVER avg: {ho['silver_avg']:.3f} | "
                 f"gap: {100*(ho['gold_avg']-ho['silver_avg']):+.1f} pp")
    else:
        L.append("\n## §4.5 Frozen-Weight Holdout: NOT FOUND")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(ROOT / "results" / "analysis"))
    args = ap.parse_args()

    runs, problems = load_runs()
    out = {
        "generated": datetime.now().isoformat(),
        "table2": table2(runs),
        "table3": table3(runs),
        "table4": table4(runs),
        "table5": table5(runs),
        "ecb": ecb(runs),
        "training_reliability": training_reliability(),
        "holdout": holdout(),
    }
    md = render(runs, problems, out)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "corpus_analysis.md").write_text(md)
    with open(outdir / "corpus_analysis.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(md)
    print(f"\nSaved to {outdir}/corpus_analysis.md and .json", file=sys.stderr)
    if sps is None:
        print("WARNING: scipy not available — significance tests skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
