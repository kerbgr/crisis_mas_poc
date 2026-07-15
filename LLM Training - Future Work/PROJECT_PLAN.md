# AEGIS LLM Training — Project Execution Plan

This folder is part of **AEGIS** (Adaptive Expert-based Group Intelligence
System — see the repository root README); "the framework" always means AEGIS
throughout these documents.

**Purpose**: convert this folder from a methodology library into an executed
project. Each stage has a binary status (EXECUTED with artifacts / NOT STARTED)
and an explicit exit criterion. Numbers are labeled **measured** or **estimated**;
nothing here may be reported as a result unless it is measured.

**Reference hardware**: MacBook Pro M4 Pro, 48GB unified memory (local, $0/run).
NVIDIA/cloud paths in the parent docs remain valid alternatives and are kept intact.

**Related documents**: [TRAINING plan evaluation → chat record 2026-07-15],
[TOOLS_VALIDATION_REPORT.md](TOOLS_VALIDATION_REPORT.md) (tooling now tested),
[BRAIN_TOPOLOGY.md](BRAIN_TOPOLOGY.md) (architecture direction),
[local_training/README.md](local_training/README.md) (executed pipeline).

---

## Stage 0 — Pipeline smoke test ✅ EXECUTED 2026-07-15

**Goal**: prove the entire loop runs on local hardware before investing in data.

**What ran** (all artifacts in `local_training/stage0/`):
data generation (128 rule-based examples) → MLX QLoRA training (Qwen2.5-0.5B-4bit,
100 iters, **66s**, ~1,350 tok/s, 1.7GB peak, val loss 3.65→0.13, all **measured**)
→ adapter behavior test (doctrine generalizes to held-out parameters) → fuse →
OpenAI-compatible serving → deployment safety checker correctly flagging a
hallucinated IDLH value from the tuned model.

**Exit criterion**: a request through the LM-Studio-compatible API returns
doctrine-consistent output from the fine-tuned weights. **Met** (see
local_training/README.md, "AEGIS integration").

**Carry-forward findings**: mlx-lm CLI drift documented; MLX-native QLoRA
available; `server --adapter-path` broken in 0.31.3 → fuse-first workaround;
tiny models scramble safety-critical facts → evaluation layer is not optional.

---

## Stage 1 — Real dataset v0.1, ONE agent 🔲 NOT STARTED

**Scope discipline**: fire on-scene commander (Pyragos) only. The other 12
agents wait until this one works end-to-end.

**Tasks**:
1. Generate 500–1,000 candidate examples via Method 5 (AI-assisted, see
   `data_collection/README.md` §Method 5), seeded from the Stage-0 scenario
   families plus SOP-derived topics.
2. Validate: self-review 100%, expert review ≥10% sample (or full review if
   expert access allows). Track agreement with
   `tools/data_collection/inter_rater_reliability.py` (target κ > 0.7).
3. Version the dataset with `tools/data_collection/data_versioning.py`
   (hash, provenance, splits).
4. **Language decision gate**: choose Greek-native (Qwen-family) vs English
   with Greek terminology — decide BEFORE collecting at scale, since it
   determines the base model. Greeklish is ruled out for safety-critical use.

**Exit criterion**: versioned dataset ≥500 examples with documented validation
status and split hashes.

**Honest risk**: this stage needs domain-expert hours. Without them, the
dataset ships marked "AI-generated, author-validated only" and the thesis
claims are scoped accordingly.

---

## Stage 2 — Real 7–8B LoRA run 🔲 NOT STARTED

**Tasks**:
1. Re-run base-model selection (`BASE_MODEL_SELECTION.md` criteria) against
   the **current** model generation — the doc's 2025 tier list is stale.
   Candidates must clear: permissive license, instruct-tuned, Greek support
   per the Stage-1 language gate, runs in 48GB. **gpt-oss-20b is a named
   candidate** (Apache 2.0, already an AEGIS runtime provider, strong Greek —
   see Stage 3 baseline note), weighed against a dense 7–8B: as a 21B-total
   MoE in harmony format it is heavier to fine-tune and its training-data
   format is more complex, so "baseline = gpt-oss-20b" does not automatically
   mean "fine-tuning base = gpt-oss-20b". Decide at this gate with measured
   Greek-benchmark numbers from Stage 1.
2. Train with MLX LoRA (or MLX QLoRA — now known to work) using Stage-1 data;
   config mirrors `examples/firefighter_example/` (r=32, α=64, 3 epochs).
3. Record **measured** wall time / tokens/sec / peak memory — these replace
   every "(estimated)" M4 Pro figure in `APPLE_SILICON_GUIDE.md`,
   `fine_tuning/README.md`, and the firefighter example.

**Exit criterion**: fused 7–8B model served locally; training log and measured
resource table committed.

---

## Stage 3 — Evaluation against baseline 🔲 NOT STARTED

**Designated baseline: `gpt-oss-20b`** (mlx-community/gpt-oss-20b-MXFP4-Q8,
Apache 2.0). Rationale, with **measured** evidence from 2026-07-15 on the M4
Pro: already an AEGIS runtime provider (root README, LLM Integration layer)
so baseline comparisons align with the existing experiment corpus; already on
disk (~11GB, LM Studio dir) and loads via mlx-lm in **4.0s**, generating at
**~50 tok/s**; produced fluent, tactically structured **Greek** on a wildfire
command prompt. Note its harmony format emits an English `analysis` channel
before the `final` answer — strip it for scoring, but consider capturing it
in the AEGIS audit trail (free reasoning traces support the anti-black-box
goal). Two baseline roles: (a) untuned reference the fine-tuned model must
beat, (b) the "generic LLM expert" arm in Stage 4's in-system comparison.

**Tasks**:
1. Held-out test set: domain accuracy of the fine-tuned model vs **untuned
   gpt-oss-20b** and vs its own untuned base model, using `tools/evaluation/`
   (now tested: ECE/calibration, robustness suite incl. the previously-missing
   contradictory/adversarial/edge-case tests, fairness suite).
2. Greek-language accuracy benchmark (new — gap identified in plan evaluation);
   gpt-oss-20b is the reference score to beat in Greek.
3. Safety gate: `detect_safety_failures.py` failure rate on test outputs.
4. Expert rating if expert access allows; otherwise scoped-down author rating,
   labeled as such.

**Exit criterion**: `evaluation_results.json` with measured deltas; the
firefighter example README rewritten around real numbers, projection banner
removed.

---

## Stage 4 — AEGIS integration + brain-topology experiment 🔲 NOT STARTED

**Tasks**:
1. Wire the fused model into one AEGIS expert agent via `lmstudio_client`
   (endpoint already proven in Stage 0); run a full scenario
   (`--scenario forest_fire_evia`) with the fine-tuned expert vs the generic
   LLM expert; compare assessment quality, consensus, DQS.
2. Shared-substrate deployment test: one base model + per-agent LoRA adapters
   (the brain-topology deployment argument — see BRAIN_TOPOLOGY.md §4).
3. Optional research experiment: structured GAT topology (BRAIN_TOPOLOGY.md §5).

**Exit criterion**: side-by-side scenario results in `results/`.

---

## Deferred (documented, not planned)

Remaining 12 agents · full fine-tuning · continual-learning pipeline
(`CONTINUAL_LEARNING.md`) · production monitoring deployment
(`DEPLOYMENT.md` §monitoring). These stay as methodology until Stages 1–4 close.

---

**Created**: 2026-07-15 · **Status**: Stage 0 complete, Stage 1 next
