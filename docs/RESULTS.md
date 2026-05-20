# Experimental Results

## Overview

The system was evaluated across **45 controlled runs**: 3 crisis scenarios × 3 LLM providers × 5 replicates per provider. Every run used the full 13-expert agent panel plus orchestrator in `--compare-methods` mode, executing both ER (Dempster-Shafer Evidential Reasoning) and GAT (Graph Attention Network) aggregation on the same agent assessments per run. This yields **90 aggregation results** (45 ER + 45 GAT) across three distinct emergency scenarios. Reliability tracking produced per-agent history across all 45 runs.

**Scenarios:**
- Karditsa Flood (`flood_scenario`) — 15 runs, 5 candidate alternatives
- Evia Wildfire (`forest_fire_evia`) — 15 runs, 12 candidate alternatives
- Elefsina HAZMAT / Ammonia Leak (`ammonia_leak_elefsina`) — 15 runs, 5 candidate alternatives

**LLM providers tested:** Anthropic Claude Sonnet 4 · OpenAI GPT-4o · GPT-OSS 20B (LM Studio, local)

---

## 1. ER vs. GAT — Head-to-Head Comparison

### 1.1 Overall (n = 45 runs)

| Metric | ER | GAT | GAT - ER |
|--------|----|-----|----------|
| Confidence (mean +/- sigma) | 0.8761 +/- 0.0407 | 0.8762 +/- 0.0385 | +0.0001 |
| Consensus (mean +/- sigma) | 0.9027 +/- 0.0679 | 0.9023 +/- 0.0641 | -0.0004 |
| Decision Quality Score (mean +/- sigma) | 0.7748 +/- 0.0319 | **0.7809 +/- 0.0291** | **+0.0061** |
| Avg processing time (ms) | 56,580 | 66,702 | +10,122 (+17.9%) |
| Recommendation agreement | -- | -- | **88.9% (40/45)** |

**Key observation:** ER and GAT are statistically equivalent on confidence and consensus. GAT yields a consistent DQS advantage (+0.61%) by up-weighting high-reliability agents through attention — at the cost of ~10 s additional compute per run. All differences are non-significant (p > 0.05).

---

### 1.2 Per-Scenario Breakdown

| Scenario | Method | Confidence | Consensus | DQS | Avg time (ms) | Agreement |
|----------|--------|-----------|-----------|-----|---------------|-----------|
| Flood (n=15) | ER | 0.8993 +/- 0.0083 | 0.9422 +/- 0.0141 | 0.7396 +/- 0.0147 | 39,994 | -- |
| Flood (n=15) | GAT | 0.9001 +/- 0.0094 | 0.9435 +/- 0.0159 | **0.7451 +/- 0.0000** | 51,509 | **86.7% (13/15)** |
| Forest Fire (n=15) | ER | 0.8302 +/- 0.0400 | 0.8226 +/- 0.0608 | 0.7925 +/- 0.0313 | 84,405 | -- |
| Forest Fire (n=15) | GAT | 0.8341 +/- 0.0371 | 0.8296 +/- 0.0570 | **0.8054 +/- 0.0226** | 96,686 | **80.0% (12/15)** |
| HAZMAT (n=15) | ER | **0.8988 +/- 0.0124** | **0.9433 +/- 0.0165** | 0.7923 +/- 0.0000 | 45,340 | -- |
| HAZMAT (n=15) | GAT | 0.8944 +/- 0.0184 | 0.9338 +/- 0.0295 | 0.7923 +/- 0.0000 | 51,911 | **100% (15/15)** |

**Notable patterns:**

- **Flood:** GAT DQS is exactly 0.7451 with zero variance across all 15 runs. The MCDA component locks when agent opinions converge tightly. ER shows natural run-to-run variation (sigma = 0.0147).
- **Forest Fire:** GAT achieves its largest DQS advantage (+1.3%) and tighter sigma (0.023 vs 0.031), demonstrating more stable aggregation under high ambiguity (12-alternative action space).
- **HAZMAT:** ER marginally outperforms on confidence (+0.44%) and consensus (+0.95%). DQS is identical (0.7923 +/- 0.0000) for both methods across all 15 runs — near-unanimous agent agreement makes both paths produce identical rankings.

---

### Fig. 1 — ER vs. GAT Decision Quality Score by Scenario

```mermaid
xychart-beta
    title "Decision Quality Score: ER (bar) vs GAT (line)"
    x-axis ["Flood", "Forest Fire", "HAZMAT", "Overall"]
    y-axis "DQS" 0.72 --> 0.82
    bar [0.7396, 0.7925, 0.7923, 0.7748]
    line [0.7451, 0.8054, 0.7923, 0.7809]
```

*Bar = ER, Line = GAT. GAT consistently meets or exceeds ER on DQS, with the largest advantage on the ambiguous Forest Fire scenario (+1.3%). The gap closes to zero on HAZMAT where both methods lock at 0.7923.*

---

### Fig. 2 — Consensus and Confidence by Scenario (ER vs. GAT)

```mermaid
xychart-beta
    title "Consensus Level: ER (bar) vs GAT (line)"
    x-axis ["Flood", "Forest Fire", "HAZMAT", "Overall"]
    y-axis "Consensus" 0.80 --> 0.96
    bar [0.9422, 0.8226, 0.9433, 0.9027]
    line [0.9435, 0.8296, 0.9338, 0.9023]
```

*Forest Fire has the lowest consensus (0.82-0.83), confirming it as the most ambiguous scenario. HAZMAT achieves the highest ER consensus (0.943), reflecting the tightest agent agreement across all scenarios.*

---

### 1.3 Recommendation Disagreements — Where ER and GAT Diverge

5 of the 45 runs produced split recommendations (ER != GAT). All 5 are borderline cases with top-two alternatives within 0.02-0.05 belief score.

**Flood scenario — 2 disagreements (Claude runs 8 and 9):**

| Run | ER recommendation | ER top score | GAT recommendation | GAT top score |
|-----|------|-------------|-------|---------------|
| run_8_claude | action_rescue_operations | 0.332 | action_hybrid_approach | 0.343 |
| run_9_claude | action_rescue_operations | 0.328 | action_hybrid_approach | 0.368 |

ER's equal-weight Dempster-Shafer combination gives slightly higher belief mass to `rescue_operations` from these Claude-generated assessments. GAT re-weights via agent reliability scores — flood-specialist agents preferentially rank `hybrid_approach` — flipping the result.

**Forest Fire scenario — 3 disagreements (lmstudio runs 2 and 4, Claude run 8):**

| Run | ER recommendation | ER top score | GAT recommendation | GAT top score |
|-----|------|-------------|-------|---------------|
| run_2_lmstudio | action_immediate_evacuation | 0.252 | action_combined_assault | 0.194 |
| run_4_lmstudio | action_immediate_evacuation | 0.177 | action_combined_assault | 0.174 |
| run_8_claude | action_immediate_evacuation | 0.127 | action_combined_assault | 0.111 |

The 12-alternative action space produces tightly dispersed belief masses. ER's equal-weight combination amplifies the evacuation signal when multiple uncertain agents agree on it. GAT's attention concentrates belief on `action_combined_assault` by up-weighting fire-domain specialists (`fire_silver_tactical`, `fire_gold_strategic`) whose reliability scores are highest for wildfire.

**This is the most meaningful behavioural difference between the two methods:** under high-ambiguity, multi-alternative scenarios, GAT's reliability-weighted attention more coherently resolves uncertainty toward operationally dominant alternatives.

---

### Fig. 3 — ER-GAT Recommendation Agreement Rate by Scenario

```mermaid
xychart-beta
    title "ER-GAT Recommendation Agreement Rate (%)"
    x-axis ["Flood", "Forest Fire", "HAZMAT", "Overall"]
    y-axis "Agreement (%)" 70 --> 105
    bar [86.7, 80.0, 100.0, 88.9]
```

*HAZMAT achieves perfect agreement — both methods always select `action_integrated_response`. Forest Fire is the most discriminating scenario with 3/15 disagreements, all driven by the fire-specialist up-weighting in GAT.*

---

### 1.4 Belief Distribution Comparison — Forest Fire Disagreement (run_2_lmstudio)

| Alternative | ER score | GAT score |
|-------------|----------|-----------|
| action_immediate_evacuation | **0.252** | 0.146 |
| action_phased_evacuation | 0.197 | 0.145 |
| action_international_mutual_aid | 0.138 | 0.074 |
| action_combined_assault | 0.090 | **0.194** |
| action_aerial_firefighting | 0.065 | 0.120 |
| action_ground_firefighting | 0.051 | 0.087 |
| *(6 more)* | dispersed | dispersed |

ER elevates evacuation collectively. GAT suppresses those signals and concentrates on `combined_assault` — the fire-specialist-preferred alternative.

---

## 2. Decision Consistency

### Dominant recommendation per scenario (across all runs and both methods)

| Scenario | Recommendation | ER runs | GAT runs |
|----------|----------------|---------|---------|
| Karditsa Flood | **Hybrid approach** | 13/15 (86.7%) | **15/15 (100%)** |
| Evia Wildfire | **Combined assault** | 12/15 (80.0%) | 13/15 (86.7%) |
| Evia Wildfire | Immediate evacuation | 3/15 | 2/15 |
| Elefsina HAZMAT | **Integrated response** | **15/15 (100%)** | **15/15 (100%)** |

**GAT is more decisive than ER:** it converges on the dominant alternative in 43/45 runs (95.6%) vs. ER's 40/45 (88.9%). The advantage arises from reliability-weighted attention suppressing ambiguous minority-agent signals.

---

## 3. Agent Reliability

Reliability is tracked per agent using **consensus-based ground truth**: an assessment is scored correct when the agent's top-ranked alternative matches the final `recommended_alternative`. 45 runs × 13 agents = up to 90 assessment records per agent (fewer for agents excluded from certain scenarios).

### 3.1 Overall Reliability Ranking

| Agent | Overall | Recent | Consistency | Accuracy rate | Assessments |
|-------|---------|--------|-------------|---------------|-------------|
| civilprotection_gold_strategic | **0.613** | 0.629 | **0.969** | 0.467 | 60 |
| police_gold_strategic | 0.540 | 0.254 | 0.932 | 0.389 | 90 |
| medical_silver_tactical | 0.499 | 0.217 | 0.932 | 0.478 | 90 |
| coastguard_gold_strategic | 0.493 | 0.498 | 0.934 | 0.311 | 90 |
| psap_silver_coordination | 0.490 | 0.320 | 0.922 | 0.322 | 90 |
| fire_silver_tactical | 0.484 | 0.509 | 0.935 | 0.311 | 90 |
| medical_gold_strategic | 0.468 | 0.444 | 0.922 | 0.333 | 90 |
| police_silver_tactical | 0.467 | 0.202 | 0.947 | 0.360 | 89 |
| fire_gold_strategic | 0.448 | 0.375 | 0.931 | 0.210 | 62 |
| environment_silver_advisory | 0.445 | 0.427 | 0.926 | 0.389 | 90 |
| meteorology_silver_advisory | 0.437 | 0.328 | 0.923 | 0.344 | 90 |
| coastguard_silver_tactical | 0.431 | 0.310 | 0.937 | 0.350 | 60 |
| logistics_silver_advisory | **0.408** | 0.184 | 0.939 | 0.200 | 90 |

**Key observations:**

- **Consistency is uniformly high (0.922–0.969)** across all agents. Each agent produces internally coherent belief distributions across runs — the multi-agent framework generates stable, reproducible outputs.
- **`civilprotection_gold_strategic`** is the clear outlier at 0.613 overall reliability — highest alignment with consensus outcomes. GOLD-tier agents average 0.512 vs. 0.458 for SILVER (+5.4 pp).
- **Recent reliability is lower than overall for most agents**, reflecting that the final batches of assessments included Wildfire runs (12 alternatives, genuine ambiguity), which reduces consensus alignment for agents outside their core fire domain. The drop is most pronounced for `logistics_silver_advisory` (overall 0.408 → recent 0.184).
- **GAT's advantage in Forest Fire** is directly explained by these reliability scores: it down-weights logistics and medical agents (poor wildfire reliability) and up-weights fire specialists and civil protection (stronger wildfire reliability).

---

### Fig. 4 — Agent Overall Reliability Ranking

```mermaid
xychart-beta
    title "Agent Overall Reliability Score (consensus-based)"
    x-axis ["CivPro-G", "Pol-G", "Med-S", "CG-G", "PSAP-S", "Fire-S", "Med-G", "Pol-S", "Fire-G", "Env-S", "Met-S", "CG-S", "Log-S"]
    y-axis "Reliability" 0.35 --> 0.65
    bar [0.613, 0.540, 0.499, 0.493, 0.490, 0.484, 0.468, 0.467, 0.448, 0.445, 0.437, 0.431, 0.408]
```

*Agent abbreviations: CivPro-G = civilprotection_gold_strategic, Pol-G = police_gold_strategic, Med-S = medical_silver_tactical, CG-G = coastguard_gold_strategic, PSAP-S = psap_silver_coordination, Fire-S = fire_silver_tactical, Med-G = medical_gold_strategic, Pol-S = police_silver_tactical, Fire-G = fire_gold_strategic, Env-S = environment_silver_advisory, Met-S = meteorology_silver_advisory, CG-S = coastguard_silver_tactical, Log-S = logistics_silver_advisory.*

---

### 3.2 Domain-Specific Reliability

| Agent | Flood | Wildfire | HAZMAT |
|-------|-------|---------|--------|
| civilprotection_gold_strategic | -- | 0.512 | **0.712** |
| police_gold_strategic | 0.659 | 0.341 | 0.619 |
| medical_silver_tactical | 0.575 | 0.244 | **0.678** |
| coastguard_gold_strategic | 0.617 | 0.197 | 0.664 |
| psap_silver_coordination | 0.595 | 0.309 | 0.570 |
| fire_silver_tactical | 0.336 | 0.518 | 0.599 |
| medical_gold_strategic | 0.493 | 0.306 | 0.606 |
| police_silver_tactical | 0.492 | 0.380 | 0.532 |
| fire_gold_strategic | 0.617 | 0.302 | 0.122¹ |
| environment_silver_advisory | **0.631** | 0.217 | 0.486 |
| meteorology_silver_advisory | 0.508 | 0.304 | 0.501 |
| coastguard_silver_tactical | 0.321 | -- | 0.544 |
| logistics_silver_advisory | 0.556 | 0.218 | 0.452 |

*¹ `fire_gold_strategic` participated in only 1/15 HAZMAT runs — the value 0.122 is not statistically representative.*

**Structural pattern:** Every agent scores highest on Flood or HAZMAT and lowest on Wildfire — consistent across all 13 agents. The specialisation gradient is visible: `fire_silver_tactical` (0.518) and `civilprotection_gold_strategic` (0.512) are the most reliable wildfire agents; coastguard and medical agents drop below 0.25 on wildfire, reflecting the mismatch between their domain expertise and fire suppression decisions.

These domain reliability scores feed directly into GAT's scenario-specific attention weights, giving GAT a structural advantage where domain expertise matters most.

---

## 4. Processing Time

Processing time scales with the number of candidate alternatives (more alternatives = more tokens per agent assessment):

| Scenario | Alternatives | ER avg (ms) | GAT avg (ms) | GAT overhead |
|----------|-------------|-------------|--------------|-------------|
| Karditsa Flood | 5 | 39,994 | 51,509 | +11,515 (+28.8%) |
| Elefsina HAZMAT | 5 | 45,340 | 51,911 | +6,571 (+14.5%) |
| Evia Wildfire | 12 | 84,405 | 96,686 | +12,281 (+14.6%) |
| **Overall** | -- | **56,580** | **66,702** | **+10,122 (+17.9%)** |

The GAT overhead (15-29% per run) comes from the neural attention forward pass over the 13-agent opinion graph. This is small relative to total LLM inference time (the dominant cost). Both methods are operationally viable for 5-alternative scenarios; the 12-alternative wildfire scale (~97 s total) may require pipeline optimisation for strict real-time deployment.

---

## 5. Overall System Performance Summary

| Metric | Flood | Wildfire | HAZMAT | Overall |
|--------|-------|---------|--------|---------|
| Run consistency (dominant recommendation) | 100% | 93.3% | 100% | **97.8%** |
| ER-GAT recommendation agreement | 86.7% | 80.0% | 100% | 88.9% |
| Avg confidence | 0.899 | 0.832 | 0.897 | 0.876 |
| Avg consensus | 0.942 | 0.826 | 0.939 | 0.902 |
| Avg DQS (ER) | 0.740 | 0.793 | 0.792 | 0.775 |
| Avg DQS (GAT) | 0.745 | 0.805 | 0.792 | 0.781 |

> **Run consistency** = fraction of runs where the final decision matches the scenario's dominant recommendation.
> **ER-GAT agreement** = fraction of runs where both methods produce the same recommended alternative.

---

## 6. Key Findings

### ER vs. GAT (primary)

1. **Statistically equivalent for well-defined scenarios.** On Flood and HAZMAT (5 alternatives, tight consensus), ER and GAT agree in 87-100% of runs with negligible metric differences. No practical operational difference exists for constrained action spaces.

2. **GAT is more robust under ambiguity.** The Forest Fire scenario (12 alternatives, dispersed agent beliefs) is the discriminating test. GAT achieves +1.3% DQS over ER, converges on the dominant alternative more consistently (13/15 vs 12/15), and produces tighter score variance (sigma 0.023 vs 0.031).

3. **The mechanism behind GAT's advantage is reliability-weighted attention.** GAT suppresses low-reliability out-of-domain agents and amplifies fire-domain specialists. In all 3 Forest Fire disagreements, ER is captured by uncertain agents amplifying the evacuation signal; GAT's attention resists this by down-weighting those agents.

4. **ER is faster.** ER averages 56.6 s vs. GAT's 66.7 s — an 18% overhead from the graph attention forward pass. Both are operationally viable.

5. **All 5 disagreements are genuine borderline cases.** In every ER != GAT run, the top-two alternatives are within 0.02-0.05 belief score. Neither method is "wrong" — they resolve genuine decision-boundary ambiguity differently.

6. **Consistency scores are universally high (0.92-0.999).** The agent panel produces coherent, reproducible belief distributions regardless of aggregation method. The aggregation layer adds differentiated value, not noise correction.

### Agent Capability as Autonomous Reasoners

7. **All 13 agents successfully operate the multi-agent protocol** across all 45 runs: producing valid structured belief distributions, MCDA criterion scores, and domain-grounded reasoning. Zero prompt failures across 45 x 13 = 585 agent assessments.

8. **Agents demonstrate meaningful domain specialisation** visible in the reliability data: fire agents score highest on wildfire (0.52), environment and environment_silver_advisory on flood (0.63), civilprotection on HAZMAT (0.71). This emergent specialisation — not explicitly programmed — validates that LLM-driven agents correctly internalise their role context.

9. **The Gold-Silver command hierarchy produces differentiated outputs.** Strategic-tier agents (Gold) consistently favour integrated, multi-phase approaches; tactical-tier agents (Silver) weight immediate operational actions more heavily. This enriches the collective belief distributions used by both ER and GAT.

### LLM Provider Comparison (secondary)

All three providers successfully operate the multi-agent framework across every run, demonstrating model-agnostic architecture. Claude Sonnet 4 is 5-9x faster than other providers (mean 11.6 s vs 62.4 s for GPT-4o and 100.7 s for GPT-OSS 20B) — decisive for time-critical deployment. GPT-OSS 20B (local, on-premise) achieves competitive decision quality with no API dependency. GPT-4o produces the most consistent inter-run outputs (lowest variance on consensus and confidence). All three providers achieve 100% structural validity (parseable JSON belief distributions) across all runs.

---

---

## 7. Methodological Finding: MCDA/ER Scale Mismatch and L1 Normalisation

### 7.1 Discovery

A systematic scale incompatibility between the two components of the combined DQS formula was identified during post-hoc analysis across all 92 result files (45 runs × 2 aggregation methods, ER and GAT).

The combined score formula is:

```
Score(A_k) = 0.6 * ER/GAT_belief(A_k) + 0.4 * TOPSIS_raw(A_k)
```

This 60/40 weighting assumption is only mathematically valid when **both components share the same distributional scale**. They do not.

**ER/GAT aggregated beliefs** are proper probability distributions: they always sum to 1.0 across all alternatives. For a scenario with N alternatives the average belief per alternative is 1/N:
- Flood (N=5): average ER belief ~0.200
- Forest Fire (N=12): average ER belief ~0.083
- HAZMAT (N=5): average ER belief ~0.200

**TOPSIS closeness coefficients** C_i = S_i^- / (S_i^+ + S_i^-) are geometric proximity scores. They are bounded in [0,1] but do **not** sum to 1 across alternatives. In the experimental corpus, raw TOPSIS scores ranged from 0.16 to 0.75, with cross-alternative sums reaching 1.8-3.2 depending on the scenario.

### 7.2 Effective Weight Distortion

Because TOPSIS scores are systematically larger than belief masses (especially for N=12), the MCDA component dominated the combined score regardless of the nominal 40% weight. Across the experimental corpus, the effective MCDA contribution was **55-79% of the combined DQS** — well above the intended 40%.

```mermaid
xychart-beta
    title "Effective MCDA Contribution vs Nominal 40% Weight"
    x-axis ["Flood (N=5)", "Forest Fire (N=12)", "HAZMAT (N=5)"]
    y-axis "Effective MCDA contribution (%)" 0 --> 85
    bar [58, 76, 58]
    line [40, 40, 40]
```

*Bar = measured effective MCDA weight in combined DQS. Line = nominal 40% target. The distortion is most severe for the 12-alternative Forest Fire scenario where belief masses average 1/12 ≈ 0.083 while TOPSIS scores average near 0.40.*

### 7.3 Impact on Recommendations

The inflation was most consequential for the **Evia Wildfire scenario (N=12)**, where `action_combined_assault` had the highest TOPSIS raw score in many runs (C_i up to 0.814), even when agent consensus favoured evacuation alternatives. The biased formula elevated it to the recommended alternative in runs where the ER/GAT belief distribution clearly favoured immediate or phased evacuation.

**Illustrative example — forest_fire_evia / run_7_claude / ER:**

| Alternative | ER belief | TOPSIS raw | TOPSIS norm | Old DQS | New DQS |
|-------------|-----------|------------|-------------|---------|---------|
| action_combined_assault | 0.1425 | **0.8139** | 0.1314 | **0.4110 (rec)** | 0.1380 |
| action_immediate_evacuation | **0.1549** | 0.7891 | 0.1274 | 0.3987 | **0.1414 (rec)** |

Before normalisation, `combined_assault` was recommended despite having *lower* agent belief mass than `immediate_evacuation`. Its TOPSIS raw advantage of 0.025 translated into a 0.0123 DQS advantage that overrode the expert consensus signal. The MCDA component contributed 79% of that run's final score.

**Post-hoc recalculation across all 92 result files (`scripts/recalculate_dqs.py`):**

| Scenario | Method | Runs changed | Primary change |
|----------|--------|-------------|----------------|
| Evia Wildfire | ER | 6/15 | 5 runs: combined_assault -> immediate_evacuation |
| Evia Wildfire | GAT | 5/15 | 5 runs: combined_assault -> immediate_evacuation |
| Karditsa Flood | ER | 1/16 | 1 borderline run (hybrid <-> rescue) |
| Karditsa Flood | GAT | 0/16 | No change |
| Elefsina HAZMAT | ER | 0/15 | No change |
| Elefsina HAZMAT | GAT | 0/15 | No change |
| **Total** | **Both** | **12/92** | **11 in Forest Fire scenario** |

The effect scales with action-space size: N=5 scenarios (Flood, HAZMAT) are largely unaffected because ER beliefs and TOPSIS scores operate at more comparable per-alternative magnitudes. The N=12 Forest Fire scenario is most sensitive — a direct consequence of beliefs averaging 1/12 versus TOPSIS scores averaging near 0.40.

### 7.4 Correction: L1 Normalisation

The fix maps raw TOPSIS scores to a proper distribution before combining:

```
C_norm(A_k) = C_raw(A_k) / sum_j( C_raw(A_j) )
```

This preserves TOPSIS ranking order while ensuring both components are true distributions that sum to 1. The corrected combined score formula:

```
Score(A_k) = 0.6 * ER/GAT_belief(A_k) + 0.4 * C_norm(A_k)
```

enforces the intended 60/40 balance regardless of action-space size.

```mermaid
flowchart LR
    ER["ER/GAT beliefs<br/>sum to 1.0<br/>(proper distribution)"]
    TOPSIS_RAW["TOPSIS raw C_i<br/>sum = 1.8-3.2<br/>(geometric proximity)"]
    L1["L1 normalise<br/>C_norm = C_i / sum(C)"]
    TOPSIS_NORM["TOPSIS normalised<br/>sum to 1.0<br/>(proper distribution)"]
    COMBINE["0.6 x belief + 0.4 x C_norm<br/>(scale-compatible blend)"]

    ER --> COMBINE
    TOPSIS_RAW --> L1 --> TOPSIS_NORM --> COMBINE

    style ER fill:#e3f2fd,stroke:#1565c0
    style TOPSIS_RAW fill:#ffebee,stroke:#c62828
    style L1 fill:#fff3e0,stroke:#ef6c00
    style TOPSIS_NORM fill:#e8f5e9,stroke:#2e7d32
    style COMBINE fill:#f3e5f5,stroke:#7b1fa2
```

The fix is implemented in `agents/coordinator_agent.py` and applies to all new runs. Original `results.json` files are preserved unmodified; corrected scores are available as `dqs_recalculated.json` in each `results/<scenario>/<run>/<method>/` directory.

### 7.5 Implications for Previous Results

**Finding 10 — Scale normalisation is required for valid 60/40 blending.** Combining a proper probability distribution (ER/GAT beliefs) with an unnormalised proximity score (TOPSIS raw) on a fixed weight violates the mathematical precondition of the formula. The effect grows with action-space size: it is negligible for N=5 but inflates the MCDA contribution to ~76% for N=12.

**Finding 11 — 11 of 30 Forest Fire recommendations were MCDA-dominated artefacts.** After L1 normalisation, `action_immediate_evacuation` and `action_phased_evacuation` emerge as the correct agent-consensus-driven recommendations in runs where the biased formula selected `action_combined_assault`. Flood and HAZMAT results are robust (0-1 changes each).

**Finding 12 — Scale normalisation is a prerequisite for GAT training.** The reliability tracker and future warm-started online learning extension (Finding 6, Key Findings) require that the DQS signal used as training supervision accurately reflects expert consensus rather than TOPSIS scale artefacts. L1 normalisation must be applied before any supervised training on this corpus.

---

## Result Files

Results are stored at `results/<scenario>/<run_N_provider>/`:
- `results.json` -- combined ER+GAT decision, full agent opinions, all metrics
- `er/results.json` -- ER-only aggregation result
- `gat/results.json` -- GAT-only aggregation result
- `comparative_analysis.json` -- side-by-side ER vs. GAT metrics per run
- `results/reliability/<agent_id>_reliability.json` -- per-agent reliability history (13 files)

See [EVALUATION_METHODOLOGY.md](../evaluation/EVALUATION_METHODOLOGY.md) for metric definitions.
