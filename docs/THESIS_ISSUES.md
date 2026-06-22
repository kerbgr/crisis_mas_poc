# AEGIS Thesis — Issues Tracker
_Source: Critical evaluation of `academic_journal_draft_long version.md` against the codebase_

Each issue is tagged **[PAPER]**, **[CODE]**, or **[BOTH]** and prioritised **HIGH / MEDIUM / LOW**.
Fix target is noted: what changes, where.

---

## 1. The "GAT" naming — HIGH PRIORITY **[BOTH]**

### What the paper says
The second aggregation method is labelled "Graph Attention Network (GAT)" and compared to ER in a "first controlled ER-vs-GAT comparison." GAT_TRAINED is presented as a trained variant.

### What the code actually does
`GATAggregator` has **no learnable W projection matrix, no `a` attention vector, no backpropagation, no gradient descent**.  
The attention score is a fixed formula:

```
score = w[0]*f_j[confidence] + w[1]*f_j[relevance] + w[2]*f_j[certainty] + w[3]*cos(f_i,f_j)
```

This is a **hand-crafted scalar weighted average** over 4 features, followed by LeakyReLU + softmax. Nothing in this formula is a neural-network component.

**Multi-head claim is hollow.** All 4 "heads" are created from identical `GraphAttentionLayer` instances with the same formula and the same weights (lines 563–570). Since the forward is deterministic, all 4 heads produce the same output. Averaging 4 identical matrices = 1 matrix. The "multi-head" label is misleading.

### GAT_TRAINED analysis
`GATTrainer` uses `scipy.optimize.minimize` (L-BFGS-B) to search a **4-dimensional weight space** `[w_conf, w_rel, w_cert, w_sim] ∈ [0,1]^4`.  
Trained result from `models/gat_weights/gat_trained_weights.json`:

| Weight | Prior | Trained | Delta |
|--------|-------|---------|-------|
| w_confidence | 0.4000 | 0.4002 | +0.0002 |
| w_relevance  | 0.3000 | 0.2860 | −0.0140 |
| w_certainty  | 0.3000 | 0.3127 | +0.0127 |
| w_similarity | 0.2000 | 0.2007 | +0.0007 |

**Top-1 accuracy: 86.67% before training, 86.67% after.** The optimizer found the prior as approximately optimal. The training did not learn anything beyond the hand-crafted initialisation.

### Summary of what "GAT" and "GAT_TRAINED" actually are
| Label in paper | Reality |
|---|---|
| GAT (untrained) | Rule-based attention: 4-scalar weighted sum of agent features |
| GAT_TRAINED | Same formula with L-BFGS-B–optimised scalars; delta < 0.014; zero accuracy gain |

### Fix options

**Option A — Rename throughout (recommended)**  
Rename to something that honestly describes the method without claiming neural-network properties.  
Proposed names:
- `RBGA` — Rule-Based Graph Aggregation (untrained)  
- `LBGA` — L-BFGS-Optimised Graph Aggregation (trained)
Or simpler:
- `RWA` — Reliability-Weighted Aggregation (untrained)
- `RWA-Opt` — RWA with Optimised Weights (trained)

This requires:
- Paper: rename GAT → RBGA throughout all text, tables, figures, abstract
- Code: rename method label in `gat_aggregator.py` return dict (`'method': 'GAT'` → `'method': 'RBGA'`)
- Code: rename `GAT_TRAINED` constant and subdirectory to `RBGA_OPT`

**Option B — Reframe as "GAT-inspired" (weaker)**  
Add explicit framing: "a GAT-*inspired* rule-based aggregator (RBGA) that inherits the graph-attention architecture but replaces gradient-trained W and a with domain-knowledge rules." Emphasise that this is a deliberate design choice for cold-start interpretability.  
The paper already says "untrained GAT" — tighten this to "GAT-architecture with rule-based parametrisation, no gradient-based learning."

**Option C — Make it a real GAT (future work only)**  
Genuine GAT requires learned W ∈ ℝ^{d×d'} and a ∈ ℝ^{2d'} via backprop. Labelling GAT_TRAINED as "trained" when only 4 scalars are optimised is insufficient for this claim. A truly trained GAT is future work (warm-started online learning, Section 6).

### Multi-head fix (separate from naming)
Either:
- Remove the multi-head claim and use a single attention computation
- OR give each head a different feature-weighting mask (e.g., head 1 = confidence-dominant, head 2 = relevance-dominant, head 3 = reliability-dominant, head 4 = similarity-dominant) so they actually produce different outputs

---

## 2. 13 vs. 12 agents mislabelling — HIGH **[PAPER]**

**Paper (abstract, intro, Table III header):** "13 specialised agents"  
**All experiments:** 12 active agents per run (auto-selection always excludes 1)

Fix: Change all references to "up to 13 specialised agents" or "12–13 active agents." Fix Table III header from "13-agent, 45 runs" to "12-agent, 45 runs."

---

## 3. Collective-vs-individual comparison uses retroactive best-agent selection — HIGH **[PAPER]**

Section 4.5 picks the "best individual" as the agent with highest belief mass on the winning alternative — identified *after* the collective result is known. This is selection bias.

**Fix:** Redefine the baseline as the **highest-reliability agent** at the time of the run, identified before knowing the collective outcome. This gives an honest pre-committed baseline. The ±pp margins may change.

---

## 4. Explainability ratings presented as "domain expert evaluation" — HIGH **[PAPER]**

**Abstract:** "preliminary domain expert evaluation, explainability and auditability received ratings of 4.2/5 and 4.5/5"  
**Section 5.3:** "a single evaluator who is also the system's designer"

**Fix:** Remove ratings from abstract entirely, or relabel as "author self-assessment." The abstract should not imply independent validation that did not occur.

---

## 5. Reliability accuracy formula inconsistency — MEDIUM **[BOTH]**

**Paper (Section 3.6):**  
`a_k = 0.4·m_i(A*) + 0.3·1[top(m_i)=A*] + 0.3·margin(m_i, A*)`  
`margin` is never formally defined.

**CLAUDE.md (code documentation):**  
"accuracy = cosine similarity of agent's belief vector vs recommended alternative vector"

These are different computations. One of them is wrong.

**Fix:** Read `agents/reliability_tracker.py` to find which formula is actually executed. Align the paper to the code or vice versa. Define `margin(m_i, A*)` formally in the paper.

---

## 6. "Combined Score" in Table V is undefined — MEDIUM **[PAPER]**

Table V uses "Combined Score (μ ± σ)" as the primary provider comparison column. This metric is not in the Section 4.1 metric definitions (DQS, CL, DC, ECB). Values (~0.47–0.50) differ from DQS (~0.775–0.781).

**Fix:** Add a formal definition. If it is `Score(A_k) = 0.6×m_agg(A_k) + 0.4×C_k^norm` for the recommended alternative, say so. If it is something else, define it.

---

## 7. Attention coefficient weights sum to 1.2, not 1.0 — MEDIUM **[BOTH]**

Paper formula: `e_ij = 0.4·f^(1) + 0.3·f^(3) + 0.3·f^(2) + 0.2·cos(...)`  
Sum: 0.4 + 0.3 + 0.3 + 0.2 = **1.2** (not a convex combination)

Code (lines 488–499): same coefficients applied literally. Softmax normalises the final output, but the intermediate formula is non-standard.

**Fix (code):** Either normalise the coefficients to [0.333, 0.25, 0.25, 0.167] (sum = 1.0), or document the deliberate unnormalised design. The paper formula should match the code exactly.

---

## 8. Singleton focal element simplification understated — MEDIUM **[PAPER]**

Section 3.3 notes the O(2^n)→O(n) complexity reduction but does not say: restricting to singleton focal elements **eliminates the defining advantage of Dempster-Shafer theory** — the ability to assign mass to non-singleton sets and explicitly represent ignorance. The ER engine is effectively a **reliability-weighted Bayesian combination** with a conflict patch, not full DST.

**Fix:** Add one sentence: "Note that singleton restriction eliminates the capacity to represent non-specific belief (mass assigned to sets of alternatives), reducing the ER engine to a weighted Bayesian combination; full DST expressivity is reserved for future extensions."

---

## 9. Conflict redistribution formula is ambiguous — MEDIUM **[PAPER]**

Paper formula:  
`m_conflict-adj(A) = m(A) + K · (w_i r_i m_i(A)) / Σ_j w_j r_j m_j(A)`

In sequential pairwise combination at step k, there are only **two** BBAs: the accumulated combined mass and agent k. The formula's `Σ_j w_j r_j m_j(A)` over all agents is inconsistent with the pairwise procedure described. This reads as a post-hoc global redistribution, but pairwise ER (Yang-Xu rule) modifies the combination itself before it happens.

**Fix:** Clarify whether the formula is applied globally (after all pairs are combined) or locally (within each pairwise step). Align notation with what the code in `evidential_reasoning.py` actually computes.

---

## 10. Pre/post L1-normalisation ambiguity in tables — MEDIUM **[PAPER]**

Section 4.7 reports 12 recommendation changes after L1 normalisation (40% of Wildfire runs). It's unclear whether Tables II–V report pre- or post-normalisation DQS.

**Fix:** Add a caption note to Table II: "All DQS values use L1-normalised TOPSIS scores as described in Section 3.5 and corrected in Section 4.7."

---

## 11. Vision/multimodal subsystem absent from paper — LOW **[PAPER]**

The codebase has `GeospatialContextAgent`, `CameraFeedAgent`, and `VisionClient` running as Step 0 of every `make_final_decision()` call. The Santorini scenario has camera feeds with tsunami and crowd analysis. None of this is described in the paper.

**Fix:** Add a subsection "3.X Vision Pre-Assessment Layer" or note it as "implemented but not evaluated in this study." Omitting it entirely creates a reproducibility gap between the code and the paper's pipeline description.

---

## 12. Decision Confidence formula unjustified — LOW **[PAPER]**

`DC = 0.6 × CL + 0.4 × c̄`

The 0.6/0.4 split has no theoretical basis. `c̄` (mean LLM self-reported confidence) has no external calibration — LLMs are known to be overconfident.

**Fix:** Either justify the split with a sensitivity analysis, or demote DC to a secondary/exploratory metric with a note that agent confidence values are uncalibrated self-reports.

---

## 13. GAT_TRAINED "trained" label misleads on what is learned — LOW **[PAPER + CODE]**

The paper implies GAT_TRAINED learned a meaningful parameterisation. The trained weights deviate < 0.014 from prior; accuracy is identical. The training found the prior as approximately optimal on the 45-run corpus.

**Fix (paper):** Add a sentence in Section 4.3 or 6: "The trained weights deviated negligibly from the prior (max delta 0.014), yielding identical top-1 accuracy on the training corpus, indicating that the rule-based prior is already near-optimal for the current scenario set."  
**Fix (code):** The `GATTrainer` is legitimate as a framework; add an assertion or warning if `|trained_weights - prior_weights| < 0.05` to surface this in future runs.

---

## Summary Matrix

| # | Issue | Severity | Fix target | Status |
|---|-------|----------|------------|--------|
| 1 | GAT naming / hollow multi-head | HIGH | BOTH | ✅ RESOLVED — renamed to RBGA throughout paper |
| 2 | 13 vs 12 agents | HIGH | PAPER | ✅ RESOLVED — changed to "up to 13 / 12 active" |
| 3 | Retroactive best-agent selection | HIGH | PAPER | ✅ RESOLVED — post-hoc disclosure + planned future baseline added |
| 4 | Self-evaluation labelled as expert study | HIGH | PAPER | ✅ RESOLVED — relabelled as "author self-assessment" in abstract |
| 5 | Reliability formula mismatch | MEDIUM | BOTH | ✅ RESOLVED — formal piecewise margin definition added to §3.6 |
| 6 | Combined Score undefined | MEDIUM | PAPER | ✅ RESOLVED — formal CS definition added to §4.1 |
| 7 | Attention weights sum to 1.2 | MEDIUM | BOTH | ✅ RESOLVED — disclosure already present in §3.4 (softmax normalises output) |
| 8 | Singleton DST limitation understated | MEDIUM | PAPER | ✅ RESOLVED — sentence added to §3.3 noting reduction to weighted Bayesian combination |
| 9 | Conflict formula ambiguous | MEDIUM | PAPER | ✅ RESOLVED — Option B: original formula retained as intended design; implementation note added |
| 10 | Pre/post L1 normalisation unclear | MEDIUM | PAPER | ✅ RESOLVED — Table II footnote updated to state all DQS values use L1-normalised TOPSIS |
| 11 | Vision subsystem absent | LOW | PAPER | ✅ RESOLVED — §3.10 Multimodal Pre-Assessment Layer present; vision layer in all diagrams |
| 12 | DC formula unjustified | LOW | PAPER | ✅ RESOLVED — DC demoted to "exploratory composite" with uncalibrated-confidence caveat in §4.1 |
| 13 | GAT_TRAINED delta near zero | LOW | BOTH | ✅ RESOLVED — RBGA-Opt §4.3 subsection added with exact weight deltas and null-result interpretation |
