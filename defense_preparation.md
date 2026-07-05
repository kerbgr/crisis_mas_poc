# AEGIS Thesis Defense Preparation
## Lessons Identified During Paper Analysis and Optimisation

**Purpose:** Committee Q&A preparation. Each section names the vulnerability, the honest acknowledgment, and the defensible response. Issues are ranked by expected severity of committee challenge.

---

## 1. Validation — Circular Ground Truth (Highest Risk)

**The weakness:**
The ReliabilityTracker scores each agent's accuracy by comparing its output to the system's own consensus recommendation. Agents that agree with the majority are rewarded, regardless of whether the majority is correct. This is a closed loop: AEGIS evaluates itself against itself.

**Why the committee will raise it:**
This is the most fundamental scientific challenge. It means the "reliability" scores are a measure of conformity to collective agreement, not a measure of decision correctness. The +4.9 pp GOLD-SILVER training gap could reflect that GOLD agents are better at producing whatever the system will recommend, rather than better at recommending what should actually be done.

**Defence:**
1. **Acknowledge it upfront before being challenged.** The paper already names it explicitly in §5.3. Raising it yourself signals scientific honesty rather than defensiveness.
2. **Ground truth is structurally unavailable in this domain.** No labelled dataset of "correct" crisis decisions exists. The same circular problem affects all decision support systems without real-world deployment data. Your proxy is standard practice in the absence of expert panels.
3. **The tracker's value is relative, not absolute.** Its purpose is to differentiate agent influence dynamically across runs, not to certify correctness. Even if the consensus proxy is imperfect, an agent that consistently diverges from every aggregated outcome is a signal worth acting on.
4. **The fix is already scoped.** §5.3 and §6 explicitly identify tabletop exercises with agency practitioners as the next empirical step. You are not claiming the tracker is validated — you are demonstrating its architecture and the differentiation it produces.

**What not to say:** Do not claim the consensus proxy is a valid substitute for external ground truth. It is a practical approximation and should be described as such.

---

## 2. Frozen-Weight Holdout — Single Scenario, Five Runs

**The weakness:**
The test set consists of 5 runs of a single crisis type (volcanic-seismic, Santorini). While the weights are properly frozen and cross-run contamination was eliminated, drawing generalisability conclusions from one scenario type and 65 agent records is statistically thin.

**Background to know:**
During analysis, the original Santorini runs were found to be sequentially contaminated — the ReliabilityTracker updated weights between "test" runs, making each run start from different weights. This was identified, the poisoned runs were deleted, and a snapshot-restore wrapper (`run_frozen_volcanic_test.py`) was written to guarantee each of the 5 clean runs starts from identical frozen training weights. The committee may ask about methodology; you can describe this correction as evidence of rigor.

**Why the committee will raise it:**
A single holdout scenario type means the generalisability claim rests on a single data point at the scenario-type level. High within-test variance — five agents, **including two GOLD-level agents** (Police Regional min 0.121, Medical Infrastructure Director min 0.064), score near zero in one run each — further weakens any claim about stability, and the GOLD-SILVER gap flattens from +4.9 pp (training) to +3.3 pp (test).

**Defence:**
1. **The holdout serves a structural claim, not a statistical one.** The purpose is to show that training-phase weights transfer to a novel crisis type with a distinct agent relevance profile — not to produce confidence intervals over many crisis types. A single scenario is sufficient to demonstrate non-zero transfer.
2. **The GOLD-SILVER ordering is directionally preserved but honestly reported as flattening** (+3.3 pp test vs +4.9 pp training). The paper does not overclaim level-based transfer; it explicitly concludes that within-test *variance* — not command level — is the robust discriminator of transferable expertise, and that coordination-centric roles (PSAP 0.696, Police On-Scene 0.685) lead the test set as domain knowledge would predict for a volcanic island emergency.
3. **The high within-test variance is a finding, not noise.** Agents whose training domains are loosely coupled to volcanic emergencies produce inconsistent assessments — and the tracker captures exactly this instability as variance, assigning them lower weights. This is correct tracker behaviour, and it applies to GOLD agents too, which is why the paper retreats from level-based claims.
4. **Honest framing is already in the paper.** §4.5 explicitly calls the volcanic test a "preliminary" zero-shot evaluation, and §5.3 will note the single-scenario limitation.
5. **The alternative was worse.** Using the volcanic runs without frozen weights (the original approach) would have produced contaminated scores. A clean 5-run frozen test on one scenario is more defensible than 15 contaminated runs on one scenario.

---

## 3. RBGA Is Not a True Graph Attention Network

**The weakness:**
The paper names the aggregator "Graph Attention-Inspired" and repeatedly references GAT [Veličković et al., 2018]. But RBGA has no learnable weight matrix **W**, no trainable attention vector **a**, and no gradient-based optimisation. It is a weighted average with four fixed scalar coefficients derived from domain rules. Calling it a GAT variant may be challenged as overclaiming.

**Why the committee will raise it:**
Reviewers with ML backgrounds will note that the RBGA-Opt experiment confirmed zero improvement from gradient-free optimisation, meaning the "graph attention" framing adds no empirical benefit over a simpler weighted-average model. The naming could be seen as borrowing credibility from a well-known architecture.

**Defence:**
1. **The paper is explicit about this distinction.** §3.4 opens with: "RBGA contains no learnable weight matrix **W** ∈ ℝ^(F×F') and no trainable attention vector **a** ∈ ℝ^(2F'). This is a deliberate design choice." The architecture is transparent about what it is.
2. **The graph structure is genuine.** Agents form a fully connected graph, attention scores are computed pairwise between nodes, and multi-head aggregation is implemented. The architecture follows GAT topology; what is replaced are the learnable parameters.
3. **The reason for fixedness is principled, not expedient.** Training a GAT requires a labelled corpus of 300–500+ decisions with externally verified ground truth. That corpus does not exist. Deploying fixed domain-knowledge coefficients is the responsible alternative; it avoids training on the circular self-generated labels.
4. **The RBGA-Opt null result is a positive finding, replicated twice.** On the current 45-run corpus the trained coefficients deviate by max 0.0051 from the hand-crafted prior with top-1 accuracy unchanged at 97.8 % (the earlier pilot fit showed the same null at max 0.014). This is not a failure — it validates that the domain-knowledge coefficients are near-optimal within the scalar architecture, meaning a practitioner can deploy them with confidence. Bonus lesson: the pilot-fitted weights transferred poorly across a scenario revision (diverging from consensus in 15/45 runs), showing that even mild tuning is decision-space-specific — a further argument for the untuned prior.
5. **The path to a proper GAT is explicitly mapped.** §6 specifies exactly what conditions would justify a full **W** ∈ ℝ^(F×F') projection: 300–500 externally validated runs, cross-crisis-type coverage. The current work establishes the architecture and the feature space for that next step.

---

## 4. Explainability Scores Are Self-Assessed

**The weakness:**
The explainability rating (4.2/5) and auditability rating (4.5/5) in §5.1 were produced by a single evaluator — the system's designer. This is acknowledged in §5.3 as "a substantial risk of confirmation bias."

**Why the committee will raise it:**
Self-assessed quality scores in a thesis are a standard target. A committee member may argue that these numbers add no evidentiary value and should be removed, or that they mislead readers.

**Defence:**
1. **The scores are not presented as externally validated findings.** The paper uses the framing "preliminary author self-assessment" and immediately flags the confirmation bias risk in the same paragraph.
2. **The purpose is scoping, not evidence.** The scores communicate the *type* of transparency produced (audit trails, attention weight visualisation, per-agent reasoning traces) and give a qualitative sense of what a practitioner would encounter. They are not offered as proof that the system is explainable.
3. **The structured audit trail is auditable independently.** A committee member can inspect the HAZMAT trace in §4.6 and form their own view. The transparency is in the output JSON, not in the self-assessment number.
4. **An independent panel is identified as future work.** Practitioners from Hellenic Fire Corps, EKAB, and the General Secretariat of Civil Protection are named. The limitation is acknowledged as the most urgent validation gap.

**If pressed hard:** Offer to remove or footnote the scores rather than defend their epistemic weight. They are supporting colour, not a load-bearing result.

---

## 5. Collective Does Not Beat the Best Individual — the RQ4 Reframing

**The weakness (and how it changed):**
On the current corpus, with both sides scored on an identical raw-TOPSIS choice-quality scale, the collective recommendation *matches* the post-hoc best individual agent (differences of 0.3-1.9 pp, within noise) rather than exceeding it. The margins live over the *mean* individual (+0.5 pp Flood, +1.9 pp Wildfire, +3.2 pp HAZMAT). The classical "collective beats best individual" GDM claim does not hold in its strong form here.

**Why the committee will raise it:**
"If your system only matches the best expert, why do we need it? Just ask the best expert." This is the sharpest attack on RQ4 as reframed.

**Defence:**
1. **You don't know who the best expert is until afterwards.** The best-individual baseline is identified post-hoc — after every agent's choice quality is known. Operationally, a commander delegating to a single arbitrarily chosen expert would get a below-collective-quality recommendation roughly one time in three in the contested Wildfire case (only 67.7 % of solo choices coincide with the system recommendation; 78.5 % Flood, 79.2 % HAZMAT). The system's contribution is *reliably landing on the strongest expert position without prior knowledge of which expert holds it*.
2. **The paper names this honestly.** §4.5 characterises aggregation as a *selection-and-stabilisation* mechanism rather than a synthesis mechanism — a distinction the classical GDM formulation does not draw, and arguably a more useful claim for deployment than an inflated superiority margin.
3. **The same-scale methodology is a strength.** Earlier drafts compared quantities on different scales; the current comparison scores both the collective and each solo agent by the raw TOPSIS coefficient of their chosen alternative. Any criticism of the metric applies equally to both sides.
4. **The pre-committed baseline is scoped.** §4.5 and §5.3 explicitly plan the stricter test: can the reliability tracker *pre-identify* the expert whose solo choice matches collective quality? That is the operationally relevant follow-up.

---

## 6. MCDA Scale Mismatch — Found and Corrected Mid-Study

**The weakness:**
Post-hoc analysis of the pilot corpus revealed that raw TOPSIS closeness coefficients were not L1-normalised before blending with belief distributions, causing the MCDA component to contribute 55–79 % of the combined score instead of the nominal 40 %. In the pilot corpus this shifted 12 of 92 recommendations (11 of them in the Wildfire scenario). The correction was integrated into the decision engine *before* the main 45-run corpus was generated, so every result in Sections 4.2-4.6 applies L1 normalisation at decision time; §4.7 documents the discovery on the pilot corpus as a methodological finding.

**Why the committee will raise it:**
A bug that changed 13 % of recommendations (12/92 result files) is a significant finding. A committee may ask: (a) how was it not caught earlier? (b) could there be other such biases? (c) does the correction undermine the validity of earlier claims?

**Defence:**
1. **Reporting it demonstrates scientific integrity.** Section 4.7 describes the discovery, root cause, and quantified impact in full. A researcher who hides this loses credibility; one who reports it gains it.
2. **The scale mismatch is a mathematically predictable consequence of TOPSIS geometry.** Closeness coefficients are bounded in [0,1] individually but sum to 1.8–3.2 across alternatives; beliefs always sum to 1.0. This is a known property of TOPSIS, not a coding error. L1 normalisation is the standard remedy.
3. **All quantitative claims in the paper use the corrected values.** No uncorrected numbers are cited as findings. The pre-correction results are reported only in §4.7 as a characterisation of the impact.
4. **The HAZMAT and Flood scenario conclusions are unchanged.** Only the most ambiguous scenario (Wildfire, 12-alternative space) was substantially affected, and even there the ER-RBGA comparison conclusions hold because both methods were equally affected by the same bias.
5. **The discovery strengthens the paper's practical contribution.** §4.7 is now a methodological warning for any researcher combining TOPSIS with probability distributions — a contribution in its own right.

---

## 7. LLM Agents May Not Be Truly Independent

**The weakness:**
All active agents (11-13 per run) call the same underlying LLM (Claude Sonnet 4.5, GPT-4o, or GPT-OSS 20B within a given run). If the LLM has a systematic bias toward certain alternatives — e.g., consistently preferring evacuation over shelter-in-place for chemical incidents — then all agents share that bias. The consensus mechanism would register high agreement without detecting that agreement is driven by a shared model artifact rather than domain expertise.

**Why the committee will raise it:**
Independence of judgements is a foundational assumption of group decision-making theory. Violated independence can make collective aggregation worse than individual judgement.

**Defence:**
1. **Role-specific prompts create genuine differentiation.** Each agent receives a distinct professional identity (Senior Meteorologist vs. Coast Guard National Director), a distinct scenario framing, and a distinct set of priorities (the 5-criterion preference vector). The belief distributions produced are observably different: the Medical Expert concentrates mass on life-safety alternatives while the Logistics Coordinator distributes more evenly. This reflects prompt-induced domain specialisation, not identical outputs.
2. **Multi-provider results are consistent.** The same aggregation pattern holds across Claude, GPT-4o, and GPT-OSS 20B. If the result were an artefact of one LLM's biases, the pattern would vary across providers. It does not.
3. **Semantic hallucination correlation is acknowledged.** §5.3 explicitly raises "correlated skewed outputs across multiple agents that remain invisible in consensus and reliability metrics" as an untested risk. The 12-agent panel provides robustness against independent failures, not correlated ones.
4. **The architectural mitigation is multi-layer aggregation.** Even if one LLM family has a systematic preference, the 60/40 blend with TOPSIS provides a second signal path that does not rely on LLM output at all.

---

## 8. Static, Well-Formed Scenarios vs. Real Crisis Dynamics

**The weakness:**
Each scenario is a fully specified JSON object: severity, affected population, alternatives, criterion scores. Real crises arrive as incomplete, contradictory, rapidly changing streams of information. The system has never been tested under partial inputs, conflicting field reports, mid-scenario changes, or failed LLM calls.

**Defence:**
1. **Controlled evaluation requires controlled inputs.** Comparing ER vs. RBGA, providers, and scenarios requires holding scenario quality constant. Introducing input noise would confound the comparison without adding insight at the PoC stage.
2. **The failure-mode architecture exists.** Auto-fallback between providers, retry logic with exponential backoff, and the minimum-panel of 3 core agents provide degradation paths. They are described in §3.7 but untested under adversarial conditions — a gap the paper acknowledges.
3. **Scope is consistent with a master's-level PoC.** The thesis is not claiming operational readiness; it is establishing a framework, demonstrating its components, and characterising its behaviour under controlled conditions. Each limitation section explicitly points to the gap between current state and deployment.

---

## 9. Sample Size — 45 Runs, Three Scenarios

**The weakness:**
5 replicates per provider per scenario gives narrow confidence intervals for within-scenario claims, but only 3 scenario types is a thin basis for cross-scenario generalisability. Provider comparisons rest on n = 15 per provider (note: on the current corpus the provider effect on Combined Score is *not* significant, H = 5.24, p = 0.073 — the paper claims provider-independence of quality, which small n makes conservative rather than fragile: a null claim is harder to inflate with low power, though the committee may note low power cuts both ways).

**Defence:**
1. **Each run costs 11-13 LLM calls with commercial API billing.** 555 calls across the full experiment is a substantial investment for a thesis-scale study, and the full corpus was regenerated once more after bug fixes (the round-2 corpus reported in the paper). The replicate count is the maximum feasible given resource constraints.
2. **Results are consistent within scenarios.** The HAZMAT scenario achieves 0.000 DQS variance across all 15 runs — complete convergence. This consistency at within-scenario level compensates for the limited scenario count.
3. **Cross-scenario claims are bounded.** The paper does not claim that 3 scenarios represent all crisis types. The "scope of scenarios" limitation in §5.3 is explicit.

---

## 10. Issues Fixed During Paper Preparation — Proactive Disclosure Points

These should be raised proactively if the committee does not ask first. They demonstrate methodological honesty and scientific maturity.

| Issue discovered | Status | What it shows |
|---|---|---|
| Sequential contamination of Santorini test runs (tracker updated weights between "test" runs, poisoning independent evaluation) | Fixed: snapshot-restore frozen-weight wrapper written; poisoned runs deleted; 5 clean runs re-executed | Ability to identify and fix a subtle methodological flaw before publication |
| MCDA-ER scale mismatch (TOPSIS raw coefficients inflated MCDA weight to 55–79 %) | Fixed: L1 normalisation applied; §4.7 added reporting full impact; 12 recommendation changes documented | Transparency about a mid-study discovery that materially affected some results |
| LaTeX symbol inconsistencies (α for both attention weight and LeakyReLU slope; r_i vs ρ_j for same reliability concept; k overloaded as alternative/assessment/run index) | Fixed: unified notation throughout | Attention to formal precision |
| References [17] and [18] (wrong titles, non-existent journals) | Fixed: replaced with verified real papers (DisasterResponseGPT arXiv:2306.17271; Otal et al. IEEE CAI 2024) | Verification rather than citation of unverified sources |
| Contribution count ("five" but six listed) and arithmetic error (86.7 % vs 87.0 %) | Fixed | Thoroughness of self-review |
| RBGA influence metric: self-attention $\alpha_{ii}$ used as agent weight instead of received attention $c_i = \frac{1}{N}\sum_j \bar{\alpha}_{ji}$ | Fixed: code and paper formula updated to column-sum received attention | Correct operationalisation of "influence" in a graph attention architecture |
| Compare-methods design collected assessments separately per aggregator (confounding aggregation with LLM sampling variance) | Fixed before round-2 corpus: single-collection refactor — one LLM pass feeds both ER and RBGA; agreement rose from 88.9 % to 93.3 % as predicted | Confound identified, eliminated, and its predicted effect confirmed empirically |
| Geospatial vision model echoed the prompt's option list ("island\|coastal_mainland\|inland"), silently triggering the deterministic fallback in all holdout runs | Fixed: response-normalisation step added and verified live against minicpm-v; fallback label agreed with vision in every case, zero decision impact | Graceful-degradation design worked as intended; failure mode eliminated and disclosed in §3.10 |
| RBGA-Opt weights fitted on the pilot corpus transferred poorly after scenario enrichment (diverged from consensus in 15/45 round-2 runs) | Fixed: re-optimised on the 45-run corpus; converged to within 0.0051 of the prior, null result replicated | Even mild tuning is decision-space-specific; the untuned prior is the robust choice |
| Paper cited "Claude Sonnet 4" while the round-2 corpus used claude-sonnet-4-5 | Fixed: model names and all provider statistics regenerated from the actual corpus | Model-version pinning and corpus-to-paper traceability |

---

## 11. Sharp Committee Questions — Anticipated Attacks

These five questions were identified during pre-submission analysis as the most technically precise challenges a reviewer with ML or crisis management expertise would raise. Ranked by how much preparation they require.

---

### Q1. Can you demonstrate from one raw run that ER and RBGA received byte-for-byte identical agent assessments?

**Status: Fixed before the round-2 corpus — now a strength, answer "yes" and show it.**
`run_comparative_analysis` collects assessments exactly once (via the ER coordinator, which also runs the Step 0 vision pre-assessment) and passes the same assessment dict to the RBGA coordinator with zero additional LLM calls. Any run's `er/results.json` and `gat/results.json` contain identical per-agent belief distributions — demonstrable live from the repository.

**The narrative bonus:** The earlier design *did* collect separately per aggregator, confounding aggregation with LLM sampling variance. The confound was identified, the refactor predicted the agreement rate would rise, and it did: 88.9 % (pilot, confounded) → 93.3 % (round-2, single-collection). A methodological prediction confirmed by data is exactly what a committee wants to hear.

---

### Q2. Which values in the abstract and tables were generated before versus after the TOPSIS normalisation correction?

**Honest answer:** All quantitative claims in the paper — abstract, tables, figures — come from the round-2 corpus, generated with the L1 normalisation already integrated in the decision engine (`coordinator_agent.py`), i.e. applied at decision time, not retrofitted. The pre-correction behaviour appears only in §4.7 as a pilot-corpus impact characterisation (12/92 recommendation shifts, 11 in the Wildfire scenario).

**Supporting detail:** The fix predates the round-2 corpus generation; `results/` contains only post-correction JSON files, and `scripts/analyze_corpus.py` regenerates every table directly from them — there is no manual number in the results section.

---

### Q3. Why is $\alpha_{ii}$, rather than an incoming-attention sum or learned pooling weight, an appropriate measure of global agent influence?

**Status: Fixed before defence.** The code previously used the diagonal (self-attention) as the aggregation weight. This has been corrected to the received-attention column sum $c_i = \frac{1}{N}\sum_j \bar{\alpha}_{ji}$, and the paper formula updated accordingly. Self-attention measures how much an agent trusts itself; received attention measures how much the collective panel defers to that agent — the latter is the correct influence proxy.

**If asked why this wasn't the original design:** The self-attention diagonal is a common shortcut in GAT implementations for interpretability reporting; its use as the aggregation weight was an oversight. In RBGA's attention formula (dominated by agent-intrinsic features), both measures rank agents identically in practice — so no historical results change in direction, only in the precise weight values.

---

### Q4. What independent evidence indicates that the recommended alternatives are operationally better rather than merely more internally consistent?

**See Issue 1 (Circular Validation).** The short answer: there is none at this stage. The honest defence is that ground-truth crisis decisions are structurally unavailable, the consensus proxy is standard practice for PoC systems without deployment data, and §5.3 explicitly identifies tabletop validation with Hellenic Fire Corps / EKAB practitioners as the immediate next step.

**Partial positive evidence you can cite:** the recommendations are not merely internally consistent — they align with doctrine-encoded criteria *without being driven by them*. In the HAZMAT scenario the agent panel overrides the TOPSIS-optimal alternative in all 15 runs, choosing the integrated response over pure evacuation for reasons (source control, ongoing-release risk) that the reasoning traces articulate and that match how HAZMAT doctrine actually argues. That is not external validation, but it is qualitative evidence of domain-plausible reasoning beyond self-consistency.

**Do not claim** the consensus proxy validates correctness. Claim it validates structural coherence and rank differentiation, and that external validation is scoped and planned.

---

### Q5. How sensitive are the principal recommendations to the 60/40 blend and to the author-selected TOPSIS matrices?

**The gap:** No sensitivity analysis was conducted. The 60/40 blend and the TOPSIS criterion weights (safety 0.30, cost 0.25, effectiveness 0.20, response speed 0.20, public acceptance 0.20; engine-normalised) are author-set and untested under perturbation.

**Defence:**
1. The 60/40 split is a design parameter, not a finding. The paper states this explicitly and frames it as a practitioner-configurable value.
2. The criterion weights reflect the life-safety priority of the Greek scenarios and are documented in `criteria_weights.json` — transparent and reproducible.
3. The gap is now named in the paper itself: §5.3 explicitly commits to a perturbation study of the five criterion weights as a low-cost, LLM-free addition for the next revision.
4. One robustness data point already exists: in the HAZMAT scenario the recommendation is belief-dominated (the panel overrides the TOPSIS-best alternative in 15/15 runs), so at least there the recommendation is insensitive to moderate criterion-weight perturbation by construction.

**For PhD continuation:** Add the sensitivity sweep before journal submission. It is a low-cost experiment on existing data (no new LLM calls needed) and directly addresses this predictable reviewer challenge.

---

### Q6. Is there a scientific basis behind the Decision Quality Score, or is the system grading itself with its own rubric?

**The challenge:** DQS is the TOPSIS closeness coefficient of the recommended alternative, and the selection criterion itself contains a 40 % normalised-TOPSIS component. A sharp committee member will argue the metric is (a) circular, (b) not a measure of real-world quality, and (c) not comparable across scenarios.

**Defence — concede the construct, then show the evidence (§4.1 "Construct validity of DQS" now covers all three points in the paper):**
1. **The ingredient is canonical.** The closeness coefficient $C_i = S^-/(S^+ + S^-)$ is the standard TOPSIS quantity (Hwang & Yoon 1981; Behzadian et al. 2012), computed on a criterion matrix sourced from GSCP doctrine and after-action analyses. DQS therefore measures *doctrine-encoded criterion satisfaction* — say exactly that phrase, never "real-world quality".
2. **Circularity is partial, and the data prove it.** If the metric were self-fulfilling, the system would always pick the TOPSIS-argmax. It does so in only 27/45 ER runs — and in the HAZMAT scenario the belief component overrides the TOPSIS-best alternative in *all 15 runs* (integrated response, $C_k = 0.792$, over evacuation, $C_k = 0.806$), at a mean DQS cost of 0.029 when overrides occur. Expert consensus, not criterion geometry, dominates the selection. These numbers are in §4.1 and regenerable via `scripts/analyze_corpus.py`.
3. **Where DQS is uninformative, the paper says so.** DQS is deterministic given the recommendation, so the ER-vs-RBGA DQS test only carries information in the 3 discordant runs — Table III's caption states this. Cross-scenario DQS averages are labelled indicative because closeness coefficients are matrix-specific.
4. **Relative comparisons survive the critique.** ER vs RBGA and collective vs individual are scored on the same scale, so any bias applies to both sides equally. The absolute scale is uncalibrated (Issue 1); the relative statements are not.

**One-line summary to deliver verbatim:** "DQS is a doctrine-alignment score built from a canonical MCDA quantity; it is partially coupled to the selection criterion, we quantified that coupling — 60 % argmax coincidence overall, 0 % in HAZMAT — and we restrict every claim to what the metric can support."

---

## Key Strengths to Emphasise When Under Pressure

- **Two aggregation paths on byte-for-byte identical inputs** — single-collection design; demonstrable from any run's `er/` and `gat/` result files. No confounding, and the fix's predicted effect (agreement 88.9 % → 93.3 %) was confirmed empirically.
- **93.3 % recommendation agreement** between ER and RBGA across 45 runs, with paired Wilcoxon tests: the choice of aggregation algorithm matters less than the quality of agent assessments — yet the §4.6 trace shows the two produce structurally different belief profiles (ER: unanimity signal at 0.9995; RBGA: calibrated spread at 0.390), which is a decision-support insight, not a redundancy.
- **No significant provider effect on decision quality** (DQS 0.771–0.791; CS Kruskal-Wallis p = 0.073): GPT-OSS 20B achieves statistically indistinguishable quality at zero API dependency — practically significant for GDPR-constrained deployments.
- **The HAZMAT belief-override result (15/15 runs)** is the single best defensive statistic: it simultaneously answers the DQS-circularity attack (Q6), the "internally consistent only" attack (Q4), and demonstrates that expert consensus dominates criterion geometry.
- **The MCDA discovery (§4.7) is a methodological contribution in its own right** — a warning that applies to any researcher combining TOPSIS with probability distributions.
- **The RBGA-Opt null result is a clean negative finding, replicated on two corpora** — the domain-knowledge prior is empirically validated as near-optimal, and the pilot-weight transfer failure shows why the untuned prior is the robust deployment choice.
- **All code is open-source**, every table regenerates from `scripts/analyze_corpus.py` over the stored JSON corpus, and the frozen-weight holdout ships with a provenance manifest (git commit, training-corpus size, run directories). Transparency is architectural, not just claimed.

---

## Phrases to Avoid

| Avoid | Say instead |
|---|---|
| "The system works well" | "The system produces consistent, self-coherent recommendations under controlled conditions" |
| "The agents are experts" | "The agents are role-prompted LLMs that model expert reasoning within the constraints of their training data" |
| "The reliability tracker is validated" | "The tracker produces differentiated scores that are consistent with expected domain specialisation — external validation against ground truth is future work" |
| "RBGA is a graph attention network" | "RBGA is a graph-structured aggregator that implements the GAT topology with domain-knowledge scalar coefficients in place of learned parameters" |
| "The results generalise" | "The results are consistent across three scenario types and three LLM providers — broader generalisation requires expanded validation" |
| "DQS measures decision quality" | "DQS measures doctrine-encoded criterion satisfaction — the alignment of the chosen alternative with GSCP-derived criteria; external outcome validation is future work" |
| "The collective beats the best expert" | "The collective reliably matches the best expert position without knowing in advance who holds it — a selection-and-stabilisation guarantee, which is the operationally relevant property" |
