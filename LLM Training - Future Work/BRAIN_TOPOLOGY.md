# Brain-Topology Framing for AEGIS: Assessment

**Question**: is there added value in establishing/deploying the multi-agent
LLM system as a brain-like topology?

**Short answer**: yes — but not as a metaphor. The value is concrete in three
places: (1) a unifying theory (Global Workspace Theory) that AEGIS
*already implements* without naming it, (2) a deployment architecture
(shared base model + per-agent LoRA adapters = one substrate, specialized
regions) that is what makes 13 experts feasible on 48GB at all, and (3) one
genuinely novel, testable experiment (structured GAT topology). Renaming
existing components with neuroscience vocabulary adds nothing; these three do.

---

## 1. What AEGIS already is, in brain terms

An honest audit of the existing architecture shows five brain-like mechanisms
that were built for engineering reasons and can be *legitimately* described in
cognitive-architecture terms — no code changes required, only framing:

| Existing AEGIS mechanism | Cognitive-architecture equivalent |
|---|---|
| 13 specialized expert agents assessing in parallel | Specialized processors / functional regions |
| CoordinatorAgent aggregating into one decision | **Global Workspace** (Baars 1988; Dehaene's Global Neuronal Workspace) — parallel specialists compete/contribute to a single broadcast decision state |
| `--expert-selection auto` (3–11 of 13 agents activated per scenario) | Sparse, task-dependent activation — brains do not run all regions at full power; selection is metabolic/attentional gating |
| GAT aggregation (attention over agent beliefs) | Attention-weighted message passing between regions |
| ReliabilityTracker (per-agent weights updated after each decision, injected into GAT) | Synaptic plasticity — connection strengths adapt with experience ("neurons that fire usefully together get weighted together") |
| Vision subsystem Step 0 (terrain/camera pre-filter before expert collection) | Early sensory processing gating downstream executive decision-making |

**Thesis value**: this gives the dissertation a single coherent theoretical
narrative (GWT / Society of Mind, Minsky 1986) connecting components that are
currently presented as separate engineering choices. That is a framing gain,
citable and defensible — but it is *descriptive*, not a contribution by itself.

## 2. Where the framing earns its keep: deployment

The plan's open question was how to deploy 13 expert models. The brain analogy
resolves it in favor of the architecture that is also the cheapest:

- **One shared base model** (the "cortex"): a single 7–8B model, 4-bit, ~4.5GB.
- **Per-agent LoRA adapters** (specialized "micro-circuitry"): ~100–300MB each;
  13 adapters ≈ 2–4GB total.
- **Adapter selection per expert turn** = task-dependent functional
  reconfiguration of one substrate, instead of 13 separate networks.

Total: **~7–9GB for all 14 agents** — fits the M4 Pro with room for the
coordinator, versus 13 separate sLLMs (13× storage, 13× maintenance,
13× drift monitoring). Stage 0 already proved the unit of this architecture
(one base + one adapter, trained/fused/served locally); Stage 4 tests
multi-adapter serving. The brain framing is not decoration here — "shared
substrate, specialized regions, gated activation" *is* the deployment design.

Practical note (verified 2026-07-15): `mlx_lm server --adapter-path` does not
currently apply adapters (mlx-lm 0.31.3), so hot-swapping adapters per request
is not yet available on the serving path; per-agent *fused* models or direct
`mlx_lm.load(model, adapter_path=...)` in-process are the working options.
In-process loading is actually the closer analog to gated activation anyway.

## 3. Where it does NOT add value

- **Renaming** (calling the coordinator a "thalamus", agents "cortical
  columns") without behavioral consequences — reviewers correctly discount
  loose neuro-analogies.
- **Biological fidelity claims**: LoRA adapters are not synapses; GAT is not
  neural tissue. The framing must stay at the *architecture-principles* level
  (modularity, sparsity, plasticity, workspace broadcast), where the analogy
  is standard in the literature (Mixture-of-Experts, Shazeer et al. 2017, is
  exactly this and never claims biology).
- **Replacing the ER/Dempster-Shafer path**: the evidential-reasoning
  aggregation is the system's transparent, auditable baseline — a regulatory
  asset under the EU AI Act analysis in REGULATORY_COMPLIANCE.md. The brain
  framing applies to the GAT path and deployment; keep ER untouched.

## 4. The one genuinely novel experiment: structured GAT topology

Today the GAT effectively attends over a fully-connected agent graph. Real
brain networks are **small-world and modular**: dense clusters (functional
domains) + sparse long-range hub connections (rich-club; Watts & Strogatz 1998;
Bassett & Bullmore 2006; van den Heuvel & Sporns 2011).

AEGIS's agent roster has an obvious modular structure to impose:

- **Fire cluster**: fire on-scene ↔ fire regional ↔ meteorologist
- **Medical cluster**: EKAB physician ↔ medical infrastructure ↔ logistics
- **Security cluster**: police on-scene ↔ police regional ↔ PSAP commander
- **Maritime cluster**: coast guard on-scene ↔ coast guard national
- **Hubs (rich-club)**: civil protection director, regional commanders — the
  only nodes with dense inter-cluster edges to the coordinator workspace.

**Testable hypotheses** (Stage 4, optional — uses existing evaluation
machinery: consensus level, DQS, decision confidence):

- H1: modular/small-world GAT graph ≥ fully-connected GAT on consensus and
  DQS, with fewer attention parameters.
- H2: modular topology degrades more gracefully when an agent fails or
  returns garbage (lesion study — a classic brain-network analysis that maps
  directly onto crisis-relevant robustness: what if the meteorologist feed
  dies mid-incident?).
- H3: attention weights concentrate on hub agents under cross-domain
  scenarios (ammonia leak = fire + medical + police) and stay intra-cluster
  for single-domain ones — measurable from GAT attention maps, giving an
  interpretability result for free.

This is a real, publishable-scale experiment: it changes computation (the
graph), has baselines (current GAT, ER), metrics that already exist in the
codebase, and a falsifiable outcome. If H1 fails, that is still a reportable
negative result about topology priors in agent aggregation.

## 5. Recommendation

1. **Adopt the GWT framing descriptively** in the thesis architecture chapter
   (§1) — low cost, real narrative value.
2. **Adopt the shared-substrate + adapters deployment** (§2) as the Stage 2/4
   plan of record — it is simultaneously the brain-inspired design and the
   only one that fits local hardware economics.
3. **Treat the structured-GAT experiment (§4) as the optional research
   contribution** — do it only after Stages 1–3 close, since it needs the
   evaluation pipeline live. It is the difference between "brain-inspired" as
   a label and as a finding.
4. **Do not** rebrand existing components or claim biological fidelity (§3).

---

**Created**: 2026-07-15 · relates to [PROJECT_PLAN.md](PROJECT_PLAN.md) Stage 4
