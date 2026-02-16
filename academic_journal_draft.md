# A Collaborative Multi-Agent Framework for Crisis Management Decision Support Using Evidential Reasoning and Graph Attention Networks

**Vasileios Kazoukas**
*Military Academy (SSE), Department of Military Sciences*
*Technical University of Crete (TUC), School of Production Engineering and Management*
Email: vkazoukas@tuc.gr, kazoukas@gmail.com

---

## Abstract

Effective crisis management depends on the ability to coordinate expert judgments rapidly and under considerable uncertainty. We present a multi-agent decision support system in which 13 specialised agents — modelled on Greek emergency response roles — generate structured assessments using Large Language Models and aggregate them through two complementary mechanisms: classical Evidential Reasoning based on Dempster-Shafer theory, and a Graph Attention Network that learns agent-to-agent attention weights from a 9-dimensional feature representation. A TOPSIS-based multi-criteria ranking and a historical reliability tracker that adjusts agent influence over successive decisions complete the pipeline.

We evaluate the system on three crisis scenarios inspired from recent Greek emergencies (Karditsa flooding, Evia wildfires, Elefsina industrial HAZMAT). Across 75 runs, multi-agent decisions surpass the best individual expert by 4.3–6.7 % in decision quality. The two aggregation paths reach comparable quality scores, though GAT produces measurably higher consensus (+2.8 %, p<0.01) and confidence (+1.5 %, p<0.05). Reliability-adjusted weighting yields a further 2.5 % quality gain over static weights when accumulated across 100 sequential runs. In a preliminary evaluation with 15 emergency management professionals, explainability and auditability received mean ratings of 4.2/5 and 4.5/5, respectively.

**Keywords:** Multi-Agent Systems, Crisis Management, Evidential Reasoning, Graph Attention Networks, Large Language Models, Decision Support Systems

---

## I. Introduction

Large-scale emergencies confront decision-makers with incomplete information, evolving hazards, and the need to synchronise responses across disciplines — medical, logistical, meteorological, environmental — within minutes rather than hours [1]. Human coordination teams, however skilled, can be overwhelmed when the number of concurrent information streams exceeds cognitive limits [2].

Multi-agent systems offer a natural computational analogy: autonomous software agents, each encoding a distinct area of expertise, can deliberate in parallel and pool their judgments. Prior work has applied agent-based models to evacuation planning [3] and group decision support in emergencies [4], yet several issues remain open. Most existing frameworks rely on a single aggregation strategy without examining alternatives, treat agent credibility as fixed, and provide limited transparency into how a collective recommendation is reached — a serious shortcoming in safety-critical settings. The recent availability of Large Language Models introduces new possibilities for richer, contextually grounded agent reasoning, but also new challenges around reliability and prompt design that the literature has only begun to address.

This paper makes the following contributions:

1. A direct, controlled comparison of weighted Evidential Reasoning and Graph Attention Networks as competing belief-aggregation mechanisms within the same multi-agent architecture.
2. A multi-provider LLM integration (Local LLM (by LM Studio), Anthropic Claude, OpenAI GPT-4) that supplies each agent with structured domain reasoning and includes automatic provider fallback.
3. A historical reliability tracker that updates agent influence weights after every decision, feeding into both the ER weighting scheme and the GAT feature vector.
4. A 13-agent model of the Greek emergency response hierarchy — spanning Paramedics,  Police, the Hellenic Fire Corps, the Coast Guard, and the General Secretariat of Civil Protection — evaluated on three scenario types inspired from recent Greek crises.
5. An explainability layer comprising attention-weight visualisation, MCDA score decomposition, and natural-language justification, assessed by domain practitioners.

Five research questions guide the evaluation: how effectively can multi-agent coordination support time-critical decisions (RQ1); how do ER and GAT compare for belief aggregation under high uncertainty (RQ2); what does LLM-powered reasoning add to agent quality (RQ3); does collective judgement measurably outperform individual experts (RQ4); and can the resulting decision trails satisfy the transparency requirements of operational crisis management (RQ5).

---

## II. Related Work

The design of the proposed system draws on several research threads that we briefly review below.

*Multi-agent systems.* Wooldridge [5] laid out the foundational properties of autonomous agents — reactivity, proactiveness, and social ability — that inform our agent design. In the emergency domain, Ren et al. [3] showed that agent-based evacuation models can capture emergent coordination patterns that monolithic simulations miss. We adopt a hierarchical coordinator-expert topology, where individual agents retain autonomy over their assessments while a coordinator agent orchestrates aggregation and final ranking.

*Dempster-Shafer theory and evidential reasoning.* Shafer's mathematical theory of evidence [6] extends Bayesian probability by allowing explicit representation of ignorance through belief and plausibility intervals. Yang and Xu [7] later refined the evidential reasoning rule to mitigate the well-known problem of counter-intuitive outputs under high inter-source conflict, while Sentz and Ferson [8] provide a systematic comparison of combination operators. In our implementation we use a simplified weighted-average formulation of ER, trading some theoretical rigour for the computational speed that real-time crisis support demands.

*Multi-criteria decision analysis.* The TOPSIS method introduced by Hwang and Yoon [9] ranks alternatives by their geometric distance to ideal and anti-ideal reference points. Behzadian et al. [10] survey its many extensions and application domains. We use TOPSIS as the primary ranker and include WSM and SAW as secondary checks.

*Graph attention networks.* Veličković et al. [11] proposed GAT as a mechanism for learning anisotropic, neighbour-dependent weights on graph-structured data. Zhang et al. [12] survey the broader landscape of deep learning on graphs. We repurpose the GAT architecture for an expert-agent graph in which each node corresponds to an agent and each edge carries an attention coefficient reflecting inter-agent relevance. A distinctive feature of our formulation is the inclusion of a historical reliability dimension in the node feature vector, enabling the network to discount agents whose past performance has been poor.

*Large language models for reasoning.* Chain-of-thought prompting [13] demonstrated that guiding a language model through intermediate reasoning steps materially improves answer quality on complex tasks. We exploit this principle by supplying each agent with a structured prompt template (approximately 5 000 characters) that encodes the agent's professional role, the relevant emergency protocols, and the required output format for machine-parseable belief distributions.

*Crisis management decision support.* Comfort et al. [1] and Kapucu and Garayev [2] highlight the central role of inter-organisational coordination in disaster response, while Levy and Taji [4] apply MCDA to hazard planning. Our work extends this line by replacing human panel deliberation with autonomous, LLM-powered agents whose collective output is aggregated through formal uncertainty calculi.

---

## III. Methodology

### A. System Architecture

The system is organised into six layers (Fig. 1). At the interface level, a command-line front end accepts scenario descriptions in JSON and returns structured decision reports. A coordination layer, built around a dedicated Coordinator agent, orchestrates the deliberation pipeline: it distributes the scenario to the expert panel, collects their assessments, invokes the chosen aggregation mechanism, checks consensus, and triggers conflict resolution when needed. Below the coordinator sit 13 expert agents, each associated with an LLM reasoning engine and a persistent reliability record. The decision framework layer houses the two aggregation mechanisms (ER and GAT), the MCDA ranker, and the consensus model. A multi-provider LLM layer manages requests to Claude, GPT-4, or a locally hosted model through LM Studio, falling back automatically when a provider is unavailable. Finally, an evaluation layer computes performance metrics and generates visualisations.

```mermaid
graph TB
    UI[User Interface Layer<br/>CLI, JSON I/O, Visualization]
    COORD[Coordination Layer<br/>Orchestration, Consensus]
    AGENTS[Agent Layer<br/>13 Experts: 6 Tactical + 7 Strategic]
    DF[Decision Framework<br/>ER, GAT, MCDA]
    LLM[LLM Integration<br/>Claude, OpenAI, LM Studio]
    EVAL[Evaluation Layer<br/>Metrics, Baseline, Visualization]

    UI -->|1. Load Scenario| COORD
    COORD -->|2. Distribute| AGENTS
    AGENTS -->|3. LLM Reasoning| LLM
    LLM -->|4. Structured Response| AGENTS
    AGENTS -->|5. Assessments| DF
    DF -->|6. Aggregated Decision| COORD
    COORD -->|7. Final Decision| EVAL
    EVAL -->|8. Results| UI

    style UI fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style COORD fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style AGENTS fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style DF fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style LLM fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style EVAL fill:#fffde7,stroke:#f9a825,stroke-width:2px
```
*Fig. 1. Six-layer system architecture and data flow. Numbered edges indicate the processing sequence for a single crisis scenario.*

The 13 agents mirror the organisational structure of Greek emergency response. They include a Meteorologist, an Emergency Physician, a Logistics Coordinator, a PSAP Commander, two Police Commanders at tactical and regional level, two Fire Commanders, a Medical Infrastructure Director, two Coast Guard Directors, an Environmental Scientist, and a Civil Protection Director.

### B. Evidential Reasoning

For computational tractability in time-critical settings, we adopt a weighted-average formulation of evidential reasoning rather than the full Dempster-Shafer combination rule. Given $n$ agents, the combined belief mass assigned to alternative $i$ is

$$b_{\text{combined}}(i) = \frac{\sum_j w_j \cdot b_j(i)}{\sum_j w_j}$$

where $w_j$ is the weight of agent $j$, composed of three factors: domain-relevance to the current scenario, historical reliability (Section III-E), and self-reported confidence. Overall decision confidence is derived from the normalised entropy of the combined distribution:

$$\text{confidence} = 1 - \frac{H}{\log_2(n_{\text{alternatives}})}, \quad H = -\sum_i p_i \log_2(p_i)$$

A low-entropy distribution — one that concentrates most belief mass on a single alternative — yields high confidence, while a near-uniform spread signals genuine ambiguity.

### C. Graph Attention Network

As an alternative to the linear ER aggregation, we construct a fully connected graph in which each agent is a node. Every node is described by a 9-dimensional feature vector $\mathbf{f}_j \in \mathbb{R}^9$ whose components capture confidence, belief certainty (inverse entropy), expertise relevance, risk tolerance, severity awareness, top-choice strength, number of concerns raised, reasoning quality, and historical reliability. The last dimension links the GAT directly to the reliability tracker described in Section III-E, allowing the network to learn from accumulated performance data.

Attention coefficients are computed with $K = 4$ heads:

$$e_{ij}^h = \text{LeakyReLU}(\mathbf{a}^h \cdot [\mathbf{W}^h \mathbf{f}_i \| \mathbf{W}^h \mathbf{f}_j])$$

$$\alpha_{ij}^h = \text{softmax}_j(e_{ij}^h)$$

The aggregated belief for alternative $i$ is then the mean over heads of the attention-weighted agent beliefs:

$$b_{\text{combined}}(i) = \frac{1}{K} \sum_{h=1}^{K} \sum_j \alpha_{ij}^h \cdot b_j(i)$$

### D. MCDA and Consensus

Once beliefs have been aggregated, TOPSIS ranks the alternatives by their relative closeness to the ideal solution in a normalised criterion space. In parallel, we measure the degree of inter-agent agreement through pairwise cosine similarity of belief vectors:

$$CL = \frac{2}{n(n-1)} \sum_i \sum_{j>i} \cos(\mathbf{b}_i, \mathbf{b}_j)$$

This consensus level serves as a gate for operational use: the default threshold is set at $CL = 0.75$, below which the system flags the decision as lacking sufficient agreement and triggers conflict identification among the agents.

Fig. 2 summarises the end-to-end decision pipeline, showing how the coordinator distributes the scenario to the tactical and strategic agents, collects their assessments in parallel, routes them through either ER or GAT aggregation, and combines the result with MCDA scores to produce a final recommendation.

```mermaid
sequenceDiagram
    participant User
    participant Coordinator
    participant Tactical as Tactical Level<br/>(6 Agents)
    participant Strategic as Strategic Level<br/>(7 Agents)
    participant ER as ER Engine
    participant GAT as GAT Aggregator
    participant MCDA as MCDA Engine
    participant Consensus

    User->>Coordinator: Submit Crisis Scenario

    par Parallel Assessment
        Coordinator->>Tactical: evaluate_scenario()
        Tactical->>Tactical: LLM Reasoning → Belief Distribution
        Tactical-->>Coordinator: {belief, confidence, reasoning}
    and
        Coordinator->>Strategic: evaluate_scenario()
        Strategic->>Strategic: LLM Reasoning → Belief Distribution
        Strategic-->>Coordinator: {belief, confidence, reasoning}
    end

    alt ER aggregation (default)
        Coordinator->>ER: combine_beliefs(beliefs, weights)
        ER-->>Coordinator: Aggregated Beliefs + Confidence
    else GAT aggregation
        Coordinator->>GAT: aggregate(9D features, beliefs)
        GAT-->>Coordinator: Aggregated Beliefs + Attention Weights
    end

    Coordinator->>MCDA: rank_alternatives (TOPSIS)
    MCDA-->>Coordinator: MCDA Scores

    Coordinator->>Consensus: check_consensus(assessments)
    Consensus-->>Coordinator: Consensus Level + Conflicts

    Note over Coordinator: final_score = 0.6 × ER/GAT + 0.4 × MCDA<br/>confidence = 0.6 × CL + 0.4 × mean(agent confidence)
    Coordinator-->>User: Final Decision + Explanation + Metrics
```
*Fig. 2. Six-step decision pipeline. Expert agents are queried in parallel; belief aggregation (ER or GAT) and MCDA scoring are combined with a 60/40 weighting to produce the final recommendation.*

### E. Historical Reliability Tracking

The reliability tracker maintains a per-agent performance history and updates it after every scenario. Because ground truth is unavailable in a simulated setting, we adopt a consensus-based proxy: the system's own final recommendation is treated as the reference outcome, and each agent's assessment is scored against it.

The reliability score for agent $j$ is computed as a temporally decayed, confidence-weighted average of past accuracy scores. For each evaluated assessment $k$ with age $d_k$ days and self-reported confidence $c_k$, the weight is

$$w_k = \gamma^{d_k} \times (0.5 + 0.5 \, c_k)$$

where $\gamma$ is a decay factor that down-weights older assessments. The overall reliability is then

$$\rho_j = \frac{\sum_k w_k \cdot a_k}{\sum_k w_k}$$

with $a_k$ the accuracy score of assessment $k$. A separate consistency score, defined as $1/(1 + \text{Var}(a))$ over recent assessments, is tracked alongside but does not enter the reliability computation directly.

Agent weights for ER aggregation are set by normalising the raw reliability scores across the panel:

$$w_j = \frac{\rho_j}{\sum_{j'} \rho_{j'}}$$

This scheme gradually concentrates influence on agents that have been consistently well-calibrated and dilutes the contribution of persistently poor performers.

As shown in Fig. 3, reliability scores feed into both aggregation paths through different mechanisms: the ER path consumes them as normalised weights, while the GAT path incorporates them as the ninth dimension of each agent's feature vector, allowing the attention mechanism to learn their importance jointly with the other eight features.

```mermaid
flowchart LR
    RT[ReliabilityTracker<br/>per agent]

    subgraph ER_PATH["ER Path"]
        direction TB
        WEIGHTS[Normalise reliability scores]
        AW[agent_weights = ρⱼ / Σρ]
        DS[Weighted belief<br/>combination]
        WEIGHTS --> AW --> DS
    end

    subgraph GAT_PATH["GAT Path"]
        direction TB
        INJECT[Inject reliability score<br/>into assessment]
        F9[Feature 9 of 9:<br/>Historical Reliability]
        ATT[Multi-head attention<br/>learns data-driven weights]
        INJECT --> F9 --> ATT
    end

    RT -->|get_reliability_score| ER_PATH
    RT -->|get_reliability_score| GAT_PATH

    style ER_PATH fill:#ffcccc,stroke:#c62828,stroke-width:2px
    style GAT_PATH fill:#ccddff,stroke:#1565c0,stroke-width:2px
    style RT fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```
*Fig. 3. Dual-path injection of reliability scores. The ER path uses normalised scores as explicit weights; the GAT path embeds them as a learned feature dimension.*

---

## IV. Experimental Setup

### A. Crisis Scenarios

The system is evaluated on three scenarios modelled influenced by real emergencies (Table I). The Karditsa flood scenario is influenced by the Thessaly inundations during Storm Daniel (September 2023); the Evia wildfire scenario is based on the North Evia fires of August 2021; and the Elefsina HAZMAT scenario is based on the industrial risk profile of the Thriasio Plain petrochemical zone. Each scenario defines five candidate response alternatives together with domain-specific evaluation criteria and their relative weights.

**TABLE I: Crisis Scenario Parameters**

| Parameter | Karditsa Flood | Evia Wildfire | Elefsina HAZMAT |
|-----------|---------------|---------------|-----------------|
| Severity | 0.8 (High) | 0.9 (Very High) | 0.85 (Very High) |
| Affected Pop. | 15,000 | 8,000 | 12,000 |
| Time Constraint | 2 hours | Immediate | 30 minutes |
| Alternatives | 5 | 12 | 5 |
| Top Criterion | Safety (0.35) | Life Safety (0.40) | Health Safety (0.45) |

### B. Evaluation Metrics

We report four metrics. The *Decision Quality Score* (DQS) is the weighted criterion-satisfaction value produced by the MCDA ranker. *Consensus Level* (CL) is the mean pairwise cosine similarity of agent belief vectors, indicating how much the experts agree before aggregation. *Decision Confidence* (DC) blends consensus and average agent confidence as $0.6 \times CL + 0.4 \times \overline{c}_{\text{agents}}$. Finally, the *Extended Comparison Bandwidth* (ECB) compares the multi-agent DQS against the score that each individual agent would have achieved alone.

### C. Configurations

We test five configurations: a compact 3-agent subset, the full 13-agent panel, an ER-versus-GAT comparison on the 13-agent panel, an LLM provider comparison (Claude, GPT-4, LLaMA 2 via LM Studio), and a single-agent baseline. Each configuration is run 25 times per scenario, yielding 75 runs per configuration.

---

## V. Results

### A. Overall System Performance

Table II summarises the aggregate results for the 3-agent and 13-agent configurations across 75 runs.

**TABLE II: System Performance (averaged across 75 runs)**

| Metric | 3-Agent | 13-Agent |
|--------|---------|----------|
| DQS | 84.7 % ± 2.3 | 86.3 % ± 1.9 |
| Consensus | 75.3 % ± 8.1 | 68.7 % ± 9.3 |
| Confidence | 79.8 % ± 6.4 | 81.2 % ± 5.8 |
| Processing Time | 12.4 s | 39.7 s |
| Cost / Scenario | $0.012 | $0.044 |

Expanding the panel from 3 to 13 agents raises DQS by 1.6 percentage points (p < 0.01) at the expense of lower consensus, which drops from 75.3 % to 68.7 %. The reduction in consensus is expected: a larger, more heterogeneous group will naturally exhibit greater disagreement, and we regard moderate consensus levels as a sign of genuine perspective diversity rather than a deficiency. Processing time scales roughly linearly with agent count.

### B. ER vs. GAT Comparison

Table III compares the two aggregation mechanisms on the full 13-agent panel.

**TABLE III: Aggregation Method Comparison (13-agent, 75 runs)**

| Metric | ER | GAT | Difference |
|--------|-----|-----|---|
| DQS | 0.861 | 0.863 | +0.002 (n.s.) |
| Consensus | 68.2 % | 71.0 % | +2.8 %** |
| Confidence | 80.4 % | 81.9 % | +1.5 %* |
| Processing Time | 38.9 s | 41.2 s | +2.3 s |

\* p < 0.05, \*\* p < 0.01, n.s. = not significant

The two methods produce nearly identical decision quality scores; the difference of 0.002 does not reach statistical significance. Where they diverge is in consensus and confidence: GAT achieves significantly higher values on both measures. Inspection of the learned attention weights offers some insight into why. In the flood and wildfire scenarios the Meteorologist receives the highest attention coefficient (0.18), whereas in the ammonia HAZMAT scenario the Medical Infrastructure Director and the Environmental Scientist together account for the largest share (0.22). In other words, the GAT learns to up-weight the agents whose expertise is most germane to each crisis type, and this adaptive weighting helps pull the expert panel towards greater agreement.

### C. Multi-Agent vs. Single-Agent

Table IV presents the extended comparison bandwidth — the gap between the collective recommendation and the best-performing individual agent on each scenario.

**TABLE IV: Extended Comparison Bandwidth**

| Scenario | Multi-Agent DQS | Best Individual | Mean Individual | ECB vs. Best |
|----------|----------------|-----------------|-----------------|--------------|
| Karditsa Flood | 0.839 | 0.772 | 0.733 | +6.7 % |
| Evia Wildfire | 0.891 | 0.837 | 0.759 | +5.4 % |
| Elefsina HAZMAT | 0.868 | 0.825 | 0.766 | +4.3 % |

In every scenario the multi-agent system outperforms not only the average individual agent but also the single best expert, with margins ranging from 4.3 to 6.7 percentage points. Notably, the identity of the best individual varies across scenarios — Meteorologist for the flood, Fire Commander for the wildfire, Medical Director for the HAZMAT event — which underscores the value of a panel that can draw on different specialisms as circumstances require.

### D. LLM Provider Comparison

**TABLE V: LLM Provider Performance**

| Provider | DQS | Consensus | Time (s) | Cost ($) |
|----------|-----|-----------|----------|----------|
| Claude 3 Sonnet | 0.863 | 71.0 % | 41.2 | 0.044 |
| GPT-4 | 0.859 | 69.3 % | 38.6 | 0.067 |
| LLaMA 2 (Local) | 0.831 | 64.2 % | 127.3 | 0.000 |

Among the three providers tested, Claude 3 Sonnet achieves the highest quality and consensus scores. GPT-4 is close behind in quality but at roughly 50 % higher API cost. The locally hosted LLaMA 2 model trails by 3.2 percentage points in DQS and is substantially slower, yet it incurs no API cost and keeps all data on-premise — a relevant consideration for agencies with strict data-sovereignty requirements.

### E. Historical Reliability Impact

Over a sequence of 100 scenarios, dynamic reliability-adjusted weighting yields a 2.5 percentage-point improvement in DQS relative to static weights (p < 0.01). The learning curve shows rapid gains during the first 30–40 scenarios and plateaus around scenario 60, after which agent reliability estimates stabilise. By the end of the sequence, reliability scores span the range 0.68 (Civil Protection Director) to 0.91 (Emergency Physician), indicating that the tracker is able to meaningfully discriminate between agents of varying calibration quality.

### F. Explainability Evaluation

Fifteen emergency management professionals reviewed system outputs from all three scenarios. Mean explainability was rated 4.2 out of 5, with auditability scoring highest at 4.5/5. Participants found the attention-weight visualisations and the natural-language justifications particularly useful for understanding why a recommendation was made. Several reviewers, however, cautioned that the system should undergo field-level validation before being considered for operational deployment.

---

## VI. Discussion

### A. Key Findings

We return to the five research questions posed in the introduction.

Regarding coordination (RQ1), the hierarchical architecture proves able to orchestrate 13 agents within processing times of 12–40 seconds, well inside the operational window of the scenarios tested. Consensus levels of 69–75 % may appear modest, but we interpret them as a healthy sign of perspective diversity; excessively high consensus would suggest that agents are redundant or that the system is converging on a single viewpoint prematurely.

On belief aggregation (RQ2), the near-identical DQS of ER and GAT (0.861 vs. 0.863) suggests that, at least for well-prompted agents, the choice of aggregation mechanism has less influence on outcome quality than the quality of the individual assessments feeding into it. The advantage of GAT lies elsewhere: its learned attention weights produce measurably higher consensus and confidence, and this benefit is likely to grow as the number of agents increases.

With respect to LLM-powered reasoning (RQ3), the structured prompt templates succeed in eliciting contextually grounded assessments that evaluators rate 4.1/5 for understandability. The multi-provider architecture mitigates availability risk through automatic fallback, though two practical concerns remain: the possibility of hallucinated facts in agent reasoning, and API latency, which accounts for over 85 % of total processing time.

The multi-agent versus single-agent comparison (RQ4) provides the clearest result of the study. Collective recommendations exceed the best individual expert by 4.3–6.7 percentage points across all three scenarios, and the identity of the best expert changes with the crisis type. This confirms that the gain is not simply a matter of error averaging but reflects genuine complementarity among specialisms.

Finally, on explainability (RQ5), the combination of attention-weight visualisation, MCDA score decomposition, and natural-language justification achieves an auditability rating of 4.5/5 from domain practitioners — the highest-rated dimension in the evaluation. This is encouraging for a domain in which post-hoc accountability is not optional.

### B. Practical Guidance on Aggregation Choice

The results do not point to a single best aggregation method for all circumstances. ER has the advantage of full transparency: every weight is explicit and deterministic, which matters in legal or regulatory settings where decisions must be auditable down to individual parameters. GAT, by contrast, is better suited to larger agent panels and to environments where scenario conditions change frequently, since it can learn context-dependent weighting patterns. A pragmatic deployment strategy would be to start with ER, accumulate a decision history, and transition to GAT once enough data are available to train the attention mechanism reliably.

### C. Limitations

Several limitations should temper the conclusions drawn above. The evaluation relies on simulated scenarios; no field deployment has yet been conducted, and real-world performance may differ in ways that simulation cannot anticipate. The agents inherit whatever biases and hallucination tendencies are present in their underlying language models. Only three crisis types are represented, all within the Greek institutional context, so generalisability to other hazard profiles or national response structures remains untested. The scenarios are treated as single-point decisions, with no modelling of how a crisis evolves over time. The stakeholder evaluation, while informative, is based on a small sample (n = 15). Finally, the consensus-based validation strategy — using the system's own recommendation as a proxy for ground truth — is a known methodological weakness; it may overstate reliability gains because the tracker rewards conformity with the majority rather than objective correctness.

---

## VII. Conclusion

We have described a multi-agent decision support system for crisis management that combines LLM-powered expert reasoning with two formal belief-aggregation mechanisms — weighted Evidential Reasoning and a Graph Attention Network — and evaluated it on three Greek emergency scenarios. The principal empirical finding is that collective multi-agent recommendations consistently outperform the best individual expert by a non-trivial margin (4.3–6.7 %), with the GAT path offering better consensus properties than ER at comparable decision quality. A historical reliability tracker that adjusts agent weights over successive decisions adds a further measurable gain, and the overall framework receives favourable explainability and auditability ratings from domain professionals.

Several directions for future work follow naturally. The current system treats each scenario as an isolated, single-shot decision; extending the architecture to model how crises evolve over time would better reflect operational reality. The scenario coverage should be broadened beyond the three types tested here, ideally in collaboration with national civil-protection agencies that can supply validated exercise data. On the technical side, multimodal inputs — satellite imagery, GIS layers, sensor feeds — could enrich the information available to each agent. Finally, structured human-AI teaming experiments, in which the system assists rather than replaces a human decision-maker, would provide the field-level evidence needed before any operational adoption can be considered responsibly.

At its core, the work is motivated by a straightforward observation: no single expert, however capable, can match a well-coordinated panel when the problem spans multiple domains under uncertainty. The challenge lies in designing the coordination mechanism so that it remains transparent, auditable, and ultimately subordinate to human judgement.

---

## References

[1] L. K. Comfort et al., "Reframing disaster policy: The global evolution of vulnerable communities," *Environ. Hazards*, vol. 5, no. 4, pp. 39–44, 2004.

[2] N. Kapucu and V. Garayev, "Collaborative decision-making in emergency and disaster management," *Int. J. Public Admin.*, vol. 34, no. 6, pp. 366–375, 2011.

[3] Z. Ren et al., "Agent-based evacuation model of large public buildings under fire conditions," *Autom. Constr.*, vol. 20, no. 7, pp. 959–965, 2011.

[4] J. K. Levy and K. Taji, "Group decision support for hazards planning and emergency management," *Math. Comput. Model.*, vol. 46, no. 7-8, pp. 906–917, 2007.

[5] M. Wooldridge, *An Introduction to MultiAgent Systems*, 2nd ed. Wiley, 2009.

[6] G. Shafer, *A Mathematical Theory of Evidence*. Princeton Univ. Press, 1976.

[7] J. B. Yang and D. L. Xu, "Evidential reasoning rule for evidence combination," *Artif. Intell.*, vol. 205, pp. 1–29, 2013.

[8] K. Sentz and S. Ferson, "Combination of evidence in Dempster-Shafer theory," Sandia Nat. Lab., SAND 2002-0835, 2002.

[9] C. L. Hwang and K. Yoon, *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag, 1981.

[10] M. Behzadian et al., "A state-of-the-art survey of TOPSIS applications," *Expert Syst. Appl.*, vol. 39, no. 17, pp. 13051–13069, 2012.

[11] P. Veličković et al., "Graph attention networks," in *Proc. ICLR*, 2018.

[12] X. Zhang et al., "Deep learning on graphs: A survey," *IEEE Trans. Knowl. Data Eng.*, vol. 34, no. 1, pp. 249–270, 2020.

[13] J. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models," in *Proc. NeurIPS*, vol. 35, pp. 24824–24837, 2022.

[14] J. Ferber, *Multi-Agent Systems: An Introduction to Distributed Artificial Intelligence*. Addison-Wesley, 1999.

[15] E. K. Zavadskas and Z. Turskis, "Multiple criteria decision making (MCDM) methods in economics: An overview," *Technol. Econ. Dev. Econ.*, vol. 17, no. 2, pp. 397–427, 2011.

---

## Acknowledgments

This research was conducted as part of a Master's thesis in Operational Research and Decision Making, jointly supervised by the Military Academy (SSE), Department of Military Sciences, and the Technical University of Crete (TUC), School of Production Engineering and Management. Supervisors: Emeritus Professor N. Matsatsinis, Associate Professor N. Papadakis, Assistant Professor E. Siskos.

---

**Contact:** vkazoukas@tuc.gr, kazoukas@gmail.com
**Institutions:** Military Academy (sse.gr), Technical University of Crete (tuc.gr)
**Repository:** https://github.com/kerbgr/crisis_mas_poc
