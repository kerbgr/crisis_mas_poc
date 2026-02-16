# A Collaborative Multi-Agent Framework for Crisis Management Decision Support Using Evidential Reasoning and Graph Attention Networks

**Vasileios Kazoukas**
*Military Academy (SSE), Department of Military Sciences*
*Technical University of Crete (TUC), School of Production Engineering and Management*
Email: vkazoukas@tuc.gr, kazoukas@gmail.com

---

## Abstract

Crisis management demands rapid, coordinated decision-making under severe uncertainty and time pressure. This paper presents a Multi-Agent System (MAS) integrating classical Evidential Reasoning (ER) with Graph Attention Networks (GAT) and Large Language Models (LLMs) for collaborative crisis response decision support. The framework features 13 specialized Greek emergency response expert agents with LLM-powered reasoning (Claude, GPT-4, local models), two belief aggregation mechanisms (Dempster-Shafer-based ER and GAT with 9-dimensional feature extraction including historical reliability tracking), and TOPSIS-based Multi-Criteria Decision Analysis.

Evaluation across three realistic Greek crisis scenarios — Karditsa flooding, Evia wildfires, and Elefsina ammonia HAZMAT — demonstrates that multi-agent collaboration achieves +4.3% to +6.7% decision quality improvement over the best individual expert. GAT aggregation achieves 92% consensus with +2.8% consensus and +1.5% confidence improvement over classical ER. Historical reliability tracking produces +2.5% quality improvement over 100 scenarios. Emergency management professionals rate system explainability at 4.2/5 and auditability at 4.5/5.

**Keywords:** Multi-Agent Systems, Crisis Management, Evidential Reasoning, Graph Attention Networks, Large Language Models, Decision Support Systems

---

## I. Introduction

Crisis situations present unique decision-making challenges: severe time constraints, incomplete information, rapidly evolving conditions, and catastrophic consequences of error [1]. Traditional crisis management relies on human expert coordination, which can become overwhelmed during large-scale emergencies requiring synchronization across medical response, logistics, meteorology, and environmental assessment [2].

While multi-agent systems have been explored for evacuation modeling [3] and collaborative disaster decision-making [4], existing approaches exhibit critical limitations: (1) limited integration of modern AI capabilities such as LLMs, (2) inadequate uncertainty handling through rigorous belief aggregation, (3) absent comparative analysis between classical and neural aggregation methods, (4) insufficient explainability for high-stakes domains, and (5) static agent weighting ignoring valuable performance history.

This work addresses these gaps through a novel hybrid framework with the following contributions:

1. **Hybrid Aggregation Framework**: First direct comparison of Evidential Reasoning versus Graph Attention Networks for multi-agent belief aggregation in crisis management
2. **LLM-Enhanced Agents**: Integration of multiple LLM providers (Claude, GPT-4, LM Studio) for contextual expert reasoning with automatic fallback
3. **Historical Reliability Tracking**: Dynamic agent weighting based on past performance, feeding into both ER weights and GAT features
4. **Greek Emergency Response Modeling**: Authentic 13-agent system representing Hellenic emergency services (EKAB, ELAS, Fire Corps, Coast Guard, Civil Protection) with tactical/strategic command hierarchy
5. **Explainability Mechanisms**: Attention visualization, consensus tracking, and decision audit trails achieving 4.5/5 auditability ratings

The paper investigates five research questions: (RQ1) effective multi-agent coordination for time-critical decisions, (RQ2) belief aggregation under high uncertainty, (RQ3) LLM enhancement of agent reasoning, (RQ4) multi-agent vs. single-agent decision quality, and (RQ5) transparent, auditable decision trails for high-stakes domains.

---

## II. Related Work

**Multi-Agent Systems.** Wooldridge [5] established foundational MAS principles including agent autonomy, social ability, and proactiveness. Ren et al. [3] demonstrated agent-based evacuation modeling, showing how emergent coordination optimizes life-saving outcomes. Our work adopts a hierarchical coordinator-expert architecture balancing agent independence with collective goal achievement.

**Dempster-Shafer Theory.** Shafer's [6] theory provides rigorous foundations for reasoning under uncertainty, permitting belief functions that explicitly represent ignorance. Yang and Xu [7] advanced the Evidential Reasoning Rule addressing limitations when highly conflicting evidence produces counterintuitive results. Sentz and Ferson [8] provide comprehensive analysis of combination operators. Our implementation adopts a simplified weighted ER approach for computational tractability in real-time crisis response.

**Multi-Criteria Decision Analysis.** TOPSIS [9], surveyed extensively by Behzadian et al. [10], provides intuitive geometric interpretation and computational efficiency for group decision-making. We implement TOPSIS alongside WSM and SAW for comparative validation.

**Graph Attention Networks.** Veličković et al. [11] introduced GAT for computing node representations through weighted attention over neighbors. Zhang et al. [12] survey deep learning on graphs. We adapt GAT to expert networks, treating agents as nodes with 9-dimensional features including historical reliability as a novel adaptive dimension.

**LLMs in Decision Support.** Chain-of-Thought prompting [13] enables structured reasoning through intermediate steps. We leverage structured prompt templates (~5,000 characters each) encoding expert roles, crisis protocols, and output format constraints for parseable belief distributions.

**Crisis Management.** Comfort et al. [1] and Kapucu and Garayev [2] emphasize multi-stakeholder coordination, while Levy and Taji [4] demonstrate MCDA in emergency management. Our work extends these with autonomous AI agents providing LLM-powered reasoning.

---

## III. Methodology

### A. System Architecture

The MAS implements a five-layer architecture: (1) CLI interface with JSON I/O, (2) 13 specialized expert agents with LLM reasoning engines and historical reliability tracking, (3) decision framework (ER, GAT, MCDA, consensus model, reliability tracker), (4) multi-provider LLM integration (Claude, GPT-4, LM Studio) with automatic fallback, and (5) evaluation layer with metrics computation and visualization.

The 13 agents represent Greek emergency response roles: Meteorologist (HNMS), Emergency Physician (EKAB), Logistics Coordinator (Civil Protection), PSAP Commander, Police Tactical/Regional Commanders (ELAS), Fire Tactical/Regional Commanders (Hellenic Fire Corps), Medical Infrastructure Director, Coast Guard Tactical/National Directors, Environmental Scientist, and Civil Engineer — plus a Coordinator agent.

### B. Evidential Reasoning

Our ER implementation uses weighted averaging of belief distributions for computational efficiency. For each alternative $i$ and agent $j$ with weight $w_j$ and belief $b_j(i)$:

$$b_{\text{combined}}(i) = \frac{\sum_j w_j \cdot b_j(i)}{\sum_j w_j}$$

Agent weights derive from expertise relevance, historical reliability (from ReliabilityTracker), and confidence scores. Decision confidence uses entropy-based quantification:

$$\text{confidence} = 1 - \frac{H}{\log_2(n_{\text{alternatives}})}, \quad H = -\sum_i p_i \log_2(p_i)$$

### C. Graph Attention Network

Our GAT extracts a 9-dimensional feature vector $\mathbf{f}_j \in \mathbb{R}^9$ per agent: (1) confidence score, (2) belief certainty (inverse entropy), (3) expertise relevance, (4) risk tolerance, (5) severity awareness, (6) top choice strength, (7) concerns raised, (8) reasoning quality, and (9) historical reliability from ReliabilityTracker — our key innovation enabling dynamic, data-driven weighting.

Multi-head attention ($K$=4 heads) computes attention coefficients:

$$e_{ij}^h = \text{LeakyReLU}(\mathbf{a}^h \cdot [\mathbf{W}^h \mathbf{f}_i \| \mathbf{W}^h \mathbf{f}_j])$$

$$\alpha_{ij}^h = \text{softmax}_j(e_{ij}^h)$$

Final aggregated belief for alternative $i$:

$$b_{\text{combined}}(i) = \frac{1}{K} \sum_{h=1}^{K} \sum_j \alpha_{ij}^h \cdot b_j(i)$$

### D. MCDA and Consensus

TOPSIS ranks alternatives by minimizing distance from ideal and maximizing distance from anti-ideal solutions in normalized criterion space. Consensus uses pairwise cosine similarity:

$$CL = \frac{2}{n(n-1)} \sum_i \sum_{j>i} \cos(\mathbf{b}_i, \mathbf{b}_j)$$

CL > 0.9 permits automated execution; CL < 0.7 triggers mandatory human review.

### E. Historical Reliability Tracking

The ReliabilityTracker maintains per-agent performance histories using consensus-based validation. Reliability scores combine exponential moving average of quality ($\alpha$=0.3), calibration alignment, and consistency:

$$\rho_j(t) = 0.7 \times \text{EMA}(q_j) + 0.2 \times c_j + 0.1 \times (1 - \sigma_j)$$

Dynamic weight adjustment boosts high-reliability agents and reduces low-reliability ones:

$$w_j = w_j^{\text{base}} \times \left(1 + 0.5 \times (\rho_j - \bar{\rho})\right)$$

---

## IV. Experimental Setup

### A. Crisis Scenarios

We evaluate on three authentic Greek emergency scenarios (Table I).

**TABLE I: Crisis Scenario Parameters**

| Parameter | Karditsa Flood | Evia Wildfire | Elefsina HAZMAT |
|-----------|---------------|---------------|-----------------|
| Severity | 0.8 (High) | 0.9 (Very High) | 0.85 (Very High) |
| Affected Pop. | 15,000 | 8,000 | 12,000 |
| Time Constraint | 2 hours | Immediate | 30 minutes |
| Alternatives | 5 | 5 | 5 |
| Top Criterion | Safety (0.35) | Life Safety (0.40) | Health Safety (0.45) |

Scenarios are inspired by historical events: Thessaly Storm Daniel flooding (2023), North Evia wildfires, and industrial HAZMAT incidents in the Elefsina zone.

### B. Evaluation Metrics

- **Decision Quality Score (DQS)**: Weighted criterion satisfaction via MCDA
- **Consensus Level (CL)**: Mean pairwise cosine similarity of agent belief vectors
- **Decision Confidence (DC)**: $0.6 \times CL + 0.4 \times \overline{c}_{\text{agents}}$
- **Extended Comparison Bandwidth (ECB)**: Multi-agent DQS vs. each individual agent

### C. Configurations

Five configurations: (1) Core 3-agent, (2) Full 13-agent, (3) ER vs. GAT comparison, (4) LLM provider comparison (Claude/GPT-4/LLaMA 2), and (5) single-agent baseline. Each executes 25 runs per scenario (75 total).

---

## V. Results

### A. Overall System Performance

**TABLE II: System Performance (Averaged across 75 runs)**

| Metric | 3-Agent | 13-Agent |
|--------|---------|----------|
| DQS | 84.7% ± 2.3 | 86.3% ± 1.9 |
| Consensus | 75.3% ± 8.1 | 68.7% ± 9.3 |
| Confidence | 79.8% ± 6.4 | 81.2% ± 5.8 |
| Processing Time | 12.4s | 39.7s |
| Cost/Scenario | $0.012 | $0.044 |

The 13-agent system achieves +1.6% DQS improvement (p<0.01) with expected consensus reduction from increased perspective diversity. Processing time scales approximately linearly.

### B. ER vs. GAT Comparison

**TABLE III: Aggregation Method Comparison (13-agent, 75 scenarios)**

| Metric | ER | GAT | Δ |
|--------|-----|-----|---|
| DQS | 0.861 | 0.863 | +0.002 (NS) |
| Consensus | 68.2% | 71.0% | +2.8%** |
| Confidence | 80.4% | 81.9% | +1.5%* |
| Processing Time | 38.9s | 41.2s | +2.3s |

*p<0.05, **p<0.01, NS=Not Significant

Both methods achieve comparable decision quality, but GAT provides significantly higher consensus and confidence through learned attention patterns that better downweight poorly-calibrated agents.

GAT attention weights appropriately reflect scenario-specific expertise: Meteorologist receives highest attention (0.18) in flood/wildfire scenarios; HAZMAT specialists dominate (0.22) in the ammonia scenario.

### C. Multi-Agent vs. Single-Agent

**TABLE IV: Extended Comparison Bandwidth**

| Scenario | Multi-Agent DQS | Best Individual | Mean Individual | ECB vs. Best |
|----------|----------------|-----------------|-----------------|--------------|
| Karditsa Flood | 0.839 | 0.772 | 0.733 | +6.7% |
| Evia Wildfire | 0.891 | 0.837 | 0.759 | +5.4% |
| Elefsina HAZMAT | 0.868 | 0.825 | 0.766 | +4.3% |

Multi-agent decisions consistently exceed even the best individual expert (+4.3% to +6.7%). The best individual varies by scenario (Meteorologist for flood, Fire Commander for wildfire, Medical Director for HAZMAT), validating the need for multi-expert systems.

### D. LLM Provider Comparison

**TABLE V: LLM Provider Performance**

| Provider | DQS | Consensus | Time (s) | Cost ($) |
|----------|-----|-----------|----------|----------|
| Claude 3 Sonnet | 0.863 | 71.0% | 41.2 | 0.044 |
| GPT-4 | 0.859 | 69.3% | 38.6 | 0.067 |
| LLaMA 2 (Local) | 0.831 | 64.2% | 127.3 | 0.000 |

Claude achieves highest quality and consensus. Local models provide zero-cost, privacy-preserving alternatives with acceptable quality degradation (-3.2% DQS).

### E. Historical Reliability Impact

Dynamic reliability-adjusted weighting produces +2.5% DQS improvement (p<0.01) over 100 sequential scenarios compared to static weights, with improvement plateauing around scenario 60. Final reliability scores range from 0.68 (Civil Engineer) to 0.91 (Emergency Physician), demonstrating effective discriminative power.

### F. Explainability Evaluation

Human evaluation by 15 emergency management professionals yields mean explainability rating of 4.2/5, with auditability rated highest (4.5/5). Stakeholders appreciate attention weight visualization and natural language reasoning but emphasize need for operational validation before full deployment.

---

## VI. Discussion

### A. Key Findings

**RQ1 (Coordination):** The hierarchical coordinator-expert architecture enables effective coordination among 13 agents with processing times (12.4–39.7s) within operational requirements. Moderate consensus levels (68.7–75.3%) reflect healthy diversity rather than groupthink.

**RQ2 (Uncertainty):** Both ER and GAT effectively aggregate beliefs with comparable DQS (0.861 vs. 0.863), suggesting aggregation method matters less than input quality for well-calibrated experts. GAT's superior consensus (+2.8%) becomes increasingly valuable as agent count grows.

**RQ3 (LLM Enhancement):** LLMs substantially enhance reasoning with contextual explanations rated 4.1/5 for understandability. Multi-provider architecture ensures availability through automatic fallback. Challenges include hallucination risks and API latency (85%+ of processing time).

**RQ4 (Decision Quality):** Multi-agent decisions consistently exceed single-agent performance (+4.3% to +6.7% over best individual), demonstrating genuine collective intelligence rather than mere error averaging.

**RQ5 (Explainability):** Multi-layered mechanisms (attention visualization, MCDA decomposition, natural language reasoning) achieve 4.5/5 auditability, addressing transparency requirements for high-stakes domains.

### B. ER vs. GAT Deployment Guidance

ER is preferred when maximum transparency is required (legal/regulatory contexts), training data is limited, or computational constraints exist. GAT is preferred for large agent networks (13+), dynamic environments, or when consensus optimization is prioritized. A hybrid approach — deploying ER initially, accumulating decision history, then transitioning to GAT — balances immediate feasibility with long-term adaptive optimization.

### C. Limitations

Key limitations include: (1) evaluation on simulated scenarios without real-world deployment validation, (2) LLM-based agents inheriting model biases and hallucination tendencies, (3) limited crisis type coverage (3 scenarios), (4) static single-point decisions without temporal evolution, (5) small stakeholder evaluation sample (n=15), and (6) criterion weights encoding subjective value judgments. The system uses consensus-based validation (the system's own recommendation as proxy ground truth) rather than real-world outcome feedback.

---

## VII. Conclusion

This paper presents a Multi-Agent System integrating Evidential Reasoning, Graph Attention Networks, and Large Language Models for crisis management decision support. Key findings demonstrate: (1) multi-agent decisions exceed single-expert judgments by +4.3% to +6.7%, (2) GAT and ER achieve comparable quality with GAT offering superior consensus (+2.8%), (3) LLM enhancement improves reasoning quality (4.1/5 understandability), (4) historical reliability tracking yields +2.5% quality improvement, and (5) explainability mechanisms achieve 4.5/5 auditability.

Future work includes temporal multi-agent systems for evolving crises, expanded crisis typology validation, multimodal LLM inputs (satellite imagery, maps), game-theoretic resource allocation, federated learning for multi-agency privacy preservation, and human-AI teaming optimization studies.

The framework demonstrates that carefully designed MAS can augment human crisis decision-making while preserving transparency and accountability — serving as collaborative partners rather than autonomous replacements for human commanders.

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
