# Limitations

## Algorithmic Limitations

### 1. Dempster-Shafer Evidential Reasoning

**Implementation:** Full Dempster-Shafer combination rule with conflict handling, optimized by ReliabilityTracker dynamic weighting.

**Features Implemented:**
- Complete Dempster's combination rule: $m_{12}(a) = \frac{1}{1-K} \sum_{x \cap y = a} m_1(x) \cdot m_2(y)$
- Conflict mass calculation: $K = \sum_{i \neq j} m_1(a_i) \cdot m_2(a_j)$
- High conflict handling (K > 0.7) via proportional redistribution
- Agent sorting by reliability for stable iterative combination
- **Dynamic agent weighting**: ReliabilityTracker provides per-agent weights based on historical accuracy, updated after each decision cycle via consensus-based validation
- **Domain-specific weighting**: Scenario-type-specific reliability scores (e.g., flood, fire) allow agents with proven track records in particular crisis types to receive higher weights
- Backward compatibility with weighted averaging (`method='weighted'`)

**Remaining Simplifications:**
- Assumes singleton focal elements (beliefs assigned to individual alternatives only)
- Does not implement frame of discernment with compound hypotheses
- Missing features: Pignistic transformation, Transferable Belief Model (TBM)

**Impact:** May not capture full complexity in scenarios requiring:
- Compound hypotheses (e.g., "A or B" as single belief)
- Hierarchical belief structures

**Mitigation:** GAT aggregation provides neural attention-based alternative for complex scenarios.

### 2. Static MCDA Weights

**Limitation:** Criteria weights are predefined and static across all scenarios.

**Impact:**
- Does not adapt to scenario-specific contexts (e.g., cost may matter less in extreme emergencies)
- Cannot learn optimal weights from historical decisions
- Assumes consistent stakeholder preferences

**Mitigation:** Manual weight adjustment per scenario type is supported but not automated.

### 3. GAT Training Data

**Limitation:** GAT uses rule-based feature extraction and a fixed attention formula, not weights learned from data via backpropagation.

**Explanation:** Unlike typical GAT implementations trained on large datasets, our GAT uses:

- Hand-crafted 9-dimensional feature functions (including historical reliability from ReliabilityTracker as feature 9)
- Fixed attention formula: $e_{ij} = 0.4 \cdot \text{confidence} + 0.3 \cdot \text{relevance} + 0.3 \cdot \text{certainty} + 0.2 \cdot \cos(\mathbf{f}_i, \mathbf{f}_j)$ (not learned weights)
- No backpropagation or gradient descent

**Partial Mitigation via ReliabilityTracker:** While the attention formula itself is fixed, the 9th feature dimension (historical reliability) introduces **data-driven adaptation**. Agents with proven track records in specific crisis types receive different feature values over time, indirectly influencing attention weights across decision cycles. This provides a lightweight learning signal without full gradient-based training.

**Remaining Impact:** May not capture complex, non-linear expert interaction patterns that end-to-end data-driven learning would discover.

**Rationale:** Insufficient training data for supervised learning in crisis domain. Rule-based approach ensures interpretability for high-stakes decisions.

## Operational Limitations

### 4. API Costs and Latency

**Limitation:** Claude API incurs costs and latency for each agent assessment.

**Costs:**
- ~$0.004-0.005 per agent assessment
- ~$0.052-0.065 per complete decision (13 agents, full team)
- Scales linearly with agent count

**Latency:**
- ~2-4 seconds per agent (API call + processing)
- ~8-16 seconds total for parallel 13-agent decision (LLM-bound)
- Network dependency introduces variability

**Impact:** Not suitable for:
- Real-time systems requiring sub-second response
- High-frequency decision scenarios (>100/hour)
- Offline/air-gapped deployments

**Mitigation:** `--no-llm` mode provides rule-based fallback (instant, free, but lower quality).

### 5. LLM Prompt Sensitivity

**Limitation:** Decision quality depends on prompt engineering and LLM stochasticity.

**Issues:**
- Different prompts can yield different recommendations for same scenario
- Temperature >0 introduces randomness (reduced but not eliminated at T=0.7)
- Model updates may change behavior (Claude versions)

**Impact:** Reproducibility challenges for scientific validation.

**Mitigation:**
- Fixed prompt templates with version control
- Seed setting for temperature control
- Logging of exact prompts and model versions

## Scope Limitations

### 6. Expert Diversity (Expanded in v0.8)

**Current (v0.8):** 13 expert types organized in emergency response command structure:
- Core: Meteorologist, Logistics, Medical Director
- Emergency Communications: PSAP Commander
- Law Enforcement: Tactical and Strategic Police Commanders
- Fire/Rescue: Tactical and Strategic Fire Commanders
- Medical Infrastructure: Healthcare System Director
- Maritime: Tactical and Strategic Coast Guard Commanders

**Still Missing for Future Enhancement:**
- Economic/financial experts (budget analysis, cost-benefit)
- Legal/regulatory experts (compliance, liability)
- Communications/media experts (public messaging, crisis communications)
- Political/governance experts (policy implications, stakeholder management)
- Psychological/social experts (community impact, trauma response)
- Civil engineering experts (infrastructure assessment)

**Impact:** The 13-agent system provides comprehensive operational command perspective, but may still miss economic, legal, and social dimensions in complex multi-faceted crises.

### 7. Simplified Scenario Representation

**Current Scenario Format:**
- Static JSON with predefined alternatives
- No dynamic environment simulation
- No real-time data integration
- No spatial/geographic information

**Impact:** Cannot handle:
- Evolving crises with changing conditions
- Spatially distributed decisions
- Information cascades and updates

### 8. Limited Learning Mechanism

**Current Implementation:** The ReliabilityTracker provides a consensus-based learning loop:

- **Historical tracking**: Per-agent assessment history with accuracy scores persisted to JSON
- **Outcome feedback**: After each decision, the consensus recommendation serves as proxy ground truth; agents are scored on probability accuracy (40%), rank accuracy (30%), and confidence calibration (30%)
- **Adaptive weighting**: Reliability scores update ER agent weights and GAT feature 9 after each decision cycle
- **Domain-specific learning**: Per-crisis-type reliability scores (flood, fire, HAZMAT) enable specialization
- **Temporal decay**: Older assessments count less ($w_t = \gamma^{(T-t)}$, $\gamma = 0.95$)

**Remaining Limitations:**

- Consensus-based validation uses the system's own recommendation as ground truth, not real-world outcomes
- No reinforcement learning from actual crisis results or expert post-incident reviews
- Agent profiles (expertise, risk tolerance) remain static; only weights adapt
- No cross-organization learning or federated knowledge sharing

**Impact:** System improves agent weighting over repeated runs, but cannot validate against real-world outcomes or adapt agent reasoning strategies.

### 9. Scalability Constraints

**Current:** Tested with 3-13 agents, 5-12 alternatives

**Scalability Limits:**
- ER complexity: O(n²) for n agents (pairwise belief combination)
- GAT complexity: O(n² · d) for n agents, d features (d=9)
- Consensus checking: O(n²) for pairwise comparisons
- LLM API latency: ~3-4s per agent (parallel calls in current implementation)

**Performance at Scale:**
- 3 agents (default): ~10-15s total decision time
- 13 agents (full team): ~35-45s total decision time
- Estimated 20 agents: ~60-80s (still practical for crisis decisions)

**Impact:** Current architecture scales well to 20-30 agents. May face performance issues with >50 agents or >10 alternatives without optimization.

### 10. Evaluation Limitations

**Limitation:** No ground truth for crisis decisions.

**Challenges:**
- Cannot validate "correctness" without real-world deployment
- Metrics measure internal consistency, not external validity
- No comparison with actual crisis management outcomes

**Current Validation:** Limited to:
- Face validity (expert review)
- Internal consistency checks
- Comparative benchmarks (single vs. multi-agent)
