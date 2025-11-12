# Evaluation Methodology

## Overview

This document describes the comprehensive evaluation framework for comparing multi-agent system (MAS) performance against single-agent baseline decisions. The framework evaluates the **Greek Emergency Response Multi-Agent System** featuring 13 expert agents responding to realistic Greek crisis scenarios.

The framework addresses critical bugs fixed in commit `09bec4c` that previously made single vs. multi-agent comparisons invalid.

### Greek Crisis Scenarios

The system is evaluated on three realistic Greek crisis scenarios:

1. **Karditsa Flood Emergency** (severity 0.8)
   - Location: Karditsa, Thessaly, Greece (39.3644°N, 21.9211°E)
   - 15,000 affected population
   - Pamisos River overflow with infrastructure damage

2. **Evia Forest Fire Emergency** (severity 0.9)
   - Location: North Evia, Central Greece (38.9231°N, 23.6578°E)
   - 8,000 affected population
   - 12,000 hectares burned, 4 active fire fronts
   - Canadair CL-415 and Chinook operations

3. **Elefsina Ammonia Leak Emergency** (severity 0.85)
   - Location: Elefsina (Eleusis), Attica, Greece (38.0411°N, 23.5461°E)
   - 12,000 affected population
   - UN1005 Anhydrous Ammonia leak (HAZMAT)
   - Toxic gas evacuation and decontamination

### Greek Expert Agents (13 Total)

The multi-agent system includes authentic Greek emergency response experts:

- **Dr. Dimitris Nikolaou** - Medical Expert (EKAB Emergency Physician)
- **Katerina Georgiou** - Logistics Coordinator (Civil Protection)
- **Dr. Eleni Papadopoulou** - Meteorologist
- **Dr. Sofia Karagianni** - Environmental Scientist
- **Taxiarchos Nikos Konstantinou** - Police Tactical Commander (ELAS)
- **Pyragos Ioanna Michaelidou** - Fire Tactical Commander (Hellenic Fire Corps)
- **Plotarchos Andreas Papadakis** - Coast Guard Operations (Hellenic Coast Guard)
- **Dr. Georgios Athanasiou** - Civil Engineer (Infrastructure)
- **Antonia Vassiliou** - Mental Health Expert (Psychologist)
- **Commander Maria Papadimitriou** - EKAB/PSAP Director (Emergency Operations)
- **Dr. Stavros Nikolaidis** - Public Health Officer (EODY)
- **Theodoros Makris** - Volunteer Coordinator (Hellenic Red Cross)
- **Konstantinos Petrou** - Disaster Recovery Specialist (Civil Protection)

---

## Critical Fix: Decision Quality Calculation

### Previous Issues (Fixed in commit 09bec4c)

**Bug #1: No Actual Quality Calculation**
- The `calculate_decision_quality()` method simply copied the `confidence` field
- Code: `weighted_score = confidence` (line 86, evaluation/metrics.py)
- Result: No actual evaluation of criteria satisfaction

**Bug #2: Multi-Agent Misused "Confidence" Field**
- The coordinator stored the ER+MCDA combined score (~0.31) as "confidence"
- This was actually a quality score, not a confidence measure
- Made multi-agent appear to have low confidence when it actually had high consensus

**Bug #3: Invalid Comparison**
- Single-agent: Used LLM's subjective confidence (~0.82)
- Multi-agent: Used ER+MCDA combined score (~0.31)
- These metrics represented different things on different scales
- Example: -62.2% "improvement" was meaningless

### Fixed Implementation

**Decision Quality Score (DQS)** now properly evaluates criteria satisfaction:

$$\text{DQS} = \begin{cases}
\frac{1}{|C|} \sum_{c \in C} s_c(a^{*}) & \text{if criteria scores available} \\
\text{MCDA}(a^{*}) & \text{if multi-agent with MCDA} \\
f(a^{*}) & \text{if only final scores available}
\end{cases}$$

Where:
- $a^{*}$ = recommended alternative
- $C$ = set of decision criteria
- $s_c(a^{*})$ = score of alternative $a^{*}$ on criterion $c$
- $\text{MCDA}(a^{*})$ = MCDA score for alternative $a^{*}$
- $f(a^{*})$ = final combined score for alternative $a^{*}$

**Confidence** is now properly separated from quality:

For multi-agent decisions:
$$\text{Confidence}_{\text{MA}} = 0.6 \times \text{Consensus} + 0.4 \times \bar{c}_{\text{agents}}$$

Where:
- $\text{Consensus}$ = pairwise agent agreement (cosine similarity)
- $\bar{c}_{\text{agents}}$ = average agent confidence

For single-agent decisions:
$$\text{Confidence}_{\text{SA}} = c_{\text{LLM}}$$

Where $c_{\text{LLM}}$ is the LLM's self-reported confidence.

---

## Performance Metrics

### 1. Decision Quality Score (DQS)

**Purpose:** Measures how well the recommended alternative satisfies the decision criteria.

**Calculation Methods:**

#### Single-Agent DQS

Uses criteria scores from the expert agent's assessment:

```python
criteria_scores = {
    'safety': {'alt1': 0.90, 'alt2': 0.85, 'alt3': 0.70},
    'cost': {'alt1': 0.50, 'alt2': 0.80, 'alt3': 0.60},
    'speed': {'alt1': 0.95, 'alt2': 0.90, 'alt3': 0.65}
}

# For recommended alternative 'alt1':
DQS = (0.90 + 0.50 + 0.95) / 3 = 0.783
```

Formula:
$$\text{DQS}_{\text{SA}} = \frac{1}{|C|} \sum_{c \in C} s_c(a^{*})$$

#### Multi-Agent DQS

Uses MCDA score of the recommended alternative:

```python
mcda_scores = {
    'alt1': 0.720,
    'alt2': 0.550,
    'alt3': 0.810
}

# For recommended alternative 'alt1':
DQS = mcda_scores['alt1'] = 0.720
```

Formula:
$$\text{DQS}_{\text{MA}} = \text{MCDA}(a^{*})$$

Where MCDA is calculated using TOPSIS method (see below).

**Weighted Version:**

If criteria weights $w_c$ are provided:

$$\text{DQS}_{\text{weighted}} = \frac{\sum_{c \in C} w_c \cdot s_c(a^{*})}{\sum_{c \in C} w_c}$$

**Output Format:**

```json
{
  "weighted_score": 0.783,
  "confidence": 0.820,
  "criteria_satisfaction": {
    "safety": 0.90,
    "cost": 0.50,
    "speed": 0.95
  },
  "recommended_alternative": "alt1",
  "ground_truth_match": null
}
```

**Interpretation:**
- **0.0-0.3:** Poor quality - recommendation weakly satisfies criteria
- **0.3-0.5:** Below average - significant trade-offs present
- **0.5-0.7:** Acceptable - reasonable balance of criteria
- **0.7-0.9:** Good quality - strong satisfaction of most criteria
- **0.9-1.0:** Excellent - optimal or near-optimal solution

---

### 2. Consensus Level (CL)

**Purpose:** Measures agreement between agents using belief distribution similarity.

**Formula:**

$$\text{CL} = \frac{2}{n(n-1)} \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \cos(\mathbf{m}_i, \mathbf{m}_j)$$

Where cosine similarity is:

$$\cos(\mathbf{m}_i, \mathbf{m}_j) = \frac{\sum_{a \in \mathcal{A}} m_i(a) \cdot m_j(a)}{\sqrt{\sum_{a \in \mathcal{A}} m_i(a)^2} \cdot \sqrt{\sum_{a \in \mathcal{A}} m_j(a)^2}}$$

**Components:**
- $n$ = number of agents
- $\mathbf{m}_i$ = belief vector of agent $i$
- $\mathcal{A}$ = set of alternatives
- $m_i(a)$ = agent $i$'s belief mass for alternative $a$

**Example (Karditsa Flood Scenario):**

```python
agent_beliefs = {
    'Dr. Dimitris Nikolaou (Medical)': {'evacuate_hospital': 0.6, 'shelter_in_place': 0.3, 'partial_evac': 0.1},
    'Katerina Georgiou (Logistics)': {'evacuate_hospital': 0.7, 'shelter_in_place': 0.2, 'partial_evac': 0.1},
    'Pyragos Ioanna Michaelidou (Fire)': {'evacuate_hospital': 0.5, 'shelter_in_place': 0.4, 'partial_evac': 0.1}
}

# Pairwise similarities:
# cos(Dimitris, Katerina) = 0.987
# cos(Dimitris, Ioanna) = 0.954
# cos(Katerina, Ioanna) = 0.921

# Consensus:
CL = (0.987 + 0.954 + 0.921) / 3 = 0.954
```

**Output Format:**

```json
{
  "consensus_level": 0.954,
  "pairwise_similarities": {
    "Dimitris_Katerina": 0.987,
    "Dimitris_Ioanna": 0.954,
    "Katerina_Ioanna": 0.921
  },
  "agreement_percentage": 1.0,
  "top_preference": "evacuate_hospital",
  "num_agents": 3
}
```

**Interpretation:**
- **CL > 0.80:** Strong consensus - agents largely agree
- **0.60 < CL ≤ 0.80:** Moderate consensus - some disagreement
- **0.40 < CL ≤ 0.60:** Low consensus - significant differences
- **CL ≤ 0.40:** No consensus - conflicting views

---

### 3. Confidence Score (CS)

**Purpose:** Measures certainty in the decision and individual agent confidence levels.

**Multi-Agent Confidence:**

$$\text{CS}_{\text{decision}} = 0.6 \times \text{CL} + 0.4 \times \frac{1}{n} \sum_{i=1}^{n} c_i$$

Where:
- $\text{CL}$ = consensus level (agreement between agents)
- $c_i$ = confidence of agent $i$
- $n$ = number of agents

**Rationale:**
- High consensus (CL) → higher confidence in collective decision
- High individual confidence → agents are certain of their assessments
- 60/40 weighting prioritizes consensus over individual certainty

**Single-Agent Confidence:**

$$\text{CS}_{\text{decision}} = c_{\text{LLM}}$$

Simply uses the LLM's self-reported confidence level.

**Uncertainty:**

$$U = 1 - \text{CS}_{\text{decision}}$$

**Confidence Variance:**

$$\sigma^2_c = \frac{1}{n} \sum_{i=1}^{n} (c_i - \bar{c})^2$$

**Output Format:**

```json
{
  "decision_confidence": 0.847,
  "average_confidence": 0.798,
  "uncertainty": 0.153,
  "confidence_variance": 0.032,
  "confidence_std": 0.179,
  "min_confidence": 0.71,
  "max_confidence": 0.88,
  "num_agents": 4,
  "agent_confidences": [0.82, 0.78, 0.88, 0.71]
}
```

**Interpretation:**
- **Low variance (<0.05):** Consistent confidence across agents
- **High variance (>0.10):** Disagreement in certainty levels
- **Decision confidence > average:** Aggregation provides benefit
- **Decision confidence < average:** Aggregation introduces doubt

---

### 4. Expert Contribution Balance (ECB)

**Purpose:** Measures fairness and diversity in expert participation.

**Balance Score:**

$$\text{ECB} = 1 - G$$

Where $G$ is the Gini coefficient:

$$G = \frac{\sum_{i=1}^{n} (2i - n - 1) \cdot w_i}{n \sum_{i=1}^{n} w_i}$$

With weights $w_1 \leq w_2 \leq \ldots \leq w_n$ (sorted).

**Contribution Score per Agent:**

$$\text{contrib}_i = \frac{c_i + H_{\text{norm}}(m_i)}{2}$$

Where:
- $c_i$ = agent $i$'s confidence
- $H_{\text{norm}}(m_i)$ = normalized entropy of belief distribution

**Diversity Score:**

$$\text{Diversity} = \frac{|\{a^{*}_1, a^{*}_2, \ldots, a^{*}_n\}|}{n}$$

Number of unique top preferences divided by total agents.

**Output Format (Evia Forest Fire - 6 Agents Selected):**

```json
{
  "balance_score": 0.923,
  "participation_distribution": {
    "Dr. Dimitris Nikolaou": 0.75,
    "Pyragos Ioanna Michaelidou": 0.82,
    "Katerina Georgiou": 0.68,
    "Dr. Eleni Papadopoulou": 0.78,
    "Taxiarchos Nikos Konstantinou": 0.71,
    "Commander Maria Papadimitriou": 0.80
  },
  "diversity_score": 0.67,
  "gini_coefficient": 0.077,
  "unique_preferences": 4,
  "num_agents": 6
}
```

**Interpretation:**
- **Balance > 0.90:** Well-balanced participation
- **Gini < 0.20:** Relatively equal influence
- **Diversity = 1.0:** All agents prefer different alternatives
- **Diversity < 0.5:** Majority agree on one alternative

---

### 5. Efficiency Metrics

**Purpose:** Track computational cost and time to consensus.

**Metrics:**

- **Time to Consensus:** Number of deliberation iterations
- **API Calls:** Total LLM API calls made
- **Processing Time:** Wall-clock time in seconds
- **Tokens Used:** Total tokens consumed
- **Cost:** Estimated USD cost

**Efficiency Score:**

$$\text{Eff} = \frac{1}{3} \left( \frac{1}{1 + I} + \frac{1}{1 + A/3} + \frac{1}{1 + T/5} \right)$$

Where:
- $I$ = iterations
- $A$ = API calls
- $T$ = time in seconds

Baselines: 1 iteration, 3 API calls, 5 seconds.

**Output Format:**

```json
{
  "time_to_consensus": 1,
  "api_calls_used": 4,
  "processing_time_seconds": 12.4,
  "efficiency_score": 0.891,
  "iteration_efficiency": 0.500,
  "api_efficiency": 0.750,
  "time_efficiency": 0.424
}
```

---

## Baseline Comparison

### Single-Agent vs Multi-Agent

**Decision Quality Comparison:**

$$\Delta \text{DQS} = \text{DQS}_{\text{MA}} - \text{DQS}_{\text{SA}}$$

$$\Delta \text{DQS}_{\%} = \frac{\text{DQS}_{\text{MA}} - \text{DQS}_{\text{SA}}}{\text{DQS}_{\text{SA}}} \times 100\%$$

**Example:**

```json
{
  "decision_quality": {
    "multi_agent": 0.720,
    "single_agent": 0.850,
    "improvement": -0.130,
    "improvement_percentage": -15.3
  },
  "confidence": {
    "multi_agent": 0.760,
    "single_agent": 0.820,
    "improvement": -0.060,
    "improvement_percentage": -7.3
  }
}
```

**Interpretation:**
- **Positive improvement:** Multi-agent outperforms single-agent
- **Negative improvement:** Single-agent outperforms multi-agent
- **±5%:** Negligible difference
- **±20%:** Significant difference
- **>50%:** Major difference (investigate causes)

**Important Note:** Both DQS scores are now calculated using the same methodology (criteria satisfaction), making them directly comparable.

---

## Ground Truth Validation

When ground truth is available:

**Ground Truth Match:**

$$\text{Match} = \begin{cases}
1 & \text{if } a^{*} = a_{\text{correct}} \\
0 & \text{otherwise}
\end{cases}$$

**Quality Boost:**

If ground truth matches:
$$\text{DQS}_{\text{final}} = \max(\text{DQS}, 0.9)$$

**Output Format:**

```json
{
  "ground_truth_match": {
    "match": true,
    "recommended": "alt1",
    "correct": "alt1"
  }
}
```

---

## Statistical Significance Testing

### T-Test for Multi-Run Comparison

**Null Hypothesis:** $H_0: \mu_{\text{MA}} = \mu_{\text{SA}}$

**Alternative:** $H_1: \mu_{\text{MA}} \neq \mu_{\text{SA}}$

**Test Statistic:**

$$t = \frac{\bar{x}_{\text{MA}} - \bar{x}_{\text{SA}}}{\sqrt{\frac{s^2_{\text{MA}}}{n_{\text{MA}}} + \frac{s^2_{\text{SA}}}{n_{\text{SA}}}}}$$

**Effect Size (Cohen's d):**

$$d = \frac{\bar{x}_{\text{MA}} - \bar{x}_{\text{SA}}}{s_{\text{pooled}}}$$

Where:

$$s_{\text{pooled}} = \sqrt{\frac{(n_{\text{MA}} - 1)s^2_{\text{MA}} + (n_{\text{SA}} - 1)s^2_{\text{SA}}}{n_{\text{MA}} + n_{\text{SA}} - 2}}$$

**Interpretation:**
- $|d| < 0.2$: Negligible effect
- $0.2 \leq |d| < 0.5$: Small effect
- $0.5 \leq |d| < 0.8$: Medium effect
- $|d| \geq 0.8$: Large effect

**Output Format:**

```json
{
  "t_statistic": 2.341,
  "p_value": 0.0234,
  "significant": true,
  "alpha": 0.05,
  "cohens_d": 0.567,
  "effect_size": "medium",
  "multi_agent_mean": 0.745,
  "single_agent_mean": 0.680,
  "multi_agent_std": 0.082,
  "single_agent_std": 0.091,
  "n_multi": 10,
  "n_single": 10
}
```

---

## Implementation Details

### Data Flow

```
Decision → calculate_decision_quality() → DQS
                                        ↓
                                    Compare → Improvement %
                                        ↑
Baseline → calculate_decision_quality() → DQS
```

### Code Example (Elefsina Ammonia Leak Scenario)

```python
from evaluation.metrics import MetricsEvaluator

evaluator = MetricsEvaluator()

# Criteria for HAZMAT response decision
criteria_weights = {
    'safety': 0.5,           # Public safety priority
    'response_speed': 0.3,   # Toxic gas requires fast action
    'resource_efficiency': 0.2
}

# Calculate multi-agent decision quality (13 Greek experts available)
multi_dqs = evaluator.calculate_decision_quality(
    decision=multi_agent_decision,  # From 5-7 selected experts
    criteria_weights=criteria_weights
)

# Calculate single-agent baseline (Dr. Dimitris Nikolaou only)
single_dqs = evaluator.calculate_decision_quality(
    decision=single_agent_decision,
    criteria_weights=criteria_weights
)

# Compare multi-agent vs single-agent performance
comparison = evaluator.compare_to_baseline(
    {'decision_quality': multi_dqs},
    {'decision_quality': single_dqs}
)

print(f"MAS vs Single-Agent Improvement: {comparison['decision_quality']['improvement_percentage']:.1f}%")
print(f"Consensus Level: {multi_agent_decision.get('consensus_level', 'N/A')}")
```

---

## Testing

Run `test_evaluation_fix.py` to verify the evaluation methodology:

```bash
python test_evaluation_fix.py
```

Expected output:
```
✓ Decision quality now calculated from criteria satisfaction
✓ Multi-agent uses MCDA scores
✓ Single-agent uses criteria scores
✓ Both scores are now COMPARABLE
✓ Confidence is separate from quality score
```

---

## References

### Key Algorithms

- **TOPSIS (MCDA):** Multi-criteria decision analysis
- **Evidential Reasoning:** Belief aggregation framework
- **Cosine Similarity:** Vector similarity measure
- **Gini Coefficient:** Inequality measure

### Related Files

- `evaluation/metrics.py` - Implementation
- `agents/coordinator_agent.py` - Multi-agent decision generation
- `agents/expert_agent.py` - Single-agent assessment
- `agents/agent_profiles.json` - 13 Greek expert profiles
- `models/data_models.py` - Pydantic models (LLMResponse, BeliefDistribution, AgentAssessment)
- `llm_integration/lmstudio_client.py` - LM Studio client with JSON cleaning
- `llm_integration/claude_client.py` - Claude API client
- `llm_integration/openai_client.py` - OpenAI API client
- `scenarios/flood_scenario.json` - Karditsa flood scenario
- `scenarios/forest_fire_evia.json` - Evia forest fire scenario
- `scenarios/ammonia_leak_elefsina.json` - Elefsina ammonia leak scenario
- `test_evaluation_fix.py` - Test suite

---

## Changelog

### v2.1.0 (2025-11-12) - Greek Emergency Response Edition

**New Features:**
- **13 Greek Expert Agents:** Authentic Greek emergency response professionals with Greek names and ranks
  - Medical: Dr. Dimitris Nikolaou (EKAB), Dr. Stavros Nikolaidis (EODY), Antonia Vassiliou (Mental Health)
  - Tactical: Taxiarchos Nikos Konstantinou (ELAS Police), Pyragos Ioanna Michaelidou (Fire), Plotarchos Andreas Papadakis (Coast Guard)
  - Coordination: Commander Maria Papadimitriou (EKAB Director), Katerina Georgiou (Civil Protection Logistics)
  - Specialists: Dr. Eleni Papadopoulou (Meteorology), Dr. Sofia Karagianni (Environment), Dr. Georgios Athanasiou (Civil Engineering)
  - Support: Theodoros Makris (Red Cross), Konstantinos Petrou (Disaster Recovery)

- **Three Greek Crisis Scenarios:**
  - Karditsa Flood Emergency (Thessaly, severity 0.8)
  - Evia Forest Fire Emergency (Central Greece, severity 0.9)
  - Elefsina Ammonia Leak HAZMAT Emergency (Attica, severity 0.85)

**Technical Improvements:**
- **Pydantic Model Validation:** LLMResponse, BeliefDistribution, and AgentAssessment now implement dict-like interfaces
  - Added `__len__()`, `__getitem__()`, `__setitem__()`, `__contains__()` methods
  - Support for `Union[Dict, LLMResponse]` in all validation functions
- **JSON Parsing Robustness:** Clean trailing commas from LM Studio responses
- **Greeklish Support:** Converted Greek characters to Latin alphabet for LLM compatibility
- **Role Mapping Fix:** Updated agent role names to match LLM prompt template expectations

**Documentation:**
- Updated all examples to use Greek expert names and scenarios
- Added Greek crisis scenario details with coordinates and severity levels
- Updated architecture diagrams to reflect 13-expert system
- Enhanced README with Greek scenario usage examples

**Evaluation Updates:**
- Examples now use realistic Greek emergency response decisions
- Multi-agent evaluations test 13-expert pool with 5-7 expert selection
- Consensus metrics validated on Greek expert belief distributions
- Criteria weights reflect Greek emergency response priorities (safety-first approach)

### v2.0.1 (2025-01-09) - Fixed DQS Fallback Logic

**Bug Fixes:**
- Fixed Bug #4: DQS fallback didn't work when criteria_scores present but recommended alternative not found
  - Symptom: Single-agent DQS returned 0.0 when alternative IDs didn't match
  - Cause: Fallback logic used elif/else instead of proper fallthrough
  - Fix: Added `quality_calculated` flag to enable proper cascade through fallback options
  - Impact: Single-agent evaluations now properly fall back to final_scores

**Testing:**
- Added `debug_single_agent_dqs.py` test suite
- All fallback scenarios now verified

### v2.0.0 (2025-01-09) - Fixed Evaluation Methodology

**Breaking Changes:**
- Decision quality now properly calculated from criteria satisfaction
- Confidence and quality are separate metrics
- Multi-agent decisions include both `confidence` and `decision_quality_score` fields

**Bug Fixes:**
- Fixed Bug #1: DQS calculation bypassed
- Fixed Bug #2: Multi-agent misused confidence field
- Fixed Bug #3: Invalid single vs multi-agent comparison

**Impact:**
- Single vs multi-agent comparisons are now valid and interpretable
- Scores are comparable (same scale, same methodology)
- Example: Previous -62% "improvement" corrected to -15%

### v1.0.0 - Initial Implementation

**Features:**
- Basic decision quality metrics
- Consensus level calculation
- Confidence scoring
- Expert contribution balance

**Known Issues:**
- Decision quality not properly calculated (fixed in v2.0.0)
- Confidence conflated with quality score (fixed in v2.0.0)
