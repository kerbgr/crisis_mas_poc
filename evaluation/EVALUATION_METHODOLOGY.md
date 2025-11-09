# Evaluation Methodology

## Overview

This document describes the comprehensive evaluation framework for comparing multi-agent system (MAS) performance against single-agent baseline decisions. The framework addresses critical bugs fixed in commit `09bec4c` that previously made single vs. multi-agent comparisons invalid.

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
\frac{1}{|C|} \sum_{c \in C} s_c(a^*) & \text{if criteria scores available} \\
\text{MCDA}(a^*) & \text{if multi-agent with MCDA} \\
f(a^*) & \text{if only final scores available}
\end{cases}$$

Where:
- $a^*$ = recommended alternative
- $C$ = set of decision criteria
- $s_c(a^*)$ = score of alternative $a^*$ on criterion $c$
- $\text{MCDA}(a^*)$ = MCDA score for alternative $a^*$
- $f(a^*)$ = final combined score for alternative $a^*$

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
$$\text{DQS}_{\text{SA}} = \frac{1}{|C|} \sum_{c \in C} s_c(a^*)$$

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
$$\text{DQS}_{\text{MA}} = \text{MCDA}(a^*)$$

Where MCDA is calculated using TOPSIS method (see below).

**Weighted Version:**

If criteria weights $w_c$ are provided:

$$\text{DQS}_{\text{weighted}} = \frac{\sum_{c \in C} w_c \cdot s_c(a^*)}{\sum_{c \in C} w_c}$$

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

**Example:**

```python
agent_beliefs = {
    'agent1': {'alt1': 0.6, 'alt2': 0.3, 'alt3': 0.1},
    'agent2': {'alt1': 0.7, 'alt2': 0.2, 'alt3': 0.1},
    'agent3': {'alt1': 0.5, 'alt2': 0.4, 'alt3': 0.1}
}

# Pairwise similarities:
# cos(m1, m2) = 0.987
# cos(m1, m3) = 0.954
# cos(m2, m3) = 0.921

# Consensus:
CL = (0.987 + 0.954 + 0.921) / 3 = 0.954
```

**Output Format:**

```json
{
  "consensus_level": 0.954,
  "pairwise_similarities": {
    "agent1_agent2": 0.987,
    "agent1_agent3": 0.954,
    "agent2_agent3": 0.921
  },
  "agreement_percentage": 1.0,
  "top_preference": "alt1",
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

$$\text{Diversity} = \frac{|\{a^*_1, a^*_2, \ldots, a^*_n\}|}{n}$$

Number of unique top preferences divided by total agents.

**Output Format:**

```json
{
  "balance_score": 0.923,
  "participation_distribution": {
    "agent1": 0.75,
    "agent2": 0.68,
    "agent3": 0.82,
    "agent4": 0.71
  },
  "diversity_score": 0.75,
  "gini_coefficient": 0.077,
  "unique_preferences": 3,
  "num_agents": 4
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
1 & \text{if } a^* = a_{\text{correct}} \\
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

### Code Example

```python
from evaluation.metrics import MetricsEvaluator

evaluator = MetricsEvaluator()

# Calculate multi-agent decision quality
multi_dqs = evaluator.calculate_decision_quality(
    decision=multi_agent_decision,
    criteria_weights={'safety': 0.4, 'cost': 0.3, 'speed': 0.3}
)

# Calculate single-agent decision quality
single_dqs = evaluator.calculate_decision_quality(
    decision=single_agent_decision,
    criteria_weights={'safety': 0.4, 'cost': 0.3, 'speed': 0.3}
)

# Compare
comparison = evaluator.compare_to_baseline(
    {'decision_quality': multi_dqs},
    {'decision_quality': single_dqs}
)

print(f"Improvement: {comparison['decision_quality']['improvement_percentage']:.1f}%")
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
- `test_evaluation_fix.py` - Test suite

---

## Changelog

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
