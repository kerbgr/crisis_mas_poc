# Evaluation Methodology

## Overview

This document describes the comprehensive evaluation framework for comparing multi-agent system (MAS) performance against individual agent decisions. The framework evaluates the **Greek Emergency Response Multi-Agent System** featuring **13 expert agents** responding to realistic Greek crisis scenarios.

### Greek Crisis Scenarios

**Important Note on Scenario Design:**

The scenarios used for evaluation are **synthetic, research-oriented scenarios inspired by real historical crisis events**, but they are **NOT retrospective evaluations of actual emergency response operations**. These scenarios:

- Draw inspiration from publicly documented Greek crises to ensure realistic modeling
- Are designed exclusively for research purposes to evaluate multi-agent system capabilities
- Do **NOT** represent testing against actual historical crisis decisions
- Make **NO** claims about the quality or correctness of actual decisions made during past crises
- Respect ethical boundaries regarding the use of sensitive crisis data without proper institutional approvals

The system is evaluated on three realistic Greek crisis scenarios:

1. **Karditsa Flood Emergency** (severity 0.8)
   - Location: Karditsa, Thessaly, Greece (39.3644°N, 21.9211°E)
   - 15,000 affected population
   - Pineios River overflow with infrastructure damage

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

## Performance Metrics

### 1. Decision Quality Score (DQS)

**Purpose:** Measures how well the recommended alternative satisfies the decision criteria.

**Formula:**

```
DQS = (1/|C|) × Σ s_c(a*)     if criteria scores available
    = MCDA(a*)                 if multi-agent with MCDA
    = f(a*)                    if only final scores available
```

Where:
- `a*` = recommended alternative
- `C` = set of decision criteria
- `s_c(a*)` = score of alternative a* on criterion c
- `MCDA(a*)` = MCDA score for alternative a*
- `f(a*)` = final combined score for alternative a*

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
```
DQS_SA = (1/|C|) × Σ s_c(a*)
```

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
```
DQS_MA = MCDA(a*)
```

Where MCDA is calculated using TOPSIS method (see below).

**Weighted Version:**

If criteria weights `w_c` are provided:

```
DQS_weighted = Σ(w_c × s_c(a*)) / Σ(w_c)
```

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

```
CL = (2 / n(n-1)) × Σᵢ Σⱼ cos(mᵢ, mⱼ)    for i < j
```

Where cosine similarity is:

```
cos(mᵢ, mⱼ) = (Σ mᵢ(a) × mⱼ(a)) / (||mᵢ|| × ||mⱼ||)
```

**Components:**
- `n` = number of agents
- `mᵢ` = belief vector of agent i
- `A` = set of alternatives
- `mᵢ(a)` = agent i's belief mass for alternative a

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
- **0.60 < CL <= 0.80:** Moderate consensus - some disagreement
- **0.40 < CL <= 0.60:** Low consensus - significant differences
- **CL <= 0.40:** No consensus - conflicting views

---

### 3. Confidence Score (CS)

**Purpose:** Measures certainty in the decision and individual agent confidence levels.

**Multi-Agent Confidence:**

```
CS_decision = 0.6 × CL + 0.4 × (1/n) × Σ cᵢ
```

Where:
- `CL` = consensus level (agreement between agents)
- `cᵢ` = confidence of agent i
- `n` = number of agents

**Rationale:**
- High consensus (CL) -> higher confidence in collective decision
- High individual confidence -> agents are certain of their assessments
- 60/40 weighting prioritizes consensus over individual certainty

**Single-Agent Confidence:**

```
CS_decision = c_LLM
```

Simply uses the LLM's self-reported confidence level.

**Uncertainty:**

```
U = 1 - CS_decision
```

**Confidence Variance:**

```
σ²_c = (1/n) × Σ(cᵢ - c̄)²
```

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

```
ECB = 1 - G
```

Where G is the Gini coefficient:

```
G = Σᵢ(2i - n - 1) × wᵢ / (n × Σwᵢ)
```

With weights w₁ ≤ w₂ ≤ ... ≤ wₙ (sorted).

**Contribution Score per Agent:**

```
contribᵢ = (cᵢ + H_norm(mᵢ)) / 2
```

Where:
- `cᵢ` = agent i's confidence
- `H_norm(mᵢ)` = normalized entropy of belief distribution

**Diversity Score:**

```
Diversity = |unique_preferences| / n
```

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

```
Eff = (1/3) × [1/(1+I) + 1/(1+A/3) + 1/(1+T/5)]
```

Where:
- `I` = iterations
- `A` = API calls
- `T` = time in seconds

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

### Multi-Agent vs Individual Agents (Comprehensive Comparison)

**Methodology:**
The system evaluates EVERY agent individually and compares the multi-agent consensus to the distribution of individual agent decisions, rather than comparing to a single arbitrary baseline agent.

**Decision Quality Comparison:**

```
ΔDQS_avg = DQS_MA - avg(DQS_individuals)

ΔDQS_% = (DQS_MA - avg(DQS_individuals)) / avg(DQS_individuals) × 100%
```

Where:
- `DQS_MA` = Multi-agent consensus quality score
- `avg(DQS_individuals)` = Average quality across all individual agents
- `N` = Number of participating agents

**Agreement Analysis:**

```
Agreement Rate = (count of agents agreeing with MA) / N × 100%
```

Where:
- `a*ᵢ` = Recommended alternative by agent i
- `a*_MA` = Multi-agent consensus recommendation

**Example Output:**

```json
{
  "multi_agent_quality": 0.774,
  "multi_agent_confidence": 0.674,
  "multi_agent_recommendation": "action_rescue_operations",
  "statistics": {
    "avg_quality": 0.521,
    "min_quality": 0.350,
    "max_quality": 0.685,
    "avg_confidence": 0.803,
    "agreement_rate_percent": 45.5,
    "num_agents_agree": 5,
    "total_agents": 11
  },
  "individual_agents": [
    {
      "agent_name": "Taxiarchos Vasilis",
      "recommended_alternative": "action_rescue_operations",
      "confidence": 0.850,
      "decision_quality": 0.685,
      "agrees_with_consensus": true
    },
    {
      "agent_name": "Commander Maria",
      "recommended_alternative": "A1",
      "confidence": 0.850,
      "decision_quality": 0.620,
      "agrees_with_consensus": false
    }
  ]
}
```

**Key Insights:**

1. **Quality Distribution:**
   - Shows how multi-agent consensus compares to ALL individual experts
   - Identifies quality range: best individual vs worst individual vs consensus
   - Example: Multi-agent (0.774) exceeds even best individual (0.685)

2. **Agreement Analysis:**
   - Shows how many experts agree with the consensus
   - Example: 5/11 (45.5%) agreement demonstrates consensus synthesizes diverse viewpoints
   - Lower agreement doesn't mean poor decision - may indicate novel synthesis

3. **Individual Rankings:**
   - Identifies which experts contribute most/least effectively
   - Enables reliability tracking and dynamic weighting
   - Shows expertise relevance to specific crisis types

**Interpretation:**
- **Large positive improvement (>30%):** Strong multi-agent advantage, consensus significantly better than average
- **Moderate positive improvement (10-30%):** Clear multi-agent benefit, validates collaborative approach
- **Small positive improvement (0-10%):** Marginal benefit, may depend on scenario complexity
- **Negative improvement:** Individual experts outperform consensus (investigate causes)

---

## Ground Truth Validation

When ground truth is available:

**Ground Truth Match:**

```
Match = 1   if a* = a_correct
      = 0   otherwise
```

**Quality Boost:**

If ground truth matches:
```
DQS_final = max(DQS, 0.9)
```

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

**Null Hypothesis:** H₀: μ_MA = μ_SA

**Alternative:** H₁: μ_MA ≠ μ_SA

**Test Statistic:**

```
t = (x̄_MA - x̄_SA) / sqrt(s²_MA/n_MA + s²_SA/n_SA)
```

**Effect Size (Cohen's d):**

```
d = (x̄_MA - x̄_SA) / s_pooled
```

Where:

```
s_pooled = sqrt[((n_MA-1)×s²_MA + (n_SA-1)×s²_SA) / (n_MA + n_SA - 2)]
```

**Interpretation:**
- `|d| < 0.2`: Negligible effect
- `0.2 ≤ |d| < 0.5`: Small effect
- `0.5 ≤ |d| < 0.8`: Medium effect
- `|d| ≥ 0.8`: Large effect

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
Decision -> calculate_decision_quality() -> DQS
                                          |
                                      Compare -> Improvement %
                                          |
Baseline -> calculate_decision_quality() -> DQS
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
