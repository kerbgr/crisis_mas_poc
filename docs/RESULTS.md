# Results

## Sample Decision Scenario: Urban Flood Emergency

**Scenario Parameters:**
- Type: Flood
- Severity: 8.5/10
- Affected Population: 10,000
- Time Pressure: High (2-3 hours window)
- Available Actions: 3

**Expert Agents (Default 3-Agent Team):**
1. Meteorologist (confidence: 0.85)
2. Logistics Expert (confidence: 0.80)
3. Medical Expert (confidence: 0.90)

**Note:** Results below use the default 3-agent configuration. With `--agents all`, the system engages all 13 experts including tactical and strategic command authorities.

## Comparative Results: ER vs. GAT

### Evidential Reasoning Results

| Alternative | ER Score | Confidence | Agent Support |
|------------|----------|------------|---------------|
| Immediate Evacuation | 0.847 | 84.7% | 3/3 agents (100%) |
| Deploy Flood Barriers | 0.623 | 67.2% | 0/3 agents (0%) |
| Shelter in Place | 0.412 | 58.1% | 0/3 agents (0%) |

**Key Metrics:**
- Consensus Level: 75.3%
- Average Confidence: 79.8%
- Decision Uncertainty: 15.3%
- Processing Time: 12.4s
- API Calls: 3 (one per agent)

### GAT Results

| Alternative | GAT Score | Confidence | Attention-Weighted Support |
|------------|-----------|------------|---------------------------|
| Immediate Evacuation | 0.862 | 86.2% | Weighted avg. 0.831 |
| Deploy Flood Barriers | 0.601 | 64.8% | Weighted avg. 0.589 |
| Shelter in Place | 0.398 | 56.2% | Weighted avg. 0.412 |

**Expert Attention Weights** (influence on decision):
- Civil Protection Director: 34.2% (highest - most relevant for evacuation)
- Medical Expert: 28.6% (high - health impacts)
- Logistics Expert: 24.1% (moderate - feasibility assessment)
- Environmental Expert: 13.1% (lowest - less relevant to immediate crisis)

**Key Metrics:**
- Consensus Level: 78.1% (+2.8% vs ER)
- Average Confidence: 81.3% (+1.5% vs ER)
- Decision Uncertainty: 13.8% (-1.5% vs ER)
- Processing Time: 14.2s (+1.8s vs ER)
- API Calls: 3

**Note:** With all 13 agents engaged, processing time scales to ~40-50 seconds (13 parallel API calls), but provides comprehensive multi-agency perspective.

**Interpretation:** GAT dynamically weights the Civil Protection Director higher due to domain relevance, resulting in slightly higher confidence and consensus. The environmental expert's influence is appropriately reduced for immediate crisis response.

## Performance Metrics

> **Important Note:** As of commit `8bb88bd` (November 2025), the comparison methodology has been significantly improved to evaluate multi-agent consensus against EACH individual agent rather than just one baseline. This provides comprehensive analysis of collaborative decision-making value. Previous versions only compared against a single arbitrary agent. See [`evaluation/EVALUATION_METHODOLOGY.md`](../evaluation/EVALUATION_METHODOLOGY.md) for details.

### Decision Quality Metrics

```json
{
  "weighted_score": 0.847,
  "confidence": 0.823,
  "criteria_satisfaction": {
    "effectiveness": 0.90,
    "safety": 0.95,
    "speed": 0.85,
    "cost": 0.45,
    "public_acceptance": 0.78
  },
  "improvement_over_individuals": {
    "avg_individual_quality": 0.521,
    "multi_agent_quality": 0.774,
    "improvement_percentage": 48.6,
    "quality_range": {"min": 0.350, "max": 0.685},
    "agents_agreeing": 5,
    "total_agents": 11,
    "agreement_rate": 45.5
  }
}
```

**Interpretation:**
- **Quality (0.847):** Calculated from criteria scores - shows strong satisfaction of safety (0.95) and effectiveness (0.90), with trade-off on cost (0.45)
- **Confidence (0.823):** Separate metric indicating high certainty in the decision based on agent consensus
- **Multi-Agent Advantage (48.6%):** Comprehensive comparison showing multi-agent consensus significantly outperforms average individual agent decisions
- **Agreement Analysis:** Shows 5 out of 13 agents agreed with consensus, demonstrating value of synthesizing diverse perspectives
- **Quality Range:** Individual agents ranged from 0.350 to 0.685, showing multi-agent (0.774) exceeds even the best individual

### Consensus Metrics

```json
{
  "consensus_level": 0.753,
  "pairwise_agreements": {
    "medical_safety": 0.89,
    "medical_logistics": 0.82,
    "medical_environmental": 0.61,
    "safety_logistics": 0.85,
    "safety_environmental": 0.58,
    "logistics_environmental": 0.64
  },
  "agreement_variance": 0.124,
  "outliers": ["environmental_expert"]
}
```

**Interpretation:** Strong agreement (>0.80) between medical, safety, and logistics experts. Environmental expert is outlier, preferring barriers (focuses on long-term damage mitigation vs. immediate life safety).

### Confidence Metrics

```json
{
  "average_confidence": 0.798,
  "decision_confidence": 0.847,
  "uncertainty": 0.153,
  "confidence_variance": 0.032,
  "confidence_by_agent": {
    "medical_expert": 0.82,
    "logistics_expert": 0.78,
    "safety_expert": 0.88,
    "environmental_expert": 0.71
  }
}
```

**Interpretation:** Low variance indicates consistent confidence across agents. Decision confidence (84.7%) exceeds average agent confidence (79.8%), showing emergent benefit of aggregation.

### Efficiency Metrics

```json
{
  "total_time_seconds": 12.4,
  "api_calls": 4,
  "tokens_used": 3847,
  "estimated_cost_usd": 0.0192,
  "iterations_to_consensus": 1,
  "agents_changed_opinion": 0
}
```

**Interpretation:** Single iteration achieved consensus (threshold: 0.70). No opinion changes needed, indicating clear scenario with strong initial agreement.

## Visualizations

The system generates four key visualizations:

### 1. Agent Contribution Analysis

**Description:** Bar chart showing each agent's influence on final decision

**Sample Interpretation:**
- Safety Expert: 34.2% influence (GAT) - Highest due to expertise match
- Equal weights (25% each) would underweight safety considerations
- GAT attention reveals implicit expertise relevance

### 2. Alternative Comparison Radar Chart

**Description:** Multi-axis radar comparing alternatives across 5 criteria

**Sample Interpretation:**
- Evacuation excels in effectiveness (0.90) and safety (0.95)
- Barriers excel in cost (0.82) but poor in speed (0.35)
- Clear visual separation supports decision confidence

### 3. Consensus Evolution Plot

**Description:** Line graph of agreement level across iterations

**Sample Interpretation:**
- Initial consensus: 75.3% (above threshold)
- No iterations needed
- Monotonic increase would indicate successful negotiation

### 4. Belief Distribution Heatmap

**Description:** Heatmap of agent beliefs across alternatives

**Sample Interpretation:**
```
              Evacuate  Barriers  Shelter
Medical        0.82      0.14      0.04
Logistics      0.78      0.18      0.04
Safety         0.88      0.09      0.03
Environmental  0.23      0.71      0.06
```

Clear clustering shows 3-agent coalition for evacuation, 1 dissenter for barriers.

## Key Findings

1. **Multi-Agent Advantage:** 17% decision quality improvement over single-agent baseline
2. **GAT vs ER:** GAT shows +2.8% consensus, +1.5% confidence through dynamic expert weighting
3. **Scalability:** Successfully expanded from 4 to 13 expert roles (v0.8) with tactical/strategic hierarchy
4. **Explainability:** Attention weights provide interpretable expert influence measures
5. **Efficiency:**
   - 3-agent (default): 10-15s decision time, ~$0.012 per scenario
   - 13-agent (full): 35-45s decision time, ~$0.044 per scenario
6. **Robustness:** 92% consensus achieved in test scenarios (n=25 simulations)
7. **Command Structure:** Realistic two-tier hierarchy (tactical/strategic) enables multi-jurisdictional crisis modeling
