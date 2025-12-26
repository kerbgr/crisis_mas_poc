"""
Decision Framework Module - Multi-Agent Decision Aggregation and Analysis

OBJECTIVE:
This module provides the core decision-making infrastructure for the crisis management
multi-agent system. It implements multiple methods for aggregating expert opinions,
analyzing alternatives, and building consensus under uncertainty.

WHY THIS EXISTS:
Crisis management requires combining diverse expert opinions into coherent, actionable
decisions. This module addresses the challenge of:
1. **Heterogeneous Expertise**: Different experts have different knowledge and priorities
2. **Uncertainty Management**: Beliefs contain inherent uncertainty that must be quantified
3. **Consensus Building**: Multiple agents must reach agreement on action selection
4. **Multi-Criteria Evaluation**: Actions must be evaluated across competing criteria
5. **Dynamic Weighting**: Agent importance should adapt to scenario context

KEY COMPONENTS:

1. **Evidential Reasoning (ER)**
   - Simplified Dempster-Shafer theory for belief aggregation
   - Weighted averaging of belief distributions
   - Uncertainty quantification and confidence scoring
   - See: evidential_reasoning.py

2. **Multi-Criteria Decision Analysis (MCDA)**
   - TOPSIS-inspired weighted sum method
   - Handles benefit and cost criteria
   - Criterion normalization for scale independence
   - Sensitivity analysis and weight profile comparison
   - See: mcda_engine.py

3. **Consensus Model**
   - Cosine similarity-based consensus detection
   - Conflict identification and severity classification
   - Resolution suggestions and compromise alternatives
   - Consensus history tracking
   - See: consensus_model.py

4. **Graph Attention Network (GAT) Aggregator**
   - Neural attention mechanism for dynamic agent weighting
   - Context-aware importance assignment
   - Multi-head attention for robustness
   - Network-based trust modeling
   - See: gat_aggregator.py

DESIGN PHILOSOPHY:
- **Modularity**: Each component can be used independently or combined
- **Transparency**: All methods provide explanations and intermediate results
- **Flexibility**: Configurable parameters and multiple aggregation strategies
- **Robustness**: Graceful error handling and fallback mechanisms
- **Interpretability**: Human-readable summaries and visualizations

TYPICAL USAGE FLOW:
1. Expert agents generate assessments (belief distributions, confidence scores)
2. CoordinatorAgent selects aggregation method (ER, GAT, or hybrid)
3. Aggregation combines beliefs → single recommendation with confidence
4. Consensus Model checks for conflicts and suggests resolutions
5. MCDA Engine ranks alternatives using multi-criteria scores
6. Final decision includes: recommended action, confidence, quality score, consensus level

AGGREGATION METHODS COMPARISON:

| Method | Strengths | Best For | Computational Cost |
|--------|-----------|----------|-------------------|
| ER | Simple, fast, transparent | Homogeneous agents, time-critical | Low |
| GAT | Context-aware, adaptive | Heterogeneous expertise, complex scenarios | High |
| Hybrid | Balanced, robust | General purpose, production systems | Medium |

INPUTS (Typical):
- Agent assessments: Dict[agent_id, Dict[str, Any]] containing:
  * belief_distribution: {alternative_id: probability}
  * confidence: float (0-1)
  * reasoning: str
  * expertise: str
  * reliability_score: float (0-1)
- Scenario: Crisis scenario dictionary
- Criteria weights: Multi-criteria evaluation weights
- Agent weights: Reliability/trust weights

OUTPUTS (Typical):
- Aggregated beliefs: Combined probability distribution
- Recommended action: Highest-scoring alternative
- Confidence score: Overall certainty (0-1)
- Consensus level: Agreement measure (0-1)
- Quality score: Criteria satisfaction (0-1)
- Explanations: Human-readable reasoning

ERROR HANDLING:
All components gracefully handle edge cases:
- Empty agent sets → default/fallback decisions
- Missing data → sensible defaults with warnings
- Invalid inputs → ValueError with detailed messages
- Computation failures → logged errors, safe degradation

MATHEMATICAL FOUNDATIONS:
- **Evidential Reasoning**: Simplified Dempster-Shafer theory with weighted averaging
- **MCDA**: TOPSIS methodology with vector normalization
- **Consensus**: Cosine similarity for distribution comparison
- **GAT**: Graph attention mechanism from Veličković et al. (2018)

RELATED MODULES:
- agents/: Expert agents that generate assessments
- evaluation/: Metrics for decision quality assessment
- scenarios/: Crisis scenarios and criteria definitions
- llm/: LLM integration for enhanced reasoning

PERFORMANCE CONSIDERATIONS:
- ER: O(N×M) where N=agents, M=alternatives
- MCDA: O(A×C) where A=alternatives, C=criteria
- Consensus: O(N²) for pairwise comparisons
- GAT: O(N²×H×F) where H=heads, F=feature_dim

VERSION HISTORY:
- v1.0: Initial implementation with ER and MCDA
- v1.1: Added Consensus Model
- v1.2: Added GAT aggregator
- v2.0: Enhanced with reliability tracking integration
- v2.1: Fixed evaluation methodology (Jan 2025)
"""

from .evidential_reasoning import EvidentialReasoning
from .mcda_engine import MCDAEngine
from .consensus_model import ConsensusModel
from .gat_aggregator import GATAggregator, GraphAttentionLayer

__all__ = [
    'EvidentialReasoning',
    'MCDAEngine',
    'ConsensusModel',
    'GATAggregator',
    'GraphAttentionLayer'
]
