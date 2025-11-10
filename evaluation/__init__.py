"""
Evaluation Module - Performance Metrics and Visualization for Multi-Agent Systems

OBJECTIVE:
This module provides comprehensive evaluation capabilities for assessing the performance
of the crisis management multi-agent system. It implements standardized metrics for
measuring decision quality, consensus, efficiency, and system behavior, along with
professional visualization tools for results presentation.

WHY THIS EXISTS:
Evaluating multi-agent systems is challenging due to:
1. **Multiple Performance Dimensions**: Quality, consensus, efficiency, balance
2. **Comparison Requirements**: Need to compare against single-agent baseline
3. **Statistical Rigor**: Must quantify significance of improvements
4. **Result Communication**: Stakeholders need clear, visual representations
5. **Reproducibility**: Standardized metrics enable consistent evaluation

This module ensures that:
- System performance is objectively measured across multiple dimensions
- Improvements over baseline are quantified with statistical significance
- Results are presented in publication-quality visualizations
- Evaluations are reproducible and auditable

KEY COMPONENTS:

1. **MetricsEvaluator** (metrics.py)
   - Decision Quality Score (DQS): Weighted criteria satisfaction
   - Consensus Level (CL): Agent agreement measurement
   - Time to Consensus (TtC): Efficiency metrics
   - Confidence Score (CS): Uncertainty quantification
   - Expert Contribution Balance (ECB): Participation fairness
   - Statistical significance testing (t-tests, Cohen's d)
   - Baseline comparison (multi-agent vs. single-agent)

2. **SystemVisualizer** (visualizations.py)
   - Belief distribution plots (stacked bar charts)
   - Consensus evolution tracking (line charts)
   - Criteria importance visualization (radar charts)
   - Performance comparison charts (bar charts)
   - Agent network graphs (network diagrams)
   - Publication-quality output (300 DPI, professional styling)

EVALUATION METHODOLOGY:
The evaluation follows a multi-dimensional approach:

1. **Decision Quality (DQS)**:
   - Measures how well decisions satisfy weighted criteria
   - Range: 0.0-1.0 (higher is better)
   - Formula: Weighted average of criteria satisfaction
   - See: EVALUATION_METHODOLOGY.md for detailed formulas

2. **Consensus Level (CL)**:
   - Measures agreement between agents using cosine similarity
   - Range: 0.0-1.0 (1.0 = perfect agreement)
   - Threshold: 0.75 (75% similarity required)
   - Pairwise comparisons across all agent pairs

3. **Confidence Score (CS)**:
   - Separate from quality, measures certainty in decision
   - Components: Agent confidence + consensus level
   - Range: 0.0-1.0 (higher = more certain)
   - Includes uncertainty quantification

4. **Time to Consensus (TtC)**:
   - Efficiency metrics: iterations, API calls, processing time
   - Lower values indicate more efficient systems
   - Trade-off with quality (faster may be lower quality)

5. **Expert Contribution Balance (ECB)**:
   - Measures fairness of agent participation
   - Uses Gini coefficient (0 = perfect equality)
   - Range: 0.0-1.0 (1.0 = perfectly balanced)
   - Includes diversity score (unique perspectives)

TYPICAL USAGE FLOW:
```python
from evaluation import MetricsEvaluator, SystemVisualizer

# 1. Initialize evaluator
evaluator = MetricsEvaluator()

# 2. Calculate metrics for multi-agent decision
metrics = {
    'decision_quality': evaluator.calculate_decision_quality(
        decision, ground_truth
    ),
    'consensus': evaluator.calculate_consensus_metrics(
        agent_assessments
    ),
    'confidence': evaluator.calculate_confidence_metrics(
        decision
    ),
    'efficiency': evaluator.calculate_efficiency_metrics(
        execution_log
    ),
    'expert_contribution_balance': evaluator.calculate_expert_contribution_balance(
        agent_assessments
    )
}

# 3. Compare to baseline
comparison = evaluator.compare_to_baseline(
    multi_agent_results, single_agent_results
)

# 4. Statistical significance
significance = evaluator.calculate_statistical_significance(
    multi_agent_scores, single_agent_scores
)

# 5. Generate report
report = evaluator.generate_report(all_metrics)
print(report)

# 6. Visualize results
visualizer = SystemVisualizer(output_dir="figures")
plots = visualizer.generate_all_plots(results)
```

INPUTS (Typical):
- decision: Dict from CoordinatorAgent.make_final_decision()
- agent_assessments: Dict from CoordinatorAgent.collect_assessments()
- execution_log: List[Dict] of execution events
- ground_truth: Optional Dict with correct alternative
- baseline_results: Dict with single-agent performance

OUTPUTS (Typical):
- decision_quality: DQS score and criteria satisfaction
- consensus: Consensus level and pairwise similarities
- confidence: Confidence scores and uncertainty
- efficiency: Iterations, API calls, processing time
- expert_contribution_balance: Balance and diversity scores
- baseline_comparison: Multi vs. single-agent comparison
- statistical_significance: P-values, effect sizes
- visualizations: Publication-quality charts (PNG, 300 DPI)

ERROR HANDLING:
All components gracefully handle edge cases:
- Missing data → Partial metrics with warnings
- Insufficient samples → Cannot compute statistical tests
- Empty assessments → Returns default values
- Visualization failures → Logged errors, continues

MATHEMATICAL FOUNDATIONS:
- **DQS**: Weighted average of normalized criteria scores
- **Consensus**: Average pairwise cosine similarity
- **Confidence**: Entropy-based certainty measure
- **ECB**: Gini coefficient for distribution equality
- **Statistical Tests**: Independent t-test, Cohen's d effect size

PERFORMANCE CONSIDERATIONS:
- MetricsEvaluator: O(N²) for consensus (pairwise comparisons)
- SystemVisualizer: O(N×M) for plotting
- Memory: Minimal (all operations on pre-aggregated data)
- I/O: Visualization saves to disk (configurable location)

VALIDATION & TESTING:
The evaluation methodology has been:
- Verified against LaTeX formulas (see FORMULA_VERIFICATION.md)
- Fixed in v2.0.1 to ensure valid comparisons
- Tested with multiple scenarios
- Validated against expected ranges

RELATED MODULES:
- agents/: Agents that produce assessments to be evaluated
- decision_framework/: Decision-making methods being evaluated
- scenarios/: Test scenarios with ground truth
- main.py: Orchestrates evaluation workflow

VISUALIZATION BEST PRACTICES:
1. **Resolution**: 300 DPI for publication quality
2. **Styling**: Professional seaborn themes
3. **Labeling**: Clear titles, axes, legends
4. **Colors**: Colorblind-friendly palettes
5. **Format**: PNG for compatibility

COMPARISON WITH EVALUATION v1.0:
The original evaluation had critical bugs (fixed in v2.0.0-2.0.1):
- Bug #1: DQS was copying confidence field
- Bug #2: Multi-agent stored ER+MCDA score as "confidence"
- Bug #3: Compared different metrics (apples to oranges)
- Bug #4: Fallback logic didn't work properly

Current version ensures:
- DQS is calculated from criteria satisfaction
- Confidence and quality are separate metrics
- Valid comparisons (same metric for both systems)
- Robust fallback logic with proper cascade

VERSION HISTORY:
- v1.0: Initial metrics implementation
- v2.0.0: Fixed critical evaluation bugs (Jan 2025)
- v2.0.1: Enhanced fallback logic for edge cases
- v2.1: Comprehensive documentation (Jan 2025)

REFERENCES:
- EVALUATION_METHODOLOGY.md: Detailed formulas and examples
- FORMULA_VERIFICATION.md: Code-to-formula verification
- Multi-agent system evaluation best practices
- Statistical testing in AI systems
"""

from .metrics import MetricsEvaluator
from .visualizations import SystemVisualizer

__all__ = ['MetricsEvaluator', 'SystemVisualizer']
