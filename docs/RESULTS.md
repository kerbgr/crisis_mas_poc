# Experimental Results

## Overview

The system was evaluated across **45 controlled runs**: 5 replicates × 3 LLM providers × 3 crisis scenarios. Every run used all 13 expert agents with `--compare-methods` mode (both ER and GAT aggregation per run), producing 135 result records in total. The local model was **GPT-OSS 20B** served via LM Studio; cloud providers were **Anthropic Claude Sonnet 4** and **OpenAI GPT-4o**.

---

## TABLE I — Performance by Scenario (all providers combined, n=15 per scenario)

| Metric | Karditsa Flood | Evia Wildfire | Elefsina HAZMAT | Overall |
|--------|---------------|---------------|-----------------|---------|
| DQS (mean ± σ) | 0.504 ± 0.020 | 0.418 ± 0.024 | 0.515 ± 0.036 | 0.479 ± 0.054 |
| Consensus (mean ± σ) | 0.941 ± 0.011 | 0.830 ± 0.057 | 0.922 ± 0.036 | 0.898 ± 0.057 |
| Confidence (mean ± σ) | 0.899 ± 0.009 | 0.835 ± 0.040 | 0.883 ± 0.020 | 0.872 ± 0.034 |
| Run consistency | 100 % | 93.3 % | 80–100 %* | 91.1 % |
| Avg processing time (s)† | 43.9 | 87.7 | 43.0 | 58.2 |

*HAZMAT per-provider: Claude 80 %, GPT-OSS 20B 100 %, GPT-4o 100 % — but GPT-4o chooses a different alternative than the other two providers (see Section 4).
†Averaged across providers; individual range: 8.7 s (Claude, Flood) to 154.6 s (GPT-OSS 20B, Forest Fire).

---

## TABLE II — Performance by LLM Provider (all scenarios, n=15 per provider)

| Provider | Model | DQS (mean ± σ) | Consensus (mean ± σ) | Confidence (mean ± σ) | Avg time (s) |
|----------|-------|----------------|---------------------|-----------------------|-------------|
| Anthropic | Claude Sonnet 4 | 0.468 ± 0.060 | 0.882 ± 0.059 | 0.866 ± 0.034 | 11.6 ± 3.7 |
| OpenAI API | GPT-4o | 0.464 ± 0.032 | **0.917 ± 0.036** | **0.886 ± 0.022** | 62.4 ± 29.3 |
| Local (LM Studio) | GPT-OSS 20B | **0.504 ± 0.051** | 0.895 ± 0.086 | 0.865 ± 0.050 | 100.7 ± 41.2 |

**Kruskal-Wallis statistical tests (df=2):**
- DQS: H = 7.39 → **significant (p < 0.05)**
- Processing time: H = 28.45 → **highly significant (p < 0.001)**
- Consensus: H = 2.69 → not significant
- Confidence: H = 1.51 → not significant

---

## TABLE III — Provider × Scenario Detail

| Scenario | Provider | n | DQS | Consensus | Confidence | Time (s) |
|----------|----------|---|-----|-----------|------------|----------|
| Flood | Claude Sonnet 4 | 5 | 0.509 ± 0.006 | 0.934 ± 0.005 | 0.898 ± 0.002 | 8.7 ± 0.3 |
| Flood | GPT-OSS 20B | 5 | 0.521 ± 0.007 | 0.935 ± 0.008 | 0.888 ± 0.006 | 74.2 ± 4.9 |
| Flood | GPT-4o | 5 | 0.482 ± 0.016 | 0.955 ± 0.013 | 0.910 ± 0.009 | 48.9 ± 14.1 |
| Forest Fire | Claude Sonnet 4 | 5 | 0.391 ± 0.011 | 0.807 ± 0.013 | 0.824 ± 0.008 | 16.3 ± 1.8 |
| Forest Fire | GPT-OSS 20B | 5 | 0.439 ± 0.018 | 0.788 ± 0.064 | 0.805 ± 0.039 | 154.6 ± 21.8 |
| Forest Fire | GPT-4o | 5 | 0.424 ± 0.007 | 0.894 ± 0.016 | 0.875 ± 0.011 | 92.2 ± 6.4 |
| HAZMAT | Claude Sonnet 4 | 5 | 0.505 ± 0.037 | 0.904 ± 0.035 | 0.875 ± 0.020 | 9.7 ± 0.8 |
| HAZMAT | GPT-OSS 20B | 5 | 0.553 ± 0.007 | 0.960 ± 0.007 | 0.903 ± 0.006 | 73.4 ± 1.1 |
| HAZMAT | GPT-4o | 5 | 0.487 ± 0.017 | 0.902 ± 0.036 | 0.872 ± 0.019 | 46.0 ± 33.2 |

---

## TABLE IV — ER vs. GAT Aggregation Comparison (n=45 each)

| Metric | ER | GAT | Difference |
|--------|----|-----|------------|
| DQS (mean ± σ) | 0.475 ± 0.049 | 0.482 ± 0.050 | +0.007 (n.s.) |
| Consensus (mean ± σ) | 0.900 ± 0.063 | 0.900 ± 0.063 | 0.000 (n.s.) |
| Confidence (mean ± σ) | 0.868 ± 0.034 | 0.873 ± 0.034 | +0.005 (n.s.) |
| Recommendation agreement | — | — | 82.2 % (37/45 runs) |

All differences are non-significant (p > 0.05, Kruskal-Wallis). ER and GAT are effectively interchangeable for this 13-agent panel. The 8 disagreements occur exclusively in ambiguous scenarios (6 × Forest Fire, 2 × HAZMAT).

---

## Section 4 — Decision Consistency

### Run-Level Recommendation (dominant alternative across all 45 runs)

| Scenario | Recommendation | Runs |
|----------|----------------|------|
| Karditsa Flood | **Hybrid approach** | 15/15 (100 %) |
| Evia Wildfire | **Combined assault** | 14/15 (93.3 %) |
| Elefsina HAZMAT | Integrated response | 9/15 (60 %) |
| Elefsina HAZMAT | Immediate downwind evacuation | 6/15 (40 %) |

### HAZMAT Provider Divergence

The HAZMAT split is not random — it is fully explained by provider identity:

| Provider | n | Recommendation | Consistency |
|----------|---|----------------|-------------|
| Claude Sonnet 4 | 5 | Integrated response (4×), Evacuation (1×) | 80 % |
| GPT-OSS 20B | 5 | Integrated response | 100 % |
| GPT-4o | 5 | Immediate downwind evacuation | 100 % |

GPT-4o reproducibly weights acute inhalation risk more heavily than logistical constraints, resulting in a clear split. Claude and GPT-OSS 20B converge on the integrated response that balances evacuation with shelter-and-monitor for lower-risk zones.

---

## Section 5 — Processing Time Analysis

| Provider | Flood (s) | Forest Fire (s) | HAZMAT (s) | Overall (s) |
|----------|-----------|----------------|------------|-------------|
| Claude Sonnet 4 | 8.7 ± 0.3 | 16.3 ± 1.8 | 9.7 ± 0.8 | 11.6 ± 3.7 |
| GPT-4o (OpenAI) | 48.9 ± 14.1 | 92.2 ± 6.4 | 46.0 ± 33.2 | 62.4 ± 29.3 |
| GPT-OSS 20B (local) | 74.2 ± 4.9 | 154.6 ± 21.8 | 73.4 ± 1.1 | 100.7 ± 41.2 |

**Forest Fire is consistently the slowest scenario** across all providers because its 12 candidate alternatives (vs 5 in the other scenarios) require more tokens per agent assessment. Claude's speed advantage (~9× faster than GPT-OSS 20B overall) is critical for time-sensitive operational use.

---

## Key Findings Summary

1. **ER and GAT are statistically equivalent** for 13-agent panels (DQS Δ = 0.007, p > 0.05; 82.2 % recommendation agreement).
2. **GPT-OSS 20B achieves the highest mean DQS** (0.504), demonstrating that competitive decision quality is achievable with local, on-premise inference at zero API cost.
3. **GPT-4o achieves the most consistent consensus** (0.917 ± 0.036) and confidence (0.886 ± 0.022) across all scenarios, suggesting the most predictable output.
4. **Claude Sonnet 4 is 5–9× faster** than the other providers (11.6 s mean), which is decisive for real-time emergency response.
5. **The Evia Wildfire scenario is the hardest** — lowest DQS (0.418), lowest consensus (0.830), most ER-GAT disagreements — reflecting genuine ambiguity in multi-front wildfire trade-offs.
6. **The HAZMAT provider divergence is the most scientifically interesting finding**: GPT-4o consistently selects a different response than Claude and GPT-OSS 20B. This is reproducible across all 5 replicates and indicates a real inter-model interpretive difference warranting domain expert review.
7. **Run-level recommendation stability is 91.1 %** across all 45 runs; the 4 deviating runs all occur at the boundary between two closely ranked alternatives.

---

## Result Files

Results are stored at `results/<scenario>/<run_N_provider>/`:
- `results.json` — combined ER+GAT decision, full agent opinions, all metrics
- `er/results.json` — ER-only aggregation result
- `gat/results.json` — GAT-only aggregation result
- `decision_comparison.png` — ER vs GAT visual comparison
- `*.png` — additional visualisations (agent contributions, belief heatmap, etc.)

See [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) for metric definitions and [REGENERATE_VISUALIZATIONS.md](REGENERATE_VISUALIZATIONS.md) for regenerating plots from existing JSON.
