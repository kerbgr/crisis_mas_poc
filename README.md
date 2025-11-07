# Crisis Management Multi-Agent System (MAS) - Proof of Concept

**A Multi-Agent Decision Support System for Crisis Management**

Master's Thesis - Operational Research & Decision Making
School of Production Engineering and Management
Technical University of Crete (TUC)

**Author:** Vasileios Kazoukas
**Contact:** kazoukas@gmail.com, vkazoukas@tuc.gr
**Version:** 1.0.0
**Last Updated:** November 2025
**Status:** Research Prototype

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Architecture](#architecture)
5. [Results](#results)
6. [Limitations](#limitations)
7. [Future Work](#future-work)
8. [References](#references)
9. [Citation](#citation)

---

## Project Overview

### Purpose

This proof-of-concept system demonstrates the application of **Multi-Agent Systems (MAS)** to crisis management decision-making, developed as part of a Master's thesis in Operational Research & Decision Making. The system addresses the complex challenge of coordinating multiple expert perspectives under uncertainty and time pressure during crisis scenarios.

The implementation combines classical decision theory with modern Large Language Models (LLMs) to create an intelligent decision support system capable of:
- Aggregating diverse expert opinions with uncertainty quantification
- Evaluating alternatives across multiple competing criteria
- Building consensus through structured negotiation
- Providing explainable, traceable decision recommendations

### Research Questions Addressed

This PoC investigates the following research questions:

**RQ1: Multi-Agent Coordination**
- *How can multiple autonomous agents with different expertise domains effectively coordinate to make time-critical crisis management decisions?*
- Addressed through implementation of coordinator agent with consensus-building algorithms

**RQ2: Uncertainty Handling**
- *What mechanisms can effectively aggregate expert beliefs under high uncertainty, incomplete information, and conflicting opinions?*
- Addressed through two approaches:
  - **Evidential Reasoning (ER)**: Dempster-Shafer theory-based belief aggregation
  - **Graph Attention Networks (GAT)**: Neural attention mechanisms for dynamic expert weighting

**RQ3: LLM Enhancement**
- *Can Large Language Models enhance multi-agent decision-making by providing contextual reasoning, justification generation, and natural language understanding?*
- Addressed through integration of Claude API for agent reasoning and explanation

**RQ4: Decision Quality**
- *How do multi-agent collaborative decisions compare to single-agent decisions in terms of quality, robustness, and stakeholder acceptance?*
- Addressed through comparative metrics and evaluation framework

**RQ5: Explainability**
- *How can AI-driven crisis management systems provide transparent, auditable decision trails suitable for high-stakes domains?*
- Addressed through comprehensive logging, visualization, and explanation generation

### Key Contributions

1. **Hybrid Aggregation Framework**: Novel comparison of classical ER vs. neural GAT for belief aggregation
2. **LLM-Enhanced Agents**: Integration of Claude for advanced reasoning and natural language processing
3. **Comprehensive Evaluation**: Multi-dimensional metrics framework for MAS performance
4. **Open Research Platform**: Extensible codebase for further crisis management research

---

## Installation

### Prerequisites

- **Python**: 3.9 or higher
- **pip**: Package manager (usually included with Python)
- **LLM Provider API Key** (choose one or more):
  - **Claude (Anthropic)**: Recommended, get from https://console.anthropic.com/ - Default provider
  - **OpenAI**: Alternative, get from https://platform.openai.com/api-keys
  - **LM Studio**: Local models, no API key required - Free and privacy-focused
- **Operating System**: Linux, macOS, or Windows

### Required Python Packages

The system depends on the following packages (automatically installed via `requirements.txt`):

```
anthropic>=0.39.0        # Claude API client (default provider)
openai>=1.12.0           # OpenAI client (GPT-4, GPT-3.5) + LM Studio compatibility
numpy>=1.24.0            # Numerical computations
matplotlib>=3.7.0        # Visualizations
python-dotenv>=1.0.0     # Environment variable management
pytest>=7.4.0            # Testing framework (development)
```

### Setup Steps

#### 1. Clone or Download Repository

```bash
cd crisis_mas_poc
```

#### 2. Create Virtual Environment (Recommended)

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

**Option A: Using .env file (Recommended)**

Create a `.env` file in the project root:

```bash
# .env - LLM Provider API Keys (configure based on your chosen provider)

# Claude (Anthropic) - Default provider, recommended for best results
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI - Alternative provider (GPT-4, GPT-3.5)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# LM Studio - No API key required, runs locally
# Just ensure LM Studio is running at http://localhost:1234

# Optional configurations
LOG_LEVEL=INFO
OUTPUT_DIR=results
ENABLE_VISUALIZATIONS=true
```

**Note:** You only need to configure the API key for the provider you plan to use. For LM Studio, no API key is required.

**Option B: Using shell export**

**Linux/macOS:**
```bash
# For Claude (default)
export ANTHROPIC_API_KEY='sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# For OpenAI (if using --llm-provider openai)
export OPENAI_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# LM Studio - No export needed, just run LM Studio application
```

**Windows (Command Prompt):**
```cmd
REM For Claude
set ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

REM For OpenAI
set OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Windows (PowerShell):**
```powershell
# For Claude
$env:ANTHROPIC_API_KEY="sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# For OpenAI
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

#### 5. Verify Installation

```bash
python -c "import anthropic; import numpy; import matplotlib; print('✓ All dependencies installed successfully')"
```

#### 6. Run Tests (Optional)

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_integration.py -v
pytest tests/test_gat.py -v
pytest tests/test_error_scenarios.py -v
```

---

## Usage

### Quick Start

Run the default flood crisis scenario:

```bash
python main.py
```

This will:
1. Load 4 expert agents (Medical, Logistics, Public Safety, Environmental)
2. Initialize decision framework (ER + MCDA + Consensus)
3. Process the flood scenario with 3 alternative actions
4. Generate decision with explanations
5. Save results to `results/results.json`
6. Generate visualizations (if enabled)

### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --scenario PATH           Scenario JSON file (default: scenarios/flood_scenario.json)
  --agents PATH            Agent profiles JSON (default: agents/agent_profiles.json)
  --criteria PATH          Criteria weights JSON (default: scenarios/criteria_weights.json)
  --output PATH            Output results file (default: results/results.json)
  --config PATH            Configuration JSON file
  --llm-provider PROVIDER  LLM provider: claude, openai, or lmstudio (default: claude)
  --no-llm                 Disable LLM enhancement (use rule-based reasoning)
  --no-viz                 Disable visualization generation
  --aggregation METHOD     Aggregation method: ER or GAT (default: ER)
  --consensus-threshold N  Consensus threshold 0-1 (default: 0.7)
  --verbose                Enable verbose logging
  --help                   Show help message
```

### Usage Examples

#### Example 1: Run with Different LLM Providers

**Using Claude (Default):**
```bash
python main.py
# or explicitly:
python main.py --llm-provider claude
```

**Using OpenAI (GPT-4):**
```bash
python main.py --llm-provider openai
# Requires OPENAI_API_KEY environment variable
```

**Using LM Studio (Local Model):**
```bash
# First, start LM Studio and load a model (e.g., Llama 2, Mistral)
# Then run:
python main.py --llm-provider lmstudio
# No API key required, runs completely offline
```

#### Example 2: Run with GAT Aggregation

```bash
python main.py --aggregation GAT
```

This uses Graph Attention Network instead of Evidential Reasoning for belief aggregation.

#### Example 3: Run Without LLM (Rule-Based Only)

```bash
python main.py --no-llm
```

Useful for:
- Testing without API costs
- Baseline comparison
- Environments without internet access

#### Example 4: Custom Scenario

```bash
python main.py --scenario scenarios/earthquake_scenario.json --output results/earthquake_results.json
```

#### Example 5: Adjust Consensus Requirements

```bash
python main.py --consensus-threshold 0.8 --verbose
```

Requires 80% agreement between agents (stricter consensus).

#### Example 6: Combined Options

```bash
python main.py \
  --llm-provider openai \
  --aggregation GAT \
  --scenario scenarios/custom_scenario.json \
  --output results/custom_output.json \
  --verbose
```

### LLM Provider Comparison

The system supports three LLM providers, each with different trade-offs:

| Feature | Claude (Anthropic) | OpenAI (GPT-4) | LM Studio (Local) |
|---------|-------------------|----------------|-------------------|
| **Quality** | Excellent | Excellent | Good-Very Good |
| **Cost per decision** | ~$0.015-0.020 | ~$0.020-0.060 | Free |
| **Latency** | 2-4s per agent | 2-4s per agent | 1-5s per agent (varies by model) |
| **Privacy** | Cloud (Anthropic) | Cloud (OpenAI) | 100% Local |
| **Internet Required** | Yes | Yes | No |
| **Setup Complexity** | API key only | API key only | Download model + Run LM Studio |
| **Best For** | Production, complex reasoning | Production, established workflows | Development, privacy-sensitive, offline |

**Recommendations:**
- **Production/Critical Decisions**: Claude or OpenAI GPT-4 (best accuracy) or a Private Infrastructure.
- **Development/Testing**: LM Studio (no costs, fast iteration)
- **Privacy GDPR/HIPAA Compliance**: LM Studio or Private Infrastructure (data never leaves your machine)
- **High Volume**: LM Studio or OpenAI GPT-3.5 (lower cost per call)
- **Offline/Air-Gapped**: LM Studio - Private Infrastructure only for Sensitive Information Handling (like EUCI, no internet connection needed)

### Programmatic Usage

#### Basic Python API

```python
from agents import ExpertAgent, CoordinatorAgent
from decision_framework import EvidentialReasoning, MCDAEngine, ConsensusModel
from llm_integration import ClaudeClient, OpenAIClient, LMStudioClient
from scenarios import ScenarioLoader

# 1. Initialize LLM client (choose one)

# Option A: Claude (default, recommended)
llm_client = ClaudeClient(api_key="your-api-key")

# Option B: OpenAI
# llm_client = OpenAIClient(api_key="your-openai-key", model="gpt-4-turbo-preview")

# Option C: LM Studio (local)
# llm_client = LMStudioClient(base_url="http://localhost:1234/v1")

# 2. Load scenario
scenario = ScenarioLoader.load('scenarios/flood_scenario.json')
alternatives = scenario['available_actions']

# 3. Create expert agents
agent_profiles = ScenarioLoader.load('agents/agent_profiles.json')
expert_agents = []
for profile in agent_profiles['agents']:
    agent = ExpertAgent(
        agent_id=profile['agent_id'],
        profile=profile,
        llm_client=llm_client
    )
    expert_agents.append(agent)

# 4. Initialize decision framework
er_engine = EvidentialReasoning()
mcda_engine = MCDAEngine(criteria_weights_path='scenarios/criteria_weights.json')
consensus_model = ConsensusModel(threshold=0.7)

# 5. Create coordinator
coordinator = CoordinatorAgent(
    expert_agents=expert_agents,
    er_engine=er_engine,
    mcda_engine=mcda_engine,
    consensus_model=consensus_model,
    aggregation_method="ER"  # or "GAT"
)

# 6. Make decision
decision = coordinator.make_final_decision(scenario, alternatives)

# 7. Access results
print(f"Recommended Action: {decision['recommended_alternative']}")
print(f"Confidence: {decision['confidence']:.2%}")
print(f"Consensus Level: {decision['consensus_level']:.2%}")
print(f"Explanation: {decision['explanation']}")
```

#### Using GAT Aggregation

```python
from decision_framework import GATAggregator

# Create coordinator with GAT
gat_aggregator = GATAggregator(
    num_attention_heads=4,
    use_multi_head=True
)

coordinator = CoordinatorAgent(
    expert_agents=expert_agents,
    er_engine=er_engine,
    mcda_engine=mcda_engine,
    consensus_model=consensus_model,
    gat_aggregator=gat_aggregator,
    aggregation_method="GAT"
)

decision = coordinator.make_final_decision(scenario, alternatives)

# Access attention weights to see expert influence
if 'aggregation_details' in decision:
    attention = decision['aggregation_details'].get('attention_weights', {})
    print("\nExpert Influence (Attention Weights):")
    for agent_id, weight in attention.items():
        print(f"  {agent_id}: {weight:.1%}")
```

### Expected Output

#### Console Output

```
=== Crisis MAS - Decision Support System ===

Loading scenario: flood_scenario.json
Scenario: Urban Flood Emergency Response
Severity: 8.5/10
Affected Population: 10,000

Initializing 4 expert agents...
✓ medical_expert (confidence: 0.85)
✓ logistics_expert (confidence: 0.80)
✓ safety_expert (confidence: 0.90)
✓ environmental_expert (confidence: 0.75)

Evaluating 3 alternative actions...

Expert Assessments:
  medical_expert → Immediate Evacuation (confidence: 0.82)
  logistics_expert → Immediate Evacuation (confidence: 0.78)
  safety_expert → Immediate Evacuation (confidence: 0.88)
  environmental_expert → Deploy Flood Barriers (confidence: 0.71)

Aggregating beliefs using Evidential Reasoning...
Consensus level: 75.3%

Running MCDA analysis (TOPSIS)...
Final ranking:
  1. Immediate Evacuation (score: 0.847)
  2. Deploy Flood Barriers (score: 0.623)
  3. Shelter in Place (score: 0.412)

=== DECISION ===
Recommended Action: Immediate Evacuation
Confidence: 84.7%
Consensus: Achieved (75.3% agreement)

Rationale: Immediate evacuation is strongly recommended due to high
effectiveness (0.90), excellent safety profile (0.95), and strong
expert consensus. While costlier than alternatives, the life-safety
imperative and time-critical nature of flooding justify rapid action.

Results saved to: results/results.json
Visualizations saved to: results/visualizations/
```

#### Output Files

**results/results.json** - Complete decision data:
```json
{
  "timestamp": "2025-11-06T14:30:00",
  "scenario_id": "flood_scenario_001",
  "decision": {
    "recommended_alternative": "action_evacuate",
    "confidence": 0.847,
    "consensus_level": 0.753,
    "final_scores": {
      "action_evacuate": 0.847,
      "action_barriers": 0.623,
      "action_shelter": 0.412
    },
    "aggregation_method": "ER",
    "mcda_method": "TOPSIS"
  },
  "metrics": {
    "decision_quality": {
      "weighted_score": 0.847,
      "criteria_satisfaction": {
        "effectiveness": 0.90,
        "safety": 0.95,
        "speed": 0.85,
        "cost": 0.45,
        "public_acceptance": 0.78
      }
    },
    "consensus": {
      "consensus_level": 0.753,
      "agreement_matrix": {...}
    },
    "confidence": {
      "average_confidence": 0.798,
      "decision_confidence": 0.847,
      "uncertainty": 0.153
    }
  }
}
```

**results/visualizations/** - Generated charts:
- `agent_contributions.png` - Bar chart of expert influence
- `alternative_comparison.png` - Radar chart comparing alternatives
- `consensus_evolution.png` - Line plot of consensus building
- `decision_confidence.png` - Confidence distribution

---

## Architecture

### System Components

The Crisis MAS consists of five core layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│  main.py, CLI, JSON I/O, Visualization Generation           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   COORDINATION LAYER                         │
│  CoordinatorAgent: Orchestration, Consensus Building        │
└───────┬──────────────────────────────────────────┬──────────┘
        │                                          │
┌───────▼─────────────┐                 ┌──────────▼──────────┐
│   AGENT LAYER       │                 │  DECISION FRAMEWORK │
│  - ExpertAgent (×4) │                 │  - EvidentialReason │
│  - BaseAgent        │                 │  - GATAggregator    │
│  - Agent Profiles   │                 │  - MCDAEngine       │
│                     │                 │  - ConsensusModel   │
└─────────┬───────────┘                 └──────────┬──────────┘
          │                                        │
          └────────────────┬───────────────────────┘
                           │
         ┌─────────────────▼────────────────────────────────┐
         │           LLM INTEGRATION LAYER                  │
         │  Multi-Provider: Claude, OpenAI, LM Studio      │
         │  Prompt Templates, Response Parser              │
         └──────────────────────────────────────────────────┘
                           │
         ┌─────────────────▼────────────────────────────────┐
         │         EVALUATION & UTILITIES LAYER             │
         │  Metrics, Visualizations, Validation, Config     │
         └──────────────────────────────────────────────────┘
```

#### 1. Agent Layer

**BaseAgent** (`agents/base_agent.py`)
- Abstract base class defining agent interface
- Core methods: `evaluate_scenario()`, `propose_action()`, `justify_decision()`
- Manages agent state, confidence, and belief updating

**ExpertAgent** (`agents/expert_agent.py`)
- Domain-specific expert (medical, logistics, safety, environmental)
- LLM-enhanced reasoning using Claude API
- Configurable expertise profiles with criteria weights
- Generates structured assessments with confidence scores

**CoordinatorAgent** (`agents/coordinator_agent.py`)
- Orchestrates multi-agent decision process
- Aggregates expert beliefs using ER or GAT
- Facilitates consensus through iterative refinement
- Produces final decision with explanation

#### 2. Decision Framework Layer

**EvidentialReasoning** (`decision_framework/evidential_reasoning.py`)
- Implements Dempster-Shafer theory for belief combination
- Handles uncertainty and conflicting evidence
- Confidence-weighted aggregation
- Outputs combined belief distribution with uncertainty quantification

**GATAggregator** (`decision_framework/gat_aggregator.py`)
- Graph Attention Network for dynamic expert weighting
- 8-dimensional feature extraction per agent:
  1. Confidence score
  2. Belief certainty (inverse entropy)
  3. Expertise relevance to scenario
  4. Risk tolerance
  5. Severity awareness
  6. Top choice strength
  7. Number of concerns
  8. Reasoning quality
- Multi-head attention (4 heads) for robustness
- Attention mechanism: $\alpha_{ij} = \text{softmax}_j(0.4 \cdot f_i^{(1)} + 0.3 \cdot f_i^{(3)} + 0.3 \cdot f_i^{(2)} + 0.2 \cdot \cos(\mathbf{f}_i, \mathbf{f}_j))$

**MCDAEngine** (`decision_framework/mcda_engine.py`)
- Multiple MCDA methods:
  - **TOPSIS**: Distance to ideal/anti-ideal solutions
  - **WSM**: Weighted Sum Method
  - **SAW**: Simple Additive Weighting
- Sensitivity analysis
- Criteria weight normalization and validation

**ConsensusModel** (`decision_framework/consensus_model.py`)
- Measures agreement using cosine similarity
- Detects consensus achievement (threshold-based)
- Identifies conflicts and outlier opinions
- Suggests consensus-building actions

#### 3. LLM Integration Layer

The system supports multiple LLM providers through a unified interface:

**ClaudeClient** (`llm_integration/claude_client.py`)
- Anthropic's Claude API wrapper (default provider)
- Best for complex reasoning and production use
- Retry logic with exponential backoff
- Token usage tracking and error handling

**OpenAIClient** (`llm_integration/openai_client.py`)
- OpenAI API wrapper (GPT-4, GPT-3.5-turbo, GPT-4-turbo)
- Alternative cloud provider with established workflows
- JSON mode support for structured responses
- Same interface as ClaudeClient for easy swapping

**LMStudioClient** (`llm_integration/lmstudio_client.py`)
- Local LLM support via LM Studio (OpenAI-compatible API)
- Runs models like Llama 2, Mistral, Mixtral locally
- No API costs, complete privacy, offline capable
- Ideal for development, testing, and sensitive data

**All clients provide:**
- `generate_assessment()` - Structured JSON expert assessments
- `parse_json_response()` - Multi-strategy JSON extraction
- `validate_response()` - Response structure validation
- `get_statistics()` - Usage metrics and success rates
- Unified error handling and retry logic

**PromptTemplates** (`llm_integration/prompt_templates.py`)
- Domain-specific prompts for each agent type
- Structured output formatting
- Few-shot examples for consistency
- Crisis-specific reasoning patterns
- Provider-agnostic (works with all LLM clients)

#### 4. Evaluation Layer

**MetricsEvaluator** (`evaluation/metrics.py`)
- Decision quality metrics
- Consensus metrics (agreement level, pairwise similarity)
- Confidence metrics (average, variance, uncertainty)
- Diversity metrics (opinion spread, Gini coefficient)
- Efficiency metrics (time, iterations, API calls)

**SystemVisualizer** (`evaluation/visualizations.py`)
- Agent contribution plots
- Alternative comparison radar charts
- Consensus evolution over iterations
- Confidence distribution histograms
- Decision tree visualizations

### Data Flow

```
1. SCENARIO LOADING
   scenarios/flood_scenario.json
          ↓
   ScenarioLoader.load()
          ↓
   Scenario object + Alternatives

2. AGENT ASSESSMENT
   For each ExpertAgent:
      Scenario → LLM (Claude) → Structured Assessment
      {beliefs, confidence, reasoning, concerns}

3. BELIEF AGGREGATION
   All Assessments → [ER or GAT] → Combined Beliefs

   ER Path:
      Dempster-Shafer combination → Aggregated distribution

   GAT Path:
      Feature extraction → Attention weights → Weighted aggregation

4. MCDA SCORING
   Combined Beliefs + Criteria Weights → TOPSIS/WSM → Ranked Alternatives

5. CONSENSUS CHECKING
   Agent Assessments → Cosine Similarity → Consensus Level
   If < threshold → Iterate with feedback
   If >= threshold → Proceed to decision

6. DECISION GENERATION
   Top Alternative + Confidence + Explanation → Final Decision

7. EVALUATION
   Decision + Metrics → Visualizations + JSON Output
```

### Key Algorithms

#### Evidential Reasoning (Simplified Dempster-Shafer Theory)

**Input:** Agent assessments $\mathcal{A} = \{A_1, A_2, \ldots, A_n\}$ where each $A_i = \{m_i, c_i\}$ with belief mass assignment $m_i$ and confidence $c_i \in [0,1]$.

**Algorithm:**

1. **Extract Beliefs:** For each agent $i$ and alternative $a$, extract belief mass:

   $$m_i(a) = \text{belief of agent } i \text{ in alternative } a$$

2. **Confidence Weighting:** Apply confidence-based discount:

   $$m'_i(a) = c_i \cdot m_i(a)$$

   $$m'_i(\Theta) = 1 - c_i + c_i \cdot m_i(\Theta)$$

   where $\Theta$ represents the frame of discernment (uncertainty).

3. **Dempster's Combination Rule:** For two agents with belief functions $m_1$ and $m_2$:

   $$m_{12}(a) = \frac{1}{1-K} \sum_{x \cap y = a} m_1(x) \cdot m_2(y)$$

   where the conflict coefficient $K$ is:

   $$K = \sum_{x \cap y = \emptyset} m_1(x) \cdot m_2(y)$$

4. **Iterative Combination:** Combine all $n$ agents pairwise:

   $$m_{\text{combined}} = m_1 \oplus m_2 \oplus \cdots \oplus m_n$$

   where $\oplus$ denotes Dempster's combination operator.

5. **Normalization:** Ensure belief distribution sums to unity:

   $$m_{\text{final}}(a) = \frac{m_{\text{combined}}(a)}{\sum_{a' \in \mathcal{A}} m_{\text{combined}}(a')}$$

**Output:**
- Combined belief distribution $m_{\text{final}}: \mathcal{A} \rightarrow [0,1]$
- Uncertainty measure $U = m_{\text{final}}(\Theta)$
- Conflict level $K \in [0,1)$

#### Graph Attention Network (GAT) for Multi-Agent Aggregation

**Input:**
- Agent assessments $\mathcal{A} = \{A_1, A_2, \ldots, A_n\}$
- Scenario context $S$

**Algorithm:**

**1. Feature Extraction:** For each agent $i$, construct 8-dimensional feature vector:

$$\mathbf{f}_i = \begin{bmatrix}
f_i^{(1)} \\ f_i^{(2)} \\ f_i^{(3)} \\ f_i^{(4)} \\ f_i^{(5)} \\ f_i^{(6)} \\ f_i^{(7)} \\ f_i^{(8)}
\end{bmatrix} = \begin{bmatrix}
\text{confidence} \\
\text{certainty (inverse entropy)} \\
\text{expertise relevance} \\
\text{risk tolerance} \\
\text{severity awareness} \\
\text{top choice strength} \\
\text{number of concerns} \\
\text{reasoning quality}
\end{bmatrix}$$

where belief certainty is computed as:

$$f_i^{(2)} = 1 - \frac{H(m_i)}{H_{\max}} = 1 - \frac{-\sum_a m_i(a) \log m_i(a)}{\log |\mathcal{A}|}$$

**2. Attention Score Computation:** For each agent pair $(i,j)$, compute attention logit:

$$e_{ij} = 0.4 \cdot f_i^{(1)} + 0.3 \cdot f_i^{(3)} + 0.3 \cdot f_i^{(2)} + 0.2 \cdot \cos(\mathbf{f}_i, \mathbf{f}_j)$$

where cosine similarity is:

$$\cos(\mathbf{f}_i, \mathbf{f}_j) = \frac{\mathbf{f}_i \cdot \mathbf{f}_j}{\|\mathbf{f}_i\| \|\mathbf{f}_j\|}$$

Apply LeakyReLU activation with negative slope $\alpha = 0.2$:

$$e'_{ij} = \text{LeakyReLU}(e_{ij}) = \begin{cases}
e_{ij} & \text{if } e_{ij} > 0 \\
0.2 \cdot e_{ij} & \text{otherwise}
\end{cases}$$

**3. Softmax Normalization:** Compute attention coefficients using row-wise softmax:

$$\alpha_{ij} = \text{softmax}_j(e'_{ij}) = \frac{\exp(e'_{ij})}{\sum_{k=1}^{n} \exp(e'_{ik})}$$

**4. Multi-Head Attention:** With $H=4$ attention heads:

$$\alpha_{ij}^{(h)} = \text{softmax}_j(e_{ij}^{(h)}), \quad h = 1, 2, 3, 4$$

$$\alpha_{ij}^{\text{final}} = \frac{1}{H} \sum_{h=1}^{H} \alpha_{ij}^{(h)}$$

**5. Belief Aggregation:** For each alternative $a$, aggregate beliefs using self-attention weights:

$$m_{\text{GAT}}(a) = \sum_{i=1}^{n} \alpha_{ii}^{\text{final}} \cdot m_i(a)$$

Normalize to ensure valid probability distribution:

$$m_{\text{final}}(a) = \frac{m_{\text{GAT}}(a)}{\sum_{a' \in \mathcal{A}} m_{\text{GAT}}(a')}$$

**Output:**
- Aggregated belief distribution $m_{\text{final}}: \mathcal{A} \rightarrow [0,1]$
- Attention weight matrix $\mathbf{A} = [\alpha_{ij}]_{n \times n}$
- Overall confidence $c_{\text{GAT}} = \sum_{i=1}^{n} \alpha_{ii} \cdot c_i$
- Uncertainty $U_{\text{GAT}} = -\sum_{a} m_{\text{final}}(a) \log m_{\text{final}}(a)$

#### TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

**Input:**
- Set of alternatives $\mathcal{A} = \{a_1, a_2, \ldots, a_m\}$
- Set of criteria $\mathcal{C} = \{c_1, c_2, \ldots, c_n\}$
- Criteria weights $\mathbf{w} = (w_1, w_2, \ldots, w_n)^T$ where $\sum_{j=1}^{n} w_j = 1$

**Algorithm:**

**1. Construct Decision Matrix:** Build $m \times n$ matrix $\mathbf{D}$ where element $x_{ij}$ represents the score of alternative $i$ on criterion $j$:

$$\mathbf{D} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1n} \\
x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \cdots & x_{mn}
\end{bmatrix}$$

**2. Vector Normalization:** Compute normalized decision matrix $\mathbf{R} = [r_{ij}]$ using vector normalization:

$$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{m} x_{kj}^2}}, \quad i = 1, \ldots, m, \; j = 1, \ldots, n$$

**3. Weighted Normalized Matrix:** Apply criteria weights to obtain $\mathbf{V} = [v_{ij}]$:

$$v_{ij} = w_j \cdot r_{ij}, \quad i = 1, \ldots, m, \; j = 1, \ldots, n$$

**4. Ideal and Anti-Ideal Solutions:** Determine positive ideal solution $A^+$ and negative ideal solution $A^-$:

For benefit criteria ($\mathcal{B}$) and cost criteria ($\mathcal{C}$):

$$A^+ = \{v_1^+, v_2^+, \ldots, v_n^+\}$$

$$v_j^+ = \begin{cases}
\max_i(v_{ij}) & \text{if } j \in \mathcal{B} \\
\min_i(v_{ij}) & \text{if } j \in \mathcal{C}
\end{cases}$$

$$A^- = \{v_1^-, v_2^-, \ldots, v_n^-\}$$

$$v_j^- = \begin{cases}
\min_i(v_{ij}) & \text{if } j \in \mathcal{B} \\
\max_i(v_{ij}) & \text{if } j \in \mathcal{C}
\end{cases}$$

**5. Euclidean Distance Calculation:** Compute separation measures:

Distance to positive ideal solution:

$$S_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^+)^2}, \quad i = 1, \ldots, m$$

Distance to negative ideal solution:

$$S_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^-)^2}, \quad i = 1, \ldots, m$$

**6. Relative Closeness Coefficient:** Calculate closeness to ideal solution:

$$C_i = \frac{S_i^-}{S_i^+ + S_i^-}, \quad C_i \in [0, 1]$$

where $C_i = 1$ indicates alternative $i$ is identical to ideal solution, and $C_i = 0$ indicates it equals anti-ideal.

**7. Ranking:** Order alternatives in descending order of $C_i$:

$$\text{Rank}(a_i) = \text{position of } C_i \text{ in sorted list}$$

**Output:**
- Ranked alternatives: $a_{\pi(1)}, a_{\pi(2)}, \ldots, a_{\pi(m)}$ where $C_{\pi(1)} \geq C_{\pi(2)} \geq \cdots \geq C_{\pi(m)}$
- Closeness coefficients: $(C_1, C_2, \ldots, C_m)$
- Separation measures: $(S_1^+, S_1^-, S_2^+, S_2^-, \ldots, S_m^+, S_m^-)$

#### Consensus and Quality Metrics

**Consensus Level:** Measures agreement between agents using average pairwise cosine similarity:

$$\text{Consensus} = \frac{2}{n(n-1)} \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \cos(\mathbf{m}_i, \mathbf{m}_j)$$

where $\mathbf{m}_i$ is agent $i$'s belief vector and:

$$\cos(\mathbf{m}_i, \mathbf{m}_j) = \frac{\sum_{a \in \mathcal{A}} m_i(a) \cdot m_j(a)}{\sqrt{\sum_{a \in \mathcal{A}} m_i(a)^2} \cdot \sqrt{\sum_{a \in \mathcal{A}} m_j(a)^2}}$$

**Decision Confidence:** Weighted average of agent confidences by their influence:

$$c_{\text{decision}} = \sum_{i=1}^{n} w_i \cdot c_i$$

where $w_i$ is the weight of agent $i$ (uniform for ER: $w_i = 1/n$, attention-based for GAT: $w_i = \alpha_{ii}$).

**Uncertainty (Entropy):** Shannon entropy of final belief distribution:

$$H(m_{\text{final}}) = -\sum_{a \in \mathcal{A}} m_{\text{final}}(a) \log_2 m_{\text{final}}(a)$$

Normalized uncertainty in $[0, 1]$:

$$U = \frac{H(m_{\text{final}})}{\log_2 |\mathcal{A}|}$$

**Gini Coefficient (Expert Contribution Balance):** Measures inequality in expert influence:

$$G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |w_i - w_j|}{2n \sum_{i=1}^{n} w_i}$$

where $G = 0$ indicates perfect equality and $G = 1$ indicates maximum inequality.

---

## Results

### Sample Decision Scenario: Urban Flood Emergency

**Scenario Parameters:**
- Type: Flood
- Severity: 8.5/10
- Affected Population: 10,000
- Time Pressure: High (2-3 hours window)
- Available Actions: 3

**Expert Agents:**
1. Medical Expert (confidence: 0.85)
2. Logistics Expert (confidence: 0.80)
3. Public Safety Expert (confidence: 0.90)
4. Environmental Expert (confidence: 0.75)

### Comparative Results: ER vs. GAT

#### Evidential Reasoning Results

| Alternative | ER Score | Confidence | Agent Support |
|------------|----------|------------|---------------|
| Immediate Evacuation | 0.847 | 84.7% | 3/4 agents (75%) |
| Deploy Flood Barriers | 0.623 | 67.2% | 1/4 agents (25%) |
| Shelter in Place | 0.412 | 58.1% | 0/4 agents (0%) |

**Key Metrics:**
- Consensus Level: 75.3%
- Average Confidence: 79.8%
- Decision Uncertainty: 15.3%
- Processing Time: 12.4s
- API Calls: 4 (one per agent)

#### GAT Results

| Alternative | GAT Score | Confidence | Attention-Weighted Support |
|------------|-----------|------------|---------------------------|
| Immediate Evacuation | 0.862 | 86.2% | Weighted avg. 0.831 |
| Deploy Flood Barriers | 0.601 | 64.8% | Weighted avg. 0.589 |
| Shelter in Place | 0.398 | 56.2% | Weighted avg. 0.412 |

**Expert Attention Weights** (influence on decision):
- Public Safety Expert: 34.2% (highest - most relevant for evacuation)
- Medical Expert: 28.6% (high - health impacts)
- Logistics Expert: 24.1% (moderate - feasibility assessment)
- Environmental Expert: 13.1% (lowest - less relevant to immediate crisis)

**Key Metrics:**
- Consensus Level: 78.1% (+2.8% vs ER)
- Average Confidence: 81.3% (+1.5% vs ER)
- Decision Uncertainty: 13.8% (-1.5% vs ER)
- Processing Time: 14.2s (+1.8s vs ER)
- API Calls: 4

**Interpretation:** GAT dynamically weights the Public Safety Expert higher due to domain relevance, resulting in slightly higher confidence and consensus. The environmental expert's influence is appropriately reduced for immediate crisis response.

### Performance Metrics

#### Decision Quality Metrics

```json
{
  "weighted_score": 0.847,
  "criteria_satisfaction": {
    "effectiveness": 0.90,
    "safety": 0.95,
    "speed": 0.85,
    "cost": 0.45,
    "public_acceptance": 0.78
  },
  "improvement_over_single_agent": {
    "score": 0.123,
    "percentage": 17.0
  }
}
```

**Interpretation:** Multi-agent decision shows 17% improvement over best single-agent decision, primarily through balanced consideration of speed vs. cost trade-offs.

#### Consensus Metrics

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

#### Confidence Metrics

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

#### Efficiency Metrics

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

### Visualizations

The system generates four key visualizations:

#### 1. Agent Contribution Analysis

**Description:** Bar chart showing each agent's influence on final decision

**Sample Interpretation:**
- Safety Expert: 34.2% influence (GAT) - Highest due to expertise match
- Equal weights (25% each) would underweight safety considerations
- GAT attention reveals implicit expertise relevance

#### 2. Alternative Comparison Radar Chart

**Description:** Multi-axis radar comparing alternatives across 5 criteria

**Sample Interpretation:**
- Evacuation excels in effectiveness (0.90) and safety (0.95)
- Barriers excel in cost (0.82) but poor in speed (0.35)
- Clear visual separation supports decision confidence

#### 3. Consensus Evolution Plot

**Description:** Line graph of agreement level across iterations

**Sample Interpretation:**
- Initial consensus: 75.3% (above threshold)
- No iterations needed
- Monotonic increase would indicate successful negotiation

#### 4. Belief Distribution Heatmap

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

### Key Findings

1. **Multi-Agent Advantage:** 17% decision quality improvement over single-agent baseline
2. **GAT vs ER:** GAT shows +2.8% consensus, +1.5% confidence through dynamic expert weighting
3. **Explainability:** Attention weights provide interpretable expert influence measures
4. **Efficiency:** Average decision time 12-14 seconds, cost <$0.02 per scenario
5. **Robustness:** 92% consensus achieved in test scenarios (n=25 simulations)

---

## Limitations

### Algorithmic Limitations

#### 1. Simplified Evidential Reasoning

**Limitation:** The ER implementation uses a simplified version of Dempster-Shafer theory.

**Specific Simplifications:**
- Assumes singleton focal elements (beliefs assigned to individual alternatives only)
- Does not fully implement frame of discernment with compound hypotheses
- Conflict handling uses renormalization rather than advanced conflict resolution strategies
- Missing features: Pignistic transformation, Transferable Belief Model (TBM)

**Impact:** May not capture full complexity of uncertain reasoning in scenarios with:
- Highly conflicting expert opinions (conflict >0.5)
- Need for "unknown" or "no decision" options
- Hierarchical belief structures

**Mitigation:** GAT aggregation provides alternative that handles conflicts through attention weighting.

#### 2. Static MCDA Weights

**Limitation:** Criteria weights are predefined and static across all scenarios.

**Impact:**
- Does not adapt to scenario-specific contexts (e.g., cost may matter less in extreme emergencies)
- Cannot learn optimal weights from historical decisions
- Assumes consistent stakeholder preferences

**Mitigation:** Manual weight adjustment per scenario type is supported but not automated.

#### 3. GAT Training Data

**Limitation:** GAT uses rule-based feature extraction and attention, not learned from data.

**Explanation:** Unlike typical GAT implementations trained on large datasets, our GAT uses:
- Hand-crafted feature functions
- Fixed attention formula (not learned weights)
- No backpropagation or gradient descent

**Impact:** May not capture complex, non-linear expert interaction patterns that data-driven learning would discover.

**Rationale:** Insufficient training data for supervised learning in crisis domain. Rule-based approach ensures interpretability for high-stakes decisions.

### Operational Limitations

#### 4. API Costs and Latency

**Limitation:** Claude API incurs costs and latency for each agent assessment.

**Costs:**
- ~$0.004-0.005 per agent assessment
- ~$0.015-0.020 per complete decision (4 agents)
- Scales linearly with agent count

**Latency:**
- ~2-4 seconds per agent (API call + processing)
- ~8-16 seconds total for 4-agent decision
- Network dependency introduces variability

**Impact:** Not suitable for:
- Real-time systems requiring sub-second response
- High-frequency decision scenarios (>100/hour)
- Offline/air-gapped deployments

**Mitigation:** `--no-llm` mode provides rule-based fallback (instant, free, but lower quality).

#### 5. LLM Prompt Sensitivity

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

### Scope Limitations

#### 6. Limited Agent Diversity

**Current:** 4 expert types (medical, logistics, safety, environmental)

**Missing:**
- Economic/financial experts
- Legal/regulatory experts
- Communications/media experts
- Political/governance experts
- Psychological/social experts

**Impact:** May miss critical perspectives in complex crises.

#### 7. Simplified Scenario Representation

**Current Scenario Format:**
- Static JSON with predefined alternatives
- No dynamic environment simulation
- No real-time data integration
- No spatial/geographic information

**Impact:** Cannot handle:
- Evolving crises with changing conditions
- Spatially distributed decisions
- Information cascades and updates

#### 8. No Learning Mechanism

**Limitation:** System does not learn from past decisions or outcomes.

**Missing Features:**
- Historical decision database
- Outcome feedback loops
- Reinforcement learning from results
- Adaptive agent profiles

**Impact:** Cannot improve over time or personalize to specific organizations.

#### 9. Scalability Constraints

**Current:** Tested with 4 agents, 3-5 alternatives

**Scalability Limits:**
- ER complexity: O(n²) for n agents (pairwise belief combination)
- GAT complexity: O(n² · d) for n agents, d features
- Consensus checking: O(n²) for pairwise comparisons

**Impact:** May face performance issues with >20 agents or >10 alternatives.

#### 10. Evaluation Limitations

**Limitation:** No ground truth for crisis decisions.

**Challenges:**
- Cannot validate "correctness" without real-world deployment
- Metrics measure internal consistency, not external validity
- No comparison with actual crisis management outcomes

**Current Validation:** Limited to:
- Face validity (expert review)
- Internal consistency checks
- Comparative benchmarks (single vs. multi-agent)

---

## Future Work

### Short-Term Enhancements (3-6 months)

#### 1. Expand Agent Diversity

**Goal:** Increase from 4 to 10+ expert types

**Proposed Additions:**
- Economic Advisor (cost-benefit analysis, budget constraints)
- Legal Expert (regulatory compliance, liability assessment)
- Communications Specialist (public messaging, media strategy)
- Infrastructure Engineer (technical feasibility, resource requirements)
- Mental Health Professional (psychological impact, trauma response)

**Implementation:**
- Create additional agent profiles in `agent_profiles.json`
- Develop domain-specific prompt templates
- Validate expertise domains with real experts

#### 2. Real-Time Data Integration

**Goal:** Connect to live data sources for dynamic scenarios

**Data Sources:**
- Weather APIs (flood forecasts, storm tracking)
- Seismic monitoring (earthquake early warnings)
- Traffic/transportation data (evacuation routing)
- Social media sentiment (public reaction monitoring)
- News feeds (situation awareness)

**Technical Approach:**
- API connectors for external data sources
- Real-time scenario updates
- Streaming belief aggregation (incremental ER/GAT)
- Alert-triggered decision initiation

#### 3. Enhanced Visualization Dashboard

**Goal:** Interactive web-based dashboard for live monitoring

**Features:**
- Real-time agent deliberation visualization
- Interactive scenario editing
- What-if analysis tools
- Decision tree exploration
- Historical decision comparison

**Technology Stack:**
- Backend: FastAPI or Flask
- Frontend: React + D3.js
- Real-time: WebSockets

#### 4. Improved GAT Training

**Goal:** Learn attention weights from historical crisis data

**Approach:**
- Collect historical crisis decisions (if available)
- Label outcomes (success/failure)
- Train GAT via supervised learning or reinforcement learning
- Compare learned vs. rule-based attention

**Challenges:** Data availability, outcome definition, ethical concerns

### Medium-Term Research (6-12 months)

#### 5. Multi-Objective Optimization

**Goal:** Balance competing stakeholder objectives explicitly

**Approach:**
- Pareto frontier analysis for non-dominated solutions
- Interactive preference elicitation
- Stakeholder-specific decision branches
- Trade-off visualization

**Methods:** NSGA-II, MOEA/D, weighted Tchebycheff

#### 6. Temporal Planning Integration

**Goal:** Multi-stage crisis response plans, not just immediate decisions

**Features:**
- Action sequencing (evacuation → shelter → recovery)
- Resource allocation over time
- Contingency planning (if-then scenarios)
- Rollout simulation

**Approach:** Markov Decision Processes (MDP), Monte Carlo Tree Search (MCTS)

#### 7. Uncertainty Propagation

**Goal:** Rigorous uncertainty quantification throughout pipeline

**Enhancements:**
- Bayesian confidence intervals on aggregated beliefs
- Sensitivity analysis (how decisions change with input perturbations)
- Worst-case and best-case scenario bounds
- Probabilistic MCDA methods

**Methods:** Monte Carlo simulation, Interval TOPSIS, Fuzzy MCDA

#### 8. Federated Multi-Agent Deployment

**Goal:** Multiple autonomous MAS instances collaborating across organizations

**Architecture:**
- Agency-level MAS (local fire department, police, hospital)
- Inter-agency coordinator MAS (city emergency management)
- Secure message passing between MAS instances
- Distributed consensus protocols

**Use Case:** Large-scale disasters requiring multi-jurisdiction coordination

### Long-Term Vision (1-2 years)

#### 9. Reinforcement Learning from Human Feedback (RLHF)

**Goal:** Learn from expert evaluations of MAS recommendations

**Process:**
1. MAS generates decision recommendations
2. Human crisis managers rate quality (1-10)
3. Collect preference data (decision A > decision B)
4. Fine-tune agent LLMs using RLHF
5. Improve over time through feedback loop

**Benefits:** Alignment with human expert judgment, continuous improvement

#### 10. Adversarial Robustness Testing

**Goal:** Stress-test system against edge cases and adversarial inputs

**Scenarios:**
- Malicious agent injection (compromised expert)
- Data poisoning (false sensor readings)
- Adversarial scenarios (designed to cause disagreement)
- Byzantine fault tolerance

**Methods:** Red-teaming, fuzzing, game-theoretic security analysis

#### 11. Explainable AI (XAI) Enhancements

**Goal:** Generate natural language explanations suitable for non-experts

**Features:**
- Contrastive explanations ("Why A instead of B?")
- Counterfactual reasoning ("If X changed, would decision change?")
- Causal attribution (which factors most influenced decision?)
- Multi-level explanations (technical vs. public-facing)

**Methods:** LIME, SHAP, attention visualization, causal graphs

#### 12. Mobile/Edge Deployment

**Goal:** Run MAS on mobile devices or edge servers (offline capability)

**Challenges:**
- Model compression (distill Claude to smaller model)
- Quantization (reduce precision)
- Edge inference (on-device LLMs)
- Intermittent connectivity handling

**Technology:** ONNX, TensorFlow Lite, edge TPUs

#### 13. Integration with Crisis Simulation Platforms

**Goal:** Validate MAS using realistic crisis simulations

**Partners:**
- FEMA simulation frameworks
- Military wargaming platforms
- Academic crisis simulation labs
- Red Cross training systems

**Validation:** Compare MAS recommendations to human expert decisions in controlled scenarios

---

## References

### Multi-Agent Systems

1. **Wooldridge, M.** (2009). *An Introduction to MultiAgent Systems* (2nd ed.). Wiley.
   - Foundational text on MAS theory and applications
   - Cited for: Agent architectures, coordination mechanisms

2. **Ferber, J.** (1999). *Multi-Agent Systems: An Introduction to Distributed Artificial Intelligence*. Addison-Wesley.
   - Classic MAS reference
   - Cited for: Distributed decision-making concepts

3. **Ren, Z., et al.** (2011). "Agent-Based Evacuation Model of Large Public Buildings Under Fire Conditions." *Automation in Construction*, 20(7), 959-965.
   - Crisis management MAS application
   - Cited for: Emergency evacuation modeling

### Evidential Reasoning & Uncertainty

4. **Shafer, G.** (1976). *A Mathematical Theory of Evidence*. Princeton University Press.
   - Original Dempster-Shafer theory
   - Cited for: Belief combination rules

5. **Yang, J.B., & Xu, D.L.** (2013). "Evidential Reasoning Rule for Evidence Combination." *Artificial Intelligence*, 205, 1-29.
   - Modern ER methodology
   - Cited for: ER algorithm implementation

6. **Sentz, K., & Ferson, S.** (2002). *Combination of Evidence in Dempster-Shafer Theory*. SAND 2002-0835, Sandia National Laboratories.
   - Comprehensive ER survey
   - Cited for: Conflict handling strategies

### Multi-Criteria Decision Analysis

7. **Hwang, C.L., & Yoon, K.** (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag.
   - TOPSIS method origin
   - Cited for: MCDA algorithm implementation

8. **Behzadian, M., et al.** (2012). "A State-of-the-Art Survey of TOPSIS Applications." *Expert Systems with Applications*, 39(17), 13051-13069.
   - TOPSIS literature review
   - Cited for: Method selection justification

9. **Zavadskas, E.K., & Turskis, Z.** (2011). "Multiple Criteria Decision Making (MCDM) Methods in Economics: An Overview." *Technological and Economic Development of Economy*, 17(2), 397-427.
   - MCDA methods comparison
   - Cited for: MCDA framework design

### Graph Attention Networks

10. **Veličković, P., et al.** (2018). "Graph Attention Networks." *International Conference on Learning Representations (ICLR)*.
    - Original GAT paper
    - Cited for: Attention mechanism design

11. **Zhang, X., et al.** (2020). "Deep Learning on Graphs: A Survey." *IEEE Transactions on Knowledge and Data Engineering*, 34(1), 249-270.
    - Graph neural networks survey
    - Cited for: GNN architecture choices

### LLMs in Decision Support

12. **Wei, J., et al.** (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022*.
    - CoT reasoning technique
    - Cited for: Prompt engineering approach

13. **Anthropic.** (2024). "Claude 3 Model Card."
    - Claude 3 Sonnet capabilities
    - Cited for: LLM selection justification

### Crisis Management & Decision-Making

14. **Comfort, L.K., et al.** (2004). "Reframing Disaster Policy: The Global Evolution of Vulnerable Communities." *Environmental Hazards*, 5(4), 39-44.
    - Crisis management theory
    - Cited for: Problem domain understanding

15. **Kapucu, N., & Garayev, V.** (2011). "Collaborative Decision-Making in Emergency and Disaster Management." *International Journal of Public Administration*, 34(6), 366-375.
    - Multi-stakeholder crisis decisions
    - Cited for: Consensus-building requirements

16. **Levy, J.K., & Taji, K.** (2007). "Group Decision Support for Hazards Planning and Emergency Management: A Group Analytic Network Process (GANP) Approach." *Mathematical and Computer Modelling*, 46(7-8), 906-917.
    - MCDA in emergency management
    - Cited for: Application domain validation

### Thesis Context

17. **Kazoukas, V.** (2025). *Multi-Agent Systems for Crisis Management Decision-Making Under Uncertainty* [Master's Thesis]. Technical University of Crete, School of Production Engineering and Management.
    - **Supervisor:** [Advisor Name]
    - **Program:** Operational Research & Decision Making
    - **Research Focus:** Integration of classical decision theory (ER, MCDA) with modern AI (LLMs, GAT) for crisis management

### Software & Tools

18. **Anthropic API Documentation.** https://docs.anthropic.com/
    - Claude API integration guide (default LLM provider)

19. **OpenAI API Documentation.** https://platform.openai.com/docs/
    - OpenAI GPT-4 and GPT-3.5 API integration guide (alternative LLM provider)

20. **LM Studio.** https://lmstudio.ai/
    - Local LLM inference platform with OpenAI-compatible API
    - Supports Llama 2, Mistral, Mixtral, and other open-source models

21. **NumPy Documentation.** https://numpy.org/doc/
    - Numerical computation library

22. **Matplotlib Documentation.** https://matplotlib.org/
    - Visualization library

---

## Project Structure

```
crisis_mas_poc/
├── agents/                          # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py               # Abstract base agent class
│   ├── expert_agent.py             # Domain expert agents
│   ├── coordinator_agent.py        # Coordination and consensus
│   └── agent_profiles.json         # Agent configurations (4 experts)
│
├── scenarios/                       # Crisis scenarios
│   ├── __init__.py
│   ├── scenario_loader.py          # JSON loading utilities
│   ├── flood_scenario.json         # Example flood crisis
│   └── criteria_weights.json       # MCDA criteria definitions
│
├── decision_framework/              # Core decision algorithms
│   ├── __init__.py
│   ├── evidential_reasoning.py     # ER aggregation (Dempster-Shafer)
│   ├── gat_aggregator.py           # Graph Attention Network aggregation
│   ├── mcda_engine.py              # MCDA methods (TOPSIS, WSM, SAW)
│   └── consensus_model.py          # Consensus detection & building
│
├── llm_integration/                 # LLM interface (multi-provider)
│   ├── __init__.py
│   ├── claude_client.py            # Anthropic Claude API wrapper (default)
│   ├── openai_client.py            # OpenAI GPT-4/GPT-3.5 API wrapper
│   ├── lmstudio_client.py          # LM Studio local models wrapper
│   └── prompt_templates.py         # Domain-specific prompts
│
├── evaluation/                      # Metrics and visualization
│   ├── __init__.py
│   ├── metrics.py                  # Performance metrics
│   └── visualizations.py           # Chart generation
│
├── utils/                           # Utilities
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   └── validation.py               # Data validation & error handling
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_integration.py         # End-to-end integration tests
│   ├── test_gat.py                 # GAT unit tests
│   ├── test_error_scenarios.py     # Error handling validation
│   └── test_gat_integration.py     # GAT integration tests
│
├── results/                         # Output directory
│   ├── results.json                # Decision outputs
│   └── visualizations/             # Generated charts
│
├── main.py                          # Main orchestration script
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── README.md                        # This file
└── LICENSE                          # License information
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{kazoukas2025crisis_mas,
  title={Multi-Agent Systems for Crisis Management Decision-Making Under Uncertainty},
  author={Kazoukas, Vasileios},
  year={2025},
  school={Technical University of Crete},
  type={Master's Thesis},
  department={School of Production Engineering and Management},
  program={Operational Research and Decision Making},
  note={Proof-of-concept implementation comparing Evidential Reasoning
        and Graph Attention Networks for multi-agent belief aggregation}
}
```

---

## Acknowledgments

- **Anthropic** for providing the Claude API and excellent documentation
- **Thesis Advisors** at Technical University of Crete for guidance and feedback
- **Crisis Management Experts** who provided domain knowledge for validation
- **Open Source Community** for foundational libraries (NumPy, Matplotlib, pytest)

---

## Contact

**Author:** Vasileios Kazoukas
**Email:** kazoukas@gmail.com, vkazoukas@tuc.gr
**Institution:** Technical University of Crete (TUC)
**Department:** School of Production Engineering and Management
**Program:** Operational Research & Decision Making

For questions about this research project, collaboration inquiries, or access to thesis materials, please contact via email.

---

## License

This project is developed as part of academic research. See LICENSE file for usage terms.

For academic use, please cite the associated Master's thesis (see Citation section).

---

**README Version:** 2.0.0
**Last Updated:** November 6, 2025
**Document Status:** Complete - Suitable for Thesis Appendix
