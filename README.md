# A Multi-Agent System (MAS) for Crisis Management and Decision-Making

A Proof-of-concept system demonstrating the application of **Multi-Agent Systems (MAS)** to crisis management decision-making, developed as part of a Master's thesis in Operational Research & Decision Making with the title:

***Development of a Collaborative Multi-Agent Framework for Decision Support in Crisis Management*** \
*Optimisation through Evidential Reasoning and Large Language Models for the Preservation and Enhancement of Collective Intelligence*


Military Academy (sse.gr) - Technical University of Crete (tuc.gr)

  ![sse2](https://github.com/user-attachments/assets/efbffb32-2926-4803-af6e-dbc92237c776)  <img width="166" height="166" alt="tuc2" src="https://github.com/user-attachments/assets/babf4568-4a60-4412-a7bf-4ca322f78197" />

Department of Military Sciences - School of Production Engineering and Management


**Author:** ***Vasileios Kazoukas***
**Contact:** kazoukas@gmail.com, vkazoukas@tuc.gr\
**Version:** 0.9.1
**Last Updated:** February 2026
**Status:** Research Prototype

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](docs/USAGE.md)
4. [Web Tools](web_tools/WEB_TOOLS.md)
5. [Architecture](docs/ARCHITECTURE.md)
6. [Results](docs/RESULTS.md)
7. [Limitations](docs/LIMITATIONS.md)
8. [Future Work](docs/FUTURE_WORK.md)
9. [References](docs/REFERENCES.md)
10. [Project Structure](#project-structure)
11. [Citation](#citation)

---
## Project Overview

### Purpose

This proof-of-concept system demonstrates the application of **Multi-Agent Systems (MAS)** to crisis management decision-making, developed as part of a Master's thesis in Operational Research & Decision Making. The system addresses the complex challenge of coordinating multiple expert perspectives under uncertainty and time pressure during crisis scenarios.

The implementation combines classical decision theory with modern Large Language Models (LLMs) to create an intelligent decision support system capable of:
- Aggregating diverse expert opinions with uncertainty quantification
- Evaluating alternatives across multiple competing criteria
- Building consensus through structured negotiation
- Providing explainable, traceable decision recommendations

### Scenario Design and Ethical Considerations

**Important Notice:** The crisis scenarios used in this PoC are **synthetic scenarios inspired by real historical crisis events** but are **not retrospective evaluations of actual emergency response operations**.

**Key Points:**

1. **Inspired by Reality, Not Testing Reality**:
   - Scenarios draw inspiration from publicly documented historical crises (e.g., floods in Thessaly Greece, forest fires in Evia, industrial incidents)
   - Parameters such as geographical locations, population sizes, and response alternatives reflect realistic crisis characteristics
   - Design informed by publicly available information about emergency response protocols and procedures

2. **No Retrospective Testing**:
   - This system is **NOT being validated against actual historical crisis decisions**
   - We do not have institutional approval to access classified emergency response data
   - We lack complete operational context, real-time intelligence, and stakeholder perspectives from historical incidents
   - We make no claims about the quality or correctness of actual decisions made during past crises

3. **Research Purpose Only**:
   - Scenarios serve as realistic test cases for evaluating multi-agent decision-making frameworks
   - The goal is to demonstrate the technical feasibility of MAS approaches to crisis management
   - Results demonstrate system capabilities, not critiques of past emergency response operations

4. **Ethical Boundaries**:
   - Respects the sensitivity of actual emergency operations and privacy of affected communities
   - Avoids inappropriate use of real tragedy data without proper authorization
   - Maintains focus on methodological advancement rather than historical analysis

This approach allows us to leverage domain knowledge and realistic crisis characteristics while maintaining ethical standards and respecting the complexity of real-world emergency response decision-making.

### Research Questions Addressed

This PoC investigates the following research questions:

**RQ1: Multi-Agent Coordination**
- *How can multiple autonomous agents with different expertise domains effectively coordinate to make time-critical crisis management decisions?*
- Addressed through the implementation of a coordinator agent with consensus-building algorithms

**RQ2: Uncertainty Handling**
- *What mechanisms can effectively aggregate expert beliefs under high uncertainty, incomplete information, and conflicting opinions?*
- Addressed through two approaches:
  - **Evidential Reasoning (ER)**: Dempster-Shafer theory-based belief aggregation
  - **Graph Attention Networks (GAT)**: Domain-parameterized, rule-based attention aggregator (untrained GAT variant with fixed attention coefficients) for interpretable expert weighting

**RQ3: LLM Enhancement**
- *Can Large Language Models enhance multi-agent decision-making by providing contextual reasoning, justification generation, and natural language understanding?*
- Addressed through integration of Claude API, OpenAI's and local LLMs for agent reasoning and explanation

**RQ4: Decision Quality**
- *How do multi-agent collaborative decisions compare to single-agent decisions in terms of quality, robustness, and stakeholder acceptance?*
- Addressed through comparative metrics and an evaluation framework

**RQ5: Explainability**
- *How can AI-driven crisis management systems provide transparent, auditable decision trails suitable for high-stakes domains?*
- Addressed through comprehensive logging, visualisation, and explanation generation

### Key Contributions

1. **Hybrid Aggregation Framework**: Novel comparison of classical ER vs. rule-based GAT (untrained, domain-parameterized) for belief aggregation
2. **LLM-Enhanced Agents**: Integration of multiple LLM providers (Claude, OpenAI, LM Studio) for advanced reasoning
3. **Historical Reliability Tracking**: Dynamic agent weighting based on proven past performance and consistency
4. **Comprehensive Evaluation**: Multi-dimensional metrics framework for MAS performance
5. **Open Research Platform**: Extensible codebase for further crisis management research

---

## Installation

### Prerequisites

- **Python**: 3.9 or higher
- **pip**: Package manager (usually included with Python)
- **LLM Provider** (choose one or more):
  - **Claude (Anthropic)**: Primary — get API key from [console.anthropic.com](https://console.anthropic.com/)
  - **OpenAI**: Alternative — get API key from [platform.openai.com](https://platform.openai.com/api-keys)
  - **LM Studio**: Local models, no API key required. Free and privacy-focused. Tested with OpenAI GPT-OSS 20B. Download at [lmstudio.ai](https://lmstudio.ai)
  - **Ollama**: Local models, no API key required. Free, CLI-driven. Download at [ollama.com](https://ollama.com). Start with `ollama serve` (macOS: starts automatically via launchd).
- **Vision Provider** (optional - required only for geospatial terrain analysis and camera-feed situational intelligence):
  - **Ollama** (recommended): `ollama pull minicpm-v` (~5.5 GB). **Do not use `llama3.2-vision`** - it is broken in Ollama 0.30.x due to an mllama architecture regression ([issue #16490](https://github.com/ollama/ollama/issues/16490)).
  - **LM Studio**: Load any vision-capable model (e.g. LLaVA, BakLLaVA) in the LM Studio UI.
  - Vision is fully optional - the pipeline continues unaffected if no vision provider is running.
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
python -c "import anthropic; import numpy; import matplotlib; print('All dependencies installed successfully')"
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

For complete usage documentation including command-line options, examples, expert roles, LLM provider comparison, and programmatic API, see **[docs/USAGE.md](docs/USAGE.md)**.

**Quick start:**
```bash
python main.py                              # Default flood scenario with 3 agents
python main.py --agents all                 # All 13 expert agents
python main.py --aggregation-method gat     # Use GAT instead of ER
python main.py --expert-selection auto      # Automatic expert selection
python main.py --llm-provider lmstudio      # Use local LLM (free, offline)
python main.py --compare-methods            # Compare ER vs GAT side-by-side
```

**Evaluation commands (4 scenarios - training benchmark + held-out evaluation):**

```bash
python main.py --scenario flood_scenario --compare-methods --expert-selection auto
python main.py --scenario forest_fire_evia --compare-methods --expert-selection auto
python main.py --scenario ammonia_leak_elefsina --compare-methods --expert-selection auto
python main.py --scenario santorini_volcanic_seismic --compare-methods --expert-selection auto
```

Add `--llm-provider claude|openai|lmstudio` to select the LLM backend (default: claude).

---

## Web Tools

The system includes a web-based interface for managing expert profiles, incident handling protocols, and crisis scenarios with OpenStreetMap integration. See **[web_tools/WEB_TOOLS.md](web_tools/WEB_TOOLS.md)** for full documentation.

```bash
cd web_tools && python app.py    # Available at http://localhost:5000
```

---

## Architecture

The system consists of six core layers: User Interface, Coordination, Agent (13 experts in Gold-Silver hierarchy), Decision Framework (ER/GAT + MCDA), LLM Integration (Claude/OpenAI/LM Studio), and Evaluation. For detailed component descriptions, diagrams, decision-making flow, and algorithm specifications (Dempster-Shafer, GAT, TOPSIS), see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Results

The system was evaluated across **45 controlled runs** (5 replicates × 3 LLM providers × 3 crisis scenarios) using all 13 expert agents with `--compare-methods` mode, producing 135 result records. Key findings:

- **ER and GAT are statistically equivalent** — DQS 0.775 ± 0.032 vs 0.781 ± 0.029 (p > 0.05); 88.9% (40/45) recommendation agreement
- **GPT-OSS 20B** (local, via LM Studio) achieves the highest mean DQS (0.504 ± 0.051) at zero API cost
- **Claude Sonnet 4** is 5–9× faster than other providers (mean 11.6 s/run), decisive for real-time use
- **GPT-4o** yields the most consistent consensus (0.917 ± 0.036, lowest variance)
- **System consensus** averages 0.902 ± 0.066; run-level recommendation stability is 97.8% (44/45 runs)
- **HAZMAT convergence**: all three providers converge on `action_integrated_response` across all 15 HAZMAT runs via both ER and GAT — the highest cross-provider agreement of any scenario

See **[docs/RESULTS.md](docs/RESULTS.md)** for full tables, statistical tests, and per-scenario detail.

---

## Limitations

Known algorithmic limitations (ER simplifications, static MCDA weights, rule-based GAT), operational constraints (API costs, LLM sensitivity), and scope boundaries (scenario representation, scalability). See **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)**.

---

## Future Work

Research roadmap including the SEAL-Enhanced Genetic Agents proposal for self-adapting agents, plus short-term (real-time data, dashboard), medium-term (multi-objective optimization, temporal planning), and long-term enhancements (RLHF, edge deployment). See **[docs/FUTURE_WORK.md](docs/FUTURE_WORK.md)**.

---

## References

Academic references for Multi-Agent Systems, Evidential Reasoning, MCDA, Graph Attention Networks, LLMs in Decision Support, and Crisis Management. See **[docs/REFERENCES.md](docs/REFERENCES.md)**.

---

## Project Structure

```
crisis_mas_poc/
├── agents/                          # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py               # Abstract base agent class
│   ├── expert_agent.py             # Domain expert agents (LLM-enhanced reasoning)
│   ├── coordinator_agent.py        # Coordination and consensus building
│   ├── reliability_tracker.py      # Historical performance tracking
│   ├── agent_template.py           # Template for creating custom agents
│   ├── agent_profiles.json         # Agent configurations (13 experts)
│   └── AGENT_DEVELOPMENT_GUIDE.md  # Documentation for custom agents
│
├── scenarios/                       # Crisis scenarios
│   ├── __init__.py
│   ├── scenario_loader.py          # JSON loading utilities
│   ├── expert_selector.py          # Automatic expert selection
│   ├── flood_scenario.json         # Karditsa flood emergency (Greek)
│   ├── forest_fire_evia.json       # Evia forest fire emergency (Greek)
│   ├── ammonia_leak_elefsina.json  # Elefsina HAZMAT incident (Greek)
│   ├── criteria_weights.json       # MCDA criteria definitions
│   ├── scenario_template.json      # Template for creating scenarios
│   └── README.md                    # Comprehensive scenario documentation
│
├── decision_framework/              # Core decision algorithms
│   ├── __init__.py
│   ├── evidential_reasoning.py     # ER aggregation (Dempster-Shafer)
│   ├── gat_aggregator.py           # Graph Attention Network aggregation
│   ├── mcda_engine.py              # MCDA methods (TOPSIS, WSM, SAW)
│   ├── consensus_model.py          # Consensus detection & building
│   └── README.md                    # Algorithm documentation
│
├── llm_integration/                 # LLM interface (multi-provider)
│   ├── __init__.py
│   ├── claude_client.py            # Anthropic Claude API wrapper (default)
│   ├── openai_client.py            # OpenAI GPT-4/GPT-3.5 API wrapper
│   ├── lmstudio_client.py          # LM Studio local models wrapper
│   ├── prompt_templates.py         # Domain-specific prompts (13 expert roles)
│   └── README.md                    # Multi-provider setup guide
│
├── evaluation/                      # Metrics and visualization
│   ├── __init__.py
│   ├── metrics.py                  # Five core metrics (DQS, Consensus, etc.)
│   ├── visualizations.py           # Publication-quality chart generation
│   ├── EVALUATION_METHODOLOGY.md   # Mathematical formulas
│   ├── FORMULA_VERIFICATION.md     # Code-to-formula verification
│   └── README.md                    # Evaluation framework documentation
│
├── utils/                           # Utilities
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration management
│   ├── validation.py               # Comprehensive data validation
│   └── UTILS.md                     # Utilities documentation
│
├── tests/                           # Test suite (17 test files)
│   ├── __init__.py
│   ├── test_integration.py         # End-to-end integration tests
│   ├── test_gat.py                 # GAT unit tests
│   ├── test_evidential_reasoning.py # ER unit tests
│   ├── test_error_scenarios.py     # Error handling validation
│   ├── test_multi_llm_clients.py   # Multi-provider tests
│   └── README.md                    # Test organization and coverage
│
├── web_tools/                       # Web-based UI for scenario management
│   ├── app.py                      # Flask application server
│   ├── protocol_integration.py     # Protocol integration module
│   ├── templates/                  # HTML templates
│   ├── static/                     # CSS and JavaScript assets
│   ├── data/                       # JSON data storage
│   ├── WEB_TOOLS.md                 # Web tools documentation
│   ├── README.md                    # Quick start guide
│   └── FEATURES.md                  # Advanced features guide
│
├── docs/                            # Technical documentation
│   ├── USAGE.md                    # Complete usage guide
│   ├── ARCHITECTURE.md             # System architecture & algorithms
│   ├── RESULTS.md                  # Experimental results (45 runs, 3 providers, 3 scenarios)
│   ├── LIMITATIONS.md              # Known limitations
│   ├── FUTURE_WORK.md              # Research roadmap
│   ├── REFERENCES.md               # Academic references
│   ├── architecture_viewer.html    # Interactive Mermaid diagram viewer
│   ├── architecture_main.mmd       # Main architecture diagram source
│   ├── BELIEF_DISTRIBUTION_INVESTIGATION.md  # Technical investigation
│   └── REGENERATE_VISUALIZATIONS.md # Visualization regeneration guide
│
├── examples/                        # Example scripts and debugging utilities
│   ├── example_llm_providers.py    # Multi-provider demonstration
│   ├── example_claude_usage.py     # Claude API examples
│   ├── debug_single_agent_dqs.py   # DQS calculation debugging
│   └── README.md                    # Examples and learning path
│
├── models/                          # Pydantic data models
│   └── data_models.py              # Type-safe data structures
│
├── LLM Training/                    # LLM training methodology
│   ├── README.md                    # Zero-to-hero training roadmap
│   ├── data_collection/            # Data collection scripts
│   ├── fine_tuning/                # Fine-tuning configurations
│   ├── evaluation/                 # Evaluation benchmarks
│   ├── tools/                      # Automation tools (14 subdirectories)
│   └── examples/                   # Complete training examples
│
├── results/                         # Output directory (auto-generated)
│   ├── results.json                # Decision outputs
│   └── visualizations/             # Generated charts
│
├── main.py                          # Main orchestration script
├── requirements.txt                 # Python dependencies
├── academic_paper.md                # Comprehensive academic paper (77KB)
├── ARCHITECTURE_DIAGRAMS.md         # Architecture documentation
├── ERROR_HANDLING_IMPROVEMENTS.md   # Technical improvements
├── SPADE_FRAMEWORK_COMPARISON.md    # Framework comparison
├── CITATION.cff                     # Machine-readable citation metadata
├── LICENSE                          # Dual-license (academic/commercial)
└── README.md                        # This file
```

---

## Citation

**MANDATORY**: If you use this software in academic research, you MUST cite it as follows:

### Software Citation (Required)

```bibtex
@software{kazoukas2025crisis,
  author = {Kazoukas, Vasileios},
  title = {Crisis Management Multi-Agent System: Graph Attention Networks
           and Evidential Reasoning for Emergency Response Coordination},
  year = {2025},
  institution = {Military Academy (SSE) and Technical University of Crete (TUC)},
  url = {https://github.com/kerbgr/crisis_mas_poc}
}
```

### Master's Thesis Citation (If Available)

```bibtex
@mastersthesis{kazoukas2025crisis_mas,
  title={Multi-Agent Systems for Crisis Management Decision-Making Under Uncertainty},
  author={Kazoukas, Vasileios},
  year={2025},
  school={Military Academy (SSE) and Technical University of Crete (TUC)},
  type={Master's Thesis},
  department={School of Production Engineering and Management},
  program={Operational Research and Decision Making},
  note={Proof-of-concept implementation comparing Evidential Reasoning
        and Graph Attention Networks for multi-agent belief aggregation}
}
```

### APA Format

Kazoukas, V. (2026). *Crisis Management Multi-Agent System: Graph Attention Networks and Evidential Reasoning for Emergency Response Coordination* [Computer software]. Military Academy (SSE) and Technical University of Crete (TUC). https://github.com/kerbgr/crisis_mas_poc

### IEEE Format

V. Kazoukas, "Crisis Management Multi-Agent System: Graph Attention Networks and Evidential Reasoning for Emergency Response Coordination," Military Academy (SSE) and Technical University of Crete (TUC), 2026. [Online]. Available: https://github.com/kerbgr/crisis_mas_poc

**Note**: See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## Acknowledgments

- **Thesis Advisors** at Military Academy (SSE) and Technical University of Crete (TUC) for guidance and feedback
- **Domain Expert Evaluation** — explainability and auditability were assessed by the lead researcher (Vasileios Kazoukas) in his capacity as a practitioner with 20 years of operational experience in crisis management technology systems (national C4I infrastructure, Hellenic Civil Protection, EENA/ITU, NCCM unit 5/112, H2020/Horizon Europe safety research). Independent validation by a broader panel of emergency management professionals remains as planned future work.
- **Open Source Community** for foundational libraries (NumPy, Matplotlib, pytest)

---

## Contact

**Author:** Vasileios Kazoukas
**Email:** kazoukas@gmail.com, vkazoukas@tuc.gr
**Institution:** Military Academy (SSE) Technical University of Crete (TUC)
**Department:** Department of Military Sciences - School of Production Engineering and Management
**Program:** Operational Research & Decision Making

For questions about this research project, collaboration inquiries, or access to thesis materials, please contact via email.

---

## License

**Proprietary License - Dual Use Model**

This software is licensed under a custom Academic/Commercial dual-license model:

### Academic Use (Free with Citation)

**FREE for academic research** - You may use, modify, and build upon this software for non-commercial academic purposes, provided you:
- **Cite the work** in all publications (see Citation section above - MANDATORY)
- Retain attribution and license notices
- Keep derivative works non-commercial

### Commercial Use (Requires License)

**Commercial license required** for:
- Operational deployment in emergency response systems
- Government/municipal crisis management infrastructure
- Integration into commercial products
- Professional consulting or training services
- Any revenue-generating or operational use

**Contact for commercial licensing**: kazoukas@gmail.com

### Full License Terms

See [LICENSE](LICENSE) file for complete terms including:
- Detailed academic use permissions and requirements
- Commercial licensing inquiry process
- Intellectual property rights
- Warranty disclaimers and liability limitations
- Prohibited activities and termination clauses

**Key Points**:
- Academic use: Free with mandatory citation
- Commercial use: Negotiated license required
- All rights reserved by copyright holder
- Governed by Greek and EU law

**Copyright 2026 Vasileios Kazoukas. All Rights Reserved.**

---

**README Version:** 0.9.1
**Last Updated:** February 16, 2026
