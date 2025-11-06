# Crisis Management Multi-Agent System (MAS) - Proof of Concept

A Multi-Agent System for crisis management decision-making, developed as part of a Master's thesis in Operational Research & Decision Making.

## Overview

This project implements a sophisticated multi-agent system that combines:
- **Expert Agents** with domain-specific knowledge (medical, logistics, public safety, environmental)
- **Coordinator Agent** for consensus building and decision aggregation
- **Evidential Reasoning** for handling uncertainty
- **Multi-Criteria Decision Analysis (MCDA)** for evaluating alternatives
- **LLM Enhancement** using Claude for advanced reasoning and justification

## Project Structure

```
crisis_mas_poc/
├── agents/                      # Agent implementations
│   ├── base_agent.py           # Base agent class
│   ├── expert_agent.py         # Expert agent with domain knowledge
│   ├── coordinator_agent.py    # Coordinator for consensus building
│   └── agent_profiles.json     # Agent configurations
├── scenarios/                   # Crisis scenarios
│   ├── scenario_loader.py      # Scenario loading utilities
│   ├── flood_scenario.json     # Example flood scenario
│   └── criteria_weights.json   # Decision criteria and weights
├── decision_framework/          # Decision-making algorithms
│   ├── evidential_reasoning.py # Evidential reasoning implementation
│   ├── mcda_engine.py          # MCDA methods (TOPSIS, WSM, SAW)
│   └── consensus_model.py      # Consensus detection and building
├── llm_integration/             # Claude API integration
│   ├── claude_client.py        # API client wrapper
│   └── prompt_templates.py     # Domain-specific prompts
├── evaluation/                  # Performance evaluation
│   ├── metrics.py              # Performance metrics
│   └── visualizations.py       # Result visualization
├── utils/                       # Utilities
│   └── config.py               # Configuration management
├── main.py                      # Main orchestration script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
cd crisis_mas_poc
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up Claude API key:**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```
Or create a `.env` file with:
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Quick Start

### Run the default flood scenario:
```bash
python main.py
```

### Run with specific scenario:
```bash
python main.py --scenario flood_scenario.json
```

### Run without LLM enhancement:
```bash
python main.py --no-llm
```

### Run without visualizations:
```bash
python main.py --no-viz
```

### Specify output file:
```bash
python main.py --output my_results.json
```

## Usage Examples

### Basic Example

```python
from utils import Config
from main import CrisisMAS

# Initialize system
config = Config()
mas = CrisisMAS(config, use_llm=True)

# Load agents
mas.load_agents('agents/agent_profiles.json')

# Run scenario
results = mas.run_scenario('flood_scenario.json')

# Evaluate
evaluation = mas.evaluate_results(results)

# Save results
mas.save_results(results, 'output.json')
```

### Custom Agent Creation

```python
from agents import ExpertAgent
from llm_integration import ClaudeClient

# Create custom expert agent
profile = {
    'agent_id': 'custom_expert',
    'expertise_domain': 'cybersecurity',
    'confidence_level': 0.85,
    'criteria_weights': {
        'effectiveness': 0.30,
        'safety': 0.25,
        'speed': 0.25,
        'cost': 0.10,
        'public_acceptance': 0.10
    }
}

llm_client = ClaudeClient()
agent = ExpertAgent('custom_expert', profile, llm_client)

# Use agent
scenario = {...}
evaluation = agent.evaluate_scenario(scenario)
proposal = agent.propose_action(scenario, criteria)
```

### Running MCDA Analysis

```python
from decision_framework import MCDAEngine

# Initialize MCDA engine
mcda = MCDAEngine(criteria_weights={
    'effectiveness': 0.30,
    'safety': 0.25,
    'speed': 0.20,
    'cost': 0.15,
    'public_acceptance': 0.10
})

# Compare alternatives using different methods
alternatives = [...]  # List of alternatives with criteria scores

# Weighted Sum Method
wsm_ranking = mcda.weighted_sum_method(alternatives)

# TOPSIS Method
topsis_ranking = mcda.topsis_method(alternatives)

# Compare all methods
comparison = mcda.compare_methods(alternatives)
```

### Evidential Reasoning Example

```python
from decision_framework import EvidentialReasoning

# Initialize ER engine
er = EvidentialReasoning()

# Expert assessments
assessments = [
    {'score': 0.8, 'confidence': 0.9},  # Expert 1
    {'score': 0.7, 'confidence': 0.7},  # Expert 2
    {'score': 0.9, 'confidence': 0.8},  # Expert 3
]

# Aggregate using evidential reasoning
result = er.aggregate_expert_assessments(assessments)

print(f"Aggregated score: {result['aggregated_score']:.2f}")
print(f"Uncertainty: {result['uncertainty']:.2f}")
```

## Configuration

The system can be configured using a JSON configuration file:

```json
{
  "decision_framework": {
    "consensus_threshold": 0.7,
    "max_iterations": 5,
    "mcda_method": "weighted_sum"
  },
  "llm": {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "temperature": 0.7,
    "enable_llm": true
  },
  "evaluation": {
    "output_dir": "results",
    "save_visualizations": true
  }
}
```

Use with:
```bash
python main.py --config my_config.json
```

## Creating Custom Scenarios

Create a new scenario JSON file in the `scenarios/` directory:

```json
{
  "id": "earthquake_001",
  "type": "earthquake",
  "name": "Urban Earthquake Emergency",
  "description": "Major earthquake in metropolitan area...",
  "severity": 0.9,
  "affected_population": 50000,
  "available_actions": [
    {
      "id": "action_1",
      "name": "Emergency Evacuation",
      "description": "...",
      "criteria_scores": {
        "effectiveness": 0.85,
        "safety": 0.90,
        "speed": 0.70,
        "cost": 0.40,
        "public_acceptance": 0.75
      }
    }
  ]
}
```

## Output

The system generates:

1. **JSON Results File** - Complete results with decisions and metrics
2. **Visualizations** (if enabled):
   - Agent contribution plots
   - Action comparison radar charts
   - Decision distribution charts
   - Consensus evolution plots

Results are saved to the `results/` directory by default.

## Key Features

### 1. Multi-Agent Decision Making
- Multiple expert agents with different domain expertise
- Coordinator agent for consensus building
- Dynamic agent interaction and opinion aggregation

### 2. Evidential Reasoning
- Handles uncertainty in expert opinions
- Dempster-Shafer combination of beliefs
- Confidence-weighted aggregation

### 3. MCDA Methods
- Weighted Sum Method (WSM)
- TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
- Simple Additive Weighting (SAW)
- Sensitivity analysis

### 4. LLM Enhancement
- Claude API integration for advanced reasoning
- Domain-specific prompt templates
- Automated justification generation
- Consensus facilitation

### 5. Comprehensive Evaluation
- Consensus metrics
- Decision quality metrics
- Agent contribution analysis
- Diversity metrics
- Performance visualization

## Research Context

This project is developed as part of a Master's thesis in Operational Research & Decision Making, focusing on:
- Multi-agent systems for crisis decision-making
- Evidential reasoning under uncertainty
- Multi-criteria decision analysis
- LLM-enhanced agent reasoning
- Consensus building algorithms

## Performance Metrics

The system evaluates:
- **Consensus Rate**: How often agents reach agreement
- **Agreement Level**: Degree of consensus achieved
- **Decision Quality**: Confidence and score metrics
- **Opinion Diversity**: Variety in agent perspectives
- **Efficiency**: Time and resource utilization
- **Robustness**: Stability under parameter variations

## Extending the System

### Adding New Agent Types

1. Create a new class inheriting from `BaseAgent`
2. Implement required methods: `evaluate_scenario()`, `propose_action()`
3. Add agent profile to `agent_profiles.json`

### Adding New Decision Methods

1. Add method to `MCDAEngine` or create new module
2. Update configuration to support new method
3. Update main orchestration to call new method

### Adding New Evaluation Metrics

1. Add metric calculation to `PerformanceMetrics`
2. Update evaluation report generation
3. Add visualization if needed

## Troubleshooting

### Common Issues

**API Key Error:**
```
ValueError: API key not provided
```
Solution: Set `ANTHROPIC_API_KEY` environment variable

**Import Errors:**
```
ModuleNotFoundError: No module named 'anthropic'
```
Solution: Run `pip install -r requirements.txt`

**Visualization Errors:**
```
RuntimeError: Invalid DISPLAY variable
```
Solution: Use `--no-viz` flag or configure matplotlib backend

## Contributing

This is a research project for a Master's thesis. For questions or collaboration inquiries, please contact the research team.

## License

See LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{crisis_mas_2025,
  title={Multi-Agent Systems for Crisis Management Decision-Making},
  author={[Vasileios Kazoukas]},
  year={2025},
  school={[SSE - TUC]},
  type={Master's Thesis},
  note={Operational Research \& Decision Making}
}
```

## Acknowledgments

- Anthropic for Claude API
- Research advisors and thesis committee
- Open-source community for foundational libraries

## Contact

For questions about this research project, please contact:
- kazoukas@gmail.com vkazoukas@tuc.gr
- SSE-TUC

---

**Version:** 1.0.0
**Last Updated:** November 2025
**Status:** Proof of Concept for Master's Thesis
