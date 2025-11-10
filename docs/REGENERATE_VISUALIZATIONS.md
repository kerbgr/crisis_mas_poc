# Regenerating Agent Network Visualizations

## Overview

The `agent_network.png` visualization is **dynamically generated** when the system runs. If you see an outdated visualization showing only 3 agents instead of the current 11 experts, you need to regenerate it.

## Quick Regeneration

Run the provided script to generate the updated 11-agent network visualization:

```bash
python generate_agent_network_visualization.py
```

**Output:**
- File: `results/visualizations/agent_network_11_experts.png`
- Shows all 11 expert agents with hierarchical trust relationships
- Color-coded by role and hierarchy

## Network Structure

The generated visualization displays:

### Core Experts (3)
- **Meteorologist** - Weather/Environmental analysis
- **Logistics Coordinator** - Supply chain and resources
- **Medical Expert** - Emergency medicine and public health

### Tactical Commanders (3)
- **Police On-Scene Commander** - Field-level law enforcement
- **Fire On-Scene Commander** - Field-level fire/rescue
- **Coast Guard On-Scene Commander** - Maritime tactical operations

### Strategic Commanders (3)
- **Regional Police Commander** - Strategic police coordination
- **Regional Fire Commander** - Strategic fire service coordination
- **National Coast Guard Director** - Strategic maritime operations

### Coordinators (1)
- **PSAP Commander** - Multi-agency emergency communications

### Infrastructure (1)
- **Medical Infrastructure Director** - Hospital system coordination

## Trust Matrix

The visualization includes a hierarchical trust matrix showing:

- **High trust (0.9-0.95)**: Same domain (police-police, fire-fire, etc.)
- **Strong trust (0.85-0.9)**: Strategic commanders and tactical-strategic pairs in same domain
- **Moderate trust (0.7)**: Core experts with all agents
- **Base trust (0.5)**: Cross-domain relationships

## Requirements

The visualization script requires:
- `matplotlib`
- `seaborn`
- `networkx`
- `numpy`

All should be installed via `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Automatic Generation

The network visualization is also automatically generated when you run the full system:

```bash
# With auto expert selection
python main.py --expert-selection auto --scenario scenarios/flood_scenario.json

# With all 11 experts manually
python main.py --agents all --scenario scenarios/flood_scenario.json
```

Visualizations are saved to `results/visualizations/` after each run.

## Customization

To customize the visualization, edit `generate_agent_network_visualization.py`:

- **Layout**: Modify `nx.spring_layout()` parameters
- **Colors**: Adjust color palette in `SystemVisualizer`
- **Trust relationships**: Edit `create_trust_matrix()` function
- **Node sizing**: Change `node_size` parameter

## Troubleshooting

### "No module named 'networkx'"

Install the missing dependency:
```bash
pip install networkx
```

### "IndexError: list index out of range"

This was fixed in commit `c21b995`. Make sure you have the latest code:
```bash
git pull origin claude/evaluation-fixes-and-documentation-011CUxMTLGqtAeV2daKw7dNJ
```

### Visualization shows only 3 agents

The old visualization is cached. Delete it and regenerate:
```bash
rm results/visualizations/agent_network*.png
python generate_agent_network_visualization.py
```

## Version History

- **v0.8** (Nov 2025): Expanded to 11 expert agents with hierarchical command structure
- **v0.7**: Original 3-agent system (Meteorologist, Logistics, Medical)

---

**Related Documentation:**
- [README.md](../README.md) - Main system documentation
- [ARCHITECTURE_DIAGRAMS.md](../ARCHITECTURE_DIAGRAMS.md) - System architecture diagrams
- [evaluation/README.md](../evaluation/README.md) - Evaluation framework documentation
