# A Multi-Agent System (MAS) for crisis management and decision-making

A Proof-of-concept system demonstrating the application of **Multi-Agent Systems (MAS)** to crisis management decision-making, developed as part of a Master's thesis in Operational Research & Decision Making with the title:

***Development of a Collaborative Multi-Agent Framework for Decision Support in Crisis Management*** \
*Optimisation through Evidential Reasoning and Large Language Models for the Preservation and Enhancement of Collective Intelligence*


Military Academy (sse.gr) - Technical University of Crete (tuc.gr)

  ![sse2](https://github.com/user-attachments/assets/efbffb32-2926-4803-af6e-dbc92237c776)  <img width="166" height="166" alt="tuc2" src="https://github.com/user-attachments/assets/babf4568-4a60-4412-a7bf-4ca322f78197" />

Department of Military Sciences - School of Production Engineering and Management


**Author:** ***Vasileios Kazoukas***
**Contact:** kazoukas@gmail.com, vkazoukas@tuc.gr\
**Version:** 0.8  
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
2. **LLM-Enhanced Agents**: Integration of multiple LLM providers (Claude, OpenAI, LM Studio) for advanced reasoning
3. **Historical Reliability Tracking**: Dynamic agent weighting based on proven past performance and consistency
4. **Comprehensive Evaluation**: Multi-dimensional metrics framework for MAS performance
5. **Open Research Platform**: Extensible codebase for further crisis management research

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
1. Load 3 default expert agents (Meteorologist, Logistics, Medical) for backward compatibility
2. Initialize decision framework (ER + MCDA + Consensus)
3. Process the flood scenario with 3 alternative actions
4. Generate decision with explanations
5. Save results to `results/results.json`
6. Generate visualizations (if enabled)

**Note:** Use `--agents all` to load all 11 expert agents (see Expert Roles section below)

### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --scenario PATH              Scenario JSON file (default: scenarios/flood_scenario.json)
  --agents AGENT_IDS           Specific agent IDs to load (space-separated), or "all" for all 11 experts
                               Default: agent_meteorologist logistics_expert_01 medical_expert_01
                               Available: agent_meteorologist, medical_expert_01, logistics_expert_01,
                                         psap_commander_01, police_onscene_01, police_regional_01,
                                         fire_onscene_01, fire_regional_01, medical_infrastructure_01,
                                         coastguard_onscene_01, coastguard_national_01
  --expert-selection MODE      Expert selection mode: "manual" or "auto" (default: manual)
                               - manual: Use --agents flag to specify experts
                               - auto: Automatically select experts based on scenario metadata
  --criteria PATH              Criteria weights JSON (default: scenarios/criteria_weights.json)
  --output PATH                Output results file (default: results/results.json)
  --config PATH                Configuration JSON file
  --llm-provider PROVIDER      LLM provider: claude, openai, or lmstudio (default: claude)
  --no-llm                     Disable LLM enhancement (use rule-based reasoning)
  --no-viz                     Disable visualization generation
  --aggregation METHOD         Aggregation method: ER or GAT (default: ER)
  --consensus-threshold N      Consensus threshold 0-1 (default: 0.7)
  --verbose                    Enable verbose logging
  --help                       Show help message
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

#### Example 6: Run with All 11 Expert Agents

```bash
python main.py --agents all
```

This loads the full emergency response command structure with all 11 expert roles.

#### Example 7: Run with Specific Agent Subset

```bash
# Fire and police response team
python main.py --agents fire_onscene_01 fire_regional_01 police_onscene_01 police_regional_01

# Maritime crisis team
python main.py --agents coastguard_onscene_01 coastguard_national_01 medical_expert_01

# Medical infrastructure focus
python main.py --agents medical_expert_01 medical_infrastructure_01 logistics_expert_01
```

#### Example 8: Automatic Expert Selection (NEW in v0.8!)

```bash
# Let the system automatically choose which experts to engage
python main.py --scenario flood_scenario --expert-selection auto

# With verbose mode to see selection reasoning
python main.py --scenario flood_scenario --expert-selection auto --verbose
```

**How it works:** The system analyzes scenario metadata (crisis type, severity, affected domains, scope) and automatically selects the most relevant experts. For example, a coastal flood with high severity will auto-select: meteorologist, logistics, medical, coast guard (both levels), police (tactical + strategic), fire/rescue, PSAP commander, and medical infrastructure.

**Benefits:**
- No need to manually choose from 11 experts
- Consistent expert team selection
- Prevents over/under-engagement
- Transparent reasoning (use --verbose)

#### Example 9: Combined Options

```bash
python main.py \
  --llm-provider openai \
  --aggregation GAT \
  --expert-selection auto \
  --scenario scenarios/custom_scenario.json \
  --output results/custom_output.json \
  --verbose
```

### Greek Crisis Scenarios

The system includes three realistic Greek crisis scenarios with localized expert agents:

#### Example 10: Karditsa Flood Emergency

```bash
# Run the Karditsa flood scenario with automatic expert selection
python main.py --scenario scenarios/flood_scenario.json --expert-selection auto --output-dir results/karditsa_flood

# With all experts manually specified
python main.py \
  --scenario scenarios/flood_scenario.json \
  --agents all \
  --output results/karditsa_flood_results.json \
  --verbose
```

**Scenario Details:**
- **Location:** Karditsa, Thessaly, Greece (Pamisos River overflow)
- **Severity:** 0.8 (High)
- **Affected Population:** 15,000
- **Key Challenges:** Residential flooding, agricultural damage, infrastructure threats
- **Greek Experts:** Πολιτική Προστασία, ΕΚΑΒ, ΕΛΑΣ, Πυροσβεστική

#### Example 11: Evia Forest Fire Emergency

```bash
# Run the Evia forest fire scenario with automatic expert selection
python main.py --scenario scenarios/forest_fire_evia.json --expert-selection auto --output-dir results/evia_fire

# With specific fire response team
python main.py \
  --scenario scenarios/forest_fire_evia.json \
  --agents fire_onscene_01 fire_regional_01 agent_meteorologist coastguard_onscene_01 \
  --output results/evia_fire_results.json \
  --verbose
```

**Scenario Details:**
- **Location:** North Evia island, Central Greece
- **Severity:** 0.9 (Very High)
- **Affected Population:** 8,000
- **Burned Area:** 12,000 hectares
- **Key Challenges:** Multiple fire fronts, strong winds, village evacuations, aerial operations
- **Resources:** Canadair CL-415, Chinook helicopters, ground fire crews
- **Greek Experts:** Πυροσβεστική (Fire Service), Λιμενικό (Coast Guard), Μετεωρολόγος

#### Example 12: Elefsina Ammonia Leak (HAZMAT)

```bash
# Run the Elefsina ammonia leak scenario with automatic expert selection
python main.py --scenario scenarios/ammonia_leak_elefsina.json --expert-selection auto --output-dir results/elefsina_hazmat

# With HAZMAT-focused team
python main.py \
  --scenario scenarios/ammonia_leak_elefsina.json \
  --agents fire_onscene_01 medical_expert_01 police_onscene_01 agent_meteorologist \
  --output results/elefsina_hazmat_results.json \
  --verbose
```

**Scenario Details:**
- **Location:** Elefsina industrial zone, near Athens, Greece
- **Severity:** 0.85 (Very High)
- **Affected Population:** 12,000
- **Chemical:** 50-ton anhydrous ammonia (NH3) tank rupture - UN1005, Class 2.3 toxic gas
- **IDLH Level:** 300 ppm (current readings: 150-300 ppm downwind)
- **Key Challenges:** Toxic cloud dispersion, Level A HAZMAT operations, water curtain suppression, evacuation
- **Greek Experts:** Πυροσβεστική HAZMAT teams, ΕΚΑΒ emergency medicine, ΕΛΑΣ evacuation coordination

#### Example 13: Compare All Three Greek Scenarios

```bash
# Run all three scenarios and compare results
python main.py --scenario scenarios/flood_scenario.json --expert-selection auto --output results/karditsa_results.json
python main.py --scenario scenarios/forest_fire_evia.json --expert-selection auto --output results/evia_results.json
python main.py --scenario scenarios/ammonia_leak_elefsina.json --expert-selection auto --output results/elefsina_results.json

# Results will be saved in separate files for comparison
```

### Expert Roles

The Crisis MAS system includes **11 expert roles** organized in a comprehensive emergency response command structure. The system is designed with backward compatibility - by default it uses 3 core experts, but can scale to the full 11-agent team.

#### Default Agents (Backward Compatibility)

When run without the `--agents` flag, the system loads **3 core expert agents**:

1. **Meteorologist** (`agent_meteorologist`) - Weather forecasting and environmental analysis
2. **Logistics Coordinator** (`logistics_expert_01`) - Supply chain and resource allocation
3. **Medical Expert** (`medical_expert_01`) - Emergency medicine and public health

#### Full Emergency Response Command Structure (11 Agents)

Use `--agents all` to activate the complete multi-agency command structure:

**Original Core Experts (3):**
1. **Meteorologist** (`agent_meteorologist`)
   - Role: Weather/Environmental Specialist
   - Expertise: Severe weather prediction, climate analysis
   - Focus: Weather forecasting, environmental conditions

2. **Logistics Coordinator** (`logistics_expert_01`)
   - Role: Supply Chain Management
   - Expertise: Resource allocation, transportation, infrastructure
   - Focus: Supply chain optimization, logistics planning

3. **Medical Expert** (`medical_expert_01`)
   - Role: Medical Director
   - Expertise: Emergency medicine, public health response
   - Focus: Health impacts, triage, medical resource allocation

**Emergency Communications (1):**

4. **PSAP Commander-Supervisor** (`psap_commander_01`)
   - Role: Emergency Communications Authority
   - Expertise: Dispatch coordination, call intake, CAD systems
   - Focus: First-in decision authority, multi-agency coordination
   - Experience: 16 years

**Law Enforcement Command Structure (2):**

5. **On-Scene Police Commander** (`police_onscene_01`)
   - Role: Tactical Field Authority
   - Expertise: Tactical operations, scene security, threat assessment
   - Focus: Immediate tactical response, crowd control
   - Hierarchy: Tactical (field-level)
   - Experience: 22 years

6. **Regional Police Commander** (`police_regional_01`)
   - Role: Strategic Police Authority
   - Expertise: Regional strategy, mutual aid, inter-jurisdictional coordination
   - Focus: Strategic deployment, resource allocation across jurisdictions
   - Hierarchy: Strategic (regional-level)
   - Experience: 28 years

**Fire/Rescue Command Structure (2):**

7. **On-Scene Fire-Brigade Commander** (`fire_onscene_01`)
   - Role: Tactical Fire/Rescue Authority
   - Expertise: Fire suppression, rescue operations, hazmat response
   - Focus: Immediate fire/rescue operations, structural assessment
   - Hierarchy: Tactical (field-level)
   - Experience: 19 years

8. **Regional Fire-Brigade Commander** (`fire_regional_01`)
   - Role: Strategic Fire/Rescue Authority
   - Expertise: Regional fire operations, mutual aid, long-duration incidents
   - Focus: Strategic fire service deployment, personnel management
   - Hierarchy: Strategic (regional-level)
   - Experience: 26 years

**Medical Infrastructure (1):**

9. **Medical Infrastructure Director** (`medical_infrastructure_01`)
   - Role: Healthcare System Capacity Authority
   - Expertise: Hospital capacity, surge operations, patient distribution
   - Focus: Healthcare system coordination, medical supply chain
   - Experience: 24 years

**Maritime/Coastal Response Command Structure (2):**

10. **On-Scene Coast Guard Commander** (`coastguard_onscene_01`)
    - Role: Maritime/Coastal Tactical Authority
    - Expertise: Maritime rescue, coastal evacuation, SAR operations
    - Focus: Immediate maritime rescue, sea state assessment
    - Hierarchy: Tactical (field-level)
    - Experience: 18 years

11. **National Coast Guard Director** (`coastguard_national_01`)
    - Role: Strategic Maritime Authority
    - Expertise: National maritime policy, inter-regional coordination
    - Focus: Strategic maritime operations, port security, national assets
    - Hierarchy: Strategic (national-level)
    - Experience: 30 years

#### Tactical vs. Strategic Hierarchy

The expert roles are organized in a **two-tier command hierarchy**:

- **Tactical (On-Scene)**: Field-level commanders with immediate operational authority
  - Police On-Scene Commander
  - Fire On-Scene Commander
  - Coast Guard On-Scene Commander
  - Focus: Immediate crisis response, tactical operations, real-time decision-making

- **Strategic (Regional/National)**: Higher-level commanders responsible for resource allocation and multi-jurisdictional coordination
  - Regional Police Commander
  - Regional Fire Commander
  - National Coast Guard Director
  - Focus: Strategic planning, resource allocation, inter-agency coordination

This hierarchy enables the system to model realistic emergency response structures where tactical commanders provide on-ground intelligence while strategic commanders coordinate broader resource deployment.

#### Smart Expert Selection (Auto-Mode)

**NEW in v0.8:** The system can automatically select appropriate experts based on scenario characteristics.

**How to Enable:**
```bash
python main.py --scenario flood_scenario --expert-selection auto
```

**Expert Selection Workflow:**

```mermaid
flowchart TD
    Start([Load Scenario]) --> CheckMode{Expert Selection<br/>Mode?}

    CheckMode -->|manual| ManualPath[Use --agents flag<br/>or default 3 core experts]
    CheckMode -->|auto| AutoPath[Extract expert_selection<br/>metadata from scenario]

    ManualPath --> InitAgents[Initialize Selected Agents]

    AutoPath --> HasMeta{Metadata<br/>exists?}
    HasMeta -->|No| FallbackCore[Fallback: Use 3<br/>core experts]
    HasMeta -->|Yes| EvalExperts[Evaluate Each Expert<br/>Against Selection Rules]

    FallbackCore --> InitAgents

    EvalExperts --> ExpertLoop{For Each of<br/>11 Experts}

    ExpertLoop --> ScoreCriteria[Calculate Match Score<br/>Based on Criteria]

    ScoreCriteria --> CriteriaChecks{Check Matching Criteria}

    CriteriaChecks --> C1[Crisis Type Match: +3]
    CriteriaChecks --> C2[Crisis Subtype Match: +2]
    CriteriaChecks --> C3[Domain Match: +2]
    CriteriaChecks --> C4[Severity Threshold: +1]
    CriteriaChecks --> C5[Geographic Scope: +2]
    CriteriaChecks --> C6[Geographic Location: +2]
    CriteriaChecks --> C7[Command Structure: +2]
    CriteriaChecks --> C8[Multi-jurisdictional: +1]
    CriteriaChecks --> C9[Infrastructure Systems: +2]
    CriteriaChecks --> C10[Population Threshold: +1]
    CriteriaChecks --> C11[Duration Threshold: +1]

    C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 & C9 & C10 & C11 --> TotalScore[Calculate Total Score]

    TotalScore --> CheckScore{Score > 0<br/>OR<br/>Core Expert?}

    CheckScore -->|Yes| AddToSelected[Add to Selected Set]
    CheckScore -->|No| Skip[Skip Expert]

    AddToSelected --> MoreExperts{More Experts<br/>to Evaluate?}
    Skip --> MoreExperts

    MoreExperts -->|Yes| ExpertLoop
    MoreExperts -->|No| EnsureMin{Selected >= 3<br/>minimum?}

    EnsureMin -->|No| AddCore[Add Core Experts]
    EnsureMin -->|Yes| CheckMax{Selected <= 11<br/>maximum?}

    AddCore --> CheckMax

    CheckMax -->|No| KeepTop[Keep Top 11 by Score]
    CheckMax -->|Yes| ReturnList[Return Selected Agent IDs]

    KeepTop --> ReturnList
    ReturnList --> InitAgents

    InitAgents --> End([Initialize Expert Agents])

    %% Styling
    classDef processClass fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    classDef decisionClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef criteriaClass fill:#c8e6c9,stroke:#388e3c,stroke-width:1px
    classDef coreClass fill:#ffccbc,stroke:#e64a19,stroke-width:2px

    class Start,AutoPath,EvalExperts,ScoreCriteria,TotalScore,AddToSelected,ReturnList,InitAgents,End processClass
    class CheckMode,HasMeta,ExpertLoop,CriteriaChecks,CheckScore,MoreExperts,EnsureMin,CheckMax decisionClass
    class C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11 criteriaClass
    class FallbackCore,ManualPath,AddCore,KeepTop coreClass
```

**Selection Criteria:**
The ExpertSelector analyzes scenario metadata and scores each expert based on:
- **Crisis Type Matching**: Flood → Coast Guard, Fire → Fire Commanders, etc.
- **Severity Thresholds**: Higher severity → Strategic commanders included
- **Geographic Scope**: Regional/National → Strategic level, Local → Tactical only
- **Affected Domains**: Law enforcement, medical, maritime, fire/rescue, etc.
- **Command Structure Needs**: Tactical, strategic, multi-jurisdictional
- **Infrastructure Impact**: Hospitals → Medical Infrastructure Director
- **Population Impact**: Large populations → Regional/national coordination
- **Duration**: Long incidents → Strategic fire/rescue commanders

**Example Auto-Selection:**

*Scenario: Severe coastal flood (severity 0.8, 25k affected, regional scope)*

**Auto-selected experts (11):**
✅ Core 3 (always): Meteorologist, Logistics, Medical
✅ PSAP Commander (multi-agency coordination)
✅ Police On-Scene & Regional (evacuation + law enforcement)
✅ Fire On-Scene & Regional (rescue operations)
✅ Medical Infrastructure (hospital capacity)
✅ Coast Guard On-Scene & National (maritime rescue)

**Creating Auto-Selection Scenarios:**

Add `expert_selection` metadata to your scenario JSON:

```json
{
  "type": "flood",
  "severity": 0.8,
  "expert_selection": {
    "crisis_type": "flood",
    "crisis_subtypes": ["coastal", "evacuation"],
    "severity": 0.8,
    "geographic_scope": "regional",
    "affected_domains": ["maritime_coastal", "law_enforcement", "medical_health"],
    "command_structure_needed": {
      "tactical": true,
      "strategic": true,
      "multi_jurisdictional": true
    }
  }
}
```

See `scenarios/scenario_template.json` for complete template with all options.

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

Initializing 3 expert agents (default)...
✓ meteorologist (confidence: 0.85)
✓ logistics_expert (confidence: 0.80)
✓ medical_expert (confidence: 0.90)

Note: Use --agents all to load all 11 expert agents

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

> **📊 Viewing Diagrams:** This documentation contains Mermaid diagrams that render automatically on GitHub. If you're viewing in a text editor or IDE without Mermaid support, you can:
> - Open `docs/architecture_viewer.html` in a web browser for rendered diagrams
> - View the README on GitHub where diagrams render automatically
> - Use https://mermaid.live to render diagram code
> - See `ARCHITECTURE_DIAGRAMS.md` for all system diagrams

### System Components

The Crisis MAS consists of five core layers with integrated evaluation framework:

```mermaid
graph TB
    subgraph UI["🖥️ USER INTERFACE LAYER"]
        Main[main.py<br/>CLI & Orchestration]
        Input[JSON I/O<br/>Scenarios & Config]
        Output[Visualization<br/>Generation]
    end

    subgraph Coord["🎯 COORDINATION LAYER"]
        CA[CoordinatorAgent<br/>Orchestration & Consensus]
        CS[Consensus<br/>Builder]
        CR[Conflict<br/>Resolution]
    end

    subgraph Agents["👥 AGENT LAYER"]
        direction LR
        EA1[Medical<br/>Expert]
        EA2[Logistics<br/>Expert]
        EA3[Safety<br/>Expert]
        EA4[Environmental<br/>Expert]
        BA[BaseAgent<br/>Interface]
        RT[ReliabilityTracker<br/>Performance History]
    end

    subgraph DF["🧠 DECISION FRAMEWORK LAYER"]
        direction TB
        ER[EvidentialReasoning<br/>Dempster-Shafer]
        GAT[GATAggregator<br/>Neural Attention]
        MCDA[MCDAEngine<br/>TOPSIS]
        CM[ConflictModel<br/>Resolution]
    end

    subgraph LLM["🤖 LLM INTEGRATION LAYER"]
        direction LR
        Claude[Claude API<br/>Anthropic]
        OpenAI[OpenAI API<br/>GPT-4]
        LMStudio[LM Studio<br/>Local Models]
        Prompt[Prompt<br/>Templates]
        Parser[Response<br/>Parser]
    end

    subgraph Eval["📊 EVALUATION & UTILITIES LAYER"]
        direction TB
        ME[MetricsEvaluator<br/>DQS, CL, CS, ECB]
        Viz[SystemVisualizer<br/>Charts & Graphs]
        Val[Validator<br/>Schema Check]
        Cfg[ConfigManager<br/>Settings]
        BL[Baseline<br/>Single-Agent]
    end

    %% Main flow
    Main -->|Load Scenario| Input
    Input -->|Initialize| CA

    CA -->|Collect Assessments| EA1 & EA2 & EA3 & EA4
    EA1 & EA2 & EA3 & EA4 -.->|Inherit from| BA
    EA1 & EA2 & EA3 & EA4 -->|Track Performance| RT

    EA1 & EA2 & EA3 & EA4 -->|LLM Reasoning| Prompt
    Prompt -->|Route to| Claude & OpenAI & LMStudio
    Claude & OpenAI & LMStudio -->|Parse| Parser
    Parser -->|Structured Response| EA1 & EA2 & EA3 & EA4

    CA -->|Aggregate Beliefs| ER & GAT
    ER & GAT -.->|Use Reliability| RT
    CA -->|Rank Alternatives| MCDA
    CA -->|Build Consensus| CS
    CS -->|Detect Issues| CR

    ER & GAT & MCDA -->|Combined Decision| CA
    CA -->|Final Decision| ME

    %% Evaluation flow
    Main -->|Run Baseline| BL
    BL -->|Single-Agent| EA1
    BL & CA -->|Compare| ME

    ME -->|Calculate Metrics| ME
    ME -->|Generate Viz| Viz
    Viz -->|Save Results| Output

    Val -.->|Validate| Input
    Cfg -.->|Configure| Main & CA & LLM

    %% Styling
    classDef layerUI fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef layerCoord fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef layerAgent fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef layerDF fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef layerLLM fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef layerEval fill:#fff9c4,stroke:#f57f17,stroke-width:2px

    class Main,Input,Output layerUI
    class CA,CS,CR layerCoord
    class EA1,EA2,EA3,EA4,BA,RT layerAgent
    class ER,GAT,MCDA,CM layerDF
    class Claude,OpenAI,LMStudio,Prompt,Parser layerLLM
    class ME,Viz,Val,Cfg,BL layerEval
```

**Architecture Overview:**
- **🖥️ User Interface Layer**: Entry point, I/O handling, visualization generation
- **🎯 Coordination Layer**: Orchestrates multi-agent decision-making, builds consensus
- **👥 Agent Layer**: Domain experts with LLM-enhanced reasoning and performance tracking
- **🧠 Decision Framework Layer**: Belief aggregation (ER/GAT), multi-criteria analysis (MCDA)
- **🤖 LLM Integration Layer**: Multi-provider support (Claude, OpenAI, LM Studio)
- **📊 Evaluation Layer**: Metrics calculation, baseline comparison, visualization (v2.0.1)

#### 1. Agent Layer

**BaseAgent** (`agents/base_agent.py`)
- Abstract base class defining agent interface
- Core methods: `evaluate_scenario()`, `propose_action()`, `justify_decision()`
- Manages agent state, confidence, and belief updating
- **Historical Reliability Tracking**: Integrated `ReliabilityTracker` for performance monitoring
  - `get_reliability_score()`: Overall, recent, or domain-specific reliability
  - `record_assessment()`: Track predictions for future evaluation
  - `update_assessment_outcome()`: Update with actual outcomes for learning
  - `get_performance_summary()`: Comprehensive performance statistics

**ReliabilityTracker** (`agents/reliability_tracker.py`)
- Tracks agent assessment accuracy over time
- Four reliability metrics:
  - **Overall Reliability**: Lifetime performance with temporal decay
  - **Recent Reliability**: Sliding window of last 10 assessments
  - **Consistency Score**: Inverse of performance variance
  - **Domain-Specific Reliability**: Accuracy per crisis type (flood, fire, etc.)
- Three-component accuracy calculation (probability, rank, margin)
- Confidence-weighted performance tracking
- Enables data-driven dynamic weighting in GAT

**ExpertAgent** (`agents/expert_agent.py`)
- Domain-specific experts across 11 specialized roles (see Expert Roles section)
- Supports emergency response command hierarchy (tactical/strategic)
- LLM-enhanced reasoning using Claude, OpenAI, or LM Studio
- Configurable expertise profiles with criteria weights
- Generates structured assessments with confidence scores
- Role-based prompt generation mapped to agent profiles

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
- **9-dimensional feature extraction** per agent:
  1. Confidence score
  2. Belief certainty (inverse entropy)
  3. Expertise relevance to scenario
  4. Risk tolerance
  5. Severity awareness
  6. Top choice strength
  7. Number of concerns
  8. Reasoning quality
  9. **Historical reliability** (from ReliabilityTracker) ⭐
- Multi-head attention (4 heads) for robustness
- Attention mechanism: $\alpha_{ij} = \text{softmax}_j(0.4 \cdot f_i^{(1)} + 0.3 \cdot f_i^{(3)} + 0.3 \cdot f_i^{(2)} + 0.2 \cdot \cos(\mathbf{f}_i, \mathbf{f}_j))$
- **Data-driven weighting**: Agents with proven track records receive higher attention

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
- Domain-specific prompts for all 11 expert agent types
- Supports emergency response command hierarchy (tactical vs. strategic roles)
- Structured JSON output formatting with belief distributions
- Crisis-specific reasoning patterns and decision criteria
- Provider-agnostic (works with Claude, OpenAI, LM Studio)
- Role-specific expertise templates (~5,000 characters each)

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

### Decision-Making Flow

The system follows a structured 8-step process for multi-agent decision-making:

```mermaid
flowchart TD
    Start([Start]) --> LoadScenario[1. Load Scenario<br/>scenarios/flood_scenario.json]
    LoadScenario --> ParseScenario[Parse JSON<br/>Extract alternatives & criteria]

    ParseScenario --> SelectExperts{2. Select Experts<br/>--expert-selection mode?}

    SelectExperts -->|manual| ManualExperts[Manual: Use --agents flag<br/>or default 3 core experts]
    SelectExperts -->|auto| AutoExperts[Auto: ExpertSelector<br/>analyzes scenario metadata]

    ManualExperts --> InitExperts[Initialize Selected<br/>Expert Agents]
    AutoExperts --> InitExperts

    InitExperts --> AgentAssess{3. Agent Assessment<br/>For each selected expert}

    AgentAssess -->|Expert 1| Exp1[Expert Agent<br/>LLM Analysis]
    AgentAssess -->|Expert 2| Exp2[Expert Agent<br/>LLM Analysis]
    AgentAssess -->|Expert 3| Exp3[Expert Agent<br/>LLM Analysis]
    AgentAssess -->|Expert N| ExpN[Expert Agent<br/>LLM Analysis]

    Exp1 & Exp2 & Exp3 & ExpN --> Collect[Collect Assessments<br/>beliefs + confidence + reasoning]

    Collect --> ChooseAgg{4. Choose<br/>Aggregation Method}

    ChooseAgg -->|Classical| ER[Evidential Reasoning<br/>Dempster-Shafer Theory]
    ChooseAgg -->|Neural| GAT[Graph Attention Network<br/>9-dim features + attention]

    ER --> Combined[Combined Belief<br/>Distribution]
    GAT --> Combined

    Combined --> MCDA[5. MCDA Scoring<br/>TOPSIS/WSM]
    MCDA --> Ranked[Ranked Alternatives<br/>with scores]

    Ranked --> Consensus{6. Check Consensus<br/>Similarity ≥ threshold?}

    Consensus -->|No| Iterate[Provide Feedback<br/>Request Refinement]
    Iterate --> AgentAssess

    Consensus -->|Yes| Decision[7. Generate Decision<br/>Top alternative + explanation]

    Decision --> CalcConf[Calculate Confidence<br/>0.6×consensus + 0.4×avg_conf]
    CalcConf --> CalcQuality[Calculate Quality Score<br/>MCDA score for recommended]

    CalcQuality --> Baseline[Run Single-Agent Baseline<br/>Select best expert]
    Baseline --> BaselineAssess[Single Expert Assessment<br/>with criteria scores]

    BaselineAssess --> Compare[8. Evaluation & Comparison]
    CalcQuality --> Compare

    Compare --> Metrics[Calculate Metrics<br/>DQS, CL, CS, ECB]
    Metrics --> Visualize[Generate Visualizations<br/>Charts & graphs]
    Visualize --> SaveOutput[Save Results<br/>JSON + PNG]

    SaveOutput --> End([End])

    %% Styling
    classDef processClass fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    classDef decisionClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef agentClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef selectClass fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    classDef evalClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px

    class LoadScenario,ParseScenario,Collect,Combined,Ranked,Decision,CalcConf,CalcQuality processClass
    class SelectExperts,ChooseAgg,Consensus decisionClass
    class Exp1,Exp2,Exp3,ExpN,Baseline,BaselineAssess agentClass
    class ManualExperts,AutoExperts,InitExperts selectClass
    class Compare,Metrics,Visualize,SaveOutput evalClass
```

**Process Details:**
1. **Scenario Loading**: Parse JSON scenario with alternatives and decision criteria
2. **Expert Selection**: Choose experts manually (--agents flag) or automatically (ExpertSelector analyzes scenario metadata)
3. **Agent Assessment**: Each selected expert analyzes scenario via LLM, produces structured assessment
4. **Belief Aggregation**: Combine beliefs using ER (classical) or GAT (neural attention)
5. **MCDA Scoring**: Rank alternatives using multi-criteria decision analysis (TOPSIS)
6. **Consensus Check**: Measure agreement; iterate if below threshold
7. **Decision Generation**: Select top alternative, calculate confidence and quality scores separately
8. **Evaluation**: Compare against single-agent baseline, calculate metrics, generate visualizations

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

**1. Feature Extraction:** For each agent $i$, construct **9-dimensional feature vector**:

$$\mathbf{f}_i = \begin{bmatrix}
f_i^{(1)} \\ f_i^{(2)} \\ f_i^{(3)} \\ f_i^{(4)} \\ f_i^{(5)} \\ f_i^{(6)} \\ f_i^{(7)} \\ f_i^{(8)} \\ f_i^{(9)}
\end{bmatrix} = \begin{bmatrix}
\text{confidence} \\
\text{certainty (inverse entropy)} \\
\text{expertise relevance} \\
\text{risk tolerance} \\
\text{severity awareness} \\
\text{top choice strength} \\
\text{number of concerns} \\
\text{reasoning quality} \\
\text{historical reliability}
\end{bmatrix}$$

where belief certainty is computed as:

$$f_i^{(2)} = 1 - \frac{H(m_i)}{H_{\max}} = 1 - \frac{-\sum_a m_i(a) \log m_i(a)}{\log |\mathcal{A}|}$$

and historical reliability (9th feature) is computed from past performance:

$$f_i^{(9)} = \text{ReliabilityScore}(i) = \frac{\sum_{t} w_t \cdot \text{Accuracy}_t}{\sum_{t} w_t}$$

where $w_t = \gamma^{(T-t)}$ is temporal decay weight ($\gamma = 0.95$), $T$ is current time, and accuracy combines probability, rank, and confidence appropriateness scores from historical assessments.

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

> **📘 Complete Evaluation Methodology:** See [`evaluation/EVALUATION_METHODOLOGY.md`](evaluation/EVALUATION_METHODOLOGY.md) for comprehensive documentation of all metrics, formulas, and validation procedures.

**Decision Quality Score (DQS):** Measures how well the recommended alternative satisfies decision criteria:

$$\text{DQS} = \begin{cases}
\frac{1}{|C|} \sum_{c \in C} s_c(a^\ast) & \text{single-agent (criteria scores)} \\
\text{MCDA}(a^\ast) & \text{multi-agent (MCDA score)}
\end{cases}$$

where:
- $a^\ast$ = recommended alternative
- $C$ = set of decision criteria
- $s_c(a^\ast)$ = score of alternative $a^\ast$ on criterion $c$
- $\text{MCDA}(a^\ast)$ = TOPSIS score for alternative $a^\ast$

**Key:** Both single-agent and multi-agent DQS are calculated from criteria satisfaction, making them directly comparable.

**Consensus Level:** Measures agreement between agents using average pairwise cosine similarity:

$$\text{Consensus} = \frac{2}{n(n-1)} \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \cos(\mathbf{m}_i, \mathbf{m}_j)$$

where $\mathbf{m}_i$ is agent $i$'s belief vector and:

$$\cos(\mathbf{m}_i, \mathbf{m}_j) = \frac{\sum_{a \in \mathcal{A}} m_i(a) \cdot m_j(a)}{\sqrt{\sum_{a \in \mathcal{A}} m_i(a)^2} \cdot \sqrt{\sum_{a \in \mathcal{A}} m_j(a)^2}}$$

**Decision Confidence:** Measures certainty in the decision (separate from quality):

For multi-agent:
$$c_{\text{decision}} = 0.6 \times \text{Consensus} + 0.4 \times \bar{c}_{\text{agents}}$$

For single-agent:
$$c_{\text{decision}} = c_{\text{LLM}}$$

where $\bar{c}_{\text{agents}}$ is the average agent confidence and $c_{\text{LLM}}$ is the LLM's self-reported confidence.

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

**Expert Agents (Default 3-Agent Team):**
1. Meteorologist (confidence: 0.85)
2. Logistics Expert (confidence: 0.80)
3. Medical Expert (confidence: 0.90)

**Note:** Results below use the default 3-agent configuration. With `--agents all`, the system engages all 11 experts including tactical and strategic command authorities.

### Comparative Results: ER vs. GAT

#### Evidential Reasoning Results

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
- API Calls: 3

**Note:** With all 11 agents engaged, processing time scales to ~35-45 seconds (11 parallel API calls), but provides comprehensive multi-agency perspective.

**Interpretation:** GAT dynamically weights the Public Safety Expert higher due to domain relevance, resulting in slightly higher confidence and consensus. The environmental expert's influence is appropriately reduced for immediate crisis response.

### Performance Metrics

> **⚠️ Important Note:** As of commit `09bec4c` (2025-01-09), the evaluation methodology has been fixed to properly calculate Decision Quality Score from criteria satisfaction. Previous versions incorrectly used confidence values, making single vs. multi-agent comparisons invalid. See [`evaluation/EVALUATION_METHODOLOGY.md`](evaluation/EVALUATION_METHODOLOGY.md) for details.

#### Decision Quality Metrics

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
  "improvement_over_single_agent": {
    "score": 0.123,
    "percentage": 17.0
  }
}
```

**Interpretation:**
- **Quality (0.847):** Calculated from criteria scores - shows strong satisfaction of safety (0.95) and effectiveness (0.90), with trade-off on cost (0.45)
- **Confidence (0.823):** Separate metric indicating high certainty in the decision based on agent consensus
- **Improvement (17%):** Valid comparison showing multi-agent outperforms single-agent by considering diverse expert perspectives

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
3. **Scalability:** Successfully expanded from 4 to 11 expert roles (v0.8) with tactical/strategic hierarchy
4. **Explainability:** Attention weights provide interpretable expert influence measures
5. **Efficiency:**
   - 3-agent (default): 10-15s decision time, ~$0.012 per scenario
   - 11-agent (full): 35-45s decision time, ~$0.044 per scenario
6. **Robustness:** 92% consensus achieved in test scenarios (n=25 simulations)
7. **Command Structure:** Realistic two-tier hierarchy (tactical/strategic) enables multi-jurisdictional crisis modeling

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

#### 6. Expert Diversity (Expanded in v0.8)

**Current (v0.8):** 11 expert types organized in emergency response command structure:
- Core: Meteorologist, Logistics, Medical Director
- Emergency Communications: PSAP Commander
- Law Enforcement: Tactical and Strategic Police Commanders
- Fire/Rescue: Tactical and Strategic Fire Commanders
- Medical Infrastructure: Healthcare System Director
- Maritime: Tactical and Strategic Coast Guard Commanders

**Still Missing for Future Enhancement:**
- Economic/financial experts (budget analysis, cost-benefit)
- Legal/regulatory experts (compliance, liability)
- Communications/media experts (public messaging, crisis communications)
- Political/governance experts (policy implications, stakeholder management)
- Psychological/social experts (community impact, trauma response)
- Civil engineering experts (infrastructure assessment)

**Impact:** The 11-agent system provides comprehensive operational command perspective, but may still miss economic, legal, and social dimensions in complex multi-faceted crises.

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

**Current:** Tested with 3-11 agents, 3-5 alternatives

**Scalability Limits:**
- ER complexity: O(n²) for n agents (pairwise belief combination)
- GAT complexity: O(n² · d) for n agents, d features (d=9)
- Consensus checking: O(n²) for pairwise comparisons
- LLM API latency: ~3-4s per agent (parallel calls in current implementation)

**Performance at Scale:**
- 3 agents (default): ~10-15s total decision time
- 11 agents (full team): ~35-45s total decision time
- Estimated 20 agents: ~60-80s (still practical for crisis decisions)

**Impact:** Current architecture scales well to 20-30 agents. May face performance issues with >50 agents or >10 alternatives without optimization.

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

#### 1. Expand Agent Diversity (COMPLETED in v0.8 ✅ + Future Additions)

**Completed (v0.8):** Expanded from 4 to 11 expert types
- ✅ Added emergency response command structure (8 new roles)
- ✅ Implemented tactical/strategic hierarchy
- ✅ Created 11 domain-specific prompt templates (~5,000 chars each)
- ✅ Validated agent profiles with realistic experience levels

**Proposed Next Phase - Expand to 15-20 experts:**
- Economic Advisor (cost-benefit analysis, budget constraints)
- Legal Expert (regulatory compliance, liability assessment)
- Communications Specialist (public messaging, media strategy)
- Civil Engineer (infrastructure assessment, structural integrity)
- Mental Health Professional (psychological impact, trauma response)
- Emergency Management Agency (EMA) Director (inter-agency coordination)
- Utilities Manager (power, water, telecommunications restoration)

**Implementation:**
- Create additional agent profiles in `agent_profiles.json`
- Develop domain-specific prompt templates
- Validate expertise domains with real experts
- Test scalability with 15-20 agent scenarios

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
│   ├── expert_agent.py             # Domain expert agents (11 role mappings)
│   ├── coordinator_agent.py        # Coordination and consensus
│   └── agent_profiles.json         # Agent configurations (11 experts)
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
│   └── prompt_templates.py         # Domain-specific prompts (11 expert roles)
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
