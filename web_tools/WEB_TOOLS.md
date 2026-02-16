# Web Tools

The Crisis MAS system includes a comprehensive web-based interface for managing expert profiles, incident handling protocols, and crisis scenarios. The web tools provide a user-friendly way to create and edit data without directly manipulating JSON files.

## Overview

**Location:** `web_tools/` directory

**Features:**
- Expert profile management (14 agents: 13 experts + 1 coordinator)
- Incident handling protocol editor (Q&A pairs for LLM training)
- Crisis scenario builder with OpenStreetMap integration
- Configurable data source paths
- Import/export functionality

## Starting the Web Interface

```bash
cd web_tools
python app.py
```

The web interface will be available at `http://localhost:5000`

## Web Tools Components

### 1. Expert Profiles (`/experts`)

**Purpose:** Manage the 14 agent profiles used by the multi-agent system

**Features:**
- View all 13 expert agents + 1 coordinator
- Add/Edit/Delete expert profiles
- Configure expertise, experience, risk tolerance, weight preferences
- Support for both nested (agent_profiles.json) and flat JSON formats

**Data Source:** `agents/agent_profiles.json` (default, configurable in Settings)

**Expert Agents Include:**
- Meteorologist, Medical Expert, Logistics Coordinator
- Public Safety Officer, Environmental Scientist
- PSAP Commander, Police (On-Scene & Regional)
- Fire-Brigade (On-Scene & Regional)
- Medical Infrastructure Director
- Coast Guard (On-Scene & National)

### 2. Incident Handling Protocols (`/protocols`)

**Purpose:** Create Q&A pairs for LLM training data

**What They Are:**
- Question & Answer pairs representing expert knowledge
- Used to train LLM agents on emergency response procedures
- Categorized by emergency type (firefighting, disaster, medical, etc.)

**Features:**
- Create/Edit/Delete protocol Q&A pairs
- Categorize by emergency type
- Associate with expert profiles
- Multi-language support

**Data Source:** `web_tools/data/scenarios.json` (configurable in Settings)

**Training Methodology:** See [LLM Training Documentation](../LLM%20Training/)

### 3. Crisis Scenarios (`/crisis-scenarios`)

**Purpose:** Build crisis scenarios for multi-agent simulations

**What They Are:**
- Crisis situations with multiple response alternatives
- Used by the multi-agent system for decision-making tests
- Include location, severity, available actions, and expert selection metadata

**Features:**
- Scenario browser with OpenStreetMap visualization
- Advanced scenario builder with template library
- Action builder with criteria scoring (effectiveness, safety, speed, cost, public acceptance)
- **AI-Powered Action Suggestion:** Leverages incident handling protocols to suggest response actions
- Expert selection metadata for automatic agent selection
- Import/Export scenarios
- Validation against core framework structure

**Data Source:** `scenarios/` directory (configurable in Settings)

**Integration with Protocols:**
The scenario builder includes AI-powered action suggestion that analyzes incident handling protocols to recommend response actions:
1. Select crisis type (e.g., wildfire, flood)
2. Click "AI Suggest from Protocols"
3. System analyzes relevant protocols and extracts actionable steps
4. Displays suggested actions with estimated criteria scores
5. Select desired actions to add to your scenario

### 4. Settings (`/settings`)

**Purpose:** Configure data source file paths

**Configurable Paths:**
- **Expert Profiles File:** Path to agent_profiles.json (default: `agents/agent_profiles.json`)
- **Protocols File:** Path to incident handling protocols (default: `web_tools/data/scenarios.json`)
- **Scenarios Directory:** Path to crisis scenarios folder (default: `scenarios/`)

**Features:**
- Real-time statistics for each data source (count, format, categories)
- Quick-select buttons for common paths
- Path validation and error checking
- Persistent configuration in `web_tools/data/config.json`

## Web Tools Architecture

```
web_tools/
├── app.py                          # Flask application
├── protocol_integration.py         # Protocol-to-scenario integration
├── templates/                      # HTML templates
│   ├── index.html                  # Dashboard
│   ├── experts.html                # Expert list
│   ├── expert_form.html            # Expert editor
│   ├── scenarios.html              # Protocols list
│   ├── scenario_form.html          # Protocol editor
│   ├── crisis_scenarios.html       # Crisis scenario browser
│   ├── crisis_scenario_view.html   # Scenario viewer with map
│   ├── crisis_scenario_form_enhanced.html  # Advanced builder
│   └── settings.html               # Configuration
├── static/
│   ├── js/
│   │   └── crisis_scenario_builder.js  # Interactive builder logic
│   └── css/
│       └── style.css               # Custom styles
└── data/
    ├── scenarios.json              # Incident protocols (LLM training)
    ├── experts.json                # Legacy expert profiles
    └── config.json                 # User configuration
```

## Use Cases

**For Scenario Creation:**
1. Create new crisis scenarios using the web builder
2. Use AI suggestions to generate response actions from protocol knowledge
3. Add location data with interactive map
4. Validate structure before saving

**For LLM Training:**
1. Create incident handling protocols (Q&A pairs)
2. Associate with expert profiles
3. Export for training data preparation

**For Expert Configuration:**
1. Add new expert agents to the system
2. Adjust risk tolerance and weight preferences
3. Configure expertise tags for automatic selection

**For Data Management:**
1. Switch between different data sources
2. Import/export data for backup or sharing
3. View statistics and validate data integrity

## Further Documentation

For detailed documentation see:
- `web_tools/README.md` - Complete web tools guide
- `web_tools/FEATURES.md` - Advanced features documentation
