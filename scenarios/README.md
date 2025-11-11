# Crisis Scenarios - Documentation

## Overview

This folder contains crisis scenario definitions and decision criteria used by the multi-agent system for decision-making evaluation. Scenarios are defined in JSON format and provide the test cases for evaluating the system's crisis response capabilities.

### Purpose

**Why This Folder Exists:**
1. **Standardized Test Cases**: Provides consistent scenarios for evaluating multi-agent decision-making
2. **Reproducible Experiments**: Same scenarios can be used across different runs for comparison
3. **Realistic Crisis Modeling**: Captures complexity of real-world emergency response situations
4. **Extensibility**: Easy to add new crisis types and scenarios without code changes
5. **Validation Framework**: Ensures scenarios have required structure before processing

### Key Components

- **Scenario Files**: JSON files defining specific crisis situations (`*_scenario.json`)
- **Criteria Weights**: Decision criteria and weights for MCDA analysis (`criteria_weights.json`)
- **Scenario Loader**: Python module for loading and validating scenarios (`scenario_loader.py`)

---

## Scenario JSON Schema

### Required Fields

Every scenario MUST contain these fields:

```json
{
  "id": "unique_scenario_identifier",
  "type": "crisis_category",
  "description": "Detailed narrative of the crisis situation"
}
```

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string | Unique identifier for this scenario | `"flood_001"`, `"earthquake_002"` |
| `type` | string | Crisis category/classification | `"flood"`, `"earthquake"`, `"fire"`, `"pandemic"` |
| `description` | string | Detailed narrative explaining the situation | `"Urban flooding affecting residential area..."` |

### Recommended Fields

These fields enhance scenario realism and enable better agent reasoning:

```json
{
  "name": "Human-readable scenario name",
  "severity": 0.7,
  "affected_population": 10000,
  "location": {
    "region": "Geographic area name",
    "coordinates": {"lat": 40.7128, "lon": -74.0060}
  },
  "tags": ["urban", "infrastructure", "time_critical"],
  "casualties": 5,
  "infrastructure_damage": true
}
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `name` | string | - | Short, descriptive scenario name |
| `severity` | float | 0.0-1.0 | Crisis severity level (0=minor, 1=catastrophic) |
| `affected_population` | integer | ≥0 | Number of people impacted |
| `location.region` | string | - | Geographic region or city name |
| `location.coordinates` | object | lat: -90 to 90, lon: -180 to 180 | GPS coordinates |
| `tags` | array of strings | - | Categorization tags for filtering |
| `casualties` | integer | ≥0 | Number of casualties (deaths + serious injuries) |
| `infrastructure_damage` | boolean | - | Whether critical infrastructure is damaged |

### Critical Field: Available Actions

The `available_actions` field defines response alternatives for the crisis. This is the most important field as it provides the options agents must evaluate.

```json
{
  "available_actions": [
    {
      "id": "action_1",
      "name": "Immediate Evacuation",
      "description": "Evacuate all residents within 2km radius",
      "required_resources": ["emergency_vehicles", "temporary_shelters", "personnel"],
      "estimated_duration": "2-4 hours",
      "risk_level": 0.3,
      "criteria_scores": {
        "effectiveness": 0.85,
        "safety": 0.90,
        "speed": 0.75,
        "cost": 0.40,
        "public_acceptance": 0.70
      }
    }
  ]
}
```

#### Action Schema

Each action in `available_actions` array must contain:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ YES | Unique action identifier (e.g., `"action_1"`, `"action_rescue_operations"`) |
| `name` | string | Recommended | Human-readable action name |
| `description` | string | Recommended | Detailed explanation of what this action involves |
| `required_resources` | array of strings | Optional | Resources needed (personnel, equipment, facilities) |
| `estimated_duration` | string | Optional | Time estimate for completion |
| `risk_level` | float (0.0-1.0) | Optional | Risk assessment (0=low risk, 1=high risk) |
| `criteria_scores` | object | **Critical** | Scores for each decision criterion (see below) |

#### Criteria Scores

The `criteria_scores` object maps decision criteria to performance scores for this action:

```json
"criteria_scores": {
  "effectiveness": 0.85,    // How well does this solve the problem? (0-1)
  "safety": 0.90,          // How safe is this for responders/public? (0-1)
  "speed": 0.75,           // How quickly can this be executed? (0-1)
  "cost": 0.40,            // Cost-effectiveness (0-1, higher = cheaper)
  "public_acceptance": 0.70 // Public support level (0-1)
}
```

**Important Notes:**
- All scores should be in range 0.0 to 1.0
- Higher score = better performance on that criterion
- For "cost", higher score means MORE cost-effective (cheaper)
- These scores are used by the MCDA engine for multi-criteria ranking
- Criteria names must match those defined in `criteria_weights.json`

### Optional Fields

```json
{
  "constraints": {
    "time_critical": true,
    "resource_limitations": ["limited_personnel", "limited_equipment"],
    "weather_conditions": "severe_storm",
    "accessibility": "limited"
  },
  "metadata": {
    "created_date": "2025-01-09",
    "author": "Emergency Planning Team",
    "version": "1.0",
    "notes": "Based on 2023 urban flood case study"
  }
}
```

---

## Criteria Weights System

The `criteria_weights.json` file defines the decision criteria and their relative importance for MCDA (Multi-Criteria Decision Analysis).

### Structure

```json
{
  "criteria": [
    {
      "id": "effectiveness",
      "name": "Effectiveness",
      "description": "How well the action addresses the crisis",
      "weight": 0.30,
      "optimization": "maximize"
    },
    {
      "id": "safety",
      "name": "Safety",
      "description": "Safety for responders and affected population",
      "weight": 0.25,
      "optimization": "maximize"
    }
  ]
}
```

### Criteria Fields

| Field | Description | Values |
|-------|-------------|--------|
| `id` | Criterion identifier (must match keys in action `criteria_scores`) | String |
| `name` | Human-readable criterion name | String |
| `description` | Detailed explanation of what this criterion measures | String |
| `weight` | Relative importance (must sum to 1.0 across all criteria) | Float (0.0-1.0) |
| `optimization` | Whether higher or lower scores are better | `"maximize"` or `"minimize"` |

### Default Criteria

The system uses these five standard criteria:

1. **Effectiveness** (30%): How well does the action solve the problem?
2. **Safety** (25%): How safe is the action for responders and public?
3. **Speed** (20%): How quickly can the action be executed?
4. **Cost** (15%): How cost-effective is the action?
5. **Public Acceptance** (10%): How well will the public accept this action?

---

## Creating New Scenarios

### Method 1: Using the Template Generator

The `ScenarioLoader` class provides a template generator:

```python
from scenarios.scenario_loader import ScenarioLoader

loader = ScenarioLoader()
loader.create_scenario_template("my_new_scenario.json")
```

This creates a fully-structured template you can edit.

### Method 2: Manual Creation

1. **Copy an existing scenario** as a starting point
2. **Update required fields**: Change `id`, `type`, and `description`
3. **Customize scenario details**: Adjust severity, location, casualties, etc.
4. **Define available actions**: Create 2-5 response alternatives
5. **Set criteria scores**: Assign scores (0.0-1.0) for each action on each criterion
6. **Validate**: Run validation before use

### Validation

Before using a scenario, validate its structure:

```python
from scenarios.scenario_loader import ScenarioLoader

loader = ScenarioLoader()
try:
    scenario = loader.load_scenario("my_scenario.json")
    print("✓ Scenario valid!")
except ValueError as e:
    print(f"✗ Validation error: {e}")
```

### Common Validation Errors

| Error Message | Cause | Fix |
|---------------|-------|-----|
| `Scenario missing required field: id` | Missing `id` field | Add unique `"id": "scenario_XXX"` |
| `Scenario missing required field: type` | Missing `type` field | Add `"type": "crisis_category"` |
| `Scenario missing required field: description` | Missing `description` | Add detailed description string |
| `available_actions must be a list` | Actions not in array format | Wrap actions in `[ ]` brackets |
| `Action 0 missing 'id' field` | Action object missing ID | Add `"id": "action_X"` to each action |

---

## Example Scenarios

### Example 1: Urban Flood Scenario

```json
{
  "id": "flood_urban_001",
  "type": "flood",
  "name": "Urban Flash Flood Emergency",
  "description": "Heavy rainfall has caused flash flooding in a residential area with 5,000 residents. Water levels rising rapidly, threatening homes and critical infrastructure.",
  "severity": 0.75,
  "affected_population": 5000,
  "location": {
    "region": "Downtown Riverside District",
    "coordinates": {"lat": 40.7128, "lon": -74.0060}
  },
  "casualties": 3,
  "infrastructure_damage": true,
  "available_actions": [
    {
      "id": "action_immediate_evacuation",
      "name": "Immediate Mass Evacuation",
      "description": "Evacuate all 5,000 residents to emergency shelters",
      "required_resources": ["buses", "shelters", "personnel"],
      "estimated_duration": "3-4 hours",
      "risk_level": 0.4,
      "criteria_scores": {
        "effectiveness": 0.85,
        "safety": 0.90,
        "speed": 0.70,
        "cost": 0.30,
        "public_acceptance": 0.65
      }
    },
    {
      "id": "action_sandbagging",
      "name": "Deploy Flood Barriers",
      "description": "Deploy sandbags and temporary barriers to protect infrastructure",
      "required_resources": ["sandbags", "volunteers", "equipment"],
      "estimated_duration": "2-3 hours",
      "risk_level": 0.6,
      "criteria_scores": {
        "effectiveness": 0.60,
        "safety": 0.70,
        "speed": 0.85,
        "cost": 0.70,
        "public_acceptance": 0.80
      }
    },
    {
      "id": "action_hybrid_response",
      "name": "Hybrid Approach",
      "description": "Evacuate high-risk areas while protecting infrastructure in lower-risk zones",
      "required_resources": ["buses", "sandbags", "personnel", "equipment"],
      "estimated_duration": "3-5 hours",
      "risk_level": 0.5,
      "criteria_scores": {
        "effectiveness": 0.80,
        "safety": 0.85,
        "speed": 0.65,
        "cost": 0.45,
        "public_acceptance": 0.75
      }
    }
  ],
  "constraints": {
    "time_critical": true,
    "resource_limitations": ["limited_shelters", "limited_personnel"],
    "weather_conditions": "heavy_rain_continuing",
    "accessibility": "roads_partially_flooded"
  }
}
```

### Example 2: Earthquake Response Scenario

```json
{
  "id": "earthquake_001",
  "type": "earthquake",
  "name": "7.2 Magnitude Earthquake",
  "description": "Major earthquake struck urban area causing widespread building damage and trapping victims in collapsed structures.",
  "severity": 0.90,
  "affected_population": 50000,
  "location": {
    "region": "Metro City Center",
    "coordinates": {"lat": 37.7749, "lon": -122.4194}
  },
  "casualties": 47,
  "infrastructure_damage": true,
  "available_actions": [
    {
      "id": "action_search_rescue",
      "name": "Urban Search and Rescue",
      "description": "Deploy specialized teams to locate and extract trapped victims",
      "required_resources": ["rescue_teams", "search_dogs", "heavy_equipment"],
      "estimated_duration": "48-72 hours",
      "risk_level": 0.7,
      "criteria_scores": {
        "effectiveness": 0.95,
        "safety": 0.60,
        "speed": 0.50,
        "cost": 0.35,
        "public_acceptance": 0.95
      }
    },
    {
      "id": "action_area_stabilization",
      "name": "Area Stabilization",
      "description": "Secure damaged buildings to prevent further collapse",
      "required_resources": ["engineering_teams", "construction_equipment", "materials"],
      "estimated_duration": "24-36 hours",
      "risk_level": 0.5,
      "criteria_scores": {
        "effectiveness": 0.70,
        "safety": 0.80,
        "speed": 0.75,
        "cost": 0.60,
        "public_acceptance": 0.70
      }
    },
    {
      "id": "action_medical_response",
      "name": "Mass Casualty Medical Response",
      "description": "Deploy field hospitals and medical teams to treat injured",
      "required_resources": ["medical_teams", "field_hospitals", "supplies"],
      "estimated_duration": "12-24 hours",
      "risk_level": 0.4,
      "criteria_scores": {
        "effectiveness": 0.85,
        "safety": 0.85,
        "speed": 0.90,
        "cost": 0.50,
        "public_acceptance": 0.90
      }
    }
  ]
}
```

---

## Available Scenarios

### Current Scenarios in Repository

| Filename | Crisis Type | Severity | Location | Affected Population | Description |
|----------|-------------|----------|----------|---------------------|-------------|
| `flood_scenario.json` | flood | 0.8 | Karditsa, Thessaly, Greece | 15,000 | Pamisos River overflow causing residential and agricultural flooding |
| `forest_fire_evia.json` | wildfire | 0.9 | North Evia island, Greece | 8,000 | Major forest fire with multiple fronts, village evacuations, aerial firefighting |
| `ammonia_leak_elefsina.json` | hazmat | 0.85 | Elefsina, Attica, Greece | 12,000 | 50-ton anhydrous ammonia tank rupture, toxic cloud, HAZMAT Level A response |

### Greek Crisis Scenarios - Detailed Overview

#### 1. Karditsa Flood Emergency (`flood_scenario.json`)

**Scenario Context:**
- **Location:** Karditsa city, Thessaly region, Central Greece
- **Coordinates:** 39.3644°N, 21.9211°E
- **Trigger:** Heavy rainfall and Pamisos River overflow
- **Severity:** 0.8 (High)
- **Affected Population:** 15,000 residents
- **Casualties:** 5 reported
- **Infrastructure Impact:** Residential areas, agricultural lands, power/water systems threatened

**Available Response Actions:**
1. **Immediate Mass Evacuation** - Large-scale resident evacuation to emergency shelters
2. **Deploy Flood Barriers** - Sandbag deployment and temporary barrier installation
3. **Prioritized Rescue Operations** - Focus on trapped individuals and medical emergencies
4. **Shelter-in-Place with Supply Distribution** - Instruct upper-floor sheltering with monitoring
5. **Hybrid Approach** - Combined rescue, selective evacuation, and barrier deployment

**Usage:**
```bash
python main.py --scenario scenarios/flood_scenario.json --expert-selection auto
```

**Relevant Greek Experts:**
- Πολιτική Προστασία (Civil Protection)
- ΕΚΑΒ (Emergency Medical Service)
- ΕΛΑΣ (Hellenic Police)
- Πυροσβεστική (Fire Service)
- Λιμενικό Σώμα (Coast Guard - for water rescue)

---

#### 2. Evia Forest Fire Emergency (`forest_fire_evia.json`)

**Scenario Context:**
- **Location:** North Evia (Euboia) island, Central Greece
- **Coordinates:** 38.9231°N, 23.6578°E
- **Trigger:** Forest fire with strong winds (40 km/h) and low humidity (35%)
- **Severity:** 0.9 (Very High)
- **Affected Population:** 8,000 residents across multiple villages
- **Casualties:** 2 deaths, 5 missing
- **Burned Area:** 12,000 hectares
- **Active Fire Fronts:** 4 simultaneous fronts
- **Fire Spread Rate:** 3-5 km/h towards inhabited areas

**Available Response Actions:**
1. **Immediate Village Evacuation** - Urgent bus/boat evacuation of threatened villages
2. **Aerial Firefighting Campaign** - Canadair CL-415 and Chinook helicopter water drops
3. **Ground Firefighting with Firebreaks** - Direct combat with fire trucks and bulldozers
4. **Controlled Backburning Operations** - Strategic fuel removal through controlled burns
5. **Combined Air-Ground Assault with Evacuation** - Simultaneous aerial/ground operations

**Usage:**
```bash
python main.py --scenario scenarios/forest_fire_evia.json --expert-selection auto
```

**Relevant Greek Experts:**
- Πυροσβεστική (Hellenic Fire Service) - Tactical and Regional commanders
- Λιμενικό Σώμα (Coast Guard) - Coastal evacuation support
- Μετεωρολόγος (Meteorologist) - Wind/weather forecasting
- ΕΛΑΣ (Police) - Evacuation coordination, road closures
- ΕΚΑΒ (Emergency Medical) - Burn victims, smoke inhalation treatment

**Special Resources:**
- **Canadair CL-415** water bombers (Greek Air Force)
- **Chinook CH-47** helicopters with Bambi buckets
- Ground fire crews and bulldozers
- Coastal evacuation boats

---

#### 3. Elefsina Ammonia Leak Emergency (`ammonia_leak_elefsina.json`)

**Scenario Context:**
- **Location:** Elefsina (Eleusis) industrial zone, Attica, near Athens
- **Coordinates:** 38.0411°N, 23.5461°E
- **Trigger:** 50-ton anhydrous ammonia (NH3) storage tank rupture at industrial facility
- **Severity:** 0.85 (Very High)
- **Affected Population:** 12,000 (industrial workers + residential areas)
- **Casualties:** 8 deaths, 45 injured with chemical exposure
- **Chemical:** Anhydrous Ammonia (NH3) - UN1005, Class 2.3 Toxic Gas + Class 8 Corrosive
- **IDLH Level:** 300 ppm (Immediately Dangerous to Life or Health)
- **Current Air Quality:**
  - 500m downwind: 250 ppm (dangerous)
  - 1km downwind: 120 ppm (evacuation level)
  - 2km downwind: 45 ppm (irritation level)
- **Leak Rate:** ~500 kg/hour from ruptured tank valve
- **Wind Direction:** Southeast towards residential areas (expected to shift northeast in 4 hours)

**Available Response Actions:**
1. **Immediate Downwind Evacuation** - Urgent evacuation within 2km radius, priority: schools/hospitals
2. **HAZMAT Team Source Containment** - Level A suit teams isolate and seal leak source
3. **Water Curtain Vapor Suppression** - Fire trucks create water curtains to absorb NH3 vapors
4. **Shelter-in-Place with Sealing** - Residents seal indoors, turn off ventilation, await instructions
5. **Integrated Response** - Simultaneous containment + water curtain + selective evacuation

**Usage:**
```bash
python main.py --scenario scenarios/ammonia_leak_elefsina.json --expert-selection auto
```

**Relevant Greek Experts:**
- Πυροσβεστική HAZMAT Teams (Fire Service specialized units) - Level A containment
- ΕΚΑΒ (Emergency Medical) - Chemical exposure treatment, decontamination
- ΕΛΑΣ (Police) - Evacuation coordination, perimeter security
- Μετεωρολόγος (Meteorologist) - Wind direction forecasting, plume modeling
- Περιβαλλοντολόγος (Environmental Scientist) - Environmental impact assessment

**Special Requirements:**
- **Level A HAZMAT suits** (fully encapsulated, SCBA)
- **Decontamination units** for exposed personnel
- **Water supply** for continuous water curtain operations
- **Air quality monitoring** equipment
- **Specialized ammonia detection** instruments

**Health Effects of Ammonia Exposure:**
- **25-50 ppm:** Irritation of eyes, nose, throat
- **100 ppm:** Severe irritation, should not be tolerated
- **300 ppm:** IDLH - immediate danger to life
- **500 ppm:** Rapidly fatal without treatment

---

### Using Greek Scenarios

All three scenarios feature:
- **Authentic Greek locations** with real coordinates
- **Greek emergency response structure** (ΕΛΑΣ, Πυροσβεστική, ΕΚΑΒ, Λιμενικό, Πολιτική Προστασία)
- **Greek expert names and ranks** (e.g., Ταξίαρχος, Πυραγός, Πλωτάρχης)
- **Realistic resource constraints** based on Greek emergency response capabilities
- **Local geography and infrastructure** considerations

**Automatic Expert Selection:**
The system can automatically select appropriate Greek experts based on scenario metadata:
```bash
python main.py --scenario scenarios/forest_fire_evia.json --expert-selection auto
```

This will intelligently choose relevant experts like Fire Service commanders, Coast Guard for coastal evacuation, Meteorologist for wind analysis, etc.

---

## Integration with Multi-Agent System

### How Scenarios Are Used

1. **Loading**: `main.py` uses `ScenarioLoader.load_scenario()` to load scenario JSON
2. **Validation**: Scenario structure is validated before processing begins
3. **Agent Distribution**: Scenario is distributed to all expert agents for evaluation
4. **Criteria Extraction**: `criteria_scores` are extracted for MCDA analysis
5. **Decision Making**: Agents evaluate actions and coordinator aggregates recommendations
6. **Evaluation**: Metrics system uses scenario ground truth for evaluation

### Data Flow

```
scenarios/*.json
    ↓
ScenarioLoader.load_scenario()
    ↓
Validation (_validate_scenario)
    ↓
main.py (orchestration)
    ↓
Expert Agents (evaluation)
    ↓
Coordinator (aggregation)
    ↓
MCDA Engine (uses criteria_scores)
    ↓
Final Decision
    ↓
Metrics Evaluation
```

### Key Integration Points

**File: `main.py`**
```python
# Load scenario
scenario_loader = ScenarioLoader(scenarios_dir="scenarios")
scenario = scenario_loader.load_scenario("flood_scenario.json")

# Load criteria
criteria = scenario_loader.load_criteria_weights()
```

**File: `decision_framework/mcda_engine.py`**
```python
# Uses action criteria_scores for TOPSIS ranking
for action in scenario['available_actions']:
    scores = action.get('criteria_scores', {})
    # Apply MCDA algorithm...
```

**File: `evaluation/metrics.py`**
```python
# Extracts criteria scores for decision quality calculation
criteria_scores = decision.get('criteria_scores', {})
# Calculate DQS from criteria satisfaction...
```

---

## Best Practices

### Scenario Design

1. **Realistic Parameters**: Use realistic population numbers, severity levels, and timelines
2. **Balanced Actions**: Provide 3-5 alternatives with different trade-offs
3. **Clear Trade-offs**: Ensure actions have distinct strengths/weaknesses
4. **Consistent Scoring**: Score all actions on all criteria (no missing values)
5. **Validation Testing**: Test scenario before use in experiments

### Criteria Scores

1. **Normalized Range**: Always use 0.0-1.0 range
2. **Relative Scoring**: Scores should be relative within the scenario
3. **Differentiation**: Avoid giving all actions similar scores
4. **Justifiable**: Scores should be defensible based on action description
5. **Consistency**: Similar actions across scenarios should have similar scoring patterns

### Documentation

1. **Descriptive IDs**: Use meaningful IDs like `flood_urban_001` not `scenario1`
2. **Detailed Descriptions**: Write clear narratives explaining the crisis situation
3. **Metadata**: Include creation date, author, and version information
4. **Comments**: Add notes in metadata explaining scenario design rationale

---

## Troubleshooting

### Scenario Won't Load

**Problem**: `FileNotFoundError: Scenario file not found`
- **Solution**: Check file path and ensure file is in `scenarios/` directory
- **Solution**: Verify filename exactly matches (case-sensitive)

### Validation Fails

**Problem**: `ValueError: Scenario missing required field`
- **Solution**: Check error message for specific field name
- **Solution**: Add missing field or fix field name spelling

**Problem**: `ValueError: available_actions must be a list`
- **Solution**: Wrap actions in array brackets: `"available_actions": [...]`

**Problem**: `ValueError: Action 0 missing 'id' field`
- **Solution**: Add unique `"id"` field to each action object

### MCDA Errors

**Problem**: MCDA engine fails with missing criteria
- **Solution**: Ensure all criteria in `criteria_weights.json` are present in action `criteria_scores`
- **Solution**: Check for typos in criterion names (e.g., "effeciveness" vs "effectiveness")

### Evaluation Errors

**Problem**: Decision quality shows 0.000
- **Solution**: Verify `criteria_scores` are properly formatted
- **Solution**: Check that action IDs match between scenario and decision results

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-09 | Initial comprehensive documentation |

---

## References

### Related Documentation

- `scenario_loader.py`: Implementation details and validation logic
- `criteria_weights.json`: Decision criteria definitions
- `../README.md`: Overall system architecture
- `../evaluation/EVALUATION_METHODOLOGY.md`: How scenarios are evaluated

### External Resources

- **Crisis Response Planning**: FEMA Emergency Response Framework
- **MCDA Methods**: TOPSIS methodology for multi-criteria ranking
- **JSON Schema**: https://json-schema.org/ for validation specifications
