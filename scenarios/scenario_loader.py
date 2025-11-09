"""
Scenario Loader Module - Crisis Scenario Management and Validation

OBJECTIVE:
This module provides the interface for loading, validating, and managing crisis scenarios
from JSON files. It ensures that scenarios have the correct structure and required fields
before being used by the multi-agent system for decision-making.

WHY THIS EXISTS:
1. **Centralized Scenario Management**: Single point for loading all scenario data
2. **Validation**: Ensures scenarios have required structure before system processing
3. **File Abstraction**: Hides file I/O complexity from the rest of the system
4. **Template Generation**: Provides scaffolding for creating new scenarios
5. **Error Prevention**: Catches malformed scenarios early, preventing downstream failures

KEY RESPONSIBILITIES:
- Load scenario JSON files and parse into dictionaries
- Validate scenario structure (required fields, correct types, valid actions)
- Load decision criteria weights for MCDA analysis
- List available scenarios in directory
- Generate scenario templates for new crisis types
- Create human-readable scenario summaries

EXPECTED INPUTS:
- Scenario JSON files following standardized schema (see SCENARIO_SCHEMA below)
- Criteria weights JSON defining decision criteria and weights
- Scenarios directory path (default: same directory as this module)

EXPECTED OUTPUTS:
- Validated scenario dictionaries ready for agent processing
- Criteria weights dictionaries for MCDA engine
- Scenario templates for new crisis creation
- Human-readable scenario summaries

SCENARIO JSON SCHEMA:
A valid scenario must contain:

REQUIRED FIELDS:
- id (string): Unique scenario identifier (e.g., "flood_001", "earthquake_002")
- type (string): Crisis type category (e.g., "flood", "earthquake", "fire", "pandemic")
- description (string): Detailed narrative of the crisis situation

RECOMMENDED FIELDS:
- name (string): Human-readable scenario name
- severity (float): Crisis severity on scale 0.0-1.0
- affected_population (int): Number of people impacted
- location (dict): Geographic information with region and coordinates
- tags (list): Categorization tags for filtering
- casualties (int): Number of casualties
- infrastructure_damage (boolean): Whether infrastructure is damaged

CRITICAL FIELD:
- available_actions (list): List of response alternatives, each containing:
  * id (string): Unique action identifier [REQUIRED]
  * name (string): Action name
  * description (string): Detailed action description
  * required_resources (list): Resources needed
  * estimated_duration (string): Time estimate
  * risk_level (float): Risk assessment 0.0-1.0
  * criteria_scores (dict): Scores for MCDA criteria {criterion: score}

OPTIONAL FIELDS:
- constraints (dict): Operational constraints (time_critical, resource_limitations, etc.)
- metadata (dict): Creation info (author, date, version)

DESIGN PATTERNS:
- **Singleton-like Loader**: One instance manages all scenario loading
- **Validation on Load**: Fail-fast principle for malformed data
- **Path Abstraction**: Handles relative/absolute paths transparently
- **Template Factory**: Generates valid scenario templates

USAGE EXAMPLE:
    # Load and validate a scenario
    loader = ScenarioLoader()
    scenario = loader.load_scenario("flood_scenario.json")

    # Load decision criteria
    criteria = loader.load_criteria_weights()

    # List available scenarios
    scenarios = loader.list_available_scenarios()
    print(f"Available: {scenarios}")

    # Create new scenario template
    loader.create_scenario_template("earthquake_scenario.json")

    # Get summary
    summary = loader.get_scenario_summary(scenario)
    print(summary)

ERROR HANDLING:
- FileNotFoundError: Scenario or criteria file doesn't exist
- ValueError: Scenario structure is invalid (missing required fields)
- json.JSONDecodeError: Malformed JSON syntax

RELATED MODULES:
- scenarios/*.json: Scenario data files
- scenarios/criteria_weights.json: MCDA criteria definitions
- main.py: Loads scenarios for multi-agent processing
- decision_framework/mcda_engine.py: Uses criteria_scores from scenarios

See also: scenarios/README.md for detailed scenario creation guide
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path


class ScenarioLoader:
    """
    Loads crisis scenarios and decision criteria from JSON files.
    """

    def __init__(self, scenarios_dir: Optional[str] = None):
        """
        Initialize the scenario loader.

        Args:
            scenarios_dir: Directory containing scenario files (default: current directory)
        """
        if scenarios_dir is None:
            self.scenarios_dir = Path(__file__).parent
        else:
            self.scenarios_dir = Path(scenarios_dir)

    def load_scenario(self, scenario_file: str) -> Dict[str, Any]:
        """
        Load a scenario from a JSON file.

        Args:
            scenario_file: Name of the scenario file

        Returns:
            Dictionary containing scenario data
        """
        file_path = self.scenarios_dir / scenario_file

        if not file_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)

        # Validate scenario structure
        self._validate_scenario(scenario_data)

        return scenario_data

    def load_criteria_weights(self, criteria_file: str = "criteria_weights.json") -> Dict[str, Any]:
        """
        Load decision criteria weights from a JSON file.

        Args:
            criteria_file: Name of the criteria file

        Returns:
            Dictionary containing criteria weights
        """
        file_path = self.scenarios_dir / criteria_file

        if not file_path.exists():
            raise FileNotFoundError(f"Criteria file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            criteria_data = json.load(f)

        return criteria_data

    def list_available_scenarios(self) -> List[str]:
        """
        List all available scenario files in the scenarios directory.

        Returns:
            List of scenario file names
        """
        scenario_files = list(self.scenarios_dir.glob("*_scenario.json"))
        return [f.name for f in scenario_files]

    def _validate_scenario(self, scenario: Dict[str, Any]) -> bool:
        """
        Validate that a scenario has the required structure and field types.

        OBJECTIVE:
        Performs comprehensive validation of scenario structure to ensure it meets
        system requirements. Catches malformed scenarios before they cause errors
        during agent processing or decision-making.

        WHY THIS EXISTS:
        Scenarios come from external JSON files that may be hand-edited or generated.
        Early validation prevents:
        - Cryptic errors deep in the system
        - Invalid scenarios reaching agents
        - MCDA engine failures due to missing scores
        - Coordinator failures due to missing action IDs

        VALIDATION RULES:
        1. Required top-level fields must exist:
           - 'id': Scenario identifier
           - 'type': Crisis category
           - 'description': Narrative explanation

        2. If 'available_actions' present:
           - Must be a list type
           - Each action must have 'id' field

        INPUTS:
        - scenario: Dictionary parsed from JSON file

        OUTPUTS:
        - Returns: True if validation passes (allows method chaining)
        - Raises: ValueError with specific message if validation fails

        VALIDATION EXAMPLES:
        VALID:
        {
          "id": "flood_001",
          "type": "flood",
          "description": "Urban flood scenario",
          "available_actions": [
            {"id": "action_1", "name": "Evacuate"}
          ]
        }

        INVALID - Missing required field:
        {
          "type": "flood",  # Missing 'id' and 'description'
          "available_actions": []
        }
        → Raises: ValueError("Scenario missing required field: id")

        INVALID - Malformed actions:
        {
          "id": "flood_001",
          "type": "flood",
          "description": "...",
          "available_actions": "not_a_list"  # Should be list
        }
        → Raises: ValueError("available_actions must be a list")

        INVALID - Action missing ID:
        {
          "id": "flood_001",
          "type": "flood",
          "description": "...",
          "available_actions": [
            {"name": "Evacuate"}  # Missing 'id'
          ]
        }
        → Raises: ValueError("Action 0 missing 'id' field")

        NOTE: This is basic structural validation. Semantic validation (e.g.,
        severity in range 0-1, valid coordinates) should be added for production use.
        """
        required_fields = ['id', 'type', 'description']

        for field in required_fields:
            if field not in scenario:
                raise ValueError(f"Scenario missing required field: {field}")

        # Check for available actions
        if 'available_actions' in scenario:
            actions = scenario['available_actions']
            if not isinstance(actions, list):
                raise ValueError("available_actions must be a list")

            for i, action in enumerate(actions):
                if 'id' not in action:
                    raise ValueError(f"Action {i} missing 'id' field")

        return True

    def create_scenario_template(self, output_file: str = "new_scenario.json"):
        """
        Create a template for a new scenario.

        Args:
            output_file: Name of the output file
        """
        template = {
            "id": "scenario_001",
            "type": "crisis_type",
            "name": "Crisis Scenario Name",
            "description": "Detailed description of the crisis scenario",
            "severity": 0.7,
            "affected_population": 10000,
            "location": {
                "region": "Region Name",
                "coordinates": {"lat": 0.0, "lon": 0.0}
            },
            "tags": ["tag1", "tag2", "tag3"],
            "casualties": 0,
            "infrastructure_damage": False,
            "available_actions": [
                {
                    "id": "action_1",
                    "name": "Action Name",
                    "description": "Action description",
                    "required_resources": ["resource1", "resource2"],
                    "estimated_duration": "2 hours",
                    "risk_level": 0.5,
                    "criteria_scores": {
                        "effectiveness": 0.8,
                        "safety": 0.7,
                        "speed": 0.6,
                        "cost": 0.5,
                        "public_acceptance": 0.7
                    }
                }
            ],
            "constraints": {
                "time_critical": True,
                "resource_limitations": ["limited_personnel", "limited_equipment"],
                "weather_conditions": "clear",
                "accessibility": "moderate"
            },
            "metadata": {
                "created_date": "2025-01-01",
                "author": "Author Name",
                "version": "1.0"
            }
        }

        output_path = self.scenarios_dir / output_file

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        print(f"Scenario template created: {output_path}")

    def get_scenario_summary(self, scenario: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of a scenario.

        Args:
            scenario: Scenario dictionary

        Returns:
            Formatted summary string
        """
        summary = f"""
=== Crisis Scenario Summary ===
ID: {scenario.get('id', 'N/A')}
Type: {scenario.get('type', 'N/A')}
Name: {scenario.get('name', 'N/A')}
Severity: {scenario.get('severity', 0.0):.1f}/1.0

Description:
{scenario.get('description', 'No description available')}

Affected Population: {scenario.get('affected_population', 'Unknown')}
Location: {scenario.get('location', {}).get('region', 'Unknown')}

Available Actions: {len(scenario.get('available_actions', []))}
"""

        # Add action summaries
        if 'available_actions' in scenario:
            summary += "\nActions:\n"
            for action in scenario['available_actions']:
                summary += f"  - {action.get('name', action.get('id'))}: {action.get('description', 'No description')}\n"

        return summary
