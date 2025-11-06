"""
Scenario Loader
Loads and parses crisis scenarios from JSON files
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
        Validate that a scenario has the required structure.

        Args:
            scenario: Scenario dictionary to validate

        Returns:
            True if valid

        Raises:
            ValueError: If scenario structure is invalid
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
