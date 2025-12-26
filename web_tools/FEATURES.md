# Crisis MAS Web Tools - Advanced Features Guide

## Overview

The Crisis MAS Web Tools now includes a comprehensive Crisis Scenario Builder with advanced features for creating, editing, and managing complex crisis scenarios for multi-agent simulation.

## New Features

### 1. Advanced Action Builder

Create multiple response actions directly in the web interface with:

- **Action Details**
  - Unique ID and descriptive name
  - Detailed description of the action
  - Required resources (comma-separated list)
  - Estimated duration
  - Risk level (0.0 - 1.0)

- **Criteria Scores** (0.0 - 1.0, higher is better)
  - Effectiveness: How well does this solve the problem?
  - Safety: How safe is this for responders and public?
  - Speed: How quickly can this be executed?
  - Cost: Cost-effectiveness (higher = cheaper)
  - Public Acceptance: Public support level

**How to Use:**
1. Navigate to Crisis Scenarios → Create New
2. Scroll to "Available Response Actions"
3. Click "Add Action" to create a new action
4. Fill in all fields and criteria scores
5. Add multiple actions (click "Add Action" again)
6. Remove actions with the "Remove" button

### 2. Expert Selection Metadata Builder

Configure automatic expert selection for scenarios:

- **Geographic Parameters**
  - Scope: Local, Regional, or National
  - Location Type: Coastal, Maritime, Inland, Urban, Rural, Wilderness

- **Affected Domains**
  - Choose from: weather_environment, medical_health, law_enforcement, fire_rescue, maritime_coastal, emergency_communications, public_order, security, hazmat, search_rescue, healthcare
  - Enter as comma-separated list

- **Command Structure**
  - Tactical (On-scene commanders)
  - Strategic (Regional/national commanders)
  - Multi-jurisdictional coordination

- **Infrastructure Systems**
  - Select affected systems: hospitals, healthcare, transportation, communications, power, water, ports, emergency_services

**How to Use:**
1. Scroll to "Expert Selection Metadata" section
2. Select geographic scope and location type from dropdowns
3. Enter affected domains (comma-separated)
4. Check command structure requirements
5. Enter infrastructure systems (comma-separated)

### 3. Scenario Template Library

Quick-start templates for common crisis types:

**Available Templates:**
- 💧 **Urban Flood**: Flash flooding in residential areas
- 🔥 **Wildfire**: Forest fire spreading towards villages
- ☢️ **Hazmat Leak**: Chemical spill at industrial facility
- 🏚️ **Earthquake**: Major earthquake with building collapses
- 🌊 **Coastal Flood**: Storm surge and coastal flooding
- 🚑 **Mass Casualty**: Multi-vehicle accident or disaster

**How to Use:**
1. Click "Use Template" button at the top
2. Select a template from the modal
3. Template populates basic fields automatically
4. Customize as needed
5. Add actions and complete the scenario

### 4. Import/Export Functionality

**Import Existing Scenarios:**
1. Click "Import JSON" button
2. Select a .json file from your computer
3. The form automatically populates with all data
4. Edit as needed and save

**Export Scenarios:**
1. Fill in the form
2. Click "Export JSON" button
3. Downloads a complete JSON file
4. Use for backup or sharing

**Supported Features:**
- Imports all basic information
- Imports location and coordinates
- Imports all available actions with criteria scores
- Imports expert selection metadata
- Updates map marker position
- Validates imported data

### 5. Real-Time Validation

The form validates your input as you work:

**Validation Checks:**
- Scenario ID format (lowercase, numbers, underscores only)
- Required fields presence
- Numeric ranges for severity and coordinates
- Data structure compliance

**Visual Feedback:**
- Success messages in green
- Error messages in red
- Severity level indicator (Minor, Moderate, Severe, Catastrophic)

**How to Use:**
- Validation runs automatically when changing key fields
- Status appears at the top of the form
- Check validation before saving

### 6. Interactive Map Integration

**OpenStreetMap Features:**
- Click anywhere to set crisis location
- Drag marker to adjust position
- Automatic coordinate updates
- Zoom controls
- Real-time lat/lon display

**How to Use:**
1. Click on the map where the crisis is located
2. OR drag the marker to the exact position
3. Coordinates update automatically
4. Map centers on Greece by default
5. Zoom in/out to find exact location

### 7. Configurable Save Location

**Features:**
- Default save to `/scenarios` directory
- Specify custom save path
- Auto-generate filename from scenario ID
- Or specify custom filename

**How to Use:**
1. Scroll to "Save Location" section
2. Default is `/home/user/crisis_mas_poc/scenarios`
3. Change path if needed
4. Leave filename blank for auto-generation
5. Or specify custom filename (e.g., `my_scenario.json`)

## Complete Workflow Example

### Creating a New Wildfire Scenario

1. **Start with Template**
   - Click "Use Template"
   - Select "Wildfire" template
   - Basic fields auto-populated

2. **Customize Details**
   - Update name: "Evia Forest Fire 2024"
   - Edit description to match your scenario
   - Adjust severity, affected population
   - Add tags: `wildfire, forest, evacuation`

3. **Set Location**
   - Click on map in North Evia area
   - Coordinates: 38.9231, 23.6578
   - Region: "North Evia, Central Greece"

4. **Add Actions**
   - Click "Add Action"
   - Action 1: "Immediate Village Evacuation"
     - Resources: `evacuation_buses, police_escorts, boats`
     - Duration: `3-5 hours`
     - Risk: 0.35
     - Criteria: effectiveness=0.90, safety=0.95, speed=0.75, cost=0.45, public_acceptance=0.70

   - Click "Add Action" again
   - Action 2: "Aerial Firefighting Campaign"
     - Resources: `canadair_aircraft, helicopters, fire_retardant`
     - Duration: `6-12 hours`
     - Risk: 0.50
     - Criteria: effectiveness=0.75, safety=0.70, speed=0.85, cost=0.30, public_acceptance=0.85

5. **Configure Expert Selection**
   - Geographic scope: Regional
   - Location type: Wilderness
   - Affected domains: `fire_rescue, weather_environment, law_enforcement`
   - Command structure: ✓ Tactical, ✓ Multi-jurisdictional
   - Infrastructure: `power, communications, roads`

6. **Save**
   - Review validation status (should be green)
   - Click "Save Crisis Scenario"
   - Scenario saved to `/scenarios/fire_001.json`
   - Redirected to scenario list

### Editing an Existing Scenario

1. Navigate to Crisis Scenarios
2. Click "Edit" on any scenario
3. Form loads with all existing data
4. Modify fields as needed
5. Add/remove actions
6. Click "Save Crisis Scenario"

### Importing a Scenario

1. Have a scenario JSON file ready
2. Click "Import JSON"
3. Select the file
4. All fields populate automatically
5. Review and modify as needed
6. Save to create a copy or update

## Technical Details

### File Locations

- **Enhanced Form**: `/web_tools/templates/crisis_scenario_form_enhanced.html`
- **JavaScript**: `/web_tools/static/js/crisis_scenario_builder.js`
- **API Endpoint**: `/api/crisis-scenarios/save` (POST)
- **Saved Scenarios**: `/scenarios/*.json`

### JSON Structure

The builder creates scenarios matching the official template structure:

```json
{
  "id": "fire_001",
  "type": "wildfire",
  "name": "Scenario Name",
  "description": "Detailed description...",
  "severity": 0.9,
  "affected_population": 8000,
  "casualties": 2,
  "infrastructure_damage": true,
  "location": {
    "region": "North Evia, Greece",
    "coordinates": {"lat": 38.9231, "lon": 23.6578}
  },
  "tags": ["wildfire", "emergency"],
  "available_actions": [
    {
      "id": "action_1",
      "name": "Action Name",
      "description": "Description...",
      "required_resources": ["resource1", "resource2"],
      "estimated_duration": "2-4 hours",
      "risk_level": 0.35,
      "criteria_scores": {
        "effectiveness": 0.90,
        "safety": 0.95,
        "speed": 0.75,
        "cost": 0.45,
        "public_acceptance": 0.70
      }
    }
  ],
  "expert_selection": {
    "crisis_type": "wildfire",
    "severity": 0.9,
    "geographic_scope": "regional",
    "geographic_location": "wilderness",
    "affected_domains": ["fire_rescue", "weather_environment"],
    "command_structure_needed": {
      "tactical": true,
      "strategic": false,
      "multi_jurisdictional": true
    },
    "infrastructure_systems": ["power", "communications"]
  },
  "metadata": {
    "created_date": "2025-11-15",
    "author": "Crisis MAS Web Tools",
    "version": "1.0"
  }
}
```

### Browser Compatibility

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Requires JavaScript enabled
- Responsive design for desktop and tablet

## Troubleshooting

### Map Not Loading
- Check internet connection (OpenStreetMap requires online access)
- Refresh the page
- Clear browser cache

### Import Fails
- Ensure JSON file is valid
- Check file format matches scenario structure
- Look for syntax errors in JSON

### Validation Errors
- Check scenario ID format (lowercase, underscores only)
- Ensure all required fields are filled
- Verify numeric values are in correct ranges

### Actions Not Saving
- Ensure at least action ID and name are filled
- Check criteria scores are between 0 and 1
- Verify form submission completed

## Tips and Best Practices

1. **Use Templates**: Start with a template for faster creation
2. **Save Often**: Export JSON periodically as backup
3. **Validate Early**: Check validation before adding many actions
4. **Meaningful IDs**: Use descriptive IDs like `fire_evia_2024` not `scenario1`
5. **Complete Actions**: Fill all criteria scores for better MCDA results
6. **Test Import**: After creating, export and re-import to verify
7. **Document**: Use description field for comprehensive details

## Future Enhancements

Potential additions:
- Constraints builder
- Real-time factors editor
- Scenario comparison tool
- Batch import/export
- Scenario validation against live data
- Integration with external crisis databases

## Support

For issues or questions:
- Check this documentation
- Review scenario template: `/scenarios/scenario_template.json`
- See scenarios README: `/scenarios/README.md`
- Report issues: https://github.com/kerbgr/crisis_mas_poc/issues
