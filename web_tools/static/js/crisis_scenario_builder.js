// Crisis Scenario Builder - Advanced JavaScript
// Handles action builder, templates, import/export, validation, and map interaction

let actionCounter = 0;
let map = null;
let marker = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeMap();
    loadTemplates();
    updateSeverityLabel();

    // Add event listeners
    document.getElementById('severity').addEventListener('input', updateSeverityLabel);
    document.getElementById('scenarioForm').addEventListener('submit', handleSubmit);
});

// ============================================================================
// Map Initialization
// ============================================================================

function initializeMap() {
    const defaultLat = parseFloat(document.getElementById('lat').value);
    const defaultLon = parseFloat(document.getElementById('lon').value);

    map = L.map('map').setView([defaultLat, defaultLon], 8);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);

    marker = L.marker([defaultLat, defaultLon], { draggable: true }).addTo(map);

    map.on('click', function(e) {
        updateMarkerPosition(e.latlng.lat, e.latlng.lng);
    });

    marker.on('dragend', function(e) {
        const position = marker.getLatLng();
        updateMarkerPosition(position.lat, position.lng);
    });
}

function updateMarkerPosition(lat, lon) {
    marker.setLatLng([lat, lon]);
    document.getElementById('lat').value = lat.toFixed(6);
    document.getElementById('lon').value = lon.toFixed(6);
}

// ============================================================================
// Action Builder
// ============================================================================

function addAction() {
    actionCounter++;
    const container = document.getElementById('actionsContainer');
    document.getElementById('noActionsMessage').classList.add('d-none');

    const actionHtml = `
        <div class="action-card" id="action_${actionCounter}">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h6 class="mb-0"><i class="bi bi-gear"></i> Action ${actionCounter}</h6>
                <button type="button" class="btn btn-sm btn-danger" onclick="removeAction(${actionCounter})">
                    <i class="bi bi-trash"></i> Remove
                </button>
            </div>

            <div class="row">
                <div class="col-md-6 mb-2">
                    <label class="form-label">Action ID</label>
                    <input type="text" class="form-control form-control-sm"
                           name="action_${actionCounter}_id"
                           placeholder="e.g., action_evacuation"
                           value="action_${actionCounter}">
                </div>
                <div class="col-md-6 mb-2">
                    <label class="form-label">Action Name</label>
                    <input type="text" class="form-control form-control-sm"
                           name="action_${actionCounter}_name"
                           placeholder="e.g., Immediate Evacuation">
                </div>
            </div>

            <div class="mb-2">
                <label class="form-label">Description</label>
                <textarea class="form-control form-control-sm"
                          name="action_${actionCounter}_description"
                          rows="2"
                          placeholder="Detailed description of this action..."></textarea>
            </div>

            <div class="row">
                <div class="col-md-6 mb-2">
                    <label class="form-label">Required Resources (comma-separated)</label>
                    <input type="text" class="form-control form-control-sm"
                           name="action_${actionCounter}_resources"
                           placeholder="e.g., fire_trucks, ambulances">
                </div>
                <div class="col-md-3 mb-2">
                    <label class="form-label">Duration</label>
                    <input type="text" class="form-control form-control-sm"
                           name="action_${actionCounter}_duration"
                           placeholder="e.g., 2-4 hours">
                </div>
                <div class="col-md-3 mb-2">
                    <label class="form-label">Risk Level (0-1)</label>
                    <input type="number" class="form-control form-control-sm"
                           name="action_${actionCounter}_risk"
                           min="0" max="1" step="0.01"
                           value="0.5">
                </div>
            </div>

            <div class="mt-3">
                <label class="form-label"><strong>Criteria Scores (0.0 - 1.0, higher is better)</strong></label>
                <div class="row">
                    <div class="col-md-4 criteria-row">
                        <label class="small">Effectiveness</label>
                        <input type="number" class="form-control form-control-sm"
                               name="action_${actionCounter}_effectiveness"
                               min="0" max="1" step="0.01" value="0.7">
                    </div>
                    <div class="col-md-4 criteria-row">
                        <label class="small">Safety</label>
                        <input type="number" class="form-control form-control-sm"
                               name="action_${actionCounter}_safety"
                               min="0" max="1" step="0.01" value="0.8">
                    </div>
                    <div class="col-md-4 criteria-row">
                        <label class="small">Speed</label>
                        <input type="number" class="form-control form-control-sm"
                               name="action_${actionCounter}_speed"
                               min="0" max="1" step="0.01" value="0.6">
                    </div>
                </div>
                <div class="row mt-2">
                    <div class="col-md-4 criteria-row">
                        <label class="small">Cost (higher = cheaper)</label>
                        <input type="number" class="form-control form-control-sm"
                               name="action_${actionCounter}_cost"
                               min="0" max="1" step="0.01" value="0.5">
                    </div>
                    <div class="col-md-4 criteria-row">
                        <label class="small">Public Acceptance</label>
                        <input type="number" class="form-control form-control-sm"
                               name="action_${actionCounter}_public_acceptance"
                               min="0" max="1" step="0.01" value="0.7">
                    </div>
                </div>
            </div>
        </div>
    `;

    container.insertAdjacentHTML('beforeend', actionHtml);
}

function removeAction(actionId) {
    const element = document.getElementById(`action_${actionId}`);
    if (element) {
        element.remove();
    }

    // Show no actions message if none left
    const container = document.getElementById('actionsContainer');
    if (container.children.length === 0) {
        document.getElementById('noActionsMessage').classList.remove('d-none');
    }
}

function getActionsData() {
    const actions = [];
    const actionCards = document.querySelectorAll('.action-card');

    actionCards.forEach(card => {
        const actionNum = card.id.split('_')[1];
        const action = {
            id: document.querySelector(`[name="action_${actionNum}_id"]`).value,
            name: document.querySelector(`[name="action_${actionNum}_name"]`).value,
            description: document.querySelector(`[name="action_${actionNum}_description"]`).value,
            required_resources: document.querySelector(`[name="action_${actionNum}_resources"]`).value.split(',').map(r => r.trim()).filter(r => r),
            estimated_duration: document.querySelector(`[name="action_${actionNum}_duration"]`).value,
            risk_level: parseFloat(document.querySelector(`[name="action_${actionNum}_risk"]`).value),
            criteria_scores: {
                effectiveness: parseFloat(document.querySelector(`[name="action_${actionNum}_effectiveness"]`).value),
                safety: parseFloat(document.querySelector(`[name="action_${actionNum}_safety"]`).value),
                speed: parseFloat(document.querySelector(`[name="action_${actionNum}_speed"]`).value),
                cost: parseFloat(document.querySelector(`[name="action_${actionNum}_cost"]`).value),
                public_acceptance: parseFloat(document.querySelector(`[name="action_${actionNum}_public_acceptance"]`).value)
            }
        };
        actions.push(action);
    });

    return actions;
}

// ============================================================================
// Template Library
// ============================================================================

const scenarioTemplates = [
    {
        name: 'Urban Flood',
        type: 'flood',
        icon: '💧',
        description: 'Flash flooding in urban residential area',
        data: {
            type: 'flood',
            severity: 0.75,
            affected_population: 5000,
            geographic_scope: 'local',
            geographic_location: 'urban'
        }
    },
    {
        name: 'Wildfire',
        type: 'wildfire',
        icon: '🔥',
        description: 'Forest fire spreading towards villages',
        data: {
            type: 'wildfire',
            severity: 0.85,
            affected_population: 8000,
            geographic_scope: 'regional',
            geographic_location: 'wilderness'
        }
    },
    {
        name: 'Hazmat Leak',
        type: 'hazmat',
        icon: '☢️',
        description: 'Chemical spill at industrial facility',
        data: {
            type: 'hazmat',
            severity: 0.80,
            affected_population: 12000,
            geographic_scope: 'local',
            geographic_location: 'industrial'
        }
    },
    {
        name: 'Earthquake',
        type: 'earthquake',
        icon: '🏚️',
        description: 'Major earthquake with building collapses',
        data: {
            type: 'earthquake',
            severity: 0.90,
            affected_population: 50000,
            geographic_scope: 'regional',
            geographic_location: 'urban'
        }
    },
    {
        name: 'Coastal Flood',
        type: 'flood',
        icon: '🌊',
        description: 'Storm surge and coastal flooding',
        data: {
            type: 'flood',
            severity: 0.70,
            affected_population: 15000,
            geographic_scope: 'regional',
            geographic_location: 'coastal'
        }
    },
    {
        name: 'Mass Casualty',
        type: 'mass_casualty',
        icon: '🚑',
        description: 'Multi-vehicle accident or disaster with many casualties',
        data: {
            type: 'mass_casualty',
            severity: 0.75,
            affected_population: 200,
            geographic_scope: 'local',
            geographic_location: 'urban'
        }
    }
];

function loadTemplates() {
    const container = document.getElementById('templatesContainer');

    scenarioTemplates.forEach((template, index) => {
        const templateHtml = `
            <div class="col-md-4 mb-3">
                <div class="card template-card h-100" onclick="applyTemplate(${index})">
                    <div class="card-body text-center">
                        <div style="font-size: 3rem;">${template.icon}</div>
                        <h6 class="mt-2">${template.name}</h6>
                        <p class="small text-muted mb-0">${template.description}</p>
                    </div>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', templateHtml);
    });
}

function applyTemplate(index) {
    const template = scenarioTemplates[index];
    const data = template.data;

    // Apply basic data
    document.getElementById('type').value = data.type;
    document.getElementById('severity').value = data.severity;
    document.getElementById('affected_population').value = data.affected_population;
    document.getElementById('geographic_scope').value = data.geographic_scope;
    document.getElementById('geographic_location').value = data.geographic_location || 'urban';

    // Generate ID
    const randomNum = Math.floor(Math.random() * 1000);
    document.getElementById('id').value = `${data.type}_${String(randomNum).padStart(3, '0')}`;

    // Set name
    document.getElementById('name').value = `${template.name} Emergency Response`;

    updateSeverityLabel();

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('templateModal'));
    modal.hide();

    showValidation('success', `Template "${template.name}" applied successfully!`);
}

// ============================================================================
// Import/Export
// ============================================================================

function importScenario(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const scenario = JSON.parse(e.target.result);
            populateFormWithScenario(scenario);
            showValidation('success', `Scenario "${scenario.name || scenario.id}" imported successfully!`);
        } catch (error) {
            showValidation('error', `Error importing scenario: ${error.message}`);
        }
    };
    reader.readAsText(file);
}

function populateFormWithScenario(scenario) {
    // Basic info
    if (scenario.id) document.getElementById('id').value = scenario.id;
    if (scenario.type) document.getElementById('type').value = scenario.type;
    if (scenario.name) document.getElementById('name').value = scenario.name;
    if (scenario.description) document.getElementById('description').value = scenario.description;
    if (scenario.severity !== undefined) document.getElementById('severity').value = scenario.severity;
    if (scenario.affected_population) document.getElementById('affected_population').value = scenario.affected_population;
    if (scenario.casualties !== undefined) document.getElementById('casualties').value = scenario.casualties;
    if (scenario.infrastructure_damage !== undefined) {
        document.getElementById('infrastructure_damage').value = scenario.infrastructure_damage.toString();
    }
    if (scenario.tags) document.getElementById('tags').value = scenario.tags.join(', ');

    // Location
    if (scenario.location) {
        if (scenario.location.region) document.getElementById('location_region').value = scenario.location.region;
        if (scenario.location.coordinates) {
            document.getElementById('lat').value = scenario.location.coordinates.lat;
            document.getElementById('lon').value = scenario.location.coordinates.lon;
            updateMarkerPosition(scenario.location.coordinates.lat, scenario.location.coordinates.lon);
            map.setView([scenario.location.coordinates.lat, scenario.location.coordinates.lon], 10);
        }
    }

    // Expert selection
    if (scenario.expert_selection) {
        const es = scenario.expert_selection;
        if (es.geographic_scope) document.getElementById('geographic_scope').value = es.geographic_scope;
        if (es.geographic_location) document.getElementById('geographic_location').value = es.geographic_location;
        if (es.affected_domains) document.getElementById('affected_domains').value = es.affected_domains.join(', ');
        if (es.infrastructure_systems) document.getElementById('infrastructure_systems').value = es.infrastructure_systems.join(', ');
        if (es.command_structure_needed) {
            document.getElementById('tactical').checked = es.command_structure_needed.tactical || false;
            document.getElementById('strategic').checked = es.command_structure_needed.strategic || false;
            document.getElementById('multi_jurisdictional').checked = es.command_structure_needed.multi_jurisdictional || false;
        }
    }

    // Import actions
    if (scenario.available_actions && scenario.available_actions.length > 0) {
        // Clear existing actions
        document.getElementById('actionsContainer').innerHTML = '';

        scenario.available_actions.forEach(action => {
            addAction();
            const actionNum = actionCounter;

            if (action.id) document.querySelector(`[name="action_${actionNum}_id"]`).value = action.id;
            if (action.name) document.querySelector(`[name="action_${actionNum}_name"]`).value = action.name;
            if (action.description) document.querySelector(`[name="action_${actionNum}_description"]`).value = action.description;
            if (action.required_resources) document.querySelector(`[name="action_${actionNum}_resources"]`).value = action.required_resources.join(', ');
            if (action.estimated_duration) document.querySelector(`[name="action_${actionNum}_duration"]`).value = action.estimated_duration;
            if (action.risk_level !== undefined) document.querySelector(`[name="action_${actionNum}_risk"]`).value = action.risk_level;

            if (action.criteria_scores) {
                const cs = action.criteria_scores;
                if (cs.effectiveness !== undefined) document.querySelector(`[name="action_${actionNum}_effectiveness"]`).value = cs.effectiveness;
                if (cs.safety !== undefined) document.querySelector(`[name="action_${actionNum}_safety"]`).value = cs.safety;
                if (cs.speed !== undefined) document.querySelector(`[name="action_${actionNum}_speed"]`).value = cs.speed;
                if (cs.cost !== undefined) document.querySelector(`[name="action_${actionNum}_cost"]`).value = cs.cost;
                if (cs.public_acceptance !== undefined) document.querySelector(`[name="action_${actionNum}_public_acceptance"]`).value = cs.public_acceptance;
            }
        });
    }

    updateSeverityLabel();
}

function exportScenario() {
    const scenario = buildScenarioObject();
    const json = JSON.stringify(scenario, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${scenario.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ============================================================================
// Form Building and Validation
// ============================================================================

function buildScenarioObject() {
    const affectedDomains = document.getElementById('affected_domains').value
        .split(',').map(d => d.trim()).filter(d => d);
    const infrastructureSystems = document.getElementById('infrastructure_systems').value
        .split(',').map(s => s.trim()).filter(s => s);
    const tags = document.getElementById('tags').value
        .split(',').map(t => t.trim()).filter(t => t);

    const scenario = {
        id: document.getElementById('id').value,
        type: document.getElementById('type').value,
        name: document.getElementById('name').value,
        description: document.getElementById('description').value,
        severity: parseFloat(document.getElementById('severity').value),
        affected_population: parseInt(document.getElementById('affected_population').value) || 0,
        casualties: parseInt(document.getElementById('casualties').value) || 0,
        infrastructure_damage: document.getElementById('infrastructure_damage').value === 'true',
        location: {
            region: document.getElementById('location_region').value,
            coordinates: {
                lat: parseFloat(document.getElementById('lat').value),
                lon: parseFloat(document.getElementById('lon').value)
            }
        },
        tags: tags.length > 0 ? tags : [document.getElementById('type').value, "emergency"],
        available_actions: getActionsData(),
        expert_selection: {
            crisis_type: document.getElementById('type').value,
            severity: parseFloat(document.getElementById('severity').value),
            geographic_scope: document.getElementById('geographic_scope').value,
            geographic_location: document.getElementById('geographic_location').value,
            affected_populations: parseInt(document.getElementById('affected_population').value) || 0,
            affected_domains: affectedDomains,
            command_structure_needed: {
                tactical: document.getElementById('tactical').checked,
                strategic: document.getElementById('strategic').checked,
                multi_jurisdictional: document.getElementById('multi_jurisdictional').checked
            },
            infrastructure_systems: infrastructureSystems
        },
        metadata: {
            created_date: new Date().toISOString().split('T')[0],
            author: "Crisis MAS Web Tools",
            version: "1.0"
        },
        _save_location: document.getElementById('save_location').value || undefined,
        _filename: document.getElementById('filename').value || undefined
    };

    return scenario;
}

function validateForm() {
    const id = document.getElementById('id').value;
    const type = document.getElementById('type').value;
    const name = document.getElementById('name').value;
    const description = document.getElementById('description').value;

    let errors = [];

    // ID validation
    if (id && !/^[a-z0-9_]+$/.test(id)) {
        errors.push('ID must contain only lowercase letters, numbers, and underscores');
    }

    // Required fields
    if (!id) errors.push('Scenario ID is required');
    if (!type) errors.push('Crisis type is required');
    if (!name) errors.push('Scenario name is required');
    if (!description) errors.push('Description is required');

    if (errors.length > 0) {
        showValidation('error', errors.join('<br>'));
        return false;
    }

    showValidation('success', 'Scenario structure is valid!');
    return true;
}

function updateSeverityLabel() {
    const severity = parseFloat(document.getElementById('severity').value);
    const label = document.getElementById('severityLabel');

    if (severity < 0.3) {
        label.textContent = 'Minor';
        label.className = 'form-text text-secondary';
    } else if (severity < 0.6) {
        label.textContent = 'Moderate';
        label.className = 'form-text text-info';
    } else if (severity < 0.8) {
        label.textContent = 'Severe';
        label.className = 'form-text text-warning';
    } else {
        label.textContent = 'Catastrophic';
        label.className = 'form-text text-danger';
    }
}

function showValidation(type, message) {
    const status = document.getElementById('validationStatus');
    const msgElem = document.getElementById('validationMessage');

    status.classList.remove('d-none', 'alert-info', 'alert-success', 'alert-danger');

    if (type === 'success') {
        status.classList.add('alert-success');
    } else if (type === 'error') {
        status.classList.add('alert-danger');
    } else {
        status.classList.add('alert-info');
    }

    msgElem.innerHTML = message;

    // Auto-hide after 5 seconds for success messages
    if (type === 'success') {
        setTimeout(() => {
            status.classList.add('d-none');
        }, 5000);
    }
}

// ============================================================================
// Form Submission
// ============================================================================

async function handleSubmit(e) {
    e.preventDefault();

    if (!validateForm()) {
        return;
    }

    const scenario = buildScenarioObject();

    try {
        const response = await fetch('/api/crisis-scenarios/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(scenario)
        });

        const result = await response.json();

        if (result.success) {
            showValidation('success', `✅ Scenario saved successfully to ${result.filename}`);
            setTimeout(() => {
                window.location.href = '/crisis-scenarios';
            }, 2000);
        } else {
            showValidation('error', `❌ Error: ${result.error}`);
        }
    } catch (error) {
        showValidation('error', `❌ Error saving scenario: ${error.message}`);
    }
}
