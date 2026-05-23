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
    if (scenario.name_greek) document.getElementById('name_greek').value = scenario.name_greek;
    if (scenario.event_reference) document.getElementById('event_reference').value = scenario.event_reference;
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
        const loc = scenario.location;
        if (loc.region) document.getElementById('location_region').value = loc.region;
        if (loc.coordinates) {
            document.getElementById('lat').value = loc.coordinates.lat;
            document.getElementById('lon').value = loc.coordinates.lon;
            updateMarkerPosition(loc.coordinates.lat, loc.coordinates.lon);
            map.setView([loc.coordinates.lat, loc.coordinates.lon], 10);
        }
        if (loc.affected_area_km2 !== undefined) document.getElementById('affected_area_km2').value = loc.affected_area_km2;
        if (loc.affected_area_description) document.getElementById('affected_area_description').value = loc.affected_area_description;
    }

    // Constraints
    if (scenario.constraints) {
        const c = scenario.constraints;
        document.getElementById('time_critical').checked = c.time_critical || false;
        if (c.weather_conditions) document.getElementById('weather_conditions').value = c.weather_conditions;
        if (c.accessibility) document.getElementById('accessibility').value = c.accessibility;
        if (c.resource_limitations) document.getElementById('resource_limitations').value = c.resource_limitations.join(', ');
        if (c.additional_concerns) document.getElementById('additional_concerns').value = c.additional_concerns.join('\n');
    }

    // Real-time factors
    if (scenario.real_time_factors) {
        const rt = scenario.real_time_factors;
        const knownKeys = ['missing_persons', 'current_evacuations', 'casualties_reported', 'forecast'];
        if (rt.missing_persons !== undefined) document.getElementById('missing_persons').value = rt.missing_persons;
        if (rt.current_evacuations) document.getElementById('current_evacuations_pct').value = rt.current_evacuations;
        if (rt.forecast) document.getElementById('forecast').value = rt.forecast;
        // Remaining dynamic factors
        const dynamicLines = Object.entries(rt)
            .filter(([k]) => !knownKeys.includes(k))
            .map(([k, v]) => `${k}: ${v}`);
        if (dynamicLines.length > 0) document.getElementById('dynamic_factors').value = dynamicLines.join('\n');
    }

    // Expert selection
    if (scenario.expert_selection) {
        const es = scenario.expert_selection;
        if (es.geographic_scope) document.getElementById('geographic_scope').value = es.geographic_scope;
        if (es.geographic_location) document.getElementById('geographic_location').value = es.geographic_location;
        if (es.crisis_subtypes) document.getElementById('crisis_subtypes').value = es.crisis_subtypes.join(', ');
        if (es.duration_estimated_hours !== undefined) document.getElementById('duration_estimated_hours').value = es.duration_estimated_hours;
        if (es.affected_domains) document.getElementById('affected_domains').value = es.affected_domains.join(', ');
        if (es.infrastructure_systems) document.getElementById('infrastructure_systems').value = es.infrastructure_systems.join(', ');
        if (es.command_structure_needed) {
            document.getElementById('tactical').checked = es.command_structure_needed.tactical || false;
            document.getElementById('strategic').checked = es.command_structure_needed.strategic || false;
            document.getElementById('multi_jurisdictional').checked = es.command_structure_needed.multi_jurisdictional || false;
        }
    }

    // Metadata
    if (scenario.metadata) {
        const m = scenario.metadata;
        if (m.author) document.getElementById('meta_author').value = m.author;
        if (m.version) document.getElementById('meta_version').value = m.version;
        if (m.scenario_complexity) document.getElementById('scenario_complexity').value = m.scenario_complexity;
        if (m.recommended_expert_domains) document.getElementById('recommended_expert_domains').value = m.recommended_expert_domains.join(', ');
        if (m.version_notes) document.getElementById('version_notes').value = m.version_notes;
    }

    // Actions
    if (scenario.available_actions && scenario.available_actions.length > 0) {
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
    const crisisSubtypes = document.getElementById('crisis_subtypes').value
        .split(',').map(s => s.trim()).filter(s => s);
    const resourceLimitations = document.getElementById('resource_limitations').value
        .split(',').map(s => s.trim()).filter(s => s);
    const additionalConcerns = document.getElementById('additional_concerns').value
        .split('\n').map(s => s.trim()).filter(s => s);
    const recommendedDomains = document.getElementById('recommended_expert_domains').value
        .split(',').map(s => s.trim()).filter(s => s);

    // Parse dynamic real-time factors (key: value lines)
    const dynamicFactors = {};
    document.getElementById('dynamic_factors').value.split('\n').forEach(line => {
        const sep = line.indexOf(':');
        if (sep > 0) {
            const k = line.slice(0, sep).trim();
            let v = line.slice(sep + 1).trim();
            if (v === 'true') v = true;
            else if (v === 'false') v = false;
            else if (!isNaN(v) && v !== '') v = parseFloat(v);
            if (k) dynamicFactors[k] = v;
        }
    });

    const areakm2 = document.getElementById('affected_area_km2').value;
    const durationHrs = document.getElementById('duration_estimated_hours').value;

    const scenario = {
        id: document.getElementById('id').value,
        type: document.getElementById('type').value,
        name: document.getElementById('name').value,
        name_greek: document.getElementById('name_greek').value || undefined,
        event_reference: document.getElementById('event_reference').value || undefined,
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
            },
            ...(areakm2 ? { affected_area_km2: parseFloat(areakm2) } : {}),
            ...(document.getElementById('affected_area_description').value ? { affected_area_description: document.getElementById('affected_area_description').value } : {})
        },
        tags: tags.length > 0 ? tags : [document.getElementById('type').value, "emergency"],
        available_actions: getActionsData(),
        constraints: {
            time_critical: document.getElementById('time_critical').checked,
            resource_limitations: resourceLimitations,
            weather_conditions: document.getElementById('weather_conditions').value || undefined,
            accessibility: document.getElementById('accessibility').value,
            additional_concerns: additionalConcerns
        },
        real_time_factors: {
            ...(document.getElementById('missing_persons').value ? { missing_persons: parseInt(document.getElementById('missing_persons').value) } : {}),
            ...(document.getElementById('current_evacuations_pct').value ? { current_evacuations: document.getElementById('current_evacuations_pct').value } : {}),
            ...(document.getElementById('forecast').value ? { forecast: document.getElementById('forecast').value } : {}),
            ...dynamicFactors
        },
        expert_selection: {
            crisis_type: document.getElementById('type').value,
            crisis_subtypes: crisisSubtypes,
            severity: parseFloat(document.getElementById('severity').value),
            geographic_scope: document.getElementById('geographic_scope').value,
            geographic_location: document.getElementById('geographic_location').value,
            affected_populations: parseInt(document.getElementById('affected_population').value) || 0,
            ...(durationHrs ? { duration_estimated_hours: parseInt(durationHrs) } : {}),
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
            author: document.getElementById('meta_author').value || "Crisis MAS Web Tools",
            version: document.getElementById('meta_version').value || "1.0",
            scenario_complexity: document.getElementById('scenario_complexity').value,
            ...(recommendedDomains.length > 0 ? { recommended_expert_domains: recommendedDomains } : {}),
            ...(document.getElementById('version_notes').value ? { version_notes: document.getElementById('version_notes').value } : {})
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

// ============================================================================
// Protocol Integration - Knowledge Base
// ============================================================================

/**
 * Load and display relevant protocols when crisis type changes
 */
async function loadRelevantProtocols(crisisType) {
    if (!crisisType) return;

    try {
        const response = await fetch(`/api/protocols/relevant/${encodeURIComponent(crisisType)}?limit=3`);
        const result = await response.json();

        if (result.success && result.protocols.length > 0) {
            showProtocolsPanel(result.protocols);
        }
    } catch (error) {
        console.error('Error loading protocols:', error);
    }
}

/**
 * Display relevant protocols in a side panel or info box
 */
function showProtocolsPanel(protocols) {
    const panelHtml = `
        <div class="alert alert-info mt-3" id="protocolsPanel">
            <h6><i class="bi bi-book"></i> Related Knowledge Base</h6>
            <p class="small mb-2">Found ${protocols.length} relevant incident handling protocol(s):</p>
            ${protocols.map(p => `
                <div class="card card-body bg-white mb-2 small">
                    <strong>${p.category.toUpperCase()}</strong>: ${p.question.substring(0, 100)}...
                    <button class="btn btn-xs btn-outline-primary mt-1" onclick="extractActionsFromProtocol('${p.id}')">
                        <i class="bi bi-lightbulb"></i> Extract Actions
                    </button>
                </div>
            `).join('')}
        </div>
    `;

    // Insert after actions container
    const actionsContainer = document.getElementById('actionsContainer').parentElement;
    const existing = document.getElementById('protocolsPanel');
    if (existing) existing.remove();

    actionsContainer.insertAdjacentHTML('afterend', panelHtml);
}

/**
 * Extract and suggest actions from a specific protocol
 */
async function extractActionsFromProtocol(protocolId) {
    try {
        const response = await fetch('/api/protocols/extract-actions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ protocol_id: protocolId })
        });

        const result = await response.json();

        if (result.success && result.extracted_actions.length > 0) {
            showExtractedActions(result.extracted_actions);
        } else {
            showValidation('info', 'No actionable steps found in this protocol');
        }
    } catch (error) {
        showValidation('error', `Error extracting actions: ${error.message}`);
    }
}

/**
 * Display extracted actions and allow user to add them
 */
function showExtractedActions(actions) {
    const modalHtml = `
        <div class="modal fade" id="extractedActionsModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title"><i class="bi bi-lightbulb"></i> Extracted Actions from Protocol</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p>Select actions to add to your scenario:</p>
                        ${actions.map((action, idx) => `
                            <div class="form-check mb-2">
                                <input class="form-check-input" type="checkbox" value="${idx}" id="extractedAction${idx}">
                                <label class="form-check-label" for="extractedAction${idx}">
                                    <strong>${action.name}</strong><br>
                                    <small class="text-muted">${action.description}</small>
                                </label>
                            </div>
                        `).join('')}
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="applyExtractedActions()">
                            Add Selected Actions
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if any
    const existing = document.getElementById('extractedActionsModal');
    if (existing) existing.remove();

    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Store extracted actions globally for later use
    window.currentExtractedActions = actions;

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('extractedActionsModal'));
    modal.show();
}

/**
 * Apply selected extracted actions to the form
 */
function applyExtractedActions() {
    const checkboxes = document.querySelectorAll('#extractedActionsModal input[type="checkbox"]:checked');

    checkboxes.forEach(checkbox => {
        const idx = parseInt(checkbox.value);
        const action = window.currentExtractedActions[idx];

        // Add action to form
        addAction();
        const actionNum = actionCounter;

        // Populate with extracted data
        document.querySelector(`[name="action_${actionNum}_name"]`).value = action.name;
        document.querySelector(`[name="action_${actionNum}_description"]`).value = action.description;
    });

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('extractedActionsModal'));
    modal.hide();

    showValidation('success', `Added ${checkboxes.length} action(s) from protocol knowledge`);
}

/**
 * Suggest complete actions with criteria scores based on crisis type
 */
async function suggestActionsFromKnowledge() {
    const crisisType = document.getElementById('type').value;

    if (!crisisType) {
        showValidation('error', 'Please select a crisis type first');
        return;
    }

    try {
        const response = await fetch(`/api/protocols/suggest-actions/${encodeURIComponent(crisisType)}`);
        const result = await response.json();

        if (result.success && result.suggested_actions.length > 0) {
            showSuggestedActions(result.suggested_actions);
        } else {
            showValidation('info', 'No action suggestions available for this crisis type');
        }
    } catch (error) {
        showValidation('error', `Error getting suggestions: ${error.message}`);
    }
}

/**
 * Display suggested actions with full details including criteria scores
 */
function showSuggestedActions(actions) {
    const modalHtml = `
        <div class="modal fade" id="suggestedActionsModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header bg-success text-white">
                        <h5 class="modal-title"><i class="bi bi-stars"></i> AI-Suggested Actions from Knowledge Base</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" style="max-height: 70vh; overflow-y: auto;">
                        <p>These actions are suggested based on expert protocols. Select to add with pre-filled criteria scores:</p>
                        ${actions.map((action, idx) => `
                            <div class="card mb-3">
                                <div class="card-header">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="${idx}" id="suggestedAction${idx}">
                                        <label class="form-check-label" for="suggestedAction${idx}">
                                            <strong>${action.name}</strong>
                                        </label>
                                    </div>
                                </div>
                                <div class="card-body">
                                    <p class="small mb-2">${action.description}</p>
                                    <div class="row small">
                                        <div class="col-6">
                                            <strong>Resources:</strong> ${action.required_resources.join(', ')}<br>
                                            <strong>Duration:</strong> ${action.estimated_duration}<br>
                                            <strong>Risk:</strong> ${(action.risk_level * 100).toFixed(0)}%
                                        </div>
                                        <div class="col-6">
                                            <strong>Criteria Scores:</strong><br>
                                            ${Object.entries(action.criteria_scores).map(([k, v]) =>
                                                `<span class="badge bg-secondary me-1">${k}: ${(v * 100).toFixed(0)}%</span>`
                                            ).join('')}
                                        </div>
                                    </div>
                                    <div class="small text-muted mt-2">
                                        <i class="bi bi-info-circle"></i> Source: Protocol ${action.source_protocol}
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-success" onclick="applySuggestedActions()">
                            <i class="bi bi-check-circle"></i> Add Selected Actions
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if any
    const existing = document.getElementById('suggestedActionsModal');
    if (existing) existing.remove();

    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Store suggested actions globally
    window.currentSuggestedActions = actions;

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('suggestedActionsModal'));
    modal.show();
}

/**
 * Apply selected suggested actions with full details
 */
function applySuggestedActions() {
    const checkboxes = document.querySelectorAll('#suggestedActionsModal input[type="checkbox"]:checked');

    checkboxes.forEach(checkbox => {
        const idx = parseInt(checkbox.value);
        const action = window.currentSuggestedActions[idx];

        // Add action to form
        addAction();
        const actionNum = actionCounter;

        // Populate ALL fields including criteria scores
        document.querySelector(`[name="action_${actionNum}_id"]`).value = action.id;
        document.querySelector(`[name="action_${actionNum}_name"]`).value = action.name;
        document.querySelector(`[name="action_${actionNum}_description"]`).value = action.description;
        document.querySelector(`[name="action_${actionNum}_resources"]`).value = action.required_resources.join(', ');
        document.querySelector(`[name="action_${actionNum}_duration"]`).value = action.estimated_duration;
        document.querySelector(`[name="action_${actionNum}_risk"]`).value = action.risk_level;

        // Fill criteria scores
        const scores = action.criteria_scores;
        document.querySelector(`[name="action_${actionNum}_effectiveness"]`).value = scores.effectiveness;
        document.querySelector(`[name="action_${actionNum}_safety"]`).value = scores.safety;
        document.querySelector(`[name="action_${actionNum}_speed"]`).value = scores.speed;
        document.querySelector(`[name="action_${actionNum}_cost"]`).value = scores.cost;
        document.querySelector(`[name="action_${actionNum}_public_acceptance"]`).value = scores.public_acceptance;
    });

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('suggestedActionsModal'));
    modal.hide();

    showValidation('success', `Added ${checkboxes.length} action(s) with AI-suggested criteria scores`);
}

// ============================================================================
// Event Listeners for Protocol Integration
// ============================================================================

// Add listener to crisis type change
document.addEventListener('DOMContentLoaded', function() {
    const typeSelect = document.getElementById('type');
    if (typeSelect) {
        typeSelect.addEventListener('change', function() {
            loadRelevantProtocols(this.value);
        });

        // Load protocols if type is already selected (edit mode)
        if (typeSelect.value) {
            loadRelevantProtocols(typeSelect.value);
        }
    }
});
