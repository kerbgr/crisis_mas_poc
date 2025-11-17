"""
Web Tools for Crisis MAS - LLM Training
User-friendly web interface for managing scenarios and expert profiles

Author: kerbgr
"""
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import json
import os
from datetime import datetime
from pathlib import Path
from protocol_integration import get_protocol_integration

app = Flask(__name__)
app.secret_key = 'crisis_mas_web_tools_secret_key_2024'

# Data directories
DATA_DIR = Path(__file__).parent / 'data'
PROTOCOLS_FILE = DATA_DIR / 'scenarios.json'  # Keep filename for backward compatibility

# Expert profiles - use agent_profiles.json from agents directory
AGENTS_PROFILES_FILE = Path(__file__).parent.parent / 'agents' / 'agent_profiles.json'
EXPERTS_FILE = AGENTS_PROFILES_FILE  # Default to agent profiles

# Crisis scenarios directory (in project root)
CRISIS_SCENARIOS_DIR = Path(__file__).parent.parent / 'scenarios'

# Configuration file for custom paths
CONFIG_FILE = DATA_DIR / 'config.json'

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Initialize data files if they don't exist
if not PROTOCOLS_FILE.exists():
    with open(PROTOCOLS_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# Load configuration
def load_config():
    """Load configuration with custom file paths."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'experts_file': str(AGENTS_PROFILES_FILE),
        'scenarios_dir': str(CRISIS_SCENARIOS_DIR)
    }

def save_config(config):
    """Save configuration."""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

# Load config and update paths
config = load_config()
if 'experts_file' in config:
    EXPERTS_FILE = Path(config['experts_file'])


# ============================================================================
# Helper Functions
# ============================================================================

def load_protocols():
    """Load incident handling protocols from JSON file."""
    with open(PROTOCOLS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_protocols(protocols):
    """Save incident handling protocols to JSON file."""
    with open(PROTOCOLS_FILE, 'w', encoding='utf-8') as f:
        json.dump(protocols, f, ensure_ascii=False, indent=2)


def load_crisis_scenarios():
    """Load crisis scenarios from /scenarios directory."""
    scenarios = []
    if CRISIS_SCENARIOS_DIR.exists():
        for json_file in CRISIS_SCENARIOS_DIR.glob('*.json'):
            # Skip template and non-scenario files
            if json_file.name in ['scenario_template.json', 'criteria_weights.json']:
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    scenario = json.load(f)
                    scenario['_filename'] = json_file.name
                    scenarios.append(scenario)
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
    return scenarios


def save_crisis_scenario(scenario, filename=None, save_dir=None):
    """Save crisis scenario to JSON file."""
    if save_dir is None:
        save_dir = CRISIS_SCENARIOS_DIR
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        # Generate filename from scenario id and type
        scenario_id = scenario.get('id', 'scenario')
        filename = f"{scenario_id}.json"

    file_path = save_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(scenario, f, ensure_ascii=False, indent=2)

    return file_path


def load_experts():
    """Load expert profiles from JSON file.
    Supports both formats:
    1. Flat array: [{"id": "expert_001", ...}, ...]
    2. Nested format: {"agents": [{"agent_id": "...", ...}, ...]}
    """
    if not EXPERTS_FILE.exists():
        return []

    with open(EXPERTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle nested format (agent_profiles.json)
    if isinstance(data, dict) and 'agents' in data:
        experts = data['agents']
        # Map agent_id to id for compatibility with web tools
        for expert in experts:
            if 'agent_id' in expert and 'id' not in expert:
                expert['id'] = expert['agent_id']
        return experts

    # Handle flat array format
    return data


def save_experts(experts):
    """Save expert profiles to JSON file.
    Detects format and saves accordingly."""
    # Check if original file has nested format
    original_format = 'flat'
    if EXPERTS_FILE.exists():
        with open(EXPERTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'agents' in data:
                original_format = 'nested'

    # Ensure agent_id is synced with id for nested format
    for expert in experts:
        if 'id' in expert and 'agent_id' not in expert:
            expert['agent_id'] = expert['id']
        elif 'agent_id' in expert and 'id' not in expert:
            expert['id'] = expert['agent_id']

    with open(EXPERTS_FILE, 'w', encoding='utf-8') as f:
        if original_format == 'nested':
            json.dump({'agents': experts}, f, ensure_ascii=False, indent=2)
        else:
            json.dump(experts, f, ensure_ascii=False, indent=2)


def generate_id(items, prefix='item'):
    """Generate unique ID for new item."""
    if not items:
        return f"{prefix}_001"
    
    # Extract numeric part from IDs
    ids = [int(item['id'].split('_')[-1]) for item in items if 'id' in item]
    next_num = max(ids) + 1 if ids else 1
    return f"{prefix}_{next_num:03d}"


# ============================================================================
# Routes - Home
# ============================================================================

@app.route('/')
def index():
    """Home page with overview."""
    protocols = load_protocols()
    experts = load_experts()
    crisis_scenarios = load_crisis_scenarios()

    stats = {
        'total_protocols': len(protocols),
        'total_experts': len(experts),
        'categories': len(set(s.get('category', 'Unknown') for s in protocols)),
        'crisis_scenarios': len(crisis_scenarios)
    }

    return render_template('index.html', stats=stats)


# ============================================================================
# Routes - Incident Handling Protocols (LLM Training)
# ============================================================================

@app.route('/protocols')
def protocols_list():
    """List all incident handling protocols."""
    protocols = load_protocols()
    return render_template('scenarios.html', protocols=protocols)


@app.route('/protocols/new', methods=['GET', 'POST'])
def protocol_new():
    """Create new incident handling protocol."""
    if request.method == 'POST':
        protocols = load_protocols()

        # Create new protocol from form data
        new_protocol = {
            'id': generate_id(protocols, 'protocol'),
            'question': request.form.get('question', '').strip(),
            'answer': request.form.get('answer', '').strip(),
            'category': request.form.get('category', 'general'),
            'location': request.form.get('location', '').strip(),
            'severity': request.form.get('severity', 'medium'),
            'resources_required': request.form.get('resources_required', '').strip(),
            'expert_id': request.form.get('expert_id', '').strip(),
            'language': request.form.get('language', 'en'),
            'tags': [tag.strip() for tag in request.form.get('tags', '').split(',') if tag.strip()],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        protocols.append(new_protocol)
        save_protocols(protocols)

        flash(f'✅ Protocol "{new_protocol["id"]}" created successfully!', 'success')
        return redirect(url_for('protocols_list'))

    experts = load_experts()
    return render_template('scenario_form.html', protocol=None, experts=experts)


@app.route('/protocols/<protocol_id>/edit', methods=['GET', 'POST'])
def protocol_edit(protocol_id):
    """Edit existing incident handling protocol."""
    protocols = load_protocols()
    protocol = next((s for s in protocols if s['id'] == protocol_id), None)

    if not protocol:
        flash(f'❌ Protocol "{protocol_id}" not found!', 'error')
        return redirect(url_for('protocols_list'))

    if request.method == 'POST':
        # Update protocol with form data
        protocol['question'] = request.form.get('question', '').strip()
        protocol['answer'] = request.form.get('answer', '').strip()
        protocol['category'] = request.form.get('category', 'general')
        protocol['location'] = request.form.get('location', '').strip()
        protocol['severity'] = request.form.get('severity', 'medium')
        protocol['resources_required'] = request.form.get('resources_required', '').strip()
        protocol['expert_id'] = request.form.get('expert_id', '').strip()
        protocol['language'] = request.form.get('language', 'en')
        protocol['tags'] = [tag.strip() for tag in request.form.get('tags', '').split(',') if tag.strip()]
        protocol['updated_at'] = datetime.now().isoformat()

        save_protocols(protocols)

        flash(f'✅ Protocol "{protocol_id}" updated successfully!', 'success')
        return redirect(url_for('protocols_list'))

    experts = load_experts()
    return render_template('scenario_form.html', protocol=protocol, experts=experts)


@app.route('/protocols/<protocol_id>/delete', methods=['POST'])
def protocol_delete(protocol_id):
    """Delete incident handling protocol."""
    protocols = load_protocols()
    protocols = [s for s in protocols if s['id'] != protocol_id]
    save_protocols(protocols)

    flash(f'✅ Protocol "{protocol_id}" deleted successfully!', 'success')
    return redirect(url_for('protocols_list'))


# ============================================================================
# Routes - Crisis Scenarios (Multi-Agent Simulation)
# ============================================================================

@app.route('/crisis-scenarios')
def crisis_scenarios_list():
    """List all crisis scenarios."""
    scenarios = load_crisis_scenarios()
    return render_template('crisis_scenarios.html', scenarios=scenarios)


@app.route('/crisis-scenarios/new', methods=['GET', 'POST'])
def crisis_scenario_new():
    """Create new crisis scenario."""
    if request.method == 'POST':
        # This will be handled by JavaScript/AJAX for complex form
        pass

    return render_template('crisis_scenario_form_enhanced.html', scenario=None, mode='create')


@app.route('/crisis-scenarios/<filename>/view')
def crisis_scenario_view(filename):
    """View crisis scenario details."""
    scenario_path = CRISIS_SCENARIOS_DIR / filename
    if not scenario_path.exists():
        flash(f'❌ Scenario "{filename}" not found!', 'error')
        return redirect(url_for('crisis_scenarios_list'))

    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario = json.load(f)

    scenario['_filename'] = filename
    return render_template('crisis_scenario_view.html', scenario=scenario)


@app.route('/crisis-scenarios/<filename>/edit', methods=['GET', 'POST'])
def crisis_scenario_edit(filename):
    """Edit existing crisis scenario."""
    scenario_path = CRISIS_SCENARIOS_DIR / filename
    if not scenario_path.exists():
        flash(f'❌ Scenario "{filename}" not found!', 'error')
        return redirect(url_for('crisis_scenarios_list'))

    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario = json.load(f)

    scenario['_filename'] = filename
    return render_template('crisis_scenario_form_enhanced.html', scenario=scenario, mode='edit')


@app.route('/api/crisis-scenarios/save', methods=['POST'])
def api_crisis_scenario_save():
    """API endpoint to save crisis scenario."""
    try:
        scenario_data = request.json
        filename = scenario_data.get('_filename')
        save_location = scenario_data.get('_save_location', str(CRISIS_SCENARIOS_DIR))

        # Remove metadata fields
        if '_filename' in scenario_data:
            del scenario_data['_filename']
        if '_save_location' in scenario_data:
            del scenario_data['_save_location']

        file_path = save_crisis_scenario(scenario_data, filename, save_location)

        return jsonify({
            'success': True,
            'message': f'Scenario saved to {file_path}',
            'filename': file_path.name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


# ============================================================================
# Routes - Experts
# ============================================================================

@app.route('/experts')
def experts_list():
    """List all expert profiles."""
    experts = load_experts()
    return render_template('experts.html', experts=experts)


@app.route('/experts/new', methods=['GET', 'POST'])
def expert_new():
    """Create new expert profile."""
    if request.method == 'POST':
        experts = load_experts()
        
        # Create new expert from form data
        new_expert = {
            'id': request.form.get('expert_id', '').strip() or generate_id(experts, 'expert'),
            'name': request.form.get('name', '').strip(),
            'role': request.form.get('role', '').strip(),
            'organization': request.form.get('organization', '').strip(),
            'location': request.form.get('location', '').strip(),
            'experience_years': int(request.form.get('experience_years', 0)),
            'specializations': [spec.strip() for spec in request.form.get('specializations', '').split(',') if spec.strip()],
            'certifications': [cert.strip() for cert in request.form.get('certifications', '').split(',') if cert.strip()],
            'email': request.form.get('email', '').strip(),
            'phone': request.form.get('phone', '').strip(),
            'languages': [lang.strip() for lang in request.form.get('languages', '').split(',') if lang.strip()],
            'availability': request.form.get('availability', 'available'),
            'bio': request.form.get('bio', '').strip(),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        experts.append(new_expert)
        save_experts(experts)
        
        flash(f'✅ Expert "{new_expert["name"]}" created successfully!', 'success')
        return redirect(url_for('experts_list'))
    
    return render_template('expert_form.html', expert=None)


@app.route('/experts/<expert_id>/edit', methods=['GET', 'POST'])
def expert_edit(expert_id):
    """Edit existing expert profile."""
    experts = load_experts()
    expert = next((e for e in experts if e['id'] == expert_id), None)
    
    if not expert:
        flash(f'❌ Expert "{expert_id}" not found!', 'error')
        return redirect(url_for('experts_list'))
    
    if request.method == 'POST':
        # Update expert with form data
        expert['name'] = request.form.get('name', '').strip()
        expert['role'] = request.form.get('role', '').strip()
        expert['organization'] = request.form.get('organization', '').strip()
        expert['location'] = request.form.get('location', '').strip()
        expert['experience_years'] = int(request.form.get('experience_years', 0))
        expert['specializations'] = [spec.strip() for spec in request.form.get('specializations', '').split(',') if spec.strip()]
        expert['certifications'] = [cert.strip() for cert in request.form.get('certifications', '').split(',') if cert.strip()]
        expert['email'] = request.form.get('email', '').strip()
        expert['phone'] = request.form.get('phone', '').strip()
        expert['languages'] = [lang.strip() for lang in request.form.get('languages', '').split(',') if lang.strip()]
        expert['availability'] = request.form.get('availability', 'available')
        expert['bio'] = request.form.get('bio', '').strip()
        expert['updated_at'] = datetime.now().isoformat()
        
        save_experts(experts)
        
        flash(f'✅ Expert "{expert["name"]}" updated successfully!', 'success')
        return redirect(url_for('experts_list'))
    
    return render_template('expert_form.html', expert=expert)


@app.route('/experts/<expert_id>/delete', methods=['POST'])
def expert_delete(expert_id):
    """Delete expert profile."""
    experts = load_experts()
    experts = [e for e in experts if e['id'] != expert_id]
    save_experts(experts)
    
    flash(f'✅ Expert profile deleted successfully!', 'success')
    return redirect(url_for('experts_list'))


# ============================================================================
# API Routes (for AJAX operations)
# ============================================================================

@app.route('/api/protocols')
def api_protocols():
    """Get all incident handling protocols as JSON."""
    protocols = load_protocols()
    return jsonify(protocols)


@app.route('/api/crisis-scenarios')
def api_crisis_scenarios():
    """Get all crisis scenarios as JSON."""
    scenarios = load_crisis_scenarios()
    return jsonify(scenarios)


@app.route('/api/experts')
def api_experts():
    """Get all experts as JSON."""
    experts = load_experts()
    return jsonify(experts)


@app.route('/api/export')
def api_export():
    """Export all data as combined JSON."""
    protocols = load_protocols()
    crisis_scenarios = load_crisis_scenarios()
    experts = load_experts()

    export_data = {
        'protocols': protocols,
        'crisis_scenarios': crisis_scenarios,
        'experts': experts,
        'exported_at': datetime.now().isoformat(),
        'version': '2.0'
    }

    return jsonify(export_data)


# ============================================================================
# API Routes - Protocol Integration
# ============================================================================

@app.route('/api/protocols/relevant/<crisis_type>')
def api_get_relevant_protocols(crisis_type):
    """
    Get protocols relevant to a crisis type.

    Args:
        crisis_type: Type of crisis (wildfire, flood, etc.)

    Returns:
        JSON list of relevant protocols
    """
    try:
        integrator = get_protocol_integration()
        category = request.args.get('category')
        limit = int(request.args.get('limit', 5))

        relevant_protocols = integrator.get_relevant_protocols(
            crisis_type=crisis_type,
            category=category,
            limit=limit
        )

        return jsonify({
            'success': True,
            'crisis_type': crisis_type,
            'count': len(relevant_protocols),
            'protocols': relevant_protocols
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/protocols/suggest-actions/<crisis_type>')
def api_suggest_actions(crisis_type):
    """
    Suggest response actions based on protocol knowledge.

    Args:
        crisis_type: Type of crisis

    Returns:
        JSON list of suggested actions
    """
    try:
        integrator = get_protocol_integration()
        context = request.args.get('context', '')

        suggestions = integrator.suggest_action_from_protocol(
            crisis_type=crisis_type,
            action_context=context
        )

        return jsonify({
            'success': True,
            'crisis_type': crisis_type,
            'count': len(suggestions),
            'suggested_actions': suggestions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/protocols/extract-actions', methods=['POST'])
def api_extract_actions():
    """
    Extract potential actions from a protocol answer.

    Expects JSON body with 'protocol_id' or 'answer' field.

    Returns:
        JSON list of extracted actions
    """
    try:
        data = request.json
        integrator = get_protocol_integration()

        if 'protocol_id' in data:
            # Find protocol by ID
            protocols = load_protocols()
            protocol = next((p for p in protocols if p['id'] == data['protocol_id']), None)
            if not protocol:
                return jsonify({
                    'success': False,
                    'error': 'Protocol not found'
                }), 404
        elif 'answer' in data:
            # Use provided answer
            protocol = {'answer': data['answer']}
        else:
            return jsonify({
                'success': False,
                'error': 'Either protocol_id or answer required'
            }), 400

        extracted = integrator.extract_actions_from_protocol(protocol)

        return jsonify({
            'success': True,
            'count': len(extracted),
            'extracted_actions': extracted
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


# ============================================================================
# Settings Routes
# ============================================================================

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """View and update application settings."""
    global EXPERTS_FILE, config

    if request.method == 'POST':
        # Update experts file path
        new_experts_path = request.form.get('experts_file', '').strip()

        if new_experts_path:
            experts_path = Path(new_experts_path)

            # Validate path exists
            if not experts_path.exists():
                flash(f'❌ File not found: {new_experts_path}', 'error')
            else:
                # Update configuration
                config['experts_file'] = new_experts_path
                save_config(config)

                # Update global variable
                EXPERTS_FILE = experts_path

                flash('✅ Settings saved successfully!', 'success')
                return redirect(url_for('settings'))

    # Load current configuration
    current_config = load_config()

    # Get expert file stats
    expert_stats = {
        'file_path': str(EXPERTS_FILE),
        'exists': EXPERTS_FILE.exists(),
        'count': 0,
        'format': 'Unknown'
    }

    if EXPERTS_FILE.exists():
        try:
            with open(EXPERTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'agents' in data:
                    expert_stats['count'] = len(data['agents'])
                    expert_stats['format'] = 'Nested (agent_profiles.json)'
                elif isinstance(data, list):
                    expert_stats['count'] = len(data)
                    expert_stats['format'] = 'Flat array'
        except Exception as e:
            expert_stats['error'] = str(e)

    return render_template('settings.html',
                          config=current_config,
                          expert_stats=expert_stats)


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    flash('❌ Page not found!', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    flash('❌ Internal server error!', 'error')
    return redirect(url_for('index'))


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Crisis MAS - Web Tools for LLM Training")
    print("=" * 60)
    print("\n🚀 Starting web server...")
    print("📱 Open your browser to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
