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

app = Flask(__name__)
app.secret_key = 'crisis_mas_web_tools_secret_key_2024'

# Data directory
DATA_DIR = Path(__file__).parent / 'data'
SCENARIOS_FILE = DATA_DIR / 'scenarios.json'
EXPERTS_FILE = DATA_DIR / 'experts.json'

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Initialize data files if they don't exist
if not SCENARIOS_FILE.exists():
    with open(SCENARIOS_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)

if not EXPERTS_FILE.exists():
    with open(EXPERTS_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)


# ============================================================================
# Helper Functions
# ============================================================================

def load_scenarios():
    """Load scenarios from JSON file."""
    with open(SCENARIOS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_scenarios(scenarios):
    """Save scenarios to JSON file."""
    with open(SCENARIOS_FILE, 'w', encoding='utf-8') as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)


def load_experts():
    """Load expert profiles from JSON file."""
    with open(EXPERTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_experts(experts):
    """Save expert profiles to JSON file."""
    with open(EXPERTS_FILE, 'w', encoding='utf-8') as f:
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
    scenarios = load_scenarios()
    experts = load_experts()
    
    stats = {
        'total_scenarios': len(scenarios),
        'total_experts': len(experts),
        'categories': len(set(s.get('category', 'Unknown') for s in scenarios)),
        'languages': sum(1 for s in scenarios if any(ord(c) > 127 for c in s.get('question', '')))
    }
    
    return render_template('index.html', stats=stats)


# ============================================================================
# Routes - Scenarios
# ============================================================================

@app.route('/scenarios')
def scenarios_list():
    """List all scenarios."""
    scenarios = load_scenarios()
    return render_template('scenarios.html', scenarios=scenarios)


@app.route('/scenarios/new', methods=['GET', 'POST'])
def scenario_new():
    """Create new scenario."""
    if request.method == 'POST':
        scenarios = load_scenarios()
        
        # Create new scenario from form data
        new_scenario = {
            'id': generate_id(scenarios, 'scenario'),
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
        
        scenarios.append(new_scenario)
        save_scenarios(scenarios)
        
        flash(f'✅ Scenario "{new_scenario["id"]}" created successfully!', 'success')
        return redirect(url_for('scenarios_list'))
    
    experts = load_experts()
    return render_template('scenario_form.html', scenario=None, experts=experts)


@app.route('/scenarios/<scenario_id>/edit', methods=['GET', 'POST'])
def scenario_edit(scenario_id):
    """Edit existing scenario."""
    scenarios = load_scenarios()
    scenario = next((s for s in scenarios if s['id'] == scenario_id), None)
    
    if not scenario:
        flash(f'❌ Scenario "{scenario_id}" not found!', 'error')
        return redirect(url_for('scenarios_list'))
    
    if request.method == 'POST':
        # Update scenario with form data
        scenario['question'] = request.form.get('question', '').strip()
        scenario['answer'] = request.form.get('answer', '').strip()
        scenario['category'] = request.form.get('category', 'general')
        scenario['location'] = request.form.get('location', '').strip()
        scenario['severity'] = request.form.get('severity', 'medium')
        scenario['resources_required'] = request.form.get('resources_required', '').strip()
        scenario['expert_id'] = request.form.get('expert_id', '').strip()
        scenario['language'] = request.form.get('language', 'en')
        scenario['tags'] = [tag.strip() for tag in request.form.get('tags', '').split(',') if tag.strip()]
        scenario['updated_at'] = datetime.now().isoformat()
        
        save_scenarios(scenarios)
        
        flash(f'✅ Scenario "{scenario_id}" updated successfully!', 'success')
        return redirect(url_for('scenarios_list'))
    
    experts = load_experts()
    return render_template('scenario_form.html', scenario=scenario, experts=experts)


@app.route('/scenarios/<scenario_id>/delete', methods=['POST'])
def scenario_delete(scenario_id):
    """Delete scenario."""
    scenarios = load_scenarios()
    scenarios = [s for s in scenarios if s['id'] != scenario_id]
    save_scenarios(scenarios)
    
    flash(f'✅ Scenario "{scenario_id}" deleted successfully!', 'success')
    return redirect(url_for('scenarios_list'))


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

@app.route('/api/scenarios')
def api_scenarios():
    """Get all scenarios as JSON."""
    scenarios = load_scenarios()
    return jsonify(scenarios)


@app.route('/api/experts')
def api_experts():
    """Get all experts as JSON."""
    experts = load_experts()
    return jsonify(experts)


@app.route('/api/export')
def api_export():
    """Export all data as combined JSON."""
    scenarios = load_scenarios()
    experts = load_experts()
    
    export_data = {
        'scenarios': scenarios,
        'experts': experts,
        'exported_at': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    return jsonify(export_data)


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
    print("📱 Open your browser to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
