#!/usr/bin/env python3
"""
Generate updated agent network visualization with all 11 expert agents.

This script creates an agent_network.png showing the complete 11-agent
emergency response command structure introduced in v0.8.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.visualizations import SystemVisualizer
from scenarios.scenario_loader import ScenarioLoader


def load_agent_profiles():
    """Load the 11 current expert agent profiles from agent_profiles.json"""
    profiles_path = Path(__file__).parent / "agents" / "agent_profiles.json"

    with open(profiles_path, 'r') as f:
        data = json.load(f)

    # Current 11 expert agents (v0.8)
    current_experts = {
        'agent_meteorologist',
        'medical_expert_01',
        'logistics_expert_01',
        'psap_commander_01',
        'police_onscene_01',
        'police_regional_01',
        'fire_onscene_01',
        'fire_regional_01',
        'medical_infrastructure_01',
        'coastguard_onscene_01',
        'coastguard_national_01'
    }

    # Convert list to dictionary keyed by agent_id (only current experts)
    profiles_dict = {}
    for agent in data['agents']:
        if agent['agent_id'] in current_experts:
            profiles_dict[agent['agent_id']] = {
                'name': agent['name'],
                'expertise': agent['role'],
                'role': agent['role'],
                'description': agent.get('description', ''),
                'hierarchy': agent.get('hierarchy', 'tactical')
            }

    return profiles_dict


def create_trust_matrix(agent_ids):
    """
    Create a hierarchical trust matrix showing command structure.

    Strategic commanders have higher trust with other strategic commanders.
    Tactical commanders have higher trust with other tactical commanders.
    Same-domain experts have strong trust (police-police, fire-fire, etc.)
    """
    trust = {}

    # Define hierarchy and domains
    strategic = {'police_regional_01', 'fire_regional_01', 'coastguard_national_01'}
    tactical = {'police_onscene_01', 'fire_onscene_01', 'coastguard_onscene_01'}
    core = {'agent_meteorologist', 'logistics_expert_01', 'medical_expert_01'}
    coordinators = {'psap_commander_01'}
    infrastructure = {'medical_infrastructure_01'}

    for agent_i in agent_ids:
        trust[agent_i] = {}

        for agent_j in agent_ids:
            if agent_i == agent_j:
                continue

            # Base trust
            weight = 0.5

            # Strategic commanders have high trust with each other
            if agent_i in strategic and agent_j in strategic:
                weight = 0.9

            # Tactical commanders have high trust with each other
            elif agent_i in tactical and agent_j in tactical:
                weight = 0.85

            # Same domain (police-police, fire-fire, coast guard-coast guard)
            elif 'police' in agent_i and 'police' in agent_j:
                weight = 0.95
            elif 'fire' in agent_i and 'fire' in agent_j:
                weight = 0.95
            elif 'coastguard' in agent_i and 'coastguard' in agent_j:
                weight = 0.95

            # Tactical-Strategic in same domain
            elif (agent_i in tactical and agent_j in strategic) or (agent_i in strategic and agent_j in tactical):
                if 'police' in agent_i and 'police' in agent_j:
                    weight = 0.9
                elif 'fire' in agent_i and 'fire' in agent_j:
                    weight = 0.9
                elif 'coastguard' in agent_i and 'coastguard' in agent_j:
                    weight = 0.9

            # PSAP commander has high trust with all tactical commanders
            elif agent_i in coordinators and agent_j in tactical:
                weight = 0.85
            elif agent_i in tactical and agent_j in coordinators:
                weight = 0.85

            # Core experts have moderate trust with everyone
            elif agent_i in core or agent_j in core:
                weight = 0.7

            # Medical infrastructure has high trust with medical expert
            elif ('medical' in agent_i and 'medical' in agent_j):
                weight = 0.9

            trust[agent_i][agent_j] = weight

    return trust


def main():
    """Generate agent network visualization"""
    print("🎨 Generating Agent Network Visualization...")
    print("=" * 60)

    # Load agent profiles
    print("\n📋 Loading agent profiles...")
    profiles = load_agent_profiles()
    print(f"✓ Loaded {len(profiles)} agent profiles")

    # Create trust matrix
    print("\n🔗 Creating hierarchical trust matrix...")
    agent_ids = list(profiles.keys())
    trust_matrix = create_trust_matrix(agent_ids)
    print(f"✓ Created trust matrix with {len(trust_matrix)} nodes")

    # Generate visualization
    print("\n🖼️  Generating network visualization...")
    viz = SystemVisualizer(output_dir="results/visualizations")

    save_path = viz.plot_agent_network(
        agent_profiles=profiles,
        trust_matrix=trust_matrix,
        save_path="agent_network_11_experts.png",
        title="Crisis MAS - 11 Expert Emergency Response Network (v0.8)"
    )

    print(f"✓ Visualization saved to: {save_path}")

    # Print network statistics
    print("\n📊 Network Statistics:")
    print(f"  • Total Agents: {len(profiles)}")
    print(f"  • Core Experts: 3 (Meteorologist, Logistics, Medical)")
    print(f"  • Tactical Commanders: 3 (Police, Fire, Coast Guard)")
    print(f"  • Strategic Commanders: 3 (Regional Police, Regional Fire, National Coast Guard)")
    print(f"  • Coordinators: 1 (PSAP Commander)")
    print(f"  • Infrastructure: 1 (Medical Infrastructure Director)")

    print("\n✅ Agent network visualization complete!")
    print(f"   View the network: {save_path}")

    return save_path


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error generating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
