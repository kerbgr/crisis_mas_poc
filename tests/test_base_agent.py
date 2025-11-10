"""
Test script for BaseAgent class
Demonstrates usage and validates functionality
"""

from agents.base_agent import BaseAgent, load_agent_by_id, list_available_agents


# Create a concrete implementation for testing
class TestAgent(BaseAgent):
    """Test implementation of BaseAgent."""

    def evaluate_scenario(self, scenario):
        """Simple test implementation."""
        return {
            'agent_id': self.agent_id,
            'evaluation': 'Test evaluation',
            'confidence': self.confidence_level
        }

    def propose_action(self, scenario, criteria):
        """Simple test implementation."""
        return {
            'agent_id': self.agent_id,
            'action': 'Test action',
            'confidence': self.confidence_level
        }


def main():
    print("="*70)
    print("Testing BaseAgent Class")
    print("="*70)

    # Test 1: List available agents
    print("\n1. Listing available agents:")
    print("-" * 70)
    try:
        agents = list_available_agents()
        for agent in agents:
            print(f"  • {agent['agent_id']:30s} - {agent['name']:25s} ({agent['role']})")
        print(f"\nTotal agents: {len(agents)}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Load agent by ID
    print("\n\n2. Loading agent profile directly:")
    print("-" * 70)
    try:
        profile = load_agent_by_id("agent_meteorologist")
        print(f"Name: {profile['name']}")
        print(f"Role: {profile['role']}")
        print(f"Expertise: {profile['expertise']}")
        print(f"Experience: {profile['experience_years']} years")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Instantiate agent and get info
    print("\n\n3. Instantiating agent 'agent_meteorologist':")
    print("-" * 70)
    try:
        agent = TestAgent("agent_meteorologist", "agents/agent_profiles.json")

        print(f"\nAgent created successfully!")
        print(f"String representation:\n{agent}")

        print(f"\n\nDetailed info:")
        info = agent.get_agent_info()
        for key, value in info.items():
            if key != 'weight_preferences':
                print(f"  {key:20s}: {value}")
            else:
                print(f"  {key:20s}:")
                for criterion, weight in value.items():
                    print(f"    • {criterion:20s}: {weight:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Test with different agents
    print("\n\n4. Testing multiple agent types:")
    print("-" * 70)
    test_agents = ["medical_expert_01", "logistics_expert_01", "public_safety_expert_01"]

    for agent_id in test_agents:
        try:
            agent = TestAgent(agent_id, "agents/agent_profiles.json")
            print(f"\n{agent.name} ({agent.role})")
            print(f"  Expertise: {agent.expertise}")
            print(f"  Experience: {agent.experience_years} years")
            print(f"  Risk Tolerance: {agent.risk_tolerance:.2f}")
            print(f"  Confidence: {agent.confidence_level:.2f}")
        except Exception as e:
            print(f"  Error loading {agent_id}: {e}")

    # Test 5: Test error handling - invalid agent ID
    print("\n\n5. Testing error handling (invalid agent ID):")
    print("-" * 70)
    try:
        agent = TestAgent("non_existent_agent", "agents/agent_profiles.json")
    except ValueError as e:
        print(f"✓ ValueError caught as expected:\n  {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # Test 6: Test error handling - invalid file path
    print("\n\n6. Testing error handling (invalid file path):")
    print("-" * 70)
    try:
        agent = TestAgent("agent_meteorologist", "nonexistent/path.json")
    except FileNotFoundError as e:
        print(f"✓ FileNotFoundError caught as expected:\n  {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # Test 7: Test decision logging
    print("\n\n7. Testing decision logging:")
    print("-" * 70)
    try:
        agent = TestAgent("agent_meteorologist", "agents/agent_profiles.json")

        # Log some decisions
        agent.log_decision({'action': 'evacuate', 'priority': 'high'})
        agent.log_decision({'action': 'monitor', 'priority': 'medium'})
        agent.log_decision({'action': 'alert', 'priority': 'high'})

        history = agent.get_decision_history()
        print(f"Logged {len(history)} decisions")

        for i, entry in enumerate(history, 1):
            print(f"\nDecision {i}:")
            print(f"  Timestamp: {entry['timestamp']}")
            print(f"  Agent: {entry['agent_name']}")
            print(f"  Action: {entry['decision']['action']}")
            print(f"  Priority: {entry['decision']['priority']}")

    except Exception as e:
        print(f"Error: {e}")

    # Test 8: Test confidence update
    print("\n\n8. Testing confidence update:")
    print("-" * 70)
    try:
        agent = TestAgent("agent_meteorologist", "agents/agent_profiles.json")
        initial_confidence = agent.confidence_level

        print(f"Initial confidence: {initial_confidence:.3f}")

        # Update with high accuracy
        agent.update_confidence({'accuracy': 0.95})
        print(f"After high accuracy feedback (0.95): {agent.confidence_level:.3f}")

        # Update with low accuracy
        agent.update_confidence({'accuracy': 0.30})
        print(f"After low accuracy feedback (0.30): {agent.confidence_level:.3f}")

        # Update with perfect accuracy
        agent.update_confidence({'accuracy': 1.0})
        print(f"After perfect accuracy feedback (1.0): {agent.confidence_level:.3f}")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
