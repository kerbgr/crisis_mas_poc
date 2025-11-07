# Agent Development Guide

**Crisis Management Multi-Agent System**

This guide explains how to create custom agents for the Crisis MAS using the provided agent template.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Agent Template Overview](#agent-template-overview)
3. [Step-by-Step Development](#step-by-step-development)
4. [Required Methods](#required-methods)
5. [LLM Integration](#llm-integration)
6. [Testing Your Agent](#testing-your-agent)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

---

## Quick Start

### 1. Copy the Template

```bash
cd agents/
cp agent_template.py my_custom_agent.py
```

### 2. Rename the Class

Edit `my_custom_agent.py`:

```python
class MyCustomAgent(BaseAgent):  # Change from CustomAgentTemplate
    """
    My custom agent for [specific domain].
    """
```

### 3. Create Agent Profile

Add to `agents/agent_profiles.json`:

```json
{
  "agent_id": "agent_custom_001",
  "name": "Custom Expert",
  "role": "Custom Specialist",
  "expertise": "custom_domain",
  "experience_years": 10,
  "risk_tolerance": 0.5,
  "weight_preferences": {
    "effectiveness": 0.30,
    "safety": 0.25,
    "speed": 0.20,
    "cost": 0.15,
    "public_acceptance": 0.10
  },
  "confidence_level": 0.85,
  "description": "Expert in custom domain analysis",
  "expertise_tags": ["custom", "specialist", "domain_expert"]
}
```

### 4. Test Your Agent

```python
from agents.my_custom_agent import MyCustomAgent

agent = MyCustomAgent(agent_id="agent_custom_001", use_llm=False)
assessment = agent.evaluate_scenario(scenario, alternatives)
print(f"Recommendation: {assessment['recommended_alternative']}")
```

---

## Agent Template Overview

The template (`agent_template.py`) provides a complete, documented starting point with:

### Core Components

1. **Base Inheritance**: Inherits from `BaseAgent` (loads profile, manages state)
2. **LLM Integration**: Optional support for Claude, OpenAI, or LM Studio
3. **Abstract Method Implementations**: Required `evaluate_scenario()` and `propose_action()`
4. **Helper Methods**: Private methods for LLM and rule-based reasoning
5. **Documentation**: Extensive comments explaining each section

### What's Included

✅ Profile loading from JSON
✅ LLM client integration (optional)
✅ Rule-based fallback logic
✅ Error handling and validation
✅ Logging and debugging
✅ Type hints and docstrings
✅ Usage examples

---

## Step-by-Step Development

### Step 1: Define Agent Purpose

**Questions to answer:**
- What domain expertise does this agent represent?
- What unique perspective does it bring to crisis decisions?
- How does it differ from existing agents (medical, logistics, safety, environmental)?

**Example:**
```
Agent Type: Economic/Financial Expert
Domain: Cost-benefit analysis, budget constraints, economic impact
Unique Value: Assesses financial feasibility and economic consequences
```

### Step 2: Configure Agent Profile

Create profile in `agents/agent_profiles.json`:

```json
{
  "agent_id": "agent_economic",
  "name": "Dr. Sarah Chen",
  "role": "Chief Economic Advisor",
  "expertise": "economic_analysis",
  "experience_years": 15,
  "risk_tolerance": 0.4,  // More conservative with finances
  "weight_preferences": {
    "effectiveness": 0.20,
    "safety": 0.20,
    "speed": 0.15,
    "cost": 0.35,          // Higher weight on cost
    "public_acceptance": 0.10
  },
  "confidence_level": 0.90,
  "description": "Expert in crisis economics and resource allocation",
  "expertise_tags": ["economics", "finance", "budget", "cost_analysis"]
}
```

### Step 3: Customize `evaluate_scenario()`

This method analyzes scenarios and returns assessments.

#### Method Signature

```python
def evaluate_scenario(
    self,
    scenario: Dict[str, Any],
    alternatives: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
```

#### What It Should Do

1. **Validate inputs**: Check scenario has required fields
2. **Analyze scenario**: Apply domain expertise
3. **Evaluate alternatives**: Score each option
4. **Generate beliefs**: Create probability distribution
5. **Provide reasoning**: Explain the assessment
6. **Return structured output**: Follow standard format

#### Implementation Strategies

**A. Rule-Based (No LLM)**

```python
def _evaluate_rule_based(self, scenario, alternatives):
    severity = scenario.get('severity', 0.5)
    affected_pop = scenario.get('affected_population', 0)

    # Domain-specific logic
    economic_impact = self._calculate_economic_impact(scenario)
    budget_constraints = self._check_budget_constraints(scenario)

    belief_dist = {}
    for alt in alternatives:
        # Score based on cost-effectiveness
        cost_score = alt.get('cost', 0.5)
        effectiveness_score = alt.get('effectiveness', 0.5)

        # Economic utility function
        utility = (effectiveness_score / cost_score) * self.risk_tolerance
        belief_dist[alt['id']] = utility

    # Normalize
    total = sum(belief_dist.values())
    belief_dist = {k: v/total for k, v in belief_dist.items()}

    return {
        'belief_distribution': belief_dist,
        'confidence': 0.8,
        'reasoning': f"Economic analysis shows...",
        # ... other required fields
    }
```

**B. LLM-Enhanced**

```python
def _evaluate_with_llm(self, scenario, alternatives):
    # Generate domain-specific prompt
    prompt = f"""As an economic expert, analyze this crisis:

    Scenario: {scenario['description']}
    Budget Impact: ${scenario.get('estimated_cost', 'unknown')}

    Alternatives:
    {self._format_alternatives(alternatives)}

    Provide cost-benefit analysis and rankings."""

    # Call LLM
    response = self.llm_client.generate_assessment(
        prompt=prompt,
        system_prompt=self._generate_system_prompt()
    )

    # Parse and return
    return self._parse_llm_assessment(response, scenario, alternatives)
```

### Step 4: Customize `propose_action()`

This method suggests actions based on agent expertise.

#### Implementation Example

```python
def _propose_rule_based(self, scenario, criteria):
    # Analyze scenario from economic perspective
    severity = scenario.get('severity', 0.5)
    budget = scenario.get('available_budget', 1000000)

    # Determine cost-effective action
    if severity > 0.8 and budget > 500000:
        action = "Full Resource Mobilization"
        estimated_cost = budget * 0.8
    elif severity > 0.5:
        action = "Targeted Intervention"
        estimated_cost = budget * 0.5
    else:
        action = "Monitoring and Preparation"
        estimated_cost = budget * 0.2

    return {
        'agent_id': self.agent_id,
        'proposed_action': {
            'name': action,
            'description': f"Allocate ${estimated_cost:,.0f} for {action.lower()}",
            'estimated_impact': {
                'cost': estimated_cost / budget,  # Fraction of budget
                'effectiveness': severity * 0.9,
                # ... other criteria
            },
            'implementation_steps': [
                "Allocate budget",
                "Procure resources",
                "Deploy personnel",
                "Monitor expenditure"
            ],
            'required_resources': [f"${estimated_cost:,.0f}"],
            'timeline': "24-48 hours"
        },
        'justification': "Cost-benefit analysis indicates...",
        'confidence': self.confidence_level
    }
```

### Step 5: Add Custom Helper Methods

Create domain-specific helper methods:

```python
def _calculate_economic_impact(self, scenario: Dict[str, Any]) -> float:
    """Calculate estimated economic impact of crisis."""
    affected_pop = scenario.get('affected_population', 0)
    severity = scenario.get('severity', 0.5)

    # Simple economic model (customize this!)
    per_capita_impact = 1000  # $1000 per person
    impact = affected_pop * per_capita_impact * severity

    return impact

def _check_budget_constraints(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Assess budget constraints for response."""
    available_budget = scenario.get('available_budget', 0)
    estimated_cost = scenario.get('estimated_cost', 0)

    return {
        'budget_available': available_budget,
        'estimated_cost': estimated_cost,
        'is_affordable': estimated_cost <= available_budget,
        'deficit': max(0, estimated_cost - available_budget)
    }

def _assess_cost_effectiveness(
    self,
    alternative: Dict[str, Any],
    scenario: Dict[str, Any]
) -> float:
    """Calculate cost-effectiveness ratio for an alternative."""
    cost = alternative.get('estimated_cost', 1)
    effectiveness = alternative.get('effectiveness', 0)

    # Prevent division by zero
    if cost == 0:
        cost = 1

    return effectiveness / cost
```

### Step 6: Customize Prompts (If Using LLM)

Tailor prompts to agent's domain:

```python
def _generate_system_prompt(self) -> str:
    return (
        f"You are {self.name}, a {self.role} with {self.experience_years} years "
        f"of experience in {self.expertise}. "
        f"Your expertise includes: {', '.join(self.expertise_tags)}. "
        f"Provide professional economic analysis focusing on cost-benefit, "
        f"resource allocation, and financial sustainability."
    )

def _generate_evaluation_prompt(self, scenario, alternatives):
    return f"""ECONOMIC CRISIS ANALYSIS

Scenario: {scenario.get('description')}
Severity: {scenario.get('severity', 'unknown')}
Estimated Economic Impact: ${self._calculate_economic_impact(scenario):,.0f}
Available Budget: ${scenario.get('available_budget', 0):,.0f}

Alternatives to Evaluate:
{self._format_alternatives_with_costs(alternatives)}

As an economic expert, provide:
1. Cost-benefit analysis for each alternative
2. Probability distribution reflecting economic efficiency
3. Budget feasibility assessment
4. Long-term economic implications
5. Confidence in your analysis (0-1)

Respond in JSON format:
{{
    "alternative_rankings": {{"A1": 0.x, "A2": 0.y, ...}},
    "reasoning": "Your economic analysis...",
    "confidence": 0.x,
    "key_concerns": ["concern1", "concern2", ...],
    "budget_recommendation": {{"total": X, "allocation": {{...}}}}
}}"""
```

---

## Required Methods

All agents **must** implement these abstract methods from `BaseAgent`:

### 1. `evaluate_scenario()`

**Purpose:** Analyze a crisis scenario and provide assessment

**Required Return Fields:**
```python
{
    'agent_id': str,                    # Required
    'belief_distribution': Dict[str, float],  # Required, must sum to ~1.0
    'recommended_alternative': str,     # Required
    'confidence': float,                # Required, 0-1
    'reasoning': str,                   # Required
    'timestamp': str                    # Required (ISO format)
}
```

**Optional but Recommended:**
```python
{
    'criteria_scores': Dict[str, Dict[str, float]],
    'key_concerns': List[str],
    'risks': List[str],
    'assumptions': List[str],
    'llm_used': bool,
    'llm_metadata': Optional[Dict]
}
```

### 2. `propose_action()`

**Purpose:** Suggest action based on scenario and criteria

**Required Return Fields:**
```python
{
    'agent_id': str,                    # Required
    'proposed_action': {                # Required
        'name': str,
        'description': str,
        'priority': str,                # 'high', 'medium', 'low'
        'estimated_impact': Dict[str, float]
    },
    'justification': str,               # Required
    'confidence': float,                # Required, 0-1
    'timestamp': str                    # Required
}
```

**Optional but Recommended:**
```python
{
    'proposed_action': {
        'implementation_steps': List[str],
        'required_resources': List[str],
        'timeline': str,
        'risks': List[str],
        'dependencies': List[str]
    },
    'alternatives_considered': List[str]
}
```

---

## LLM Integration

### Choosing LLM Provider

The template supports three LLM providers:

#### 1. Claude (Anthropic) - Default, Best Quality

```python
from llm_integration import ClaudeClient

llm_client = ClaudeClient(api_key="your-key")
agent = MyCustomAgent(agent_id="agent_custom", llm_client=llm_client)
```

**Pros:** Excellent reasoning, reliable JSON output, best for production
**Cons:** Requires API key, costs per call

#### 2. OpenAI (GPT-4/GPT-3.5)

```python
from llm_integration import OpenAIClient

llm_client = OpenAIClient(api_key="your-key", model="gpt-4-turbo-preview")
agent = MyCustomAgent(agent_id="agent_custom", llm_client=llm_client)
```

**Pros:** Excellent quality, established platform, JSON mode
**Cons:** Requires API key, costs per call

#### 3. LM Studio (Local Models) - Free

```python
from llm_integration import LMStudioClient

llm_client = LMStudioClient(base_url="http://localhost:1234/v1")
agent = MyCustomAgent(agent_id="agent_custom", llm_client=llm_client)
```

**Pros:** Free, private, offline capable
**Cons:** Requires LM Studio setup, quality depends on model

### No LLM (Rule-Based Only)

```python
agent = MyCustomAgent(agent_id="agent_custom", use_llm=False)
```

**Pros:** No API costs, instant, fully deterministic
**Cons:** Lower quality, requires manual logic implementation

---

## Testing Your Agent

### Unit Testing

Create `tests/test_my_custom_agent.py`:

```python
import unittest
from agents.my_custom_agent import MyCustomAgent

class TestMyCustomAgent(unittest.TestCase):
    def setUp(self):
        """Initialize agent for testing."""
        self.agent = MyCustomAgent(
            agent_id="agent_custom_001",
            use_llm=False  # Test without LLM
        )

    def test_evaluate_scenario(self):
        """Test scenario evaluation."""
        scenario = {
            'type': 'flood',
            'severity': 0.8,
            'affected_population': 10000,
            'description': 'Test flood scenario'
        }

        alternatives = [
            {'id': 'A1', 'name': 'Evacuate'},
            {'id': 'A2', 'name': 'Barriers'}
        ]

        assessment = self.agent.evaluate_scenario(scenario, alternatives)

        # Validate required fields
        self.assertIn('belief_distribution', assessment)
        self.assertIn('confidence', assessment)
        self.assertIn('recommended_alternative', assessment)

        # Validate belief distribution sums to ~1.0
        total = sum(assessment['belief_distribution'].values())
        self.assertAlmostEqual(total, 1.0, places=2)

        # Validate confidence in range
        self.assertGreaterEqual(assessment['confidence'], 0.0)
        self.assertLessEqual(assessment['confidence'], 1.0)

    def test_propose_action(self):
        """Test action proposal."""
        scenario = {
            'type': 'earthquake',
            'severity': 0.9
        }

        criteria = {
            'effectiveness': 0.3,
            'safety': 0.3,
            'cost': 0.4
        }

        proposal = self.agent.propose_action(scenario, criteria)

        # Validate required fields
        self.assertIn('proposed_action', proposal)
        self.assertIn('justification', proposal)
        self.assertIn('confidence', proposal)

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m pytest tests/test_my_custom_agent.py -v
```

### Integration Testing

Test with full MAS:

```python
from agents.my_custom_agent import MyCustomAgent
from agents.expert_agent import ExpertAgent
from agents.coordinator_agent import CoordinatorAgent
from decision_framework import EvidentialReasoning, MCDAEngine, ConsensusModel
from llm_integration import ClaudeClient

# Initialize agents
llm_client = ClaudeClient()
custom_agent = MyCustomAgent("agent_custom_001", llm_client=llm_client)
medical_agent = ExpertAgent("agent_medical_expert", llm_client=llm_client)

# Initialize framework
er_engine = EvidentialReasoning()
mcda_engine = MCDAEngine(criteria_weights_path="scenarios/criteria_weights.json")
consensus_model = ConsensusModel()

# Create coordinator
coordinator = CoordinatorAgent(
    expert_agents=[custom_agent, medical_agent],
    er_engine=er_engine,
    mcda_engine=mcda_engine,
    consensus_model=consensus_model
)

# Test decision
decision = coordinator.make_final_decision(scenario, alternatives)
print(f"Decision: {decision['recommended_alternative']}")
```

---

## Best Practices

### 1. Domain Expertise

✅ **Do:**
- Focus on specific domain knowledge
- Provide unique perspectives
- Use domain-specific terminology
- Reference real expertise areas

❌ **Don't:**
- Create generic agents
- Duplicate existing agent capabilities
- Use vague or unclear expertise

### 2. Belief Distributions

✅ **Do:**
- Ensure beliefs sum to approximately 1.0
- Provide meaningful differences between alternatives
- Base beliefs on sound reasoning

❌ **Don't:**
- Return uniform distributions (uninformative)
- Create beliefs that don't sum to 1.0
- Ignore agent's expertise in scoring

### 3. Confidence Levels

✅ **Do:**
- Reflect genuine uncertainty
- Lower confidence when LLM unavailable
- Adjust based on scenario clarity

❌ **Don't:**
- Always return maximum confidence
- Ignore scenario complexity
- Return confidence < 0 or > 1

### 4. Error Handling

✅ **Do:**
- Validate all inputs
- Provide fallback logic
- Log errors meaningfully
- Return graceful degradation

❌ **Don't:**
- Crash on invalid inputs
- Silently ignore errors
- Return empty dictionaries

### 5. Documentation

✅ **Do:**
- Document all methods
- Explain domain-specific logic
- Provide usage examples
- Include type hints

❌ **Don't:**
- Leave methods undocumented
- Use unclear variable names
- Omit examples

---

## Examples

### Example 1: Simple Rule-Based Agent

```python
class SimpleEconomicAgent(BaseAgent):
    """Simple economic agent using only rule-based logic."""

    def evaluate_scenario(self, scenario, alternatives=None, **kwargs):
        """Evaluate based on cost-effectiveness."""
        belief_dist = {}

        for alt in alternatives:
            cost = alt.get('estimated_cost', 1)
            effectiveness = alt.get('effectiveness', 0.5)

            # Simple cost-effectiveness ratio
            score = effectiveness / (cost + 1)  # +1 to avoid division by zero
            belief_dist[alt['id']] = score

        # Normalize
        total = sum(belief_dist.values())
        belief_dist = {k: v/total for k, v in belief_dist.items()}

        recommended = max(belief_dist.items(), key=lambda x: x[1])[0]

        return {
            'agent_id': self.agent_id,
            'belief_distribution': belief_dist,
            'recommended_alternative': recommended,
            'confidence': 0.7,
            'reasoning': "Cost-effectiveness analysis favors low-cost, high-impact options",
            'timestamp': datetime.now().isoformat()
        }

    def propose_action(self, scenario, criteria, **kwargs):
        """Propose cost-effective action."""
        return {
            'agent_id': self.agent_id,
            'proposed_action': {
                'name': "Budget-Conscious Response",
                'description': "Maximize impact within budget constraints",
                'priority': 'high',
                'estimated_impact': {'cost': 0.3, 'effectiveness': 0.8}
            },
            'justification': "Optimal resource allocation",
            'confidence': 0.75,
            'timestamp': datetime.now().isoformat()
        }
```

### Example 2: LLM-Enhanced Agent with Custom Prompts

```python
class AdvancedLegalAgent(BaseAgent):
    """Legal expert agent with specialized LLM prompts."""

    def _generate_evaluation_prompt(self, scenario, alternatives):
        """Generate legal analysis prompt."""
        return f"""LEGAL CRISIS ANALYSIS

As a legal expert with {self.experience_years} years experience:

SCENARIO:
{scenario.get('description')}

LEGAL CONSIDERATIONS:
- Liability exposure
- Regulatory compliance
- Public safety obligations
- Documentation requirements

RESPONSE OPTIONS:
{self._format_alternatives(alternatives)}

Provide:
1. Legal risk assessment for each option
2. Compliance analysis
3. Liability minimization strategy
4. Recommended action from legal perspective

Format: JSON with alternative_rankings, reasoning, confidence, legal_concerns"""

        return prompt

    def _evaluate_with_llm(self, scenario, alternatives):
        """LLM-based legal analysis."""
        prompt = self._generate_evaluation_prompt(scenario, alternatives)

        response = self.llm_client.generate_assessment(
            prompt=prompt,
            system_prompt=f"You are {self.name}, a legal expert specializing in crisis law."
        )

        # Add legal-specific fields
        assessment = self._parse_llm_assessment(response, scenario, alternatives)
        assessment['legal_concerns'] = response.get('legal_concerns', [])
        assessment['compliance_status'] = self._check_compliance(scenario)

        return assessment
```

### Example 3: Hybrid Agent (Both LLM and Rules)

```python
class HybridPsychologicalAgent(BaseAgent):
    """Psychological expert using both LLM and psychological models."""

    def evaluate_scenario(self, scenario, alternatives=None, **kwargs):
        """Combine LLM reasoning with psychological models."""

        # First: Apply psychological stress model
        stress_level = self._calculate_population_stress(scenario)
        trauma_risk = self._assess_trauma_risk(scenario)

        # Second: Get LLM analysis if available
        if self.use_llm:
            llm_assessment = self._evaluate_with_llm(scenario, alternatives)
            base_confidence = llm_assessment['confidence']
        else:
            llm_assessment = self._evaluate_rule_based(scenario, alternatives)
            base_confidence = 0.7

        # Third: Adjust based on psychological models
        adjusted_beliefs = self._adjust_beliefs_for_psychology(
            llm_assessment['belief_distribution'],
            stress_level,
            trauma_risk
        )

        return {
            **llm_assessment,
            'belief_distribution': adjusted_beliefs,
            'psychological_metrics': {
                'population_stress': stress_level,
                'trauma_risk': trauma_risk
            },
            'confidence': base_confidence * 0.95  # Slightly conservative
        }

    def _calculate_population_stress(self, scenario):
        """Psychological stress model."""
        severity = scenario.get('severity', 0.5)
        affected = scenario.get('affected_population', 0)

        # Simple psychological stress model
        stress = severity * (1 + math.log10(max(1, affected / 1000)))
        return min(1.0, stress)
```

---

## Troubleshooting

### Common Issues

**Issue:** Agent can't find profile
```
ValueError: Agent ID 'agent_custom' not found in profile file
```
**Solution:** Add agent to `agents/agent_profiles.json`

**Issue:** Belief distribution doesn't sum to 1.0
```
AssertionError: Beliefs sum to 1.34, expected ~1.0
```
**Solution:** Normalize beliefs:
```python
total = sum(belief_dist.values())
belief_dist = {k: v/total for k, v in belief_dist.items()}
```

**Issue:** LLM client error
```
RuntimeError: LLM assessment failed: API key not provided
```
**Solution:** Set environment variable or pass API key:
```bash
export ANTHROPIC_API_KEY='your-key'
```

---

## Additional Resources

- **BaseAgent Source**: `agents/base_agent.py`
- **ExpertAgent Source**: `agents/expert_agent.py`
- **Agent Template**: `agents/agent_template.py`
- **Agent Profiles**: `agents/agent_profiles.json`
- **LLM Clients**: `llm_integration/`
- **System Documentation**: `README.md`

---

## Support

For questions or issues:
1. Check the agent template documentation
2. Review existing agents (ExpertAgent, CoordinatorAgent)
3. Run the template demo: `python agents/agent_template.py`
4. Check integration tests: `tests/test_integration.py`

---

**Version:** 1.0
**Last Updated:** November 2025
**Status:** Production Ready
