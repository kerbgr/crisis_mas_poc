# Belief Distribution Type Investigation

## Problem Statement

During conflict resolution with 11 agents, the error `'str' object has no attribute 'get'` occurs when trying to access `agent_beliefs[agent_id].get(alt_id, 0.0)`.

This indicates that some agents' `belief_distribution` field contains a **string** (or other non-dict type) instead of the expected **dictionary**.

---

## Investigation Approach

###  1. Data Flow Tracing

**belief_distribution** flows through the system like this:

```
LLM Response (JSON string)
    ↓
parse_json_response()  [LLM Client]
    ↓
llm_response['alternative_rankings']
    ↓
generate_belief_distribution()  [Expert Agent]
    ↓
assessment['belief_distribution'] = belief_dist  [Dict[str, float]]
    ↓
collect_assessments()  [Coordinator]
    ↓
agent_assessments[agent_id]
    ↓
resolve_conflicts()  [tries to call .get() on it]
```

### 2. Potential Failure Points

| Stage | Possible Issue | Likelihood |
|-------|---------------|-----------|
| **LLM JSON Parsing** | Malformed JSON, parsing fails, returns string | HIGH ⚠️ |
| **Ranking Extraction** | `alternative_rankings` is not a dict | MEDIUM |
| **Belief Generation** | Logic error, returns string instead of dict | LOW |
| **Serialization** | JSON dump/load somewhere in pipeline | MEDIUM |
| **ThreadPool** | Race condition corrupting data | LOW |
| **Alternative IDs** | Mismatched IDs between scenario & LLM response | MEDIUM |

### 3. Diagnostic Script

Run the diagnostic script to identify the source:

```bash
python diagnose_belief_distribution_types.py
```

This will:
- ✅ Test single agent assessment
- ✅ Inspect type at every level
- ✅ Test multiple agents
- ✅ Identify which agent(s) have the issue
- ✅ Show actual malformed data

---

## Hypothesis: LLM JSON Parsing Failure

**Most likely cause:** LLM returns malformed JSON, parser fails, falls back to something that's not a dict.

### Evidence to Look For:

1. **Check `parse_json_response()` in LLM clients:**
   - Does it have a fallback that returns non-dict?
   - What happens when all parsing strategies fail?

2. **Check `generate_belief_distribution()`:**
   - Does it validate that `llm_response['alternative_rankings']` is a dict?
   - What if `alternative_rankings` is missing or wrong type?

3. **Check `_validate_llm_response()`:**
   - Does it actually validate the types?
   - Or just check for key presence?

---

## Proposed Solutions

### Solution 1: Strict Validation (Recommended)

**Add type validation at source** - Ensure belief_distribution is always a dict:

```python
# In ExpertAgent.generate_belief_distribution()

def generate_belief_distribution(self, llm_response: Dict[str, Any]) -> Dict[str, float]:
    """Generate belief distribution from LLM response."""

    # VALIDATION: Check alternative_rankings exists and is dict
    rankings = llm_response.get('alternative_rankings')

    if not isinstance(rankings, dict):
        logger.error(
            f"alternative_rankings is not a dict: {type(rankings).__name__}. "
            f"Value: {str(rankings)[:200]}"
        )
        # Fallback: return uniform distribution
        alternatives = llm_response.get('_alternatives', [])
        if not alternatives:
            raise ValueError("Cannot create belief distribution: no alternatives available")

        n = len(alternatives)
        uniform_dist = {alt['id']: 1.0 / n for alt in alternatives}
        logger.warning(f"Using uniform distribution as fallback: {uniform_dist}")
        return uniform_dist

    if not rankings:
        logger.warning("rankings dict is empty")
        # Handle empty dict case...

    # Continue with normal processing...
    total = sum(rankings.values())
    if total == 0:
        # Handle zero sum...

    # Normalize
    belief_distribution = {
        alt_id: score / total
        for alt_id, score in rankings.items()
    }

    # FINAL VALIDATION: Ensure return type is dict
    assert isinstance(belief_distribution, dict), \
        f"belief_distribution must be dict, got {type(belief_distribution)}"

    return belief_distribution
```

**Pros:**
- ✅ Catches issue at source
- ✅ Provides meaningful fallback (uniform distribution)
- ✅ Clear error messages for debugging
- ✅ System continues to function

**Cons:**
- ❌ Fallback might mask underlying LLM issues
- ❌ Uniform distribution might not be appropriate

---

### Solution 2: Better JSON Parsing

**Improve `parse_json_response()` to never return malformed data:**

```python
def parse_json_response(self, response_text: str) -> Dict[str, Any]:
    """Parse JSON with strict type guarantees."""

    # Try all parsing strategies...
    parsed = self._try_all_parsing_strategies(response_text)

    if parsed is None:
        logger.error(f"All JSON parsing strategies failed for: {response_text[:500]}")

        # Return a structured error response (not raw text!)
        return {
            'error': True,
            'error_message': 'JSON parsing failed',
            'alternative_rankings': {},  # Empty dict, not None or string!
            'reasoning': 'LLM response could not be parsed',
            'confidence': 0.5,
            'key_concerns': ['JSON parsing failure']
        }

    # VALIDATION: Ensure alternative_rankings is dict
    if 'alternative_rankings' in parsed:
        if not isinstance(parsed['alternative_rankings'], dict):
            logger.warning(
                f"alternative_rankings is {type(parsed['alternative_rankings'])}, converting to dict"
            )
            # Try to fix it
            parsed['alternative_rankings'] = {}

    return parsed
```

**Pros:**
- ✅ Ensures parse_json_response always returns valid structure
- ✅ Handles LLM malformed responses gracefully
- ✅ Prevents corruption propagating downstream

**Cons:**
- ❌ Might hide LLM prompt engineering issues
- ❌ Empty rankings produce meaningless results

---

### Solution 3: Pydantic Data Models (Best Long-term)

**Use Pydantic for type validation and serialization:**

```python
from pydantic import BaseModel, validator
from typing import Dict

class AgentAssessment(BaseModel):
    """Validated agent assessment with type guarantees."""

    agent_id: str
    agent_name: str
    belief_distribution: Dict[str, float]  # Guaranteed to be dict
    confidence: float
    reasoning: str
    key_concerns: list[str] = []

    @validator('belief_distribution')
    def validate_beliefs(cls, v):
        """Ensure belief_distribution is valid."""
        if not isinstance(v, dict):
            raise ValueError(f"belief_distribution must be dict, got {type(v)}")

        if not v:
            raise ValueError("belief_distribution cannot be empty")

        # Check all values are floats
        for alt_id, belief in v.items():
            if not isinstance(belief, (int, float)):
                raise ValueError(f"Belief for {alt_id} must be numeric, got {type(belief)}")

        # Check sum is approximately 1.0
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Beliefs must sum to ~1.0, got {total}")

        return v

    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is in [0, 1]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be in [0,1], got {v}")
        return v


# Usage in ExpertAgent
def _build_assessment(self, belief_distribution, ...) -> AgentAssessment:
    """Build validated assessment."""
    try:
        assessment = AgentAssessment(
            agent_id=self.agent_id,
            agent_name=self.name,
            belief_distribution=belief_distribution,
            confidence=confidence,
            reasoning=reasoning,
            key_concerns=key_concerns
        )
        return assessment.dict()  # Convert to dict for backward compatibility
    except ValidationError as e:
        logger.error(f"Assessment validation failed: {e}")
        # Return valid fallback or raise
```

**Pros:**
- ✅ Compile-time type checking
- ✅ Runtime validation
- ✅ Automatic serialization/deserialization
- ✅ Clear error messages
- ✅ Self-documenting code
- ✅ Prevents many classes of bugs

**Cons:**
- ❌ Requires refactoring
- ❌ Adds dependency (Pydantic)
- ❌ More verbose

---

## Immediate Action Plan

1. **Run Diagnostics** (5 min)
   ```bash
   python diagnose_belief_distribution_types.py
   ```

2. **Identify Root Cause** (10 min)
   - Which agent(s) have non-dict beliefs?
   - What's the actual type and value?
   - Is it consistent or sporadic?

3. **Apply Quick Fix** (15 min)
   - Implement Solution 1 (strict validation) in `generate_belief_distribution()`
   - Add validation in `_build_assessment()`

4. **Test Fix** (10 min)
   ```bash
   python main.py --agents all --scenario scenarios/flood_scenario.json --verbose
   ```

5. **Long-term Improvement** (future)
   - Implement Solution 3 (Pydantic) in next major refactor
   - Add unit tests for belief distribution generation
   - Improve LLM prompt engineering to reduce malformed responses

---

## Testing Strategy

### Unit Tests to Add:

```python
def test_belief_distribution_always_dict():
    """Ensure belief_distribution is always a dict."""
    agent = create_test_agent()

    # Test with valid response
    valid_response = {'alternative_rankings': {'A1': 0.7, 'A2': 0.3}}
    belief_dist = agent.generate_belief_distribution(valid_response)
    assert isinstance(belief_dist, dict)

    # Test with invalid response (string)
    invalid_response = {'alternative_rankings': "A1: 0.7, A2: 0.3"}
    belief_dist = agent.generate_belief_distribution(invalid_response)
    assert isinstance(belief_dist, dict)  # Should still return dict (fallback)

    # Test with missing rankings
    missing_response = {}
    with pytest.raises(ValueError):
        agent.generate_belief_distribution(missing_response)

def test_assessment_structure():
    """Validate complete assessment structure."""
    assessment = agent.evaluate_scenario(scenario, alternatives)

    assert 'belief_distribution' in assessment
    assert isinstance(assessment['belief_distribution'], dict)
    assert len(assessment['belief_distribution']) > 0
    assert all(isinstance(v, float) for v in assessment['belief_distribution'].values())
```

---

## Related Files

- `agents/expert_agent.py` - generates belief_distribution
- `llm_integration/claude_client.py` - parses JSON from LLM
- `llm_integration/openai_client.py` - parses JSON from LLM
- `llm_integration/lmstudio_client.py` - parses JSON from LLM
- `agents/coordinator_agent.py` - consumes belief_distribution
- `decision_framework/consensus_model.py` - uses belief_distribution

---

## Expected Outcomes

After implementing fixes:

1. ✅ No more `'str' object has no attribute 'get'` errors
2. ✅ Clear log messages when LLM returns malformed data
3. ✅ System continues to function with fallback distributions
4. ✅ Easier debugging with type validation messages
5. ✅ More robust handling of LLM variability

---

**Last Updated:** November 10, 2025
**Status:** Investigation in progress
**Priority:** HIGH - Blocks 11-agent consensus
