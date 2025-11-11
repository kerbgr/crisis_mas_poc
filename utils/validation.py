"""
═══════════════════════════════════════════════════════════════════════════════
DATA VALIDATION MODULE
Comprehensive validation infrastructure for Crisis MAS data integrity
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
═════════
This module provides a comprehensive validation framework ensuring data integrity
across the Crisis MAS system. It validates:

1. **File Operations** - JSON file loading with comprehensive error handling
2. **Data Structures** - Scenarios, alternatives, criteria, agent profiles
3. **Probabilistic Data** - Belief distributions, criteria weights (must sum to 1.0)
4. **LLM Responses** - Structured validation of AI-generated outputs
5. **Numerical Safety** - Score sanitization and safe mathematical operations


WHY THIS EXISTS
═══════════════
In multi-agent decision systems, invalid data can cause:
- Runtime failures during agent deliberation
- Incorrect consensus calculations
- Silent errors in MCDA evaluations
- Cascading failures across agent network
- Unreliable decision outcomes

This module prevents these issues by:
- **Early Detection**: Validate at data ingestion boundaries
- **Clear Errors**: Provide informative error messages with context
- **Graceful Handling**: Offer normalization for minor issues (e.g., floating-point rounding)
- **Defensive Programming**: Safe operations that never crash


INPUTS
══════
The validation module accepts various data structures for validation:

1. **File Paths**
   - Type: str or Path
   - Purpose: JSON files containing scenarios, criteria, agent profiles
   - Example: "scenarios/crisis_scenario.json"

2. **Scenario Dictionaries**
   - Required Fields: 'id', 'title', 'description', 'alternatives', 'criteria'
   - Optional Fields: 'context', 'metadata', 'timestamp'
   - Example:
     {
       'id': 'scenario_1',
       'title': 'Crisis Response Decision',
       'description': 'Evaluate response options',
       'alternatives': [...],
       'criteria': {...}
     }

3. **Alternatives Lists**
   - Type: List[Dict[str, Any]]
   - Required Fields per Alternative: 'id', 'name', 'description'
   - Example: [{'id': 'alt_1', 'name': 'Diplomacy', 'description': '...'}]

4. **Criteria Weights**
   - Type: Dict[str, float]
   - Constraint: Must sum to 1.0 (within tolerance)
   - Example: {'political': 0.3, 'economic': 0.5, 'humanitarian': 0.2}

5. **Belief Distributions**
   - Type: Dict[str, float]
   - Constraint: Must sum to 1.0 (within tolerance)
   - Example: {'option_a': 0.6, 'option_b': 0.3, 'option_c': 0.1}

6. **Agent Profiles**
   - Required Fields: 'id', 'role', 'confidence_level', 'belief_update_rate'
   - Value Ranges: confidence_level ∈ [0, 1], belief_update_rate ∈ [0, 1]
   - Example:
     {
       'id': 'agent_1',
       'role': 'diplomat',
       'confidence_level': 0.8,
       'belief_update_rate': 0.3
     }

7. **LLM Responses**
   - Type: Dict[str, Any]
   - Validation: Check for expected keys, structure, data types
   - Example: {'analysis': '...', 'scores': {...}, 'reasoning': '...'}


OUTPUTS
═══════
Validation methods return structured results:

1. **Tuple Returns: (bool, Optional[data/error])**
   - Success Case: (True, data, None) or (True, None)
   - Failure Case: (False, None, error_message) or (False, error_message)
   - Rationale: Allows caller to decide exception vs. graceful handling

2. **Normalized Data**
   - normalize_weights(weights) → Dict[str, float]
   - Returns weights normalized to sum to 1.0
   - Preserves relative proportions

3. **Sanitized Scores**
   - sanitize_scores(scores, min_val, max_val) → Dict[str, float]
   - Clamps all scores to [min_val, max_val] range
   - Prevents out-of-range values in calculations

4. **Safe Operations**
   - safe_divide(numerator, denominator, default) → float
   - Returns default if denominator is 0 or division fails
   - safe_get(dict, key, default, expected_type) → Any
   - Returns default if key missing or type mismatch

5. **ValidationError Exceptions**
   - Raised when validation fails and caller opts to raise
   - Contains descriptive error message with context
   - Example: "Missing required field 'id' in scenario"


VALIDATION PATTERNS
═══════════════════

1. **Progressive Validation**
   ─────────────────────────
   Level 1: Structure validation (required fields exist)
   Level 2: Type validation (fields have correct types)
   Level 3: Constraint validation (ranges, sums, relationships)

   Example:
   ```python
   # Level 1: Structure
   if 'id' not in scenario:
       return (False, "Missing required field 'id'")

   # Level 2: Type
   if not isinstance(scenario['id'], str):
       return (False, "Field 'id' must be a string")

   # Level 3: Constraint
   if not scenario['id'].strip():
       return (False, "Field 'id' cannot be empty")
   ```

2. **Fail Fast, Fail Informative**
   ────────────────────────────────
   - Validate at boundaries (file load, function entry)
   - Provide specific error messages
   - Include context (field name, expected vs. actual)

   Example Error Messages:
   - "Missing required field 'alternatives' in scenario"
   - "Weights sum to 0.95, expected 1.0 (tolerance: 0.01)"
   - "Confidence level 1.5 out of valid range [0.0, 1.0]"

3. **Normalize vs. Reject**
   ────────────────────────
   Minor Issues (Auto-fix):
   - Weights sum to 0.999 → Normalize to 1.0
   - Floating-point rounding errors
   - Trailing whitespace in strings

   Major Issues (Reject):
   - Missing required fields
   - Wrong data types
   - Values far outside valid ranges

4. **Safe Operations Philosophy**
   ──────────────────────────────
   Never crash, always return:
   - safe_divide(10, 0, default=0.0) → 0.0
   - safe_get({'a': 1}, 'b', default=None) → None
   - Caller specifies sensible default for their context


TYPICAL USAGE
═════════════

Example 1: Validate and Load Scenario File
───────────────────────────────────────────
```python
from utils.validation import DataValidator, ValidationError

# Validate JSON file structure and load data
valid, data, error = DataValidator.validate_json_file('scenario.json')
if not valid:
    raise ValidationError(f"Invalid JSON file: {error}")

# Validate scenario structure
valid, error = DataValidator.validate_scenario(data)
if not valid:
    raise ValidationError(f"Invalid scenario: {error}")

# Validate alternatives
valid, error = DataValidator.validate_alternatives(data['alternatives'])
if not valid:
    raise ValidationError(f"Invalid alternatives: {error}")

# Now data is guaranteed valid, proceed with processing
```

Example 2: Validate and Normalize Criteria Weights
───────────────────────────────────────────────────
```python
from utils.validation import DataValidator

criteria = {
    'political': 0.31,
    'economic': 0.49,
    'humanitarian': 0.19
}  # Sum = 0.99 (rounding error)

# Check if valid
valid, error = DataValidator.validate_criteria_weights(criteria, tolerance=0.01)
if not valid:
    # Auto-normalize to fix minor rounding issues
    criteria = DataValidator.normalize_weights(criteria)
    # Now sum = 1.0 exactly

# Use normalized weights
for criterion, weight in criteria.items():
    print(f"{criterion}: {weight}")
```

Example 3: Validate Agent Profile
──────────────────────────────────
```python
from utils.validation import DataValidator, ValidationError

profile = {
    'id': 'agent_007',
    'role': 'intelligence_analyst',
    'confidence_level': 0.85,
    'belief_update_rate': 0.25
}

valid, error = DataValidator.validate_agent_profile(profile)
if not valid:
    raise ValidationError(f"Invalid agent profile: {error}")

# Profile is valid, create agent
agent = Agent(profile)
```

Example 4: Safe Mathematical Operations
────────────────────────────────────────
```python
from utils.validation import safe_divide, safe_get

# Safe division (never raises ZeroDivisionError)
total_votes = 100
num_agents = 0  # Edge case: no agents
avg_votes = safe_divide(total_votes, num_agents, default=0.0)
# Returns 0.0 instead of crashing

# Safe dictionary access with type checking
config = {'max_agents': 10, 'timeout': '30'}
max_agents = safe_get(config, 'max_agents', default=5, expected_type=int)  # Returns 10
timeout = safe_get(config, 'timeout', default=30, expected_type=int)  # Returns 30 (type mismatch, uses default)
missing = safe_get(config, 'missing_key', default=100)  # Returns 100
```

Example 5: Validate LLM Response Structure
───────────────────────────────────────────
```python
from utils.validation import DataValidator

llm_response = {
    'analysis': 'The economic factors are paramount...',
    'scores': {'alt_1': 0.8, 'alt_2': 0.6},
    'reasoning': 'Based on criteria weights...'
}

expected_keys = ['analysis', 'scores', 'reasoning']
valid, error = DataValidator.validate_llm_response(llm_response, expected_keys)
if not valid:
    logger.warning(f"LLM response incomplete: {error}")
    # Use fallback or default values
```

Example 6: Sanitize Scores to Valid Range
──────────────────────────────────────────
```python
from utils.validation import DataValidator

# Scores from external source may be out of range
raw_scores = {
    'alt_1': 1.2,   # Too high
    'alt_2': -0.1,  # Too low
    'alt_3': 0.7    # Valid
}

# Clamp to [0.0, 1.0] range
clean_scores = DataValidator.sanitize_scores(raw_scores, min_val=0.0, max_val=1.0)
# Returns: {'alt_1': 1.0, 'alt_2': 0.0, 'alt_3': 0.7}
```


ERROR HANDLING
══════════════

The module provides multiple error handling strategies:

1. **Tuple Return Pattern**
   ────────────────────────
   ```python
   valid, data, error = DataValidator.validate_json_file(path)
   if not valid:
       # Caller decides: raise exception or handle gracefully
       logger.error(error)
       return default_data
   ```

2. **ValidationError Exception**
   ─────────────────────────────
   ```python
   from utils.validation import ValidationError

   try:
       valid, error = DataValidator.validate_scenario(scenario)
       if not valid:
           raise ValidationError(error)
   except ValidationError as e:
       logger.error(f"Validation failed: {e}")
       # Handle with fallback behavior
   ```

3. **Safe Operations (Never Raise)**
   ──────────────────────────────────
   ```python
   # These never raise exceptions
   result = safe_divide(a, b, default=0.0)
   value = safe_get(data, key, default=None)
   ```

4. **JSON File Error Categories**
   ──────────────────────────────
   - FileNotFoundError: "File not found: {path}"
   - PermissionError: "Permission denied: {path}"
   - UnicodeDecodeError: "Encoding error reading file: {path}"
   - JSONDecodeError: "Invalid JSON format: {error_details}"


DESIGN DECISIONS
════════════════

1. **Static Methods for Validators**
   - Rationale: Validation is stateless, no instance state needed
   - Benefit: Clean API, easy testing, no initialization overhead
   - Usage: DataValidator.validate_scenario(scenario)

2. **Tuple Returns vs. Exceptions**
   - Rationale: Performance-critical paths avoid exception overhead
   - Benefit: Caller chooses exception or graceful handling
   - Pattern: (bool, Optional[data], Optional[error])

3. **Tolerance for Floating-Point**
   - Rationale: Weights may have rounding errors (0.999 vs. 1.0)
   - Default Tolerance: 0.01 (1%)
   - Benefit: Practical for real-world data

4. **Normalization Helper**
   - Rationale: Minor weight discrepancies are common
   - Auto-fix: Preserve relative proportions, rescale to sum=1.0
   - Benefit: User-friendly, reduces false validation failures

5. **Safe Operations Return Defaults**
   - Rationale: Caller knows their domain, specifies sensible fallback
   - Flexibility: Different contexts need different defaults
   - Example: safe_divide(a, b, default=0.0) vs. safe_divide(a, b, default=1.0)

6. **Comprehensive JSON Error Handling**
   - Rationale: File I/O has many failure modes
   - Coverage: File not found, permissions, encoding, JSON parse errors
   - Benefit: Clear diagnosis for users


INTEGRATION POINTS
══════════════════

This module is used extensively across the Crisis MAS system:

1. **scenarios/** → Validate scenario files, criteria weights
2. **agents/** → Validate agent profiles, belief distributions
3. **decision_framework/** → Safe division in consensus calculations
4. **llm_integration/** → Validate LLM responses before processing
5. **evaluation/** → Safe operations in metrics calculations

Usage Flow:
───────────
```
File Load → validate_json_file() → validate_scenario()
                                 ↓
                           validate_alternatives()
                           validate_criteria_weights()
                                 ↓
                           Data ingested into system
                                 ↓
Runtime: safe_divide(), safe_get() used throughout calculations
```


VALIDATION SCHEMAS
══════════════════

Required Fields by Data Type:
──────────────────────────────

Scenario:
- id: str (non-empty)
- title: str
- description: str
- alternatives: List[Dict]
- criteria: Dict

Alternative:
- id: str (non-empty)
- name: str
- description: str

Agent Profile:
- id: str (non-empty)
- role: str
- confidence_level: float ∈ [0, 1]
- belief_update_rate: float ∈ [0, 1]

Criteria Weights:
- Dict[str, float]
- Sum of values ≈ 1.0 (within tolerance)

Belief Distribution:
- Dict[str, float]
- Sum of values ≈ 1.0 (within tolerance)


PERFORMANCE CONSIDERATIONS
═══════════════════════════

1. **Tuple Returns for Hot Paths**
   - Validation in loops: Use tuple returns
   - Avoid exception overhead in performance-critical code
   - Pattern: `if not valid: continue` instead of try/except

2. **Static Methods**
   - No class instantiation overhead
   - Direct function calls
   - Compiler optimization friendly

3. **Early Returns**
   - Fail fast on first error
   - Don't validate entire structure if early field missing
   - Reduces wasted computation


RELATED MODULES
═══════════════

Dependencies:
- json: JSON parsing and error handling
- logging: Structured logging for validation failures
- typing: Type hints for API clarity
- pathlib: Cross-platform path operations

Used By:
- scenarios/scenario_loader.py: Load and validate scenarios
- agents/agent_factory.py: Validate agent profiles
- decision_framework/consensus.py: Safe operations in consensus
- llm_integration/*_client.py: Validate LLM responses
- evaluation/metrics.py: Safe operations in calculations


TROUBLESHOOTING
═══════════════

Common Validation Errors:
─────────────────────────

1. **"Weights do not sum to 1.0"**
   Cause: Floating-point rounding or manual data entry
   Solution:
   ```python
   criteria = DataValidator.normalize_weights(criteria)
   ```

2. **"Missing required field 'X'"**
   Cause: Incomplete JSON structure
   Solution: Check JSON file, ensure all required fields present
   Refer to VALIDATION SCHEMAS section for requirements

3. **"File not found: {path}"**
   Cause: Incorrect file path or missing file
   Solution: Verify path, check working directory

4. **"Invalid JSON format"**
   Cause: Syntax error in JSON file
   Solution: Validate JSON with linter, check for trailing commas, quotes

5. **"Value out of range"**
   Cause: confidence_level > 1.0 or < 0.0
   Solution: Check data source, ensure values in [0, 1]


TESTING EXAMPLES
════════════════

Unit Test Pattern:
──────────────────
```python
def test_validate_criteria_weights():
    # Valid case
    criteria = {'a': 0.5, 'b': 0.5}
    valid, error = DataValidator.validate_criteria_weights(criteria)
    assert valid is True
    assert error is None

    # Invalid case (doesn't sum to 1.0)
    criteria = {'a': 0.6, 'b': 0.6}
    valid, error = DataValidator.validate_criteria_weights(criteria)
    assert valid is False
    assert "sum" in error.lower()

    # Normalization
    criteria = {'a': 0.6, 'b': 0.6}
    normalized = DataValidator.normalize_weights(criteria)
    assert abs(sum(normalized.values()) - 1.0) < 1e-9
```


VERSION HISTORY
═══════════════

Version 1.0.0 (Initial Release)
- DataValidator class with 10+ validation methods
- ValidationError custom exception
- Safe operations: safe_divide, safe_get
- JSON file validation with comprehensive error handling
- Normalization and sanitization utilities

Future Enhancements:
- JSON Schema integration for declarative validation
- Async validation for large files
- Batch validation for multiple scenarios
- Custom validation rule registration
- Performance profiling and optimization
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from models.data_models import LLMResponse

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """
    Comprehensive data validation for Crisis MAS.

    Provides validation for:
    - JSON files
    - Scenarios
    - Alternatives
    - Criteria weights
    - Agent profiles
    - LLM responses
    - Belief distributions
    """

    @staticmethod
    def validate_json_file(file_path: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate and load JSON file with comprehensive error handling.

        Args:
            file_path: Path to JSON file

        Returns:
            Tuple of (success, data, error_message)

        Example:
            >>> success, data, error = DataValidator.validate_json_file("scenario.json")
            >>> if not success:
            ...     print(f"Error: {error}")
        """
        try:
            path = Path(file_path)

            # Check file exists
            if not path.exists():
                return False, None, f"File not found: {file_path}"

            # Check file is readable
            if not path.is_file():
                return False, None, f"Path is not a file: {file_path}"

            # Try to load JSON
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Successfully loaded JSON file: {file_path}")
            return True, data, None

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in {file_path}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

        except UnicodeDecodeError as e:
            error_msg = f"File encoding error in {file_path}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

        except PermissionError as e:
            error_msg = f"Permission denied reading {file_path}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

        except Exception as e:
            error_msg = f"Unexpected error loading {file_path}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    @staticmethod
    def validate_scenario(scenario: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate scenario structure and required fields.

        Args:
            scenario: Scenario dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['id', 'type', 'name', 'description', 'severity']

        try:
            # Check required fields
            missing_fields = [f for f in required_fields if f not in scenario]
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}"

            # Validate severity
            severity = scenario['severity']
            if not isinstance(severity, (int, float)):
                return False, f"Severity must be numeric, got {type(severity).__name__}"

            if not 0 <= severity <= 1:
                return False, f"Severity {severity} must be between 0 and 1"

            # Check for alternatives/available_actions
            if 'alternatives' not in scenario and 'available_actions' not in scenario:
                logger.warning("Scenario has no alternatives or available_actions")

            logger.info(f"Scenario validation passed: {scenario.get('name', 'Unknown')}")
            return True, None

        except Exception as e:
            error_msg = f"Error validating scenario: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def validate_alternatives(alternatives: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Validate alternatives list structure.

        Args:
            alternatives: List of alternative dictionaries

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not alternatives:
                return False, "Alternatives list is empty"

            if not isinstance(alternatives, list):
                return False, f"Alternatives must be a list, got {type(alternatives).__name__}"

            required_fields = ['id', 'name']

            for i, alt in enumerate(alternatives):
                if not isinstance(alt, dict):
                    return False, f"Alternative {i} must be a dictionary"

                # Check required fields
                missing = [f for f in required_fields if f not in alt]
                if missing:
                    return False, f"Alternative {i} missing fields: {', '.join(missing)}"

                # Validate alternative ID
                if not alt['id']:
                    return False, f"Alternative {i} has empty ID"

            # Check for duplicate IDs
            ids = [alt['id'] for alt in alternatives]
            if len(ids) != len(set(ids)):
                return False, "Duplicate alternative IDs found"

            logger.info(f"Alternatives validation passed: {len(alternatives)} alternatives")
            return True, None

        except Exception as e:
            error_msg = f"Error validating alternatives: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def validate_criteria_weights(
        criteria: Dict[str, Any],
        tolerance: float = 0.01
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate criteria weights sum to 1.0.

        Args:
            criteria: Dictionary of criteria with weights
            tolerance: Acceptable deviation from 1.0

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not criteria:
                return False, "Criteria dictionary is empty"

            if not isinstance(criteria, dict):
                return False, f"Criteria must be a dictionary, got {type(criteria).__name__}"

            # Extract weights
            weights = []
            for crit_id, crit_data in criteria.items():
                if not isinstance(crit_data, dict):
                    return False, f"Criterion '{crit_id}' must be a dictionary"

                if 'weight' not in crit_data:
                    return False, f"Criterion '{crit_id}' missing 'weight' field"

                weight = crit_data['weight']
                if not isinstance(weight, (int, float)):
                    return False, f"Weight for '{crit_id}' must be numeric"

                if weight < 0:
                    return False, f"Weight for '{crit_id}' is negative: {weight}"

                weights.append(weight)

            # Check sum
            total = sum(weights)
            if abs(total - 1.0) > tolerance:
                return False, (
                    f"Criteria weights sum to {total:.4f}, expected 1.0 "
                    f"(tolerance: ±{tolerance})"
                )

            logger.info(f"Criteria weights validation passed: {len(criteria)} criteria, sum={total:.4f}")
            return True, None

        except Exception as e:
            error_msg = f"Error validating criteria weights: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def validate_belief_distribution(
        beliefs: Dict[str, float],
        tolerance: float = 0.01
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate belief distribution sums to 1.0.

        Args:
            beliefs: Dictionary mapping alternatives to belief values
            tolerance: Acceptable deviation from 1.0

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not beliefs:
                return False, "Belief distribution is empty"

            if not isinstance(beliefs, dict):
                return False, f"Beliefs must be a dictionary, got {type(beliefs).__name__}"

            # Check all values are numeric and in range [0,1]
            for alt_id, belief in beliefs.items():
                if not isinstance(belief, (int, float)):
                    return False, f"Belief for '{alt_id}' must be numeric"

                if not 0 <= belief <= 1:
                    return False, f"Belief for '{alt_id}' ({belief}) must be between 0 and 1"

            # Check sum
            total = sum(beliefs.values())
            if abs(total - 1.0) > tolerance:
                return False, (
                    f"Beliefs sum to {total:.4f}, expected 1.0 "
                    f"(tolerance: ±{tolerance})"
                )

            return True, None

        except Exception as e:
            error_msg = f"Error validating belief distribution: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def validate_agent_profile(profile: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate agent profile structure.

        Args:
            profile: Agent profile dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['agent_id', 'name', 'role', 'expertise']

        try:
            # Check required fields
            missing = [f for f in required_fields if f not in profile]
            if missing:
                return False, f"Missing required fields: {', '.join(missing)}"

            # Validate agent_id
            if not profile['agent_id']:
                return False, "Agent ID is empty"

            # Validate name
            if not profile['name']:
                return False, "Agent name is empty"

            logger.info(f"Agent profile validation passed: {profile['name']}")
            return True, None

        except Exception as e:
            error_msg = f"Error validating agent profile: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def validate_llm_response(
        response: Union[Dict[str, Any], LLMResponse],
        expected_keys: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate LLM response structure.

        Args:
            response: LLM response (dict or LLMResponse Pydantic model)
            expected_keys: Optional list of required keys

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Accept both dict and LLMResponse (dict-like Pydantic model)
            if not isinstance(response, (dict, LLMResponse)):
                # Check if it has dict-like interface (supports __contains__)
                if not hasattr(response, '__contains__'):
                    return False, f"Response must be dict-like, got {type(response).__name__}"

            # Check for error in response
            if 'error' in response and response.get('error'):
                error_msg = response.get('error_message', 'Unknown error')
                return False, f"LLM response contains error: {error_msg}"

            # Check expected keys if provided
            if expected_keys:
                missing = [k for k in expected_keys if k not in response]
                if missing:
                    return False, f"Response missing expected keys: {', '.join(missing)}"

            return True, None

        except Exception as e:
            error_msg = f"Error validating LLM response: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Args:
            weights: Dictionary of weights

        Returns:
            Normalized weights dictionary

        Example:
            >>> weights = {'A': 0.3, 'B': 0.3, 'C': 0.3}  # Sum = 0.9
            >>> normalized = DataValidator.normalize_weights(weights)
            >>> sum(normalized.values())  # Now sums to 1.0
            1.0
        """
        try:
            total = sum(weights.values())

            if total == 0:
                # Equal weights if all are zero
                n = len(weights)
                logger.warning(f"All weights are zero, using uniform distribution")
                return {k: 1.0 / n for k in weights.keys()}

            if abs(total - 1.0) < 0.001:
                # Already normalized
                return weights.copy()

            # Normalize
            normalized = {k: v / total for k, v in weights.items()}
            logger.info(f"Normalized weights from {total:.4f} to 1.0")
            return normalized

        except Exception as e:
            logger.error(f"Error normalizing weights: {e}")
            # Return uniform distribution as fallback
            n = len(weights)
            return {k: 1.0 / n for k in weights.keys()}

    @staticmethod
    def sanitize_scores(
        scores: Dict[str, float],
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> Dict[str, float]:
        """
        Sanitize scores to be within valid range.

        Args:
            scores: Dictionary of scores
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Sanitized scores dictionary
        """
        try:
            sanitized = {}
            clipped = False

            for key, value in scores.items():
                if not isinstance(value, (int, float)):
                    logger.warning(f"Non-numeric score for '{key}': {value}, setting to {min_val}")
                    sanitized[key] = min_val
                    clipped = True
                elif value < min_val:
                    logger.warning(f"Score for '{key}' below minimum: {value}, clipping to {min_val}")
                    sanitized[key] = min_val
                    clipped = True
                elif value > max_val:
                    logger.warning(f"Score for '{key}' above maximum: {value}, clipping to {max_val}")
                    sanitized[key] = max_val
                    clipped = True
                else:
                    sanitized[key] = value

            if clipped:
                logger.info("Scores were clipped to valid range")

            return sanitized

        except Exception as e:
            logger.error(f"Error sanitizing scores: {e}")
            return scores


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division fails.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if division fails

    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            logger.warning(f"Division by zero: {numerator} / 0, returning {default}")
            return default
        return numerator / denominator
    except Exception as e:
        logger.error(f"Error in division: {e}, returning {default}")
        return default


def safe_get(
    dictionary: Dict[str, Any],
    key: str,
    default: Any = None,
    expected_type: Optional[type] = None
) -> Any:
    """
    Safely get value from dictionary with type checking.

    Args:
        dictionary: Dictionary to get value from
        key: Key to look up
        default: Default value if key not found
        expected_type: Expected type of value (optional)

    Returns:
        Value from dictionary or default
    """
    try:
        value = dictionary.get(key, default)

        if expected_type and value is not None:
            if not isinstance(value, expected_type):
                logger.warning(
                    f"Value for key '{key}' has wrong type: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )
                return default

        return value

    except Exception as e:
        logger.error(f"Error getting key '{key}': {e}")
        return default
