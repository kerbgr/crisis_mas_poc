"""
═══════════════════════════════════════════════════════════════════════════════
UTILITIES MODULE
Core infrastructure for configuration management, data validation, and safe operations
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
═════════
This module provides essential utility functions and classes that form the
foundational infrastructure for the Crisis MAS system. It centralizes:

1. Configuration Management - Unified config with dot notation access
2. Data Validation - Comprehensive validation for all data structures
3. Safe Operations - Defensive programming utilities for robust error handling
4. Error Handling - Structured exception types and graceful degradation


WHY THIS EXISTS
═══════════════
The utils module addresses several critical needs in distributed multi-agent systems:

1. **Configuration Centralization**
   - Problem: Hardcoded values scattered across modules
   - Solution: Single source of truth for all system parameters
   - Benefit: Easy tuning, consistent behavior, environment-specific configs

2. **Data Integrity**
   - Problem: Invalid data causing runtime failures in agent deliberation
   - Solution: Comprehensive validation before processing
   - Benefit: Early error detection, informative error messages

3. **Defensive Programming**
   - Problem: Division by zero, missing keys, type errors in agent calculations
   - Solution: Safe operations with sensible defaults
   - Benefit: System resilience, graceful degradation

4. **Maintainability**
   - Problem: Validation logic duplicated across modules
   - Solution: Centralized validation utilities
   - Benefit: DRY principle, easier testing, consistent validation


KEY COMPONENTS
══════════════

1. Config Class (config.py)
   ─────────────────────────
   Centralized configuration manager with:
   - Dot notation access: config.get('agents.max_agents')
   - Deep merge support for hierarchical configs
   - JSON persistence for config files
   - Default configuration fallback
   - Environment-specific overrides
   - Directory structure management

2. DataValidator Class (validation.py)
   ──────────────────────────────────
   Comprehensive validation for:
   - JSON file loading with error handling
   - Scenario structure validation
   - Alternative options validation
   - Criteria weights normalization
   - Belief distribution validation
   - Agent profile validation
   - LLM response validation
   - Score sanitization

3. Safe Operations (validation.py)
   ────────────────────────────────
   Defensive utilities:
   - safe_divide: Division with zero-handling
   - safe_get: Dictionary access with type checking

4. ValidationError Exception
   ─────────────────────────
   Custom exception for validation failures


TYPICAL USAGE
═════════════

Example 1: Configuration Management
───────────────────────────────────
```python
from utils import Config

# Load configuration (uses default or specified file)
config = Config('config.json')

# Access nested values with dot notation
max_agents = config.get('agents.max_agents', default=10)
llm_model = config.get('llm.model', default='gpt-4')

# Modify configuration
config.set('agents.max_agents', 15)

# Get specialized config sections
agent_config = config.get_agent_config()
llm_config = config.get_llm_config()

# Ensure required directories exist
config.ensure_directories()
```

Example 2: Data Validation
──────────────────────────
```python
from utils import DataValidator, ValidationError

# Validate and load JSON file
valid, data, error = DataValidator.validate_json_file('scenario.json')
if not valid:
    raise ValidationError(f"Invalid scenario file: {error}")

# Validate scenario structure
valid, error = DataValidator.validate_scenario(data)
if not valid:
    raise ValidationError(error)

# Validate criteria weights sum to 1.0
criteria = {'political': 0.3, 'economic': 0.5, 'humanitarian': 0.2}
valid, error = DataValidator.validate_criteria_weights(criteria)
if not valid:
    # Automatically normalize
    criteria = DataValidator.normalize_weights(criteria)

# Validate belief distribution
beliefs = {'option_a': 0.6, 'option_b': 0.4}
valid, error = DataValidator.validate_belief_distribution(beliefs)

# Validate agent profile
profile = {
    'id': 'agent_1',
    'role': 'diplomat',
    'confidence_level': 0.8,
    'belief_update_rate': 0.3
}
valid, error = DataValidator.validate_agent_profile(profile)
```

Example 3: Safe Operations
──────────────────────────
```python
from utils import safe_divide, safe_get

# Safe division (handles zero denominator)
avg_score = safe_divide(total_score, num_evaluations, default=0.0)
# Returns default=0.0 if num_evaluations == 0

# Safe dictionary access with type checking
agent_data = {'name': 'Agent1', 'confidence': 0.8}

name = safe_get(agent_data, 'name', default='Unknown', expected_type=str)
confidence = safe_get(agent_data, 'confidence', default=0.5, expected_type=float)
missing = safe_get(agent_data, 'missing_key', default=None)
# Returns None if key doesn't exist

# Type validation
age = safe_get(agent_data, 'confidence', expected_type=int, default=0)
# Returns 0 because confidence is float, not int
```

Example 4: Integration in Agent Module
───────────────────────────────────────
```python
from utils import Config, DataValidator, safe_divide, ValidationError

class Agent:
    def __init__(self, profile, config):
        # Validate profile before use
        valid, error = DataValidator.validate_agent_profile(profile)
        if not valid:
            raise ValidationError(f"Invalid agent profile: {error}")

        # Use config for parameters
        self.max_iterations = config.get('decision_framework.max_iterations', 5)
        self.confidence = profile.get('confidence_level', 0.7)

    def calculate_consensus(self, beliefs):
        # Validate belief distribution
        valid, error = DataValidator.validate_belief_distribution(beliefs)
        if not valid:
            # Normalize if needed
            beliefs = DataValidator.normalize_weights(beliefs)

        # Safe operations
        total = sum(beliefs.values())
        consensus = safe_divide(max(beliefs.values()), total, default=0.0)
        return consensus
```


INTEGRATION POINTS
══════════════════

The utils module is foundational and used by all other modules:

1. **agents/** → Uses Config for agent parameters, DataValidator for profiles
2. **scenarios/** → Uses DataValidator for scenario files, criteria validation
3. **decision_framework/** → Uses safe_divide, safe_get for calculations
4. **llm_integration/** → Uses Config for LLM settings, DataValidator for responses
5. **evaluation/** → Uses safe operations for metrics, Config for output settings

Data Flow:
──────────
1. Config loads at system startup
2. Validation occurs at data ingestion boundaries
3. Safe operations used throughout computation
4. ValidationError propagates to calling modules


ERROR HANDLING
══════════════

Validation Errors
─────────────────
```python
from utils import ValidationError

try:
    valid, data, error = DataValidator.validate_json_file('missing.json')
    if not valid:
        raise ValidationError(error)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Handle gracefully with defaults
```

Configuration Errors
────────────────────
```python
from utils import Config

config = Config('config.json')  # Falls back to defaults if file missing
value = config.get('some.nested.key', default='fallback')  # Always has default
```

Safe Operation Patterns
───────────────────────
```python
from utils import safe_divide, safe_get

# Never raises ZeroDivisionError
result = safe_divide(numerator, 0, default=0.0)

# Never raises KeyError or TypeError
value = safe_get(data, 'key', default=None, expected_type=int)
```


DESIGN DECISIONS
════════════════

1. **Static Methods for Validation**
   - Rationale: Validation is stateless, no need for instance state
   - Benefit: Easy to use, clear API, testable

2. **Tuple Returns (bool, data/error)**
   - Rationale: Allow validation without exceptions in performance-critical paths
   - Benefit: Caller decides whether to raise or handle gracefully

3. **Dot Notation for Config**
   - Rationale: More readable than nested dict access
   - Benefit: 'agents.max_agents' vs config['agents']['max_agents']

4. **Default Configuration Embedded**
   - Rationale: System can run without external config files
   - Benefit: Quick start, self-documenting defaults

5. **Normalization vs. Error**
   - Rationale: Weights often have floating-point rounding issues
   - Benefit: Normalize_weights fixes minor discrepancies automatically

6. **Safe Operations with Defaults**
   - Rationale: Let caller specify sensible fallback for their context
   - Benefit: Flexible, explicit, no hidden assumptions


VALIDATION PHILOSOPHY
═════════════════════

The validation approach follows these principles:

1. **Fail Fast, Fail Informative**
   - Detect errors at boundaries (file load, data ingestion)
   - Provide clear error messages with context
   - Guide users to fix issues

2. **Progressive Validation**
   - Level 1: Structure (required fields exist)
   - Level 2: Types (fields have correct types)
   - Level 3: Constraints (weights sum to 1.0, scores in range)

3. **Normalize vs. Reject**
   - Minor issues (0.999 weights): Normalize automatically
   - Major issues (missing fields): Reject with error
   - Rationale: Usability vs. correctness trade-off

4. **Validation Context**
   - Validate at appropriate granularity
   - JSON file: validate_json_file (file + parse + structure)
   - Data structure: validate_scenario (structure only)
   - Single field: validate_belief_distribution (constraints only)


CONFIGURATION PHILOSOPHY
═════════════════════════

1. **Hierarchy of Configuration**
   ```
   Hardcoded Defaults (in code)
       ↓
   Default Config File (config.json)
       ↓
   Environment Variables (future)
       ↓
   Runtime Overrides (config.set())
   ```

2. **Configuration Sections**
   - system: Global system settings
   - agents: Agent-specific parameters
   - scenarios: Scenario and criteria settings
   - decision_framework: MCDA and consensus parameters
   - llm: LLM provider and model settings
   - evaluation: Output and visualization settings
   - paths: Directory structure

3. **Best Practices**
   - Always provide defaults in get() calls
   - Use dot notation for readability
   - Group related settings in sections
   - Document configuration changes in VERSION HISTORY


RELATED MODULES
═══════════════

Core Dependencies:
- pathlib: Cross-platform path handling
- json: Configuration and data persistence
- logging: Structured logging for validation failures
- typing: Type hints for API clarity

Module Relationships:
- **utils** (this module) - Foundation
  ↓
- **agents**, **scenarios**, **decision_framework**, **llm_integration** - Use utils
  ↓
- **evaluation** - Uses utils for metrics calculations


TROUBLESHOOTING
═══════════════

Common Issues:

1. **"Config file not found"**
   - Cause: config.json missing or wrong path
   - Solution: Config falls back to defaults, check paths with config.get('paths.base_dir')

2. **"Weights do not sum to 1.0"**
   - Cause: Floating-point rounding or data entry error
   - Solution: Use DataValidator.normalize_weights() to auto-fix

3. **"ValidationError: Missing required field"**
   - Cause: JSON structure incomplete
   - Solution: Check error message for specific field, refer to validation.py for schema

4. **Division by zero in metrics**
   - Cause: Empty data or zero denominators
   - Solution: Always use safe_divide() with appropriate default

5. **Type mismatch in safe_get**
   - Cause: Expected type doesn't match actual type
   - Solution: Check expected_type parameter, or set to None to disable checking


VERSION HISTORY
═══════════════

Version 1.0.0 (Initial Release)
- Config class with dot notation and JSON persistence
- DataValidator with comprehensive validation methods
- Safe operations (safe_divide, safe_get)
- ValidationError exception type
- Default configuration embedded in code

Future Enhancements:
- Environment variable support in Config
- JSON Schema validation for complex structures
- Configuration migration utilities
- Performance profiling for validation
- Async validation for large files
"""

from .config import Config
from .validation import (
    DataValidator,
    ValidationError,
    safe_divide,
    safe_get
)

__all__ = [
    'Config',
    'DataValidator',
    'ValidationError',
    'safe_divide',
    'safe_get'
]
