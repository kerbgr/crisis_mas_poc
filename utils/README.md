# Utils Module

**Foundational Infrastructure for Configuration and Validation**

The Utils module provides the essential infrastructure components that power the entire Crisis MAS system. It centralizes configuration management, data validation, and defensive programming utilities that ensure system reliability and maintainability.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
  - [Config Class](#config-class)
  - [DataValidator Class](#datavalidator-class)
  - [Safe Operations](#safe-operations)
- [Architecture](#architecture)
- [Configuration Management](#configuration-management)
- [Validation Patterns](#validation-patterns)
- [Common Use Cases](#common-use-cases)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Related Modules](#related-modules)

---

## Overview

### What is Utils?

The Utils module is the **foundational layer** of the Crisis MAS system, providing:

1. **Configuration Management** - Centralized settings with dot notation access
2. **Data Validation** - Comprehensive validation for all data structures
3. **Safe Operations** - Defensive programming utilities (safe_divide, safe_get)
4. **Error Handling** - Structured exceptions and graceful degradation

### Why Utils?

Multi-agent systems require robust infrastructure to handle:

- **Configuration Complexity**: Dozens of parameters across agents, scenarios, LLMs, evaluation
- **Data Integrity**: Invalid data causes silent failures in agent deliberation
- **Error Resilience**: System must handle edge cases gracefully (division by zero, missing keys)
- **Maintainability**: Centralized utilities reduce code duplication

The Utils module solves these challenges by providing battle-tested, reusable components.

### Key Features

- ✅ **Zero-Config Startup** - Embedded defaults allow immediate use
- ✅ **Dot Notation** - Intuitive config access: `config.get('agents.max_agents')`
- ✅ **Deep Merge** - Partial config overrides preserve defaults
- ✅ **Comprehensive Validation** - 10+ validation methods for all data types
- ✅ **Safe Operations** - Never crash on division by zero or missing keys
- ✅ **Type Safety** - Type hints throughout, optional runtime type checking

---

## Quick Start

### Installation

The Utils module is part of the Crisis MAS system. No separate installation needed.

```python
# Import what you need
from utils import Config, DataValidator, ValidationError, safe_divide, safe_get
```

### Basic Configuration

```python
from utils import Config

# Load with defaults (no config file needed)
config = Config()

# Or load from file (merges with defaults)
config = Config('config.json')

# Access nested values with dot notation
max_agents = config.get('agents.max_agents', default=10)
llm_model = config.get('llm.model', default='claude-3-5-sonnet-20241022')
```

### Basic Validation

```python
from utils import DataValidator, ValidationError

# Validate JSON file
valid, data, error = DataValidator.validate_json_file('scenario.json')
if not valid:
    raise ValidationError(f"Invalid file: {error}")

# Validate criteria weights
criteria = {'political': 0.3, 'economic': 0.5, 'humanitarian': 0.2}
valid, error = DataValidator.validate_criteria_weights(criteria)
if not valid:
    # Auto-normalize if needed
    criteria = DataValidator.normalize_weights(criteria)
```

### Safe Operations

```python
from utils import safe_divide, safe_get

# Safe division (handles zero denominator)
average = safe_divide(total, count, default=0.0)

# Safe dictionary access
value = safe_get(data, 'key', default=None, expected_type=int)
```

---

## Core Components

### Config Class

**Purpose**: Centralized configuration management with hierarchical structure.

**Key Features**:
- Embedded default configuration (zero-config startup)
- JSON file loading with deep merge
- Dot notation access to nested values
- Runtime configuration changes
- Directory structure management

**Configuration Sections**:

```python
{
  "system": {          # Global system settings
    "name": "Crisis MAS",
    "version": "1.0.0",
    "log_level": "INFO"
  },
  "agents": {          # Agent management
    "profiles_file": "agents/agent_profiles.json",
    "max_agents": 10,
    "default_confidence": 0.7
  },
  "scenarios": {       # Scenario handling
    "scenarios_dir": "scenarios",
    "criteria_file": "scenarios/criteria_weights.json"
  },
  "decision_framework": {  # MCDA and consensus
    "consensus_threshold": 0.7,
    "max_iterations": 5,
    "convergence_rate": 0.3,
    "mcda_method": "weighted_sum"
  },
  "llm": {             # LLM integration
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "temperature": 0.7,
    "enable_llm": true,
    "retry_attempts": 3,
    "timeout": 30
  },
  "evaluation": {      # Output and metrics
    "output_dir": "results",
    "save_visualizations": true,
    "save_metrics": true,
    "generate_dashboard": true
  },
  "paths": {           # Directory structure
    "base_dir": "/path/to/project",
    "data_dir": "data",
    "output_dir": "output",
    "logs_dir": "logs"
  }
}
```

**Example Usage**:

```python
from utils import Config

# Initialize
config = Config('config.json')

# Access with dot notation
threshold = config.get('decision_framework.consensus_threshold', default=0.7)

# Get entire sections
agent_config = config.get_agent_config()
llm_config = config.get_llm_config()

# Runtime changes
config.set('agents.max_agents', 15)

# Ensure directories exist
config.ensure_directories()
```

---

### DataValidator Class

**Purpose**: Comprehensive validation for all data structures in Crisis MAS.

**Validation Methods**:

| Method | Purpose | Returns |
|--------|---------|---------|
| `validate_json_file(path)` | Load and validate JSON file | `(bool, data, error)` |
| `validate_scenario(scenario)` | Validate scenario structure | `(bool, error)` |
| `validate_alternatives(alts)` | Validate alternatives list | `(bool, error)` |
| `validate_criteria_weights(weights)` | Validate weights sum to 1.0 | `(bool, error)` |
| `validate_belief_distribution(beliefs)` | Validate beliefs sum to 1.0 | `(bool, error)` |
| `validate_agent_profile(profile)` | Validate agent profile | `(bool, error)` |
| `validate_llm_response(response, keys)` | Validate LLM response | `(bool, error)` |
| `normalize_weights(weights)` | Normalize to sum to 1.0 | `Dict[str, float]` |
| `sanitize_scores(scores, min, max)` | Clamp to range | `Dict[str, float]` |

**Validation Philosophy**:

1. **Progressive Validation**:
   - Level 1: Structure (required fields exist)
   - Level 2: Type (fields have correct types)
   - Level 3: Constraints (values in valid ranges)

2. **Fail Fast, Fail Informative**:
   - Detect errors at boundaries
   - Provide specific error messages
   - Include context for debugging

3. **Normalize vs. Reject**:
   - Minor issues (0.999 → 1.0): Auto-normalize
   - Major issues (missing fields): Reject with error

**Example Usage**:

```python
from utils import DataValidator, ValidationError

# Validate and load scenario
valid, data, error = DataValidator.validate_json_file('scenario.json')
if not valid:
    raise ValidationError(f"Invalid scenario file: {error}")

valid, error = DataValidator.validate_scenario(data)
if not valid:
    raise ValidationError(f"Invalid scenario structure: {error}")

# Validate and normalize weights
criteria = {'political': 0.31, 'economic': 0.49, 'humanitarian': 0.19}
valid, error = DataValidator.validate_criteria_weights(criteria, tolerance=0.01)
if not valid:
    criteria = DataValidator.normalize_weights(criteria)

# Sanitize scores to valid range
scores = {'alt_1': 1.2, 'alt_2': -0.1, 'alt_3': 0.7}
clean_scores = DataValidator.sanitize_scores(scores, min_val=0.0, max_val=1.0)
# Returns: {'alt_1': 1.0, 'alt_2': 0.0, 'alt_3': 0.7}
```

---

### Safe Operations

**Purpose**: Defensive programming utilities that never crash.

#### safe_divide

```python
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division fails.

    Handles:
    - Division by zero
    - None values
    - Type errors

    Example:
        avg = safe_divide(total_score, num_evaluations, default=0.0)
    """
```

**Use Cases**:
- Calculating averages with potentially empty datasets
- Computing ratios where denominator might be zero
- Metrics calculations in evaluation

**Example**:

```python
from utils import safe_divide

# Safe even if num_agents is 0
avg_confidence = safe_divide(sum_confidence, num_agents, default=0.5)

# Consensus calculation
max_belief = max(beliefs.values())
total = sum(beliefs.values())
consensus_score = safe_divide(max_belief, total, default=0.0)
```

#### safe_get

```python
def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None,
             expected_type: Optional[type] = None) -> Any:
    """
    Safely get value from dictionary with type checking.

    Handles:
    - Missing keys
    - Type mismatches
    - None values

    Example:
        confidence = safe_get(profile, 'confidence_level', default=0.7, expected_type=float)
    """
```

**Use Cases**:
- Accessing optional configuration values
- Extracting data from partially-filled dictionaries
- Type-safe access to dynamic data structures

**Example**:

```python
from utils import safe_get

agent_data = {'name': 'Agent1', 'confidence': 0.8}

# Basic access
name = safe_get(agent_data, 'name', default='Unknown')

# Type-checked access
confidence = safe_get(agent_data, 'confidence', default=0.5, expected_type=float)

# Missing key (returns default)
role = safe_get(agent_data, 'role', default='general')

# Type mismatch (returns default)
age = safe_get(agent_data, 'confidence', default=0, expected_type=int)
# Returns 0 because confidence is float, not int
```

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Utils Module                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Config    │  │ DataValidator│  │Safe Operations│     │
│  │              │  │              │  │              │      │
│  │ • Defaults   │  │ • JSON Load  │  │ • safe_divide│      │
│  │ • File Load  │  │ • Scenarios  │  │ • safe_get   │      │
│  │ • Dot Access │  │ • Weights    │  │              │      │
│  │ • Deep Merge │  │ • Profiles   │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │             │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌─────────────────────────────────────────────────┐
    │         Crisis MAS System Modules               │
    ├─────────────────────────────────────────────────┤
    │  agents/  scenarios/  decision_framework/       │
    │  llm_integration/  evaluation/                  │
    └─────────────────────────────────────────────────┘
```

### Data Flow

```
1. System Startup
   ↓
   Config Initialization (load defaults + file)
   ↓
2. Data Ingestion
   ↓
   Validation (JSON → Scenario → Alternatives → Weights)
   ↓
3. Agent Deliberation
   ↓
   Safe Operations (calculations with defensive checks)
   ↓
4. Runtime
   ↓
   Config Access (dot notation, specialized getters)
```

### Integration Points

**Config is used by**:
- System initialization: Log level, version
- Agent factory: Max agents, confidence defaults
- LLM clients: Model, tokens, temperature
- Decision framework: Thresholds, iterations
- Evaluation: Output paths, visualization flags

**DataValidator is used by**:
- Scenario loaders: Validate scenario files
- Agent factory: Validate agent profiles
- Decision framework: Validate criteria weights
- LLM integration: Validate LLM responses
- Evaluation: Validate metric calculations

**Safe Operations are used by**:
- Decision framework: Consensus calculations
- Evaluation: Metrics computations
- Agents: Belief updates
- All modules: Defensive programming

---

## Configuration Management

### Configuration Hierarchy

Configuration follows a precedence order:

```
Hardcoded Defaults (in code)
    ↓ override
Config File (config.json)
    ↓ override (future)
Environment Variables
    ↓ override
Runtime Changes (config.set())
```

### Deep Merge Behavior

When loading from a file, configurations are **deep-merged** with defaults:

**Default Config**:
```json
{
  "agents": {
    "max_agents": 10,
    "default_confidence": 0.7
  }
}
```

**File Config** (`config.json`):
```json
{
  "agents": {
    "max_agents": 15
  }
}
```

**Result** (after merge):
```json
{
  "agents": {
    "max_agents": 15,           // From file (override)
    "default_confidence": 0.7   // From defaults (preserved)
  }
}
```

**Benefits**:
- Partial overrides possible
- Don't need to specify all values in config file
- Sensible defaults preserved
- Only override what you need to change

### Dot Notation

Access nested configuration values intuitively:

```python
# Preferred: Dot notation
consensus = config.get('decision_framework.consensus_threshold', default=0.7)

# Equivalent: Nested dict access (verbose)
consensus = config.config.get('decision_framework', {}).get('consensus_threshold', 0.7)
```

**How it works**:
```python
Key path: 'llm.model'
Split on '.': ['llm', 'model']
Traverse: config['llm']['model']
Return: 'claude-3-5-sonnet-20241022'
```

### Environment-Specific Configs

Use different config files per environment:

```bash
# Development
python main.py --config configs/dev.json

# Testing
python main.py --config configs/test.json

# Production
python main.py --config configs/prod.json
```

---

## Validation Patterns

### Progressive Validation

Validate in three levels:

**Level 1: Structure** (Required fields exist)
```python
if 'id' not in scenario:
    return (False, "Missing required field 'id'")
```

**Level 2: Type** (Fields have correct types)
```python
if not isinstance(scenario['id'], str):
    return (False, "Field 'id' must be a string")
```

**Level 3: Constraint** (Values in valid ranges)
```python
if not scenario['id'].strip():
    return (False, "Field 'id' cannot be empty")
```

### Validation Workflow

```python
from utils import DataValidator, ValidationError

# Step 1: Validate JSON file
valid, data, error = DataValidator.validate_json_file('scenario.json')
if not valid:
    raise ValidationError(f"JSON error: {error}")

# Step 2: Validate scenario structure
valid, error = DataValidator.validate_scenario(data)
if not valid:
    raise ValidationError(f"Scenario error: {error}")

# Step 3: Validate alternatives
valid, error = DataValidator.validate_alternatives(data['alternatives'])
if not valid:
    raise ValidationError(f"Alternatives error: {error}")

# Step 4: Validate criteria weights
valid, error = DataValidator.validate_criteria_weights(data['criteria'])
if not valid:
    # Minor issue: normalize
    data['criteria'] = DataValidator.normalize_weights(data['criteria'])

# Data is now validated and safe to use
```

### Error Handling Strategies

**1. Tuple Return Pattern** (Flexible)
```python
valid, data, error = DataValidator.validate_json_file(path)
if not valid:
    # Caller decides: exception or graceful handling
    logger.error(error)
    return default_data
```

**2. Exception Pattern** (Fail Fast)
```python
from utils import ValidationError

valid, error = DataValidator.validate_scenario(scenario)
if not valid:
    raise ValidationError(error)
```

**3. Safe Operations** (Never Fail)
```python
# These never raise exceptions
result = safe_divide(a, b, default=0.0)
value = safe_get(data, key, default=None)
```

---

## Common Use Cases

### Use Case 1: System Initialization

```python
from utils import Config
import logging

# Load configuration
config = Config('config.json')

# Setup logging
log_level = config.get('system.log_level', default='INFO')
logging.basicConfig(level=getattr(logging, log_level))

# Ensure directories
config.ensure_directories()

# Initialize system
system_name = config.get('system.name', default='Crisis MAS')
version = config.get('system.version', default='1.0.0')
print(f"{system_name} v{version} initialized")
```

### Use Case 2: Agent Creation

```python
from utils import Config, DataValidator, ValidationError, safe_get

def create_agent(profile_path: str, config: Config):
    # Validate profile file
    valid, profile, error = DataValidator.validate_json_file(profile_path)
    if not valid:
        raise ValidationError(f"Invalid profile file: {error}")

    # Validate profile structure
    valid, error = DataValidator.validate_agent_profile(profile)
    if not valid:
        raise ValidationError(f"Invalid profile: {error}")

    # Get config parameters
    max_agents = config.get('agents.max_agents', default=10)
    default_confidence = config.get('agents.default_confidence', default=0.7)

    # Extract profile data with safe defaults
    agent_id = safe_get(profile, 'id', default='unknown')
    role = safe_get(profile, 'role', default='general')
    confidence = safe_get(profile, 'confidence_level',
                         default=default_confidence,
                         expected_type=float)

    # Create agent
    agent = Agent(
        id=agent_id,
        role=role,
        confidence=confidence
    )
    return agent
```

### Use Case 3: Consensus Calculation

```python
from utils import safe_divide

def calculate_consensus(agent_beliefs: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate consensus across agent beliefs.

    Uses safe_divide to handle edge cases:
    - Empty agent list
    - Zero total belief
    """
    if not agent_beliefs:
        return {}

    # Aggregate beliefs
    aggregated = {}
    for beliefs in agent_beliefs:
        for option, belief in beliefs.items():
            aggregated[option] = aggregated.get(option, 0.0) + belief

    # Calculate consensus scores
    total_agents = len(agent_beliefs)
    consensus = {}

    for option, total_belief in aggregated.items():
        # Safe division handles total_agents = 0 gracefully
        avg_belief = safe_divide(total_belief, total_agents, default=0.0)
        consensus[option] = avg_belief

    return consensus
```

### Use Case 4: Metrics Calculation

```python
from utils import safe_divide, safe_get

def calculate_decision_quality_metrics(evaluations: List[Dict]) -> Dict[str, float]:
    """
    Calculate decision quality metrics.

    Uses safe operations to handle:
    - Empty evaluations list
    - Missing keys in evaluation dicts
    - Division by zero
    """
    if not evaluations:
        return {
            'avg_consensus': 0.0,
            'avg_confidence': 0.0,
            'convergence_rate': 0.0
        }

    total_consensus = 0.0
    total_confidence = 0.0

    for eval_data in evaluations:
        # Safe extraction with defaults
        consensus = safe_get(eval_data, 'consensus_score', default=0.0, expected_type=float)
        confidence = safe_get(eval_data, 'confidence_score', default=0.0, expected_type=float)

        total_consensus += consensus
        total_confidence += confidence

    num_evaluations = len(evaluations)

    # Safe division
    avg_consensus = safe_divide(total_consensus, num_evaluations, default=0.0)
    avg_confidence = safe_divide(total_confidence, num_evaluations, default=0.0)

    return {
        'avg_consensus': avg_consensus,
        'avg_confidence': avg_confidence,
        'num_evaluations': num_evaluations
    }
```

---

## Best Practices

### Configuration

✅ **Always provide defaults in get() calls**
```python
# Good
value = config.get('key.path', default=42)

# Bad (could return None)
value = config.get('key.path')
```

✅ **Use specialized getters for common sections**
```python
# Good
agent_config = config.get_agent_config()

# Acceptable
agent_config = config.get('agents', default={})
```

✅ **Group related settings in config file**
```json
// Good - grouped
{
  "llm": {
    "model": "...",
    "temperature": 0.7,
    "max_tokens": 1024
  }
}

// Bad - scattered
{
  "llm_model": "...",
  "llm_temperature": 0.7,
  "llm_max_tokens": 1024
}
```

### Validation

✅ **Validate at boundaries (data ingestion points)**
```python
# Validate when loading files
valid, data, error = DataValidator.validate_json_file('scenario.json')
if not valid:
    raise ValidationError(error)

# Validate before processing
valid, error = DataValidator.validate_scenario(data)
if not valid:
    raise ValidationError(error)
```

✅ **Use normalization for minor issues**
```python
# Check weights
valid, error = DataValidator.validate_criteria_weights(weights)
if not valid:
    # Auto-fix floating-point rounding
    weights = DataValidator.normalize_weights(weights)
```

✅ **Provide informative error messages**
```python
# Good
if not valid:
    raise ValidationError(f"Invalid scenario '{scenario_id}': {error}")

# Bad
if not valid:
    raise ValidationError(error)
```

### Safe Operations

✅ **Use safe_divide for all division operations**
```python
# Good
average = safe_divide(total, count, default=0.0)

# Bad (crashes if count = 0)
average = total / count
```

✅ **Use safe_get for dynamic data structures**
```python
# Good
confidence = safe_get(profile, 'confidence_level',
                     default=0.7, expected_type=float)

# Bad (crashes if key missing or wrong type)
confidence = profile['confidence_level']
```

✅ **Choose sensible defaults for your context**
```python
# For averages, use 0.0
avg_score = safe_divide(total_score, count, default=0.0)

# For ratios, might use 1.0
ratio = safe_divide(actual, expected, default=1.0)

# Context-dependent!
```

---

## Troubleshooting

### Configuration Issues

**Problem**: "Config file not found"
```
Cause: Incorrect path or file doesn't exist
Solution: Config falls back to defaults automatically
Debug: Check config.config_file attribute
```

**Problem**: "Wrong value type"
```
Cause: Config file has string where int expected
Solution: Validate types after get(), use defaults
Prevention: Use JSON schema validation
```

**Problem**: "Missing configuration key"
```
Cause: Typo in key path or key doesn't exist
Solution: Always use default parameter in get()
Debug: Print config.config to see all keys
```

### Validation Issues

**Problem**: "Weights do not sum to 1.0"
```python
# Solution: Normalize weights
criteria = DataValidator.normalize_weights(criteria)
```

**Problem**: "Missing required field 'X'"
```
Cause: Incomplete JSON structure
Solution: Check JSON file, ensure all required fields present
Reference: See validation.py docstrings for required fields
```

**Problem**: "Value out of range"
```
Cause: confidence_level > 1.0 or < 0.0
Solution: Check data source, ensure values in [0, 1]
Or use: sanitize_scores() to clamp to valid range
```

### Safe Operations Issues

**Problem**: "safe_get returns default even though key exists"
```python
# Likely cause: Type mismatch with expected_type
value = safe_get(data, 'confidence', default=0, expected_type=int)
# If data['confidence'] is float (e.g., 0.8), returns default (0)

# Solution: Remove expected_type or use correct type
value = safe_get(data, 'confidence', default=0.5, expected_type=float)
```

---

## API Reference

### Config Class

```python
class Config:
    def __init__(self, config_file: Optional[str] = None)
    def get(self, key_path: str, default: Any = None) -> Any
    def set(self, key_path: str, value: Any) -> None
    def get_agent_config(self) -> Dict[str, Any]
    def get_llm_config(self) -> Dict[str, Any]
    def ensure_directories(self) -> None
```

### DataValidator Class

```python
class DataValidator:
    @staticmethod
    def validate_json_file(file_path: str) -> Tuple[bool, Optional[Dict], Optional[str]]

    @staticmethod
    def validate_scenario(scenario: Dict[str, Any]) -> Tuple[bool, Optional[str]]

    @staticmethod
    def validate_alternatives(alternatives: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]

    @staticmethod
    def validate_criteria_weights(criteria: Dict[str, Any], tolerance: float = 0.01) -> Tuple[bool, Optional[str]]

    @staticmethod
    def validate_belief_distribution(beliefs: Dict[str, float], tolerance: float = 0.01) -> Tuple[bool, Optional[str]]

    @staticmethod
    def validate_agent_profile(profile: Dict[str, Any]) -> Tuple[bool, Optional[str]]

    @staticmethod
    def validate_llm_response(response: Dict[str, Any], expected_keys: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]

    @staticmethod
    def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]

    @staticmethod
    def sanitize_scores(scores: Dict[str, float], min_val: float = 0.0, max_val: float = 1.0) -> Dict[str, float]
```

### Safe Operations

```python
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float
    """Safely divide two numbers, returning default if division fails."""

def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None, expected_type: Optional[type] = None) -> Any
    """Safely get value from dictionary with type checking."""
```

### ValidationError Exception

```python
class ValidationError(Exception):
    """Custom exception for validation errors."""
```

---

## Related Modules

### Dependencies

- **json**: Configuration file parsing, JSON validation
- **logging**: Structured logging for validation failures
- **pathlib**: Cross-platform path operations
- **typing**: Type hints for API clarity
- **os**: Operating system interface

### Used By

- **agents/**: Config for agent parameters, validation for profiles
- **scenarios/**: Config for scenario paths, validation for scenario files
- **decision_framework/**: Config for MCDA parameters, safe operations for calculations
- **llm_integration/**: Config for LLM settings, validation for responses
- **evaluation/**: Config for output settings, safe operations for metrics

### Integration Example

```python
# System-wide integration
from utils import Config, DataValidator, safe_divide, safe_get

# 1. Load configuration
config = Config('config.json')
config.ensure_directories()

# 2. Validate scenario
valid, scenario, error = DataValidator.validate_json_file(
    config.get('scenarios.scenarios_dir') + '/scenario1.json'
)

# 3. Create agents with safe defaults
max_agents = config.get('agents.max_agents', default=10)
for i in range(max_agents):
    # Use safe_get for profile data
    agent = create_agent(profile, config)

# 4. Run deliberation with safe operations
consensus = safe_divide(max_belief, total_belief, default=0.0)
```

---

## Summary

The Utils module provides the **foundational infrastructure** for the Crisis MAS system:

- **Config**: Centralized configuration with dot notation, deep merge, and sensible defaults
- **DataValidator**: Comprehensive validation for all data structures with progressive validation levels
- **Safe Operations**: Defensive programming utilities that never crash

**Key Principles**:
1. **Zero-Config Startup**: Embedded defaults enable immediate use
2. **Fail Fast, Fail Informative**: Early detection with clear error messages
3. **Defensive Programming**: Safe operations handle edge cases gracefully
4. **Maintainability**: Centralized utilities reduce code duplication

**Remember**:
- Always provide defaults in `config.get()`
- Validate at boundaries (data ingestion points)
- Use safe operations for calculations
- Normalize minor issues, reject major ones

For detailed API documentation, see the docstrings in each module file.
