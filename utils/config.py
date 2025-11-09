"""
═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION MANAGEMENT MODULE
Centralized configuration system with hierarchical structure and dot notation
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
═════════
This module provides a centralized configuration management system for the
Crisis MAS platform. It handles:

1. **Default Configuration** - Embedded defaults for zero-config startup
2. **File-Based Configuration** - Load settings from JSON files
3. **Dot Notation Access** - Intuitive nested key access (e.g., 'agents.max_agents')
4. **Deep Merge** - Combine file configs with defaults intelligently
5. **Directory Management** - Ensure required directories exist


WHY THIS EXISTS
═══════════════
Configuration management is critical for multi-agent systems:

1. **Avoid Hardcoded Values**
   - Problem: Magic numbers scattered across modules
   - Solution: Centralized, named configuration parameters
   - Benefit: Easy tuning without code changes

2. **Environment Flexibility**
   - Problem: Different settings for dev/test/prod
   - Solution: Load different config files per environment
   - Benefit: Same code runs in all environments

3. **Maintainability**
   - Problem: Finding and changing system parameters
   - Solution: Single source of truth for all settings
   - Benefit: Quick configuration changes, clear documentation

4. **Sensible Defaults**
   - Problem: System requires config file to run
   - Solution: Embedded defaults for immediate use
   - Benefit: Zero-config startup for demos and testing

5. **Type Safety**
   - Problem: Runtime errors from wrong config types
   - Solution: Structured config with clear sections
   - Benefit: Early error detection, IDE autocomplete


INPUTS
══════
The Config class accepts:

1. **Config File Path** (Optional)
   - Type: str or None
   - Format: JSON file
   - Purpose: Override default configuration
   - Example: 'config.json', 'configs/production.json'

2. **Configuration Structure** (JSON format)
   ```json
   {
     "system": {
       "name": "Crisis MAS",
       "version": "1.0.0",
       "log_level": "INFO"
     },
     "agents": {
       "profiles_file": "agents/agent_profiles.json",
       "max_agents": 10,
       "default_confidence": 0.7
     },
     "scenarios": {
       "scenarios_dir": "scenarios",
       "criteria_file": "scenarios/criteria_weights.json"
     },
     "decision_framework": {
       "consensus_threshold": 0.7,
       "max_iterations": 5,
       "convergence_rate": 0.3,
       "mcda_method": "weighted_sum"
     },
     "llm": {
       "model": "claude-3-5-sonnet-20241022",
       "max_tokens": 1024,
       "temperature": 0.7,
       "enable_llm": true,
       "retry_attempts": 3,
       "timeout": 30
     },
     "evaluation": {
       "output_dir": "results",
       "save_visualizations": true,
       "save_metrics": true,
       "generate_dashboard": true
     },
     "paths": {
       "base_dir": "/path/to/project",
       "data_dir": "data",
       "output_dir": "output",
       "logs_dir": "logs"
     }
   }
   ```

3. **Key Paths** (for get/set operations)
   - Type: str with dot notation
   - Format: 'section.subsection.key'
   - Examples:
     * 'agents.max_agents'
     * 'llm.model'
     * 'decision_framework.consensus_threshold'
     * 'system.log_level'


OUTPUTS
═══════
The Config class provides:

1. **Configuration Values**
   - get(key_path, default) → Any
   - Returns value at key_path or default if not found
   - Example: config.get('agents.max_agents', default=10) → 10

2. **Specialized Getters**
   - get_agent_config() → Dict[str, Any]
   - get_llm_config() → Dict[str, Any]
   - Returns entire configuration section

3. **Configuration Sections**
   - Hierarchical dictionary structure
   - Organized by functional area
   - Easy to navigate and understand

4. **Boolean Indicators**
   - Methods can check if paths exist, files loaded, etc.
   - Example: Check if LLM is enabled


CONFIGURATION HIERARCHY
═══════════════════════
Configuration follows a precedence order:

```
1. Hardcoded Defaults (in _load_default_config)
        ↓ (override)
2. Config File (if provided and exists)
        ↓ (override - future)
3. Environment Variables (planned feature)
        ↓ (override)
4. Runtime Changes (via set method)
```

This allows:
- Zero-config startup (uses defaults)
- File-based customization (override defaults)
- Runtime tuning (dynamic adjustments)


CONFIGURATION SECTIONS
══════════════════════

1. **system**
   ──────────
   Global system-level settings:
   - name: System name identifier
   - version: Version string for tracking
   - log_level: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

2. **agents**
   ─────────
   Agent-related configuration:
   - profiles_file: Path to agent profiles JSON
   - max_agents: Maximum number of concurrent agents
   - default_confidence: Default confidence level for new agents

3. **scenarios**
   ────────────
   Scenario management settings:
   - scenarios_dir: Directory containing scenario files
   - criteria_file: Path to criteria weights JSON

4. **decision_framework**
   ─────────────────────
   MCDA and consensus parameters:
   - consensus_threshold: Minimum agreement level (0.0-1.0)
   - max_iterations: Maximum deliberation rounds
   - convergence_rate: Speed of belief updates (0.0-1.0)
   - mcda_method: Multi-criteria method ('weighted_sum', 'topsis', etc.)

5. **llm**
   ──────
   LLM integration settings:
   - model: Model identifier (e.g., 'claude-3-5-sonnet-20241022')
   - max_tokens: Maximum response length
   - temperature: Randomness in generation (0.0-1.0)
   - enable_llm: Global LLM enable/disable flag
   - retry_attempts: Number of retry attempts on failure
   - timeout: Request timeout in seconds

6. **evaluation**
   ─────────────
   Evaluation and output settings:
   - output_dir: Directory for results
   - save_visualizations: Enable/disable chart generation
   - save_metrics: Enable/disable metrics logging
   - generate_dashboard: Enable/disable dashboard creation

7. **paths**
   ────────
   Directory structure configuration:
   - base_dir: Project root directory
   - data_dir: Data files location
   - output_dir: Output files location
   - logs_dir: Log files location


TYPICAL USAGE
═════════════

Example 1: Basic Configuration Loading
───────────────────────────────────────
```python
from utils.config import Config

# Load with defaults (no config file)
config = Config()

# Load from file (merges with defaults)
config = Config('config.json')

# Access values with dot notation
max_agents = config.get('agents.max_agents', default=10)
llm_model = config.get('llm.model', default='gpt-4')
log_level = config.get('system.log_level', default='INFO')
```

Example 2: Accessing Nested Configuration
──────────────────────────────────────────
```python
from utils.config import Config

config = Config('config.json')

# Deep nested access
consensus = config.get('decision_framework.consensus_threshold', default=0.7)
mcda_method = config.get('decision_framework.mcda_method', default='weighted_sum')

# Check if LLM is enabled
if config.get('llm.enable_llm', default=True):
    model = config.get('llm.model')
    print(f"Using LLM model: {model}")
```

Example 3: Specialized Configuration Getters
─────────────────────────────────────────────
```python
from utils.config import Config

config = Config()

# Get entire agent configuration section
agent_config = config.get_agent_config()
# Returns: {'profiles_file': '...', 'max_agents': 10, 'default_confidence': 0.7}

# Get entire LLM configuration section
llm_config = config.get_llm_config()
# Returns: {'model': '...', 'max_tokens': 1024, ...}

# Use in agent initialization
agent = Agent(agent_config)
```

Example 4: Runtime Configuration Changes
─────────────────────────────────────────
```python
from utils.config import Config

config = Config()

# Modify configuration at runtime
config.set('agents.max_agents', 15)
config.set('llm.temperature', 0.9)
config.set('decision_framework.consensus_threshold', 0.8)

# Changes are immediately available
new_max = config.get('agents.max_agents')  # Returns 15
```

Example 5: Directory Management
────────────────────────────────
```python
from utils.config import Config

config = Config('config.json')

# Ensure all required directories exist
config.ensure_directories()
# Creates: data/, output/, logs/ directories if missing

# Safe to proceed with file operations
output_path = Path(config.get('paths.output_dir')) / 'results.json'
```

Example 6: Using Convenience Function
──────────────────────────────────────
```python
from utils.config import load_config

# Quick one-liner to load config
config = load_config('config.json')

# Equivalent to:
# config = Config('config.json')
```


DOT NOTATION
════════════
The Config class uses dot notation for intuitive nested access:

**Why Dot Notation?**
- More readable than nested dictionary access
- Easier to type and remember
- Consistent with common configuration patterns

**Examples:**

Dot Notation (Preferred):
```python
value = config.get('decision_framework.consensus_threshold', default=0.7)
```

Equivalent Dictionary Access (Verbose):
```python
value = config.config.get('decision_framework', {}).get('consensus_threshold', 0.7)
```

**How It Works:**
```python
Key path: 'llm.model'
Split on '.': ['llm', 'model']
Traverse: config['llm']['model']
Return: 'claude-3-5-sonnet-20241022'
```

**Setting Values:**
```python
config.set('agents.max_agents', 20)
# Creates path if needed: config['agents']['max_agents'] = 20
```


DEEP MERGE BEHAVIOR
═══════════════════
When loading from file, configuration is deep-merged with defaults:

**Default Config:**
```python
{
  'agents': {
    'max_agents': 10,
    'default_confidence': 0.7
  }
}
```

**File Config:**
```python
{
  'agents': {
    'max_agents': 15
  }
}
```

**Result (Deep Merge):**
```python
{
  'agents': {
    'max_agents': 15,           # From file (override)
    'default_confidence': 0.7   # From defaults (preserved)
  }
}
```

**Benefits:**
- Partial overrides possible
- Don't need to specify all values in config file
- Sensible defaults preserved
- Only override what you need


ERROR HANDLING
══════════════

The Config class handles errors gracefully:

1. **Missing Config File**
   - Behavior: Falls back to defaults silently
   - Example:
     ```python
     config = Config('missing.json')  # No error
     # Uses defaults, logs warning
     ```

2. **Invalid JSON**
   - Behavior: Falls back to defaults, logs error
   - Example:
     ```python
     config = Config('malformed.json')
     # Uses defaults, logs parse error
     ```

3. **Missing Keys**
   - Behavior: Returns default value from get()
   - Example:
     ```python
     value = config.get('nonexistent.key', default=42)
     # Returns 42, no error
     ```

4. **Type Mismatches**
   - Behavior: Returns value as-is, caller validates
   - Example:
     ```python
     # Config has string, caller expects int
     max_agents = config.get('agents.max_agents', default=10)
     # Might return string if config has wrong type
     # Caller should validate
     ```


DESIGN DECISIONS
════════════════

1. **Embedded Defaults**
   - Rationale: System can run without any config file
   - Benefit: Quick start, self-documenting, demos work out-of-box
   - Trade-off: Defaults may become outdated

2. **Dot Notation**
   - Rationale: More readable than nested dict access
   - Benefit: Easier to use, less error-prone
   - Implementation: Split on '.', traverse dict

3. **Deep Merge**
   - Rationale: Allow partial overrides in config files
   - Benefit: Only specify what you want to change
   - Complexity: Requires recursive merge logic

4. **Graceful Fallback**
   - Rationale: Avoid crashes on missing/invalid config
   - Benefit: System always runs, easier debugging
   - Trade-off: May hide configuration errors

5. **Specialized Getters**
   - Rationale: Common access patterns deserve dedicated methods
   - Benefit: get_agent_config() clearer than get('agents')
   - Maintenance: Need to add getter for each new section

6. **No Write-to-File**
   - Rationale: Runtime changes are temporary
   - Benefit: Config file remains source of truth
   - Future: Could add save() method if needed


INTEGRATION POINTS
══════════════════

The Config class is used throughout the system:

1. **System Initialization**
   ```python
   config = Config('config.json')
   setup_logging(config.get('system.log_level'))
   ```

2. **Agent Factory**
   ```python
   agent_config = config.get_agent_config()
   max_agents = agent_config['max_agents']
   ```

3. **LLM Clients**
   ```python
   llm_config = config.get_llm_config()
   client = ClaudeClient(
       model=llm_config['model'],
       max_tokens=llm_config['max_tokens']
   )
   ```

4. **Decision Framework**
   ```python
   threshold = config.get('decision_framework.consensus_threshold')
   max_iter = config.get('decision_framework.max_iterations')
   ```

5. **Evaluation**
   ```python
   output_dir = config.get('evaluation.output_dir')
   save_viz = config.get('evaluation.save_visualizations')
   ```


BEST PRACTICES
══════════════

1. **Always Provide Defaults**
   ```python
   # Good
   value = config.get('key.path', default=42)

   # Bad
   value = config.get('key.path')  # Could return None
   ```

2. **Use Specialized Getters**
   ```python
   # Good
   agent_config = config.get_agent_config()

   # Acceptable
   agent_config = config.get('agents', default={})
   ```

3. **Group Related Settings**
   ```python
   # Good - grouped in config file
   {
     "llm": {
       "model": "...",
       "temperature": 0.7,
       "max_tokens": 1024
     }
   }

   # Bad - scattered
   {
     "llm_model": "...",
     "llm_temperature": 0.7,
     "llm_max_tokens": 1024
   }
   ```

4. **Document Configuration Changes**
   - Update default config when adding new parameters
   - Document in VERSION HISTORY
   - Provide migration guide for breaking changes

5. **Validate Configuration**
   ```python
   config = Config('config.json')
   max_agents = config.get('agents.max_agents', default=10)

   # Validate type and range
   if not isinstance(max_agents, int) or max_agents < 1:
       logger.warning(f"Invalid max_agents: {max_agents}, using default")
       max_agents = 10
   ```


RELATED MODULES
═══════════════

Dependencies:
- os: Operating system interface for paths
- json: Configuration file parsing
- pathlib: Cross-platform path handling
- typing: Type hints for API clarity

Used By:
- All modules in Crisis MAS
- System initialization and setup
- Agent factory and managers
- LLM integration layer
- Evaluation and metrics
- Scenario loaders


TROUBLESHOOTING
═══════════════

Common Issues:
──────────────

1. **"Config file not found"**
   - Cause: Incorrect path or file doesn't exist
   - Solution: Config falls back to defaults, check path
   - Debug: Check config.config_file attribute

2. **"Wrong value type"**
   - Cause: Config file has string where int expected
   - Solution: Validate types after get(), use defaults
   - Prevention: Use schema validation in config file

3. **"Missing configuration key"**
   - Cause: Typo in key path or key doesn't exist
   - Solution: Always use default parameter in get()
   - Debug: Print config.config to see all keys

4. **"Directories not created"**
   - Cause: Forgot to call ensure_directories()
   - Solution: Call config.ensure_directories() after init
   - Prevention: Add to system initialization sequence

5. **"Runtime changes not persisted"**
   - Cause: set() only modifies in-memory config
   - Solution: Manual edit of config file, or implement save()
   - Understanding: This is by design, config file is source of truth


VERSION HISTORY
═══════════════

Version 1.0.0 (Initial Release)
- Config class with dot notation access
- Default configuration embedded in code
- JSON file loading with deep merge
- Specialized getters: get_agent_config(), get_llm_config()
- Directory management: ensure_directories()
- Convenience function: load_config()

Configuration Sections Added:
- system: Global settings
- agents: Agent management
- scenarios: Scenario handling
- decision_framework: MCDA and consensus
- llm: LLM integration
- evaluation: Output and metrics
- paths: Directory structure

Future Enhancements:
- Environment variable support (e.g., CRISIS_MAS_MAX_AGENTS)
- Configuration validation against JSON schema
- save() method to persist runtime changes
- Configuration migration utilities for version upgrades
- Configuration templates for common scenarios
- Hot reload for configuration changes
- Configuration diff and merge tools
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration manager for the Crisis MAS system.
    Handles loading and accessing configuration parameters.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_default_config()

        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)

    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration.

        Returns:
            Dictionary with default configuration
        """
        return {
            'system': {
                'name': 'Crisis MAS',
                'version': '1.0.0',
                'log_level': 'INFO'
            },
            'agents': {
                'profiles_file': 'agents/agent_profiles.json',
                'max_agents': 10,
                'default_confidence': 0.7
            },
            'scenarios': {
                'scenarios_dir': 'scenarios',
                'criteria_file': 'scenarios/criteria_weights.json'
            },
            'decision_framework': {
                'consensus_threshold': 0.7,
                'max_iterations': 5,
                'convergence_rate': 0.3,
                'mcda_method': 'weighted_sum'  # Options: weighted_sum, topsis, saw
            },
            'llm': {
                'model': 'claude-3-5-sonnet-20241022',
                'max_tokens': 1024,
                'temperature': 0.7,
                'enable_llm': True,  # Set to False to run without LLM
                'retry_attempts': 3,
                'timeout': 30
            },
            'evaluation': {
                'output_dir': 'results',
                'save_visualizations': True,
                'save_metrics': True,
                'generate_dashboard': True
            },
            'paths': {
                'base_dir': os.getcwd(),
                'data_dir': 'data',
                'output_dir': 'output',
                'logs_dir': 'logs'
            }
        }

    def _load_from_file(self, config_file: str):
        """
        Load configuration from JSON file and merge with defaults.

        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)

            # Deep merge with default config
            self._deep_merge(self.config, file_config)

        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration")

    def _deep_merge(self, base: Dict, update: Dict):
        """
        Deep merge update dict into base dict.

        Args:
            base: Base dictionary (modified in place)
            update: Update dictionary
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Path to config value (e.g., 'agents.max_agents')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Path to config value (e.g., 'agents.max_agents')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

    def save(self, output_file: str):
        """
        Save current configuration to file.

        Args:
            output_file: Path to output file
        """
        with open(output_file, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Configuration saved to: {output_file}")

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent-specific configuration."""
        return self.config.get('agents', {})

    def get_scenario_config(self) -> Dict[str, Any]:
        """Get scenario-specific configuration."""
        return self.config.get('scenarios', {})

    def get_decision_framework_config(self) -> Dict[str, Any]:
        """Get decision framework configuration."""
        return self.config.get('decision_framework', {})

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.config.get('llm', {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})

    def get_path(self, path_key: str) -> Path:
        """
        Get path from configuration.

        Args:
            path_key: Key for the path

        Returns:
            Path object
        """
        path_str = self.get(f'paths.{path_key}', '.')
        return Path(path_str)

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.get_path('data_dir'),
            self.get_path('output_dir'),
            self.get_path('logs_dir'),
            Path(self.get('evaluation.output_dir', 'results'))
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Config(file={self.config_file}, sections={list(self.config.keys())})"

    def print_config(self):
        """Print current configuration in a readable format."""
        print("\n" + "="*60)
        print("Crisis MAS Configuration")
        print("="*60)
        self._print_dict(self.config)
        print("="*60 + "\n")

    def _print_dict(self, d: Dict, indent: int = 0):
        """
        Recursively print dictionary with indentation.

        Args:
            d: Dictionary to print
            indent: Current indentation level
        """
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


def load_config(config_file: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration.

    Args:
        config_file: Optional path to configuration file

    Returns:
        Config object
    """
    return Config(config_file)
