"""
Configuration Management
Centralized configuration for the MAS system
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
