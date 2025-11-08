"""
Validation Utilities for Crisis Management Multi-Agent System
Provides comprehensive validation and error handling functions.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

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
        response: Dict[str, Any],
        expected_keys: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate LLM response structure.

        Args:
            response: LLM response dictionary
            expected_keys: Optional list of required keys

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not isinstance(response, dict):
                return False, f"Response must be a dictionary, got {type(response).__name__}"

            # Check for error in response
            if 'error' in response and response['error']:
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
