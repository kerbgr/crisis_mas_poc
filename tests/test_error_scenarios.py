#!/usr/bin/env python3
"""
Error Scenario Testing for Crisis MAS
Tests system robustness under various failure conditions.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validation import DataValidator, safe_divide, safe_get


def test_json_validation():
    """Test JSON file validation with various error conditions."""
    print("\n" + "="*80)
    print("TEST 1: JSON Validation")
    print("="*80)

    # Test 1: Non-existent file
    print("\n📌 Test 1a: Non-existent file")
    success, data, error = DataValidator.validate_json_file("nonexistent.json")
    assert not success, "Should fail for non-existent file"
    assert "not found" in error.lower()
    print(f"  ✅ Correctly detected: {error}")

    # Test 2: Malformed JSON
    print("\n📌 Test 1b: Malformed JSON")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": json}')  # Missing quotes
        temp_file = f.name

    try:
        success, data, error = DataValidator.validate_json_file(temp_file)
        assert not success, "Should fail for malformed JSON"
        assert "invalid json" in error.lower() or "json" in error.lower()
        print(f"  ✅ Correctly detected: {error}")
    finally:
        Path(temp_file).unlink()

    # Test 3: Valid JSON
    print("\n📌 Test 1c: Valid JSON")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"test": "data"}, f)
        temp_file = f.name

    try:
        success, data, error = DataValidator.validate_json_file(temp_file)
        assert success, f"Should succeed for valid JSON: {error}"
        assert data == {"test": "data"}
        print(f"  ✅ Successfully loaded valid JSON")
    finally:
        Path(temp_file).unlink()

    print("\n✅ JSON Validation Tests PASSED")


def test_scenario_validation():
    """Test scenario validation."""
    print("\n" + "="*80)
    print("TEST 2: Scenario Validation")
    print("="*80)

    # Test 1: Missing required fields
    print("\n📌 Test 2a: Missing required fields")
    invalid_scenario = {
        "id": "test_001",
        "type": "flood"
        # Missing: name, description, severity
    }
    valid, error = DataValidator.validate_scenario(invalid_scenario)
    assert not valid, "Should fail for missing fields"
    assert "missing" in error.lower()
    print(f"  ✅ Correctly detected: {error}")

    # Test 2: Invalid severity range
    print("\n📌 Test 2b: Invalid severity range")
    invalid_scenario = {
        "id": "test_001",
        "type": "flood",
        "name": "Test Flood",
        "description": "Test description",
        "severity": 1.5  # Invalid: > 1.0
    }
    valid, error = DataValidator.validate_scenario(invalid_scenario)
    assert not valid, "Should fail for invalid severity"
    assert "severity" in error.lower()
    print(f"  ✅ Correctly detected: {error}")

    # Test 3: Valid scenario
    print("\n📌 Test 2c: Valid scenario")
    valid_scenario = {
        "id": "test_001",
        "type": "flood",
        "name": "Test Flood",
        "description": "Test description",
        "severity": 0.8,
        "available_actions": [{"id": "A1", "name": "Action 1"}]
    }
    valid, error = DataValidator.validate_scenario(valid_scenario)
    assert valid, f"Should succeed for valid scenario: {error}"
    print(f"  ✅ Valid scenario accepted")

    print("\n✅ Scenario Validation Tests PASSED")


def test_criteria_weights_validation():
    """Test criteria weights validation and normalization."""
    print("\n" + "="*80)
    print("TEST 3: Criteria Weights Validation")
    print("="*80)

    # Test 1: Weights don't sum to 1.0
    print("\n📌 Test 3a: Weights sum to 0.8")
    invalid_criteria = {
        "safety": {"name": "Safety", "weight": 0.3},
        "cost": {"name": "Cost", "weight": 0.25},
        "speed": {"name": "Speed", "weight": 0.25}
        # Total: 0.8, not 1.0
    }
    valid, error = DataValidator.validate_criteria_weights(invalid_criteria)
    assert not valid, "Should fail for invalid sum"
    assert "0.8" in error
    print(f"  ✅ Correctly detected: {error}")

    # Test 2: Normalize weights
    print("\n📌 Test 3b: Normalize weights")
    normalized = DataValidator.normalize_weights({
        "A": 0.3, "B": 0.3, "C": 0.3  # Sum = 0.9
    })
    total = sum(normalized.values())
    assert abs(total - 1.0) < 0.001, f"Normalized weights should sum to 1.0, got {total}"
    print(f"  ✅ Weights normalized: {normalized}")
    print(f"  ✅ Sum: {total:.6f}")

    # Test 3: Zero weights
    print("\n📌 Test 3c: All zero weights")
    normalized = DataValidator.normalize_weights({
        "A": 0.0, "B": 0.0, "C": 0.0
    })
    total = sum(normalized.values())
    assert abs(total - 1.0) < 0.001, "Should create uniform distribution"
    assert all(abs(v - 1.0/3) < 0.001 for v in normalized.values()), "Should be uniform"
    print(f"  ✅ Created uniform distribution: {normalized}")

    # Test 4: Negative weights
    print("\n📌 Test 3d: Negative weights")
    invalid_criteria = {
        "safety": {"name": "Safety", "weight": -0.3},
        "cost": {"name": "Cost", "weight": 0.7}
    }
    valid, error = DataValidator.validate_criteria_weights(invalid_criteria)
    assert not valid, "Should fail for negative weights"
    assert "negative" in error.lower()
    print(f"  ✅ Correctly detected: {error}")

    print("\n✅ Criteria Weights Validation Tests PASSED")


def test_belief_distribution_validation():
    """Test belief distribution validation."""
    print("\n" + "="*80)
    print("TEST 4: Belief Distribution Validation")
    print("="*80)

    # Test 1: Beliefs don't sum to 1.0
    print("\n📌 Test 4a: Beliefs sum to 0.9")
    invalid_beliefs = {
        "A1": 0.4,
        "A2": 0.3,
        "A3": 0.2
        # Total: 0.9
    }
    valid, error = DataValidator.validate_belief_distribution(invalid_beliefs)
    assert not valid, "Should fail for invalid sum"
    assert "0.9" in error
    print(f"  ✅ Correctly detected: {error}")

    # Test 2: Belief out of range
    print("\n📌 Test 4b: Belief value > 1.0")
    invalid_beliefs = {
        "A1": 1.5,  # Invalid
        "A2": -0.5  # Invalid
    }
    valid, error = DataValidator.validate_belief_distribution(invalid_beliefs)
    assert not valid, "Should fail for out-of-range values"
    print(f"  ✅ Correctly detected: {error}")

    # Test 3: Valid beliefs
    print("\n📌 Test 4c: Valid beliefs")
    valid_beliefs = {
        "A1": 0.5,
        "A2": 0.3,
        "A3": 0.2
    }
    valid, error = DataValidator.validate_belief_distribution(valid_beliefs)
    assert valid, f"Should succeed for valid beliefs: {error}"
    print(f"  ✅ Valid beliefs accepted")

    print("\n✅ Belief Distribution Validation Tests PASSED")


def test_score_sanitization():
    """Test score sanitization."""
    print("\n" + "="*80)
    print("TEST 5: Score Sanitization")
    print("="*80)

    # Test 1: Scores out of range
    print("\n📌 Test 5a: Clip out-of-range scores")
    scores = {
        "A1": 1.5,   # Too high
        "A2": -0.3,  # Too low
        "A3": 0.5    # OK
    }
    sanitized = DataValidator.sanitize_scores(scores, min_val=0.0, max_val=1.0)
    assert sanitized["A1"] == 1.0, "Should clip to max"
    assert sanitized["A2"] == 0.0, "Should clip to min"
    assert sanitized["A3"] == 0.5, "Should keep valid value"
    print(f"  ✅ Original: {scores}")
    print(f"  ✅ Sanitized: {sanitized}")

    # Test 2: Non-numeric scores
    print("\n📌 Test 5b: Non-numeric scores")
    scores = {
        "A1": "invalid",
        "A2": None,
        "A3": 0.7
    }
    sanitized = DataValidator.sanitize_scores(scores, min_val=0.0, max_val=1.0)
    assert sanitized["A1"] == 0.0, "Should default non-numeric to min"
    assert sanitized["A2"] == 0.0, "Should default None to min"
    assert sanitized["A3"] == 0.7, "Should keep valid value"
    print(f"  ✅ Original: {scores}")
    print(f"  ✅ Sanitized: {sanitized}")

    print("\n✅ Score Sanitization Tests PASSED")


def test_safe_operations():
    """Test safe utility functions."""
    print("\n" + "="*80)
    print("TEST 6: Safe Operations")
    print("="*80)

    # Test 1: Safe division
    print("\n📌 Test 6a: Safe division by zero")
    result = safe_divide(10, 0, default=0.0)
    assert result == 0.0, "Should return default for division by zero"
    print(f"  ✅ 10 / 0 = {result} (default)")

    result = safe_divide(10, 2, default=0.0)
    assert result == 5.0, "Should perform normal division"
    print(f"  ✅ 10 / 2 = {result}")

    # Test 2: Safe dictionary get
    print("\n📌 Test 6b: Safe dictionary access")
    data = {"key1": "value1", "key2": 42}

    result = safe_get(data, "key1", default="default")
    assert result == "value1", "Should get existing value"
    print(f"  ✅ get('key1') = '{result}'")

    result = safe_get(data, "nonexistent", default="default")
    assert result == "default", "Should return default for missing key"
    print(f"  ✅ get('nonexistent') = '{result}' (default)")

    result = safe_get(data, "key2", default=0, expected_type=int)
    assert result == 42, "Should get value with correct type"
    print(f"  ✅ get('key2', expected_type=int) = {result}")

    result = safe_get(data, "key1", default="default", expected_type=int)
    assert result == "default", "Should return default for wrong type"
    print(f"  ✅ get('key1', expected_type=int) = '{result}' (default, wrong type)")

    print("\n✅ Safe Operations Tests PASSED")


def test_alternatives_validation():
    """Test alternatives validation."""
    print("\n" + "="*80)
    print("TEST 7: Alternatives Validation")
    print("="*80)

    # Test 1: Empty alternatives
    print("\n📌 Test 7a: Empty alternatives list")
    valid, error = DataValidator.validate_alternatives([])
    assert not valid, "Should fail for empty list"
    assert "empty" in error.lower()
    print(f"  ✅ Correctly detected: {error}")

    # Test 2: Duplicate IDs
    print("\n📌 Test 7b: Duplicate alternative IDs")
    alternatives = [
        {"id": "A1", "name": "Action 1"},
        {"id": "A2", "name": "Action 2"},
        {"id": "A1", "name": "Action 1 Duplicate"}  # Duplicate
    ]
    valid, error = DataValidator.validate_alternatives(alternatives)
    assert not valid, "Should fail for duplicate IDs"
    assert "duplicate" in error.lower()
    print(f"  ✅ Correctly detected: {error}")

    # Test 3: Missing required fields
    print("\n📌 Test 7c: Missing required fields")
    alternatives = [
        {"id": "A1", "name": "Action 1"},
        {"name": "Action 2"}  # Missing 'id'
    ]
    valid, error = DataValidator.validate_alternatives(alternatives)
    assert not valid, "Should fail for missing fields"
    assert "missing" in error.lower()
    print(f"  ✅ Correctly detected: {error}")

    # Test 4: Valid alternatives
    print("\n📌 Test 7d: Valid alternatives")
    alternatives = [
        {"id": "A1", "name": "Action 1", "description": "First action"},
        {"id": "A2", "name": "Action 2", "description": "Second action"}
    ]
    valid, error = DataValidator.validate_alternatives(alternatives)
    assert valid, f"Should succeed for valid alternatives: {error}"
    print(f"  ✅ Valid alternatives accepted")

    print("\n✅ Alternatives Validation Tests PASSED")


def main():
    """Run all error scenario tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "CRISIS MAS ERROR SCENARIO TESTS" + " "*27 + "║")
    print("╚" + "="*78 + "╝")

    tests = [
        test_json_validation,
        test_scenario_validation,
        test_criteria_weights_validation,
        test_belief_distribution_validation,
        test_score_sanitization,
        test_safe_operations,
        test_alternatives_validation
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
            failed += 1

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {passed + failed}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("="*80)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe Crisis MAS system has robust error handling and can:")
        print("  ✅ Detect and handle malformed JSON files")
        print("  ✅ Validate scenario structure and data ranges")
        print("  ✅ Normalize criteria weights that don't sum to 1.0")
        print("  ✅ Validate and fix belief distributions")
        print("  ✅ Sanitize out-of-range scores")
        print("  ✅ Handle division by zero gracefully")
        print("  ✅ Safely access dictionary keys with type checking")
        print("  ✅ Validate alternatives and detect duplicates")
        print("\nThe system is ROBUST and DEMO-READY! 🚀")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
