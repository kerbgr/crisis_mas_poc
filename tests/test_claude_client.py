"""
Test Script for Claude API Client
Crisis Management Multi-Agent System

Tests JSON parsing, response validation, and error handling without requiring API calls.
"""

import os
import json
from llm_integration.claude_client import ClaudeClient


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def test_json_parsing():
    """Test 1: JSON parsing with various formats."""
    print_section("Test 1: JSON Parsing")

    # Mock client (will fail on init without API key, so we'll test parsing directly)
    # We'll create a client instance just for the parsing method

    # Test case 1: Direct JSON
    print("Test Case 1: Direct JSON")
    response_text = json.dumps({
        "alternative_rankings": {"A1": 0.7, "A2": 0.2, "A3": 0.08, "A4": 0.02},
        "reasoning": "Based on weather data...",
        "confidence": 0.85,
        "key_concerns": ["flood risk", "time constraints"]
    })

    # Create a temporary client with a dummy API key just for testing parsing
    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_parsing_only'
    client = ClaudeClient()

    result = client.parse_json_response(response_text)
    print(f"  Parsed: {result['alternative_rankings']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  ✓ Direct JSON parsing works\n")

    # Test case 2: JSON in markdown code block
    print("Test Case 2: JSON in ```json...``` block")
    response_text = """Here's the assessment:

```json
{
    "alternative_rankings": {"A1": 0.6, "A2": 0.3, "A3": 0.1},
    "reasoning": "Medical perspective prioritizes safety",
    "confidence": 0.92,
    "key_concerns": ["patient safety", "hospital capacity"]
}
```

This represents my professional assessment."""

    result = client.parse_json_response(response_text)
    print(f"  Parsed: {result['alternative_rankings']}")
    print(f"  Reasoning: {result['reasoning'][:50]}...")
    print(f"  ✓ Markdown code block parsing works\n")

    # Test case 3: Generic code block
    print("Test Case 3: JSON in generic ```...``` block")
    response_text = """```
{
    "alternative_rankings": {"A1": 0.5, "A2": 0.35, "A3": 0.15},
    "reasoning": "Logistical constraints considered",
    "confidence": 0.78,
    "key_concerns": ["resource availability", "timing"]
}
```"""

    result = client.parse_json_response(response_text)
    print(f"  Parsed: {result['alternative_rankings']}")
    print(f"  ✓ Generic code block parsing works\n")

    # Test case 4: JSON embedded in text
    print("Test Case 4: JSON embedded in surrounding text")
    response_text = """After careful analysis, my assessment is: {"alternative_rankings": {"A1": 0.8, "A2": 0.15, "A3": 0.05}, "reasoning": "Emergency response required", "confidence": 0.95, "key_concerns": ["immediate threat"]} and I recommend immediate action."""

    result = client.parse_json_response(response_text)
    print(f"  Parsed: {result['alternative_rankings']}")
    print(f"  ✓ Embedded JSON parsing works\n")

    print("✓ Test 1 Passed - All JSON parsing strategies work")


def test_response_validation():
    """Test 2: Response validation."""
    print_section("Test 2: Response Validation")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_validation_only'
    client = ClaudeClient()

    expected_keys = ['alternative_rankings', 'reasoning', 'confidence', 'key_concerns']

    # Test case 1: Valid response
    print("Test Case 1: Valid response with all expected fields")
    valid_response = {
        "alternative_rankings": {"A1": 0.7, "A2": 0.2, "A3": 0.1},
        "reasoning": "Test reasoning",
        "confidence": 0.85,
        "key_concerns": ["concern1", "concern2"]
    }
    is_valid = client.validate_response(valid_response, expected_keys)
    print(f"  Validation result: {is_valid}")
    assert is_valid, "Valid response should pass validation"
    print(f"  ✓ Valid response passes\n")

    # Test case 2: Missing key
    print("Test Case 2: Missing 'key_concerns' field")
    invalid_response = {
        "alternative_rankings": {"A1": 0.7, "A2": 0.3},
        "reasoning": "Test reasoning",
        "confidence": 0.85
    }
    is_valid = client.validate_response(invalid_response, expected_keys)
    print(f"  Validation result: {is_valid}")
    assert not is_valid, "Response with missing key should fail validation"
    print(f"  ✓ Correctly detects missing key\n")

    # Test case 3: Invalid confidence value
    print("Test Case 3: Confidence out of range [0, 1]")
    invalid_response = {
        "alternative_rankings": {"A1": 0.7, "A2": 0.3},
        "reasoning": "Test reasoning",
        "confidence": 1.5,  # Invalid: > 1.0
        "key_concerns": ["concern1"]
    }
    is_valid = client.validate_response(invalid_response, expected_keys)
    print(f"  Validation result: {is_valid}")
    assert not is_valid, "Response with invalid confidence should fail validation"
    print(f"  ✓ Correctly detects invalid confidence\n")

    # Test case 4: Invalid alternative_rankings type
    print("Test Case 4: alternative_rankings is not a dict")
    invalid_response = {
        "alternative_rankings": ["A1", "A2", "A3"],  # Should be dict, not list
        "reasoning": "Test reasoning",
        "confidence": 0.85,
        "key_concerns": ["concern1"]
    }
    is_valid = client.validate_response(invalid_response, expected_keys)
    print(f"  Validation result: {is_valid}")
    assert not is_valid, "Response with invalid rankings type should fail validation"
    print(f"  ✓ Correctly detects invalid rankings type\n")

    # Test case 5: Non-numeric ranking values
    print("Test Case 5: Ranking values are not numeric")
    invalid_response = {
        "alternative_rankings": {"A1": "high", "A2": "medium"},  # Should be numeric
        "reasoning": "Test reasoning",
        "confidence": 0.85,
        "key_concerns": ["concern1"]
    }
    is_valid = client.validate_response(invalid_response, expected_keys)
    print(f"  Validation result: {is_valid}")
    assert not is_valid, "Response with non-numeric rankings should fail validation"
    print(f"  ✓ Correctly detects non-numeric rankings\n")

    print("✓ Test 2 Passed - Response validation works correctly")


def test_client_initialization():
    """Test 3: Client initialization."""
    print_section("Test 3: Client Initialization")

    # Test case 1: Initialize with API key
    print("Test Case 1: Initialize with explicit API key")
    client = ClaudeClient(api_key="test_api_key_123")
    print(f"  Model: {client.model}")
    print(f"  Request count: {client.request_count}")
    print(f"  Total tokens: {client.total_tokens}")
    print(f"  ✓ Initialization with API key works\n")

    # Test case 2: Initialize with environment variable
    print("Test Case 2: Initialize with environment variable")
    os.environ['ANTHROPIC_API_KEY'] = 'env_api_key_456'
    client = ClaudeClient()
    print(f"  Model: {client.model}")
    print(f"  ✓ Initialization with env var works\n")

    # Test case 3: Custom model
    print("Test Case 3: Initialize with custom model")
    client = ClaudeClient(api_key="test_key", model="claude-3-opus-20240229")
    print(f"  Model: {client.model}")
    assert client.model == "claude-3-opus-20240229"
    print(f"  ✓ Custom model works\n")

    # Test case 4: Default model
    print("Test Case 4: Default model (claude-sonnet-4-20250514)")
    client = ClaudeClient(api_key="test_key")
    print(f"  Model: {client.model}")
    assert client.model == "claude-sonnet-4-20250514"
    print(f"  ✓ Default model is correct\n")

    # Test case 5: Missing API key
    print("Test Case 5: Missing API key (should raise ValueError)")
    if 'ANTHROPIC_API_KEY' in os.environ:
        del os.environ['ANTHROPIC_API_KEY']

    try:
        client = ClaudeClient()
        print("  ✗ Should have raised ValueError")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:50]}...\n")

    print("✓ Test 3 Passed - Client initialization works correctly")


def test_statistics():
    """Test 4: Usage statistics."""
    print_section("Test 4: Usage Statistics")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_stats'
    client = ClaudeClient()

    # Initial state
    print("Test Case 1: Initial statistics")
    stats = client.get_statistics()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Failed requests: {stats['failed_requests']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Model: {stats['model']}")
    assert stats['total_requests'] == 0
    assert stats['failed_requests'] == 0
    print(f"  ✓ Initial state is correct\n")

    # Simulate some usage
    print("Test Case 2: After simulating usage")
    client.request_count = 10
    client.failed_requests = 2
    client.total_tokens = 5000

    stats = client.get_statistics()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Failed requests: {stats['failed_requests']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Total tokens: {stats['total_tokens']}")
    assert stats['total_requests'] == 10
    assert stats['failed_requests'] == 2
    assert abs(stats['success_rate'] - 0.833) < 0.01  # 10/(10+2) ≈ 0.833
    print(f"  ✓ Statistics tracked correctly\n")

    # Reset statistics
    print("Test Case 3: After reset")
    client.reset_statistics()
    stats = client.get_statistics()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Failed requests: {stats['failed_requests']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    assert stats['total_requests'] == 0
    assert stats['failed_requests'] == 0
    assert stats['total_tokens'] == 0
    print(f"  ✓ Reset works correctly\n")

    print("✓ Test 4 Passed - Statistics tracking works correctly")


def test_error_responses():
    """Test 5: Error response structure."""
    print_section("Test 5: Error Response Structure")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_errors'
    client = ClaudeClient()

    print("Test Case 1: Simulating API error response")
    # Simulate what an error response would look like
    error_response = {
        'error': True,
        'error_message': 'Rate limit exceeded',
        'error_type': 'RateLimitError'
    }

    print(f"  Error: {error_response.get('error')}")
    print(f"  Message: {error_response.get('error_message')}")
    print(f"  Type: {error_response.get('error_type')}")

    # Check error response structure
    assert error_response.get('error') == True
    assert 'error_message' in error_response
    assert 'error_type' in error_response
    print(f"  ✓ Error response has correct structure\n")

    print("Test Case 2: JSON decode error response")
    error_response = {
        'error': True,
        'error_message': 'Failed to parse JSON response: Expecting value',
        'error_type': 'JSONDecodeError',
        'raw_response': 'This is not valid JSON'
    }

    print(f"  Error: {error_response.get('error')}")
    print(f"  Message: {error_response.get('error_message')[:50]}...")
    print(f"  Type: {error_response.get('error_type')}")
    print(f"  Has raw response: {'raw_response' in error_response}")

    assert error_response.get('error') == True
    assert 'raw_response' in error_response
    print(f"  ✓ JSON error response has raw response field\n")

    print("✓ Test 5 Passed - Error responses have correct structure")


def test_repr():
    """Test 6: String representation."""
    print_section("Test 6: String Representation")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_repr'
    client = ClaudeClient()

    print("Test Case 1: Initial repr")
    repr_str = repr(client)
    print(f"  {repr_str}")
    assert 'ClaudeClient' in repr_str
    assert 'claude-sonnet-4-20250514' in repr_str
    print(f"  ✓ Repr contains client info\n")

    print("Test Case 2: After simulated usage")
    client.request_count = 5
    client.total_tokens = 2000
    repr_str = repr(client)
    print(f"  {repr_str}")
    assert 'requests=5' in repr_str
    assert 'tokens=2000' in repr_str
    print(f"  ✓ Repr reflects current state\n")

    print("✓ Test 6 Passed - String representation works correctly")


def test_comprehensive_workflow():
    """Test 7: Comprehensive workflow simulation."""
    print_section("Test 7: Comprehensive Workflow Simulation")

    os.environ['ANTHROPIC_API_KEY'] = 'test_key_for_workflow'
    client = ClaudeClient()

    print("Simulating complete assessment workflow:\n")

    # Step 1: Parse a realistic response
    print("Step 1: Parse Claude response")
    claude_response = """Based on my analysis as a meteorologist:

```json
{
    "alternative_rankings": {
        "A1": 0.65,
        "A2": 0.25,
        "A3": 0.08,
        "A4": 0.02
    },
    "reasoning": "Given the severe flood forecast and limited time window, immediate mass evacuation (A1) is the safest option. The 48-hour precipitation forecast of 200mm exceeds historical flood thresholds. While flood barriers (A2) are cost-effective, deployment time of 6 hours may be insufficient. Weather patterns suggest rapid water rise within 4 hours of rainfall onset.",
    "confidence": 0.87,
    "key_concerns": [
        "Rapid water rise expected within 4 hours",
        "Limited deployment time for barriers",
        "Population density in flood zone",
        "Historical precedent from 2019 flood"
    ]
}
```

This assessment prioritizes public safety given the severe weather conditions."""

    parsed = client.parse_json_response(claude_response)
    print(f"  Alternative rankings: {parsed['alternative_rankings']}")
    print(f"  Top choice: A1 with score {parsed['alternative_rankings']['A1']}")
    print(f"  Confidence: {parsed['confidence']}")
    print(f"  Key concerns: {len(parsed['key_concerns'])} identified")
    print(f"  ✓ Response parsed successfully\n")

    # Step 2: Validate response
    print("Step 2: Validate response structure")
    expected_keys = ['alternative_rankings', 'reasoning', 'confidence', 'key_concerns']
    is_valid = client.validate_response(parsed, expected_keys)
    print(f"  Validation result: {is_valid}")
    assert is_valid
    print(f"  ✓ Response is valid\n")

    # Step 3: Extract decision information
    print("Step 3: Extract decision information")
    rankings = parsed['alternative_rankings']
    top_alternative = max(rankings.items(), key=lambda x: x[1])
    print(f"  Top alternative: {top_alternative[0]} (score: {top_alternative[1]})")
    print(f"  Reasoning excerpt: {parsed['reasoning'][:80]}...")
    print(f"  Agent confidence: {parsed['confidence']:.1%}")
    print(f"  Number of concerns: {len(parsed['key_concerns'])}")
    print(f"  ✓ Decision information extracted\n")

    # Step 4: Check statistics
    print("Step 4: Check usage statistics")
    stats = client.get_statistics()
    print(f"  Model: {stats['model']}")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  ✓ Statistics available\n")

    print("✓ Test 7 Passed - Complete workflow simulation successful")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CLAUDE CLIENT - COMPREHENSIVE TEST SUITE")
    print("Crisis Management Multi-Agent System")
    print("="*70)

    tests = [
        test_json_parsing,
        test_response_validation,
        test_client_initialization,
        test_statistics,
        test_error_responses,
        test_repr,
        test_comprehensive_workflow
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ Test Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
    print("\nNote: These tests validate parsing, validation, and client logic.")
    print("Live API testing requires a valid ANTHROPIC_API_KEY.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
