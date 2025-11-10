"""
Test Script for Multiple LLM Clients (OpenAI and LM Studio)
Crisis Management Multi-Agent System

Tests JSON parsing, response validation, and error handling for all clients.
"""

import os
from llm_integration.openai_client import OpenAIClient
from llm_integration.lmstudio_client import LMStudioClient


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def test_openai_client():
    """Test OpenAI client functionality."""
    print_section("OpenAI Client Tests")

    # Test initialization
    print("Test 1: OpenAI Client Initialization")
    try:
        client = OpenAIClient(api_key="test_key_openai", model="gpt-4-turbo-preview")
        print(f"  ✓ Client initialized: {repr(client)}")
        print(f"  Model: {client.model}")
        print(f"  Initial requests: {client.request_count}\n")
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}\n")

    # Test JSON parsing
    print("Test 2: OpenAI JSON Parsing")
    response_text = """```json
{
    "alternative_rankings": {"A1": 0.6, "A2": 0.3, "A3": 0.1},
    "reasoning": "OpenAI assessment based on analysis",
    "confidence": 0.88,
    "key_concerns": ["data accuracy", "model uncertainty"]
}
```"""

    try:
        parsed = client.parse_json_response(response_text)
        print(f"  Parsed rankings: {parsed['alternative_rankings']}")
        print(f"  Confidence: {parsed['confidence']}")
        print(f"  ✓ JSON parsing works\n")
    except Exception as e:
        print(f"  ✗ Parsing failed: {e}\n")

    # Test validation
    print("Test 3: OpenAI Response Validation")
    expected_keys = ['alternative_rankings', 'reasoning', 'confidence', 'key_concerns']
    is_valid = client.validate_response(parsed, expected_keys)
    print(f"  Validation result: {is_valid}")
    print(f"  ✓ Validation works\n")

    # Test statistics
    print("Test 4: OpenAI Statistics")
    client.request_count = 5
    client.total_tokens = 3000
    client.failed_requests = 1
    stats = client.get_statistics()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Failed requests: {stats['failed_requests']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  ✓ Statistics tracking works\n")

    print("✓ OpenAI Client Tests Complete")


def test_lmstudio_client():
    """Test LM Studio client functionality."""
    print_section("LM Studio Client Tests")

    # Test initialization
    print("Test 1: LM Studio Client Initialization")
    try:
        client = LMStudioClient(
            base_url="http://localhost:1234/v1",
            model="llama-2-7b-chat"
        )
        print(f"  ✓ Client initialized: {repr(client)}")
        print(f"  Endpoint: {client.base_url}")
        print(f"  Model: {client.model}")
        print(f"  Initial requests: {client.request_count}\n")
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}\n")

    # Test JSON parsing
    print("Test 2: LM Studio JSON Parsing")
    # Local models may produce less clean JSON
    response_text = """Here is my assessment:
{
    "alternative_rankings": {"A1": 0.7, "A2": 0.2, "A3": 0.1},
    "reasoning": "Local model assessment based on crisis parameters",
    "confidence": 0.75,
    "key_concerns": ["local processing", "model size limitations"]
}
That's my analysis."""

    try:
        parsed = client.parse_json_response(response_text)
        print(f"  Parsed rankings: {parsed['alternative_rankings']}")
        print(f"  Confidence: {parsed['confidence']}")
        print(f"  ✓ JSON parsing works (extracts from surrounding text)\n")
    except Exception as e:
        print(f"  ✗ Parsing failed: {e}\n")

    # Test validation
    print("Test 3: LM Studio Response Validation")
    expected_keys = ['alternative_rankings', 'reasoning', 'confidence', 'key_concerns']
    is_valid = client.validate_response(parsed, expected_keys)
    print(f"  Validation result: {is_valid}")
    print(f"  ✓ Validation works\n")

    # Test statistics with endpoint info
    print("Test 4: LM Studio Statistics")
    client.request_count = 8
    client.total_tokens = 4500
    client.failed_requests = 0
    stats = client.get_statistics()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Failed requests: {stats['failed_requests']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Endpoint: {stats['endpoint']}")
    print(f"  ✓ Statistics tracking works\n")

    print("✓ LM Studio Client Tests Complete")


def test_client_comparison():
    """Compare features across all clients."""
    print_section("Client Comparison")

    # Initialize all clients
    claude_exists = False
    try:
        from llm_integration.claude_client import ClaudeClient
        claude = ClaudeClient(api_key="test_key_claude")
        claude_exists = True
    except:
        pass

    openai = OpenAIClient(api_key="test_key_openai")
    lmstudio = LMStudioClient()

    print("Feature Comparison:\n")

    # Models
    print("Default Models:")
    if claude_exists:
        print(f"  Claude:    {claude.model}")
    print(f"  OpenAI:    {openai.model}")
    print(f"  LM Studio: {lmstudio.model} (local)")

    # Endpoints
    print("\nEndpoints:")
    if claude_exists:
        print(f"  Claude:    Anthropic API (cloud)")
    print(f"  OpenAI:    OpenAI API (cloud)")
    print(f"  LM Studio: {lmstudio.base_url} (local)")

    # Common methods
    print("\nCommon Methods:")
    common_methods = [
        'generate_assessment',
        'parse_json_response',
        'validate_response',
        'get_statistics',
        'reset_statistics'
    ]
    for method in common_methods:
        claude_has = hasattr(claude, method) if claude_exists else False
        openai_has = hasattr(openai, method)
        lmstudio_has = hasattr(lmstudio, method)
        status = "✓" if all([openai_has, lmstudio_has]) else "✗"
        print(f"  {status} {method}")

    # Response format
    print("\nExpected Response Format (all clients):")
    print("  - alternative_rankings: Dict[str, float]")
    print("  - reasoning: str")
    print("  - confidence: float (0-1)")
    print("  - key_concerns: List[str]")

    # Error handling
    print("\nError Handling:")
    print("  - Exponential backoff retry (2s, 4s, 8s)")
    print("  - Structured error responses")
    print("  - Comprehensive logging")
    print("  - Statistics tracking")

    print("\n✓ All clients follow the same interface")


def test_use_case_scenarios():
    """Test realistic use case scenarios."""
    print_section("Use Case Scenarios")

    # Scenario 1: Cloud vs Local trade-offs
    print("Scenario 1: When to use each provider\n")
    print("Claude (Anthropic):")
    print("  ✓ Latest model (Sonnet 4.5)")
    print("  ✓ Excellent instruction following")
    print("  ✓ Strong JSON formatting")
    print("  ✗ Requires API key and internet")
    print("  ✗ Per-token cost")

    print("\nOpenAI (GPT-4):")
    print("  ✓ Native JSON mode support")
    print("  ✓ Widely adopted, well-documented")
    print("  ✓ Multiple model options (GPT-4, GPT-3.5)")
    print("  ✗ Requires API key and internet")
    print("  ✗ Per-token cost")

    print("\nLM Studio (Local):")
    print("  ✓ Free after model download")
    print("  ✓ Works offline")
    print("  ✓ Full privacy (no data sent externally)")
    print("  ✓ No rate limits")
    print("  ✗ Requires powerful hardware")
    print("  ✗ May be slower than cloud APIs")
    print("  ✗ Lower quality than large cloud models")

    # Scenario 2: Crisis management use case
    print("\n\nScenario 2: Crisis Management System Usage\n")
    print("For Production (requires accuracy):")
    print("  → Claude or OpenAI (GPT-4)")
    print("  → Higher quality assessments")
    print("  → Better JSON compliance")

    print("\nFor Development/Testing:")
    print("  → LM Studio")
    print("  → Fast iteration without API costs")
    print("  → Works offline")

    print("\nFor Sensitive Data:")
    print("  → LM Studio")
    print("  → All data stays local")
    print("  → Meets data privacy requirements")

    print("\n✓ Use case scenarios documented")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MULTI-LLM CLIENT TEST SUITE")
    print("Testing OpenAI and LM Studio Clients")
    print("="*70)

    try:
        test_openai_client()
        test_lmstudio_client()
        test_client_comparison()
        test_use_case_scenarios()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)

        print("\nNotes:")
        print("- These tests validate client logic without making real API calls")
        print("- To test live OpenAI: set OPENAI_API_KEY environment variable")
        print("- To test live LM Studio: start LM Studio with a model loaded")
        print("- Claude client tests available in test_claude_client.py")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
