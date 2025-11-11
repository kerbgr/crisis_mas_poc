"""
OpenAI API Client - GPT-4/GPT-3.5 Integration for Expert Agent Reasoning

OBJECTIVE:
This module provides a robust, production-ready wrapper for the OpenAI Chat Completions API,
enabling expert agents in the crisis management multi-agent system to leverage GPT-4 and
GPT-3.5 models for advanced reasoning. It handles API communication, error recovery, JSON
parsing, response validation, and usage tracking with an interface identical to ClaudeClient.

WHY OPENAI:
OpenAI's GPT models (specifically GPT-4) serve as an **alternative cloud LLM provider**
for this crisis management system:

1. **Alternative to Claude**: Provides fallback when Claude is unavailable or rate limited
2. **Built-in JSON Mode**: Native JSON response format support (response_format parameter)
3. **Fast Iteration**: Well-documented API, wide ecosystem, familiar to developers
4. **Good Reasoning**: GPT-4 provides strong reasoning quality, comparable to Claude
5. **Model Variety**: Multiple models (GPT-4, GPT-4-turbo, GPT-4o, GPT-3.5) with cost/quality trade-offs

While Claude is the default provider, OpenAI provides a reliable alternative with similar
capabilities and cost structure.

WHY THIS CLIENT:
A custom client (rather than using the bare OpenAI SDK) provides:

1. **Error Resilience**: Exponential backoff retry for rate limits and transient errors
2. **Robust JSON Parsing**: Multi-strategy parsing (though JSON mode reduces need)
3. **Response Validation**: Ensures responses contain expected fields with correct types
4. **Usage Tracking**: Monitor requests, tokens, costs, and failure rates
5. **Consistent Interface**: Same API as ClaudeClient (drop-in replacement)
6. **Logging**: Comprehensive logging for debugging and monitoring
7. **Crisis-Optimized**: Designed for structured expert assessments

TYPICAL USAGE:

```python
from llm_integration import OpenAIClient, PromptTemplates

# 1. Initialize client
client = OpenAIClient()  # Reads OPENAI_API_KEY from environment
# Or with explicit API key and model:
# client = OpenAIClient(api_key="sk-...", model="gpt-4-turbo-preview")

# 2. Create expert prompt
templates = PromptTemplates()
prompt = templates.generate_operations_prompt(scenario, alternatives)

# 3. Generate assessment
response = client.generate_assessment(
    prompt=prompt,
    max_tokens=2000,
    temperature=0.7
)

# 4. Handle response
if response.get('error'):
    # Error handling
    print(f"API error: {response['error_message']}")
    # Retry or fallback logic
else:
    # Success - structured response
    rankings = response['alternative_rankings']
    reasoning = response['reasoning']
    confidence = response['confidence']
    concerns = response['key_concerns']

    # Check metadata
    print(f"Model: {response['_metadata']['model']}")
    print(f"Tokens: {response['_metadata']['input_tokens']} in, "
          f"{response['_metadata']['output_tokens']} out")

# 5. Monitor usage
stats = client.get_statistics()
print(f"Total requests: {stats['total_requests']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Success rate: {stats['success_rate']:.1%}")
```

SUPPORTED MODELS:

1. **gpt-4-turbo-preview** (Default)
   - Best balance of cost and quality
   - 128k context window
   - Recommended for crisis assessment

2. **gpt-4**
   - Original GPT-4 model
   - 8k context window
   - Higher cost than turbo

3. **gpt-4o**
   - Optimized GPT-4 variant
   - Faster, cheaper than GPT-4
   - Good for high-volume usage

4. **gpt-3.5-turbo**
   - Fastest, cheapest option
   - Lower reasoning quality
   - Good for testing or simple tasks

MODEL SELECTION GUIDE:

```python
# Default: GPT-4 Turbo (balanced)
client = OpenAIClient()

# Budget-conscious: GPT-3.5 Turbo
client = OpenAIClient(model="gpt-3.5-turbo")

# Quality-focused: GPT-4
client = OpenAIClient(model="gpt-4")

# High-volume: GPT-4o
client = OpenAIClient(model="gpt-4o")
```

INPUTS:
The primary method `generate_assessment()` expects:

- **prompt** (str): Expert assessment prompt (from PromptTemplates)
  - Should request JSON response format
  - Should specify expected fields
  - Typically 1000-2000 characters

- **max_tokens** (int, default 2000): Maximum response length
  - 2000 sufficient for expert assessments
  - Increase if responses truncated

- **system_prompt** (str, optional): System-level instructions
  - Default: "You are an expert providing structured assessments for crisis management..."
  - Customize for specific expert roles

- **temperature** (float, default 0.7): Sampling randomness
  - 0.7: Balanced (default for crisis assessment)
  - 0.3-0.5: More focused, deterministic
  - 0.8-1.0: More creative, diverse

OUTPUTS:
On success, returns Dict with:

```python
{
    'alternative_rankings': {
        'A1': 0.7,
        'A2': 0.2,
        'A3': 0.08,
        'A4': 0.02
    },
    'reasoning': str,
    'confidence': float,
    'key_concerns': List[str],
    '_metadata': {
        'model': str,              # e.g., 'gpt-4-turbo-preview'
        'input_tokens': int,       # prompt_tokens
        'output_tokens': int,      # completion_tokens
        'finish_reason': str,      # 'stop', 'length', etc.
        'validated': bool
    }
}
```

On error, returns Dict with:

```python
{
    'error': True,
    'error_message': str,
    'error_type': str             # 'RateLimitError', 'APIConnectionError', etc.
}
```

BUILT-IN JSON MODE:

OpenAI provides native JSON mode support:

```python
request_params = {
    "model": "gpt-4-turbo-preview",
    "messages": [...],
    "response_format": {"type": "json_object"}  # Forces JSON output
}
```

**Benefits**:
- Guarantees valid JSON (no parsing errors)
- More reliable than prompt-based JSON requests
- Reduces need for fallback parsing strategies

**Requirements**:
- System/user prompt must request JSON
- Only works with newer models (gpt-4-turbo, gpt-3.5-turbo-1106+)
- Not available for older GPT-3.5/GPT-4 versions

This client automatically enables JSON mode when calling `generate_assessment()`.

ERROR HANDLING:

The client implements multi-layered error recovery:

1. **Rate Limit Errors (RateLimitError)**:
   - Automatic retry with exponential backoff (2s, 4s, 8s)
   - Up to 3 attempts
   - Logs warnings with retry countdown

2. **Connection Errors (APIConnectionError)**:
   - Network issues, timeouts
   - Exponential backoff retry (3 attempts)
   - Helpful error messages

3. **API Errors (APIError)**:
   - 4xx errors (client errors): No retry
   - 5xx errors (server errors): Retry with backoff
   - Detailed error logging

4. **JSON Parsing Failures** (rare due to JSON mode):
   - Multi-strategy parsing as fallback
   - Returns error dict if all strategies fail

5. **Validation Failures**:
   - Logs warnings but returns response
   - Sets validated=False in metadata

PARSING STRATEGIES:

While JSON mode reduces parsing issues, the client still implements fallbacks:

1. **Direct JSON Parsing**: `json.loads(response_text)` (usually succeeds)
2. **Markdown JSON Blocks**: Extract from ```json...```
3. **Generic Code Blocks**: Extract from ```...```
4. **Regex Pattern Matching**: Find first {...} or [...]

With JSON mode enabled, strategy #1 almost always succeeds.

VALIDATION:

The `validate_response()` method checks:

1. **Required Keys**: alternative_rankings, reasoning, confidence, key_concerns
2. **Type Checking**:
   - alternative_rankings: Dict[str, float]
   - reasoning: str
   - confidence: float (0.0-1.0)
   - key_concerns: List[str]
3. **Value Ranges**:
   - confidence must be in [0, 1]
   - alternative_rankings values must be numeric
4. **Structure Validation**:
   - alternative_rankings must be a dictionary
   - key_concerns must be a list

USAGE TRACKING:

The client tracks:

- **request_count**: Number of successful API calls
- **failed_requests**: Number of failed API calls
- **total_tokens**: Cumulative prompt_tokens + completion_tokens
- **success_rate**: request_count / (request_count + failed_requests)

Access via `get_statistics()`, reset via `reset_statistics()`.

COST TRACKING:

Estimate costs using token counts:

**GPT-4 Turbo**:
```python
stats = client.get_statistics()
input_cost = (stats['total_tokens'] * 0.5) * 10 / 1_000_000   # $10/M input
output_cost = (stats['total_tokens'] * 0.5) * 30 / 1_000_000  # $30/M output
total_cost = input_cost + output_cost
```

**GPT-3.5 Turbo** (cheaper):
```python
input_cost = (stats['total_tokens'] * 0.5) * 0.5 / 1_000_000   # $0.50/M input
output_cost = (stats['total_tokens'] * 0.5) * 1.5 / 1_000_000  # $1.50/M output
```

(Assumes 50/50 input/output split; adjust based on actual usage)

PERFORMANCE CHARACTERISTICS:

- **Latency**: 2-5 seconds per request (depends on model, load)
- **Throughput**: ~60 requests/minute (rate limit, varies by tier)
- **Token Limits**:
  - GPT-4 Turbo: 128k context
  - GPT-4: 8k context
  - GPT-3.5 Turbo: 16k context
- **Retries**: 3 attempts max with exponential backoff
- **Memory**: Minimal (stateless except counters)

COMPARISON TO ALTERNATIVES:

**vs. ClaudeClient**:
- OpenAI: Built-in JSON mode, faster iteration, more familiar
- Claude: Slightly better reasoning quality, more nuanced
- Both: Similar latency, cost, reliability

**vs. LMStudioClient**:
- OpenAI: Superior quality, cloud-based, costs money
- LM Studio: Free, local, privacy, but variable quality

**vs. Raw OpenAI SDK**:
- OpenAIClient: Retry logic, validation, tracking, crisis-optimized
- Raw SDK: Lower-level, more flexible

DESIGN DECISIONS:

1. **Why GPT-4 Turbo (not GPT-4)?**: Better cost/quality balance, larger context
2. **Why force JSON mode?**: Guarantees valid JSON, reduces parsing errors
3. **Why 2000 max_tokens default?**: Sufficient for assessments, limits costs
4. **Why 0.7 temperature default?**: Balance between consistency and diversity
5. **Why 3 retries?**: Handles transient errors without excessive delay
6. **Why system+user messages?**: OpenAI best practice (vs. Claude's single message)

OPENAI-SPECIFIC FEATURES:

1. **JSON Mode**: Native support via `response_format` parameter
2. **System Messages**: Separate system role (Claude combines system into single message)
3. **Function Calling**: Could be added for structured extraction (not currently used)
4. **Streaming**: Not implemented but could be added
5. **Multiple Models**: Easy model switching via constructor parameter

INTEGRATION POINTS:

This client integrates with:

- **agents/expert_agent.py**: Expert agents use this as Claude alternative
- **llm_integration/prompt_templates.py**: Provides prompts for this client
- **agents/coordinator_agent.py**: May use for aggregation logic
- **evaluation/metrics.py**: Tracks metadata for evaluation

TESTING & DEBUGGING:

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
client = OpenAIClient()
```

Check response metadata:
```python
response = client.generate_assessment(prompt)
print(response.get('_metadata'))
```

Monitor failure rate:
```python
stats = client.get_statistics()
if stats['failed_requests'] > 0:
    print(f"Warning: {stats['failed_requests']} failed requests")
```

LIMITATIONS & KNOWN ISSUES:

1. **Rate Limits**: Varies by account tier (~60 req/min for standard)
2. **Cost**: GPT-4 more expensive than Claude for same quality
3. **Non-Deterministic**: Same prompt may yield different responses
4. **No Streaming**: Waits for full response (could add)
5. **JSON Mode Requirements**: Needs newer models, must request JSON in prompt
6. **Thread Safety**: Not thread-safe (create separate clients per thread)

SECURITY CONSIDERATIONS:

1. **API Key Storage**: Use environment variables, never hardcode
2. **PII in Prompts**: Avoid sending sensitive data to cloud API
3. **Output Validation**: Always validate responses
4. **Rate Limiting**: Respect OpenAI's rate limits

RELATED FILES:

- **llm_integration/__init__.py**: Module overview and provider comparison
- **llm_integration/claude_client.py**: Default cloud provider (similar interface)
- **llm_integration/lmstudio_client.py**: Local model provider
- **llm_integration/prompt_templates.py**: Generates prompts for this client

VERSION HISTORY:

- v1.0: Initial implementation
- v1.1: Added JSON mode support
- v1.2: Improved validation logic
- v1.3: Added usage statistics tracking
- v1.4: Enhanced error handling
- v2.0: Comprehensive documentation (Jan 2025)

REFERENCES:

- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat
- OpenAI Pricing: https://openai.com/pricing
- JSON Mode Documentation: https://platform.openai.com/docs/guides/text-generation/json-mode
- Best practices for LLM API integration
"""

import os
import time
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from pydantic import ValidationError

from models.data_models import LLMResponse

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError
except ImportError:
    raise ImportError(
        "OpenAI package not installed. Install with: pip install openai"
    )


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Client for interacting with OpenAI API.

    Provides methods for generating expert assessments with structured JSON responses,
    parsing and validating responses, and handling errors with retry logic.

    Example:
        >>> client = OpenAIClient()
        >>> prompt = "Assess flood risk alternatives..."
        >>> response = client.generate_assessment(prompt)
        >>> print(response['alternative_rankings'])
        {'A1': 0.7, 'A2': 0.2, 'A3': 0.08, 'A4': 0.02}
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4-turbo-preview)
                   Other options: gpt-4, gpt-3.5-turbo, gpt-4o

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.request_count = 0
        self.total_tokens = 0
        self.failed_requests = 0

        logger.info(f"OpenAIClient initialized with model={model}")

    def generate_assessment(self, prompt: str, max_tokens: int = 2000,
                          system_prompt: Optional[str] = None,
                          temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate expert assessment from OpenAI with structured JSON response.

        Expected response format:
        {
            "alternative_rankings": {"A1": 0.7, "A2": 0.2, "A3": 0.08, "A4": 0.02},
            "reasoning": "Based on weather data...",
            "confidence": 0.85,
            "key_concerns": ["flood risk", "time constraints"]
        }

        Args:
            prompt: The assessment prompt (should request JSON format)
            max_tokens: Maximum tokens to generate (default: 2000)
            system_prompt: Optional system prompt for expert context
            temperature: Sampling temperature 0-1 (default: 0.7)

        Returns:
            Dictionary with parsed response or error information:
            - On success: Parsed JSON with alternative_rankings, reasoning, confidence, key_concerns
            - On failure: {'error': True, 'error_message': str, 'error_type': str}

        Example:
            >>> client = OpenAIClient()
            >>> prompt = '''You are a meteorologist assessing flood response alternatives.
            ... Respond with JSON containing: alternative_rankings, reasoning, confidence, key_concerns.
            ... Alternatives: A1 (evacuate), A2 (barriers), A3 (shelter-in-place)'''
            >>> result = client.generate_assessment(prompt)
            >>> if not result.get('error'):
            ...     print(f"Top choice: {max(result['alternative_rankings'].items(), key=lambda x: x[1])}")
        """
        logger.info(f"Generating assessment (prompt length: {len(prompt)} chars)")

        try:
            # Prepare messages
            messages = []

            # Default system prompt emphasizes JSON format
            if system_prompt is None:
                system_prompt = (
                    "You are an expert providing structured assessments for crisis management. "
                    "Always respond with valid JSON format as specified in the prompt."
                )

            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Create request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": {"type": "json_object"}  # Force JSON mode in GPT-4
            }

            # Make API call with retry logic
            response = self._api_call_with_retry(request_params)

            # Extract response text
            response_text = response.choices[0].message.content

            # Update statistics
            self.request_count += 1
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self.total_tokens += input_tokens + output_tokens

            logger.info(
                f"API call successful (request #{self.request_count}, "
                f"tokens: {input_tokens} in / {output_tokens} out)"
            )

            # Parse JSON response
            parsed_response = self.parse_json_response(response_text)

            # Validate response structure
            expected_keys = ['alternative_rankings', 'reasoning', 'confidence', 'key_concerns']
            is_valid = self.validate_response(parsed_response, expected_keys)

            if not is_valid:
                logger.warning(
                    f"Response missing expected keys. Got: {list(parsed_response.keys())}"
                )

            # Add metadata
            parsed_response['_metadata'] = {
                'model': response.model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'finish_reason': response.choices[0].finish_reason,
                'validated': is_valid
            }

            return parsed_response

        except (APIError, APIConnectionError, RateLimitError) as e:
            self.failed_requests += 1
            error_response = {
                'error': True,
                'error_message': str(e),
                'error_type': type(e).__name__
            }
            logger.error(f"API error: {error_response}")
            return error_response

        except json.JSONDecodeError as e:
            self.failed_requests += 1
            error_response = {
                'error': True,
                'error_message': f"Failed to parse JSON response: {str(e)}",
                'error_type': 'JSONDecodeError',
                'raw_response': response_text if 'response_text' in locals() else None
            }
            logger.error(f"JSON parsing error: {error_response}")
            return error_response

        except Exception as e:
            self.failed_requests += 1
            error_response = {
                'error': True,
                'error_message': f"Unexpected error: {str(e)}",
                'error_type': type(e).__name__
            }
            logger.error(f"Unexpected error: {error_response}")
            return error_response

    def parse_json_response(self, response_text: str) -> Union[LLMResponse, Dict[str, Any]]:
        """
        Parse and validate JSON from OpenAI's response text.

        Handles responses that may include markdown code blocks or extra text.
        Attempts multiple parsing strategies:
        1. Direct JSON parsing
        2. Extract from ```json...``` blocks
        3. Extract from ```...``` blocks
        4. Find first {...} or [...] pattern

        Then validates the parsed data using Pydantic LLMResponse model.

        Args:
            response_text: Raw response text from OpenAI

        Returns:
            LLMResponse: Validated Pydantic model (preferred)
            Dict[str, Any]: Fallback to dict if validation fails (backward compatibility)

        Raises:
            json.JSONDecodeError: If JSON cannot be parsed

        Example:
            >>> client = OpenAIClient()
            >>> text = '```json\\n{"alternative_rankings": {"A1": 0.7}, "reasoning": "...", "confidence": 0.85}\\n```'
            >>> result = client.parse_json_response(text)
            >>> print(type(result))
            <class 'models.data_models.LLMResponse'>
        """
        # First, parse the JSON using multiple strategies
        parsed_data = None

        # Strategy 1: Try direct parsing
        try:
            parsed_data = json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from ```json...``` blocks
        if parsed_data is None:
            json_block_pattern = r'```json\s*\n(.*?)\n```'
            match = re.search(json_block_pattern, response_text, re.DOTALL)
            if match:
                try:
                    parsed_data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Extract from generic ```...``` blocks
        if parsed_data is None:
            code_block_pattern = r'```\s*\n(.*?)\n```'
            match = re.search(code_block_pattern, response_text, re.DOTALL)
            if match:
                try:
                    parsed_data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # Strategy 4: Find first {...} or [...] pattern
        if parsed_data is None:
            json_pattern = r'(\{.*\}|\[.*\])'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                try:
                    parsed_data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # If all strategies fail, raise error
        if parsed_data is None:
            logger.error(f"Failed to parse JSON from response: {response_text[:200]}...")
            raise json.JSONDecodeError(
                f"Could not extract valid JSON from response",
                response_text,
                0
            )

        # Now validate with Pydantic model
        try:
            # Add raw_response for debugging
            if isinstance(parsed_data, dict):
                parsed_data['raw_response'] = response_text[:500]  # Store first 500 chars

            llm_response = LLMResponse(**parsed_data)
            logger.debug(f"Successfully validated LLM response with Pydantic")
            return llm_response

        except ValidationError as e:
            logger.warning(
                f"LLM response failed Pydantic validation: {e}\n"
                f"Falling back to dict for backward compatibility"
            )
            # Fallback to plain dict for backward compatibility
            return parsed_data

    def validate_response(self, response: Union[Dict[str, Any], LLMResponse], expected_keys: List[str]) -> bool:
        """
        Validate that response contains all expected keys.

        Args:
            response: Parsed response (dict or LLMResponse Pydantic model)
            expected_keys: List of required keys

        Returns:
            True if all expected keys are present, False otherwise

        Example:
            >>> client = OpenAIClient()
            >>> response = {'alternative_rankings': {}, 'reasoning': 'text', 'confidence': 0.8}
            >>> expected = ['alternative_rankings', 'reasoning', 'confidence', 'key_concerns']
            >>> client.validate_response(response, expected)
            False  # Missing 'key_concerns'
        """
        # Accept both dict and LLMResponse (dict-like Pydantic model)
        if not isinstance(response, (dict, LLMResponse)):
            # Check if it has dict-like interface (supports __contains__)
            if not hasattr(response, '__contains__'):
                logger.warning(f"Response is not dict-like: {type(response)}")
                return False

        missing_keys = [key for key in expected_keys if key not in response]

        if missing_keys:
            logger.warning(f"Response missing keys: {missing_keys}")
            return False

        # Additional validation for alternative_rankings
        if 'alternative_rankings' in response:
            rankings = response['alternative_rankings']
            if not isinstance(rankings, dict):
                logger.warning(
                    f"alternative_rankings should be dict, got {type(rankings)}"
                )
                return False

            # Check that values are numeric
            for alt_id, score in rankings.items():
                if not isinstance(score, (int, float)):
                    logger.warning(
                        f"Ranking for {alt_id} should be numeric, got {type(score)}"
                    )
                    return False

        # Validate confidence is a number between 0 and 1
        if 'confidence' in response:
            confidence = response['confidence']
            if not isinstance(confidence, (int, float)):
                logger.warning(f"Confidence should be numeric, got {type(confidence)}")
                return False
            if not 0 <= confidence <= 1:
                logger.warning(f"Confidence should be in [0, 1], got {confidence}")
                return False

        # Validate key_concerns is a list
        if 'key_concerns' in response:
            if not isinstance(response['key_concerns'], list):
                logger.warning(
                    f"key_concerns should be list, got {type(response['key_concerns'])}"
                )
                return False

        logger.debug("Response validation passed")
        return True

    def _api_call_with_retry(self, request_params: Dict[str, Any],
                            max_retries: int = 3,
                            base_delay: float = 2.0) -> Any:
        """
        Make API call with exponential backoff retry logic.

        Retries on rate limit errors and transient API errors.
        Uses exponential backoff: delay = base_delay * (2 ** attempt)

        Args:
            request_params: Parameters for API call
            max_retries: Maximum number of retries (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 2.0)

        Returns:
            API response object

        Raises:
            APIError: If all retries fail
            RateLimitError: If rate limit persists after all retries
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{max_retries}")
                response = self.client.chat.completions.create(**request_params)
                return response

            except RateLimitError as e:
                last_exception = e
                if attempt == max_retries - 1:
                    logger.error(f"Rate limit error after {max_retries} attempts")
                    raise

                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

            except APIConnectionError as e:
                last_exception = e
                if attempt == max_retries - 1:
                    logger.error(f"Connection error after {max_retries} attempts")
                    raise

                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Connection error (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

            except APIError as e:
                last_exception = e
                # Don't retry on client errors (4xx)
                if hasattr(e, 'status_code') and 400 <= e.status_code < 500:
                    logger.error(f"Client error (status {e.status_code}), not retrying")
                    raise

                if attempt == max_retries - 1:
                    logger.error(f"API error after {max_retries} attempts")
                    raise

                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"API error (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception

    def generate_response(self, prompt: str,
                         max_tokens: int = 1024,
                         temperature: float = 0.7,
                         system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a general response from OpenAI (non-structured).

        For structured assessments, use generate_assessment() instead.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Optional system prompt

        Returns:
            Dictionary with response text and metadata

        Example:
            >>> client = OpenAIClient()
            >>> result = client.generate_response("Explain flood risk factors")
            >>> print(result['response'])
            "Flood risk factors include..."
        """
        logger.info(f"Generating general response (prompt length: {len(prompt)} chars)")

        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Create request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            # Make API call with retry logic
            response = self._api_call_with_retry(request_params)

            # Extract response
            response_text = response.choices[0].message.content

            # Update statistics
            self.request_count += 1
            self.total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens

            logger.info(f"General response generated successfully (request #{self.request_count})")

            return {
                'response': response_text,
                'model': response.model,
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'finish_reason': response.choices[0].finish_reason
            }

        except (APIError, APIConnectionError, RateLimitError) as e:
            self.failed_requests += 1
            error_response = {
                'error': True,
                'error_message': str(e),
                'error_type': type(e).__name__
            }
            logger.error(f"API error in generate_response: {error_response}")
            return error_response

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with usage stats including:
            - total_requests: Number of successful requests
            - failed_requests: Number of failed requests
            - total_tokens: Cumulative token usage
            - model: Model name being used

        Example:
            >>> client = OpenAIClient()
            >>> # ... make some API calls ...
            >>> stats = client.get_statistics()
            >>> print(f"Made {stats['total_requests']} requests, used {stats['total_tokens']} tokens")
        """
        return {
            'total_requests': self.request_count,
            'failed_requests': self.failed_requests,
            'success_rate': (
                self.request_count / (self.request_count + self.failed_requests)
                if (self.request_count + self.failed_requests) > 0
                else 0.0
            ),
            'total_tokens': self.total_tokens,
            'model': self.model
        }

    def reset_statistics(self):
        """
        Reset usage statistics.

        Example:
            >>> client = OpenAIClient()
            >>> client.reset_statistics()
            >>> stats = client.get_statistics()
            >>> print(stats['total_requests'])
            0
        """
        self.request_count = 0
        self.total_tokens = 0
        self.failed_requests = 0
        logger.info("Statistics reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OpenAIClient(model={self.model}, "
            f"requests={self.request_count}, "
            f"tokens={self.total_tokens})"
        )
