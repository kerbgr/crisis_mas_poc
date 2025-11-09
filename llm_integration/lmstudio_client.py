"""
LM Studio API Client - Local LLM Integration for Expert Agent Reasoning

OBJECTIVE:
This module provides a robust wrapper for LM Studio's local API, enabling expert agents
in the crisis management multi-agent system to leverage locally-hosted open-source LLMs.
It handles local API communication, error recovery, JSON parsing (with extra robustness for
local models), response validation, and usage tracking—all while maintaining the same
interface as ClaudeClient and OpenAIClient.

WHY LM STUDIO:
LM Studio serves as the **local/offline LLM provider** for this crisis management system:

1. **Zero API Costs**: No per-token charges—run unlimited assessments for free
2. **Data Privacy**: All data stays local—critical for sensitive crisis scenarios
3. **Offline Operation**: Works without internet—essential for disaster scenarios
4. **Open-Source Models**: Leverage Llama, Mistral, Phi, and other open models
5. **Development/Testing**: Free experimentation without burning API credits
6. **Budget Constraints**: Enables crisis management for under-resourced organizations

While local models typically have lower reasoning quality than Claude/GPT-4, LM Studio
provides a cost-effective alternative for budget-constrained scenarios, development,
testing, and privacy-sensitive use cases.

WHY THIS CLIENT:
A custom client (rather than using the bare OpenAI SDK pointed at localhost) provides:

1. **Robust JSON Parsing**: Local models less consistent—need extra fallback strategies
2. **Helpful Error Messages**: "Is LM Studio running?" instead of generic connection errors
3. **Response Validation**: Ensures responses meet crisis assessment requirements
4. **Usage Tracking**: Monitor local model performance
5. **Consistent Interface**: Same API as ClaudeClient/OpenAIClient (drop-in replacement)
6. **Graceful Degradation**: Handles missing token counts, varied output formats
7. **Crisis-Optimized**: Designed for structured expert assessments

TYPICAL USAGE:

```python
from llm_integration import LMStudioClient, PromptTemplates

# 1. Start LM Studio and load a model (e.g., Llama 3, Mistral 7B)

# 2. Initialize client
client = LMStudioClient()  # Connects to localhost:1234 by default
# Or with custom endpoint:
# client = LMStudioClient(base_url="http://192.168.1.100:1234/v1")

# 3. Create expert prompt
templates = PromptTemplates()
prompt = templates.generate_medical_prompt(scenario, alternatives)

# 4. Generate assessment
response = client.generate_assessment(
    prompt=prompt,
    max_tokens=2000,
    temperature=0.7
)

# 5. Handle response
if response.get('error'):
    # Error handling - often connection issues
    print(f"Error: {response['error_message']}")
    # Check if LM Studio is running
else:
    # Success - structured response
    rankings = response['alternative_rankings']
    reasoning = response['reasoning']
    confidence = response['confidence']

# 6. Monitor usage
stats = client.get_statistics()
print(f"Total requests: {stats['total_requests']}")
# Note: Local models may not return accurate token counts
```

SETUP REQUIREMENTS:

1. **Install LM Studio**:
   - Download from https://lmstudio.ai/
   - Available for Windows, macOS, Linux

2. **Load a Model**:
   - Open LM Studio
   - Search for models (e.g., "Llama-3", "Mistral-7B-Instruct")
   - Download a model (7B models work on most hardware)
   - Load the model

3. **Start Local Server**:
   - In LM Studio, go to "Local Server" tab
   - Click "Start Server" (default port: 1234)
   - Verify server running: http://localhost:1234/v1/models

4. **Use This Client**:
   ```python
   client = LMStudioClient()  # Connects automatically
   ```

MODEL RECOMMENDATIONS:

For crisis management assessments, recommended models:

**Best Quality (if you have GPU)**:
- **Llama-3-8B-Instruct**: Excellent reasoning, instruction following
- **Mistral-7B-Instruct-v0.2**: Good balance of quality and speed
- **Phi-3-Medium-4K-Instruct**: Strong reasoning, smaller size

**Fastest (CPU-friendly)**:
- **Llama-3-8B-Instruct-Q4**: Quantized, runs on CPU
- **Phi-3-Mini-4K-Instruct**: Small but capable
- **TinyLlama-1.1B**: Very fast, lower quality

**Avoid**: Base models (non-instruct)—they don't follow instructions well

INPUTS:
The primary method `generate_assessment()` expects:

- **prompt** (str): Expert assessment prompt
  - Should explicitly request JSON format (local models need clear instructions)
  - Should provide examples if possible (helps with consistency)
  - Typically 1000-2000 characters

- **max_tokens** (int, default 2000): Maximum response length
  - 2000 sufficient for assessments
  - Local models may be slower for longer outputs

- **system_prompt** (str, optional): System-level instructions
  - Default: "You are an expert providing structured assessments..."
  - Local models benefit from clear, explicit instructions

- **temperature** (float, default 0.7): Sampling randomness
  - 0.7: Balanced (default)
  - 0.3-0.5: More focused (recommended for local models)
  - Lower temperature → more consistent output

OUTPUTS:
Same structure as ClaudeClient/OpenAIClient:

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
        'model': str,              # Local model identifier
        'input_tokens': int,       # May be 0 if unavailable
        'output_tokens': int,      # May be 0 if unavailable
        'finish_reason': str,      # 'stop', 'length', etc.
        'validated': bool,
        'endpoint': str            # localhost:1234
    }
}
```

On error:
```python
{
    'error': True,
    'error_message': str,
    'error_type': str,
    'endpoint': str
}
```

LOCAL MODEL CHALLENGES:

Local models present unique challenges compared to cloud APIs:

1. **Inconsistent JSON Output**: May include extra text, formatting variations
   → Solution: Multi-strategy parsing with extra robustness

2. **Instruction Following**: Less reliable than Claude/GPT-4
   → Solution: Very explicit prompts, examples, clear structure

3. **Token Count Tracking**: Not all models return accurate token counts
   → Solution: Gracefully handle missing token data (return 0)

4. **Variable Quality**: Depends heavily on model size and quantization
   → Solution: Recommend quality models, validate outputs

5. **Connection Issues**: Server may not be running
   → Solution: Helpful error messages ("Is LM Studio running?")

6. **No Built-in JSON Mode**: Unlike OpenAI's response_format
   → Solution: Rely on prompt engineering and robust parsing

ROBUST JSON PARSING:

Because local models are less consistent, this client implements extra-robust parsing:

1. **Direct JSON Parsing**: Try `json.loads(response_text)` first
2. **Markdown JSON Blocks**: Extract from ```json...```
3. **Generic Code Blocks**: Extract from ```...```
4. **Regex Pattern Matching**: Find first {...} or [...]
5. **Whitespace Handling**: Strip extra spaces, newlines
6. **Error Recovery**: Detailed logging to diagnose parsing failures

This multi-layered approach ensures maximum reliability with local models.

ERROR HANDLING:

1. **Connection Errors (APIConnectionError)**:
   - Most common error (LM Studio not running)
   - Helpful message: "Is LM Studio running at localhost:1234?"
   - Retry with exponential backoff (3 attempts)

2. **API Errors (APIError)**:
   - Model not loaded, invalid request
   - Retry with backoff

3. **JSON Parsing Failures**:
   - More common with local models
   - Multi-strategy parsing with detailed logs
   - Returns error dict if all strategies fail

4. **Validation Failures**:
   - Logs warnings but returns response
   - Sets validated=False in metadata
   - Allows caller to handle incomplete responses

NO RATE LIMITS:

Unlike cloud APIs, LM Studio has:
- **No rate limits**: Send as many requests as you want
- **No costs**: Unlimited free usage
- **Only limit**: Local hardware speed (GPU/CPU throughput)

This makes LM Studio ideal for:
- Development and testing (unlimited experimentation)
- High-volume evaluation (run 1000+ scenarios)
- Budget-constrained organizations

COST SAVINGS EXAMPLE:

**Cloud APIs (Claude/OpenAI)**:
- 1000 expert assessments × 4000 tokens × $0.01/1k tokens = **$40**

**LM Studio**:
- 1000 expert assessments = **$0** (only electricity costs)

For organizations running hundreds or thousands of crisis simulations, LM Studio can
save significant costs while maintaining reasonable quality with good models.

PERFORMANCE CHARACTERISTICS:

Varies dramatically based on hardware:

**GPU (RTX 3090, M2 Max)**:
- Latency: 5-15 seconds per request
- Throughput: Unlimited (hardware-bound)

**CPU (Modern Intel/AMD)**:
- Latency: 15-60 seconds per request
- Throughput: Unlimited but slow

**Optimization Tips**:
- Use quantized models (Q4, Q5) for faster inference
- Reduce max_tokens to speed up generation
- Use smaller models (7B vs 13B) if quality acceptable
- Enable GPU acceleration in LM Studio settings

COMPARISON TO ALTERNATIVES:

**vs. ClaudeClient**:
- LM Studio: Free, private, offline | Claude: Better quality, faster
- LM Studio: Variable quality | Claude: Consistent high quality
- LM Studio: No rate limits | Claude: ~50 req/min

**vs. OpenAIClient**:
- LM Studio: Free, private, offline | OpenAI: Better quality, JSON mode
- LM Studio: Slower | OpenAI: Faster (cloud infrastructure)
- LM Studio: No rate limits | OpenAI: ~60 req/min

**vs. Raw OpenAI SDK (pointed at localhost)**:
- LMStudioClient: Helpful errors, robust parsing, crisis-optimized
- Raw SDK: Generic errors, basic parsing

DESIGN DECISIONS:

1. **Why localhost:1234 default?**: LM Studio's default local server port
2. **Why no JSON mode?**: Not supported by LM Studio/local models (prompt-based only)
3. **Why extra parsing strategies?**: Local models less consistent than cloud APIs
4. **Why graceful token handling?**: Some local models don't return token counts
5. **Why helpful connection errors?**: Users often forget to start LM Studio
6. **Why system+user messages?**: OpenAI-compatible API expects this format

INTEGRATION POINTS:

This client integrates with:

- **agents/expert_agent.py**: Expert agents use this for free local inference
- **llm_integration/prompt_templates.py**: Provides prompts (same as other clients)
- **evaluation/metrics.py**: Tracks metadata (may have missing token counts)

TESTING & DEBUGGING:

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
client = LMStudioClient()
# Will log: Connection attempts, parsing strategies, validation results
```

Check if LM Studio is running:
```python
import requests
try:
    response = requests.get("http://localhost:1234/v1/models")
    print("LM Studio is running")
    print(f"Loaded model: {response.json()}")
except:
    print("LM Studio is NOT running - start it first!")
```

Monitor local model quality:
```python
stats = client.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
# Low success rate? Try different model or explicit prompts
```

LIMITATIONS & KNOWN ISSUES:

1. **Quality Variable**: Depends on model choice, hardware, quantization
2. **Slower than Cloud**: GPU: 5-15s, CPU: 15-60s (vs. cloud 2-5s)
3. **Less Consistent**: JSON output may need cleanup
4. **No Built-in JSON Mode**: Must rely on prompt engineering
5. **Token Counts**: May be inaccurate or missing
6. **Hardware Requirements**: Good performance needs GPU
7. **Setup Complexity**: Must install and configure LM Studio

SECURITY & PRIVACY:

**Advantages**:
- **Data Privacy**: All processing local, no data sent to cloud
- **No API Keys**: No credentials to manage
- **Air-gapped**: Can run completely offline

**Use Cases**:
- Government/military crisis planning (sensitive data)
- Medical crisis scenarios (HIPAA compliance)
- Disaster scenarios (no internet access)
- Development/testing (no data leaks)

TROUBLESHOOTING:

**Connection Failed**:
1. Open LM Studio
2. Load a model
3. Start local server (port 1234)
4. Try again

**JSON Parsing Errors**:
1. Use instruct-tuned models (not base models)
2. Make prompts more explicit (request JSON clearly)
3. Try lower temperature (0.3-0.5)
4. Try different model (Llama-3, Mistral preferred)

**Low Quality Outputs**:
1. Upgrade to larger model (13B vs. 7B)
2. Use higher quantization (Q5, Q6 vs. Q4)
3. Enable GPU acceleration
4. Provide examples in prompt

RELATED FILES:

- **llm_integration/__init__.py**: Module overview and provider comparison
- **llm_integration/claude_client.py**: Default cloud provider
- **llm_integration/openai_client.py**: Alternative cloud provider
- **llm_integration/prompt_templates.py**: Generates prompts for this client

VERSION HISTORY:

- v1.0: Initial implementation
- v1.1: Enhanced JSON parsing robustness
- v1.2: Graceful handling of missing token counts
- v1.3: Improved connection error messages
- v1.4: Added usage statistics tracking
- v2.0: Comprehensive documentation (Jan 2025)

REFERENCES:

- LM Studio: https://lmstudio.ai/
- LM Studio Docs: https://lmstudio.ai/docs
- OpenAI-Compatible API Spec: https://platform.openai.com/docs/api-reference
- Recommended open models: Llama 3, Mistral, Phi-3
- Model quantization guide: GGUF formats
"""

import os
import time
import json
import logging
import re
from typing import Dict, Any, List, Optional
try:
    from openai import OpenAI, APIError, APIConnectionError
except ImportError:
    raise ImportError(
        "OpenAI package not installed (required for LM Studio compatibility). "
        "Install with: pip install openai"
    )


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LMStudioClient:
    """
    Client for interacting with LM Studio local API.

    LM Studio provides an OpenAI-compatible API for running local LLMs.
    Default endpoint: http://localhost:1234/v1

    Provides methods for generating expert assessments with structured JSON responses,
    parsing and validating responses, and handling errors with retry logic.

    Example:
        >>> client = LMStudioClient()  # Uses default localhost:1234
        >>> prompt = "Assess flood risk alternatives..."
        >>> response = client.generate_assessment(prompt)
        >>> print(response['alternative_rankings'])
        {'A1': 0.7, 'A2': 0.2, 'A3': 0.08, 'A4': 0.02}
    """

    def __init__(self,
                 base_url: str = "http://localhost:1234/v1",
                 api_key: str = "lm-studio",  # LM Studio doesn't require real key
                 model: Optional[str] = None):
        """
        Initialize LM Studio client.

        Args:
            base_url: LM Studio API base URL (default: http://localhost:1234/v1)
            api_key: API key (LM Studio doesn't require authentication, any value works)
            model: Model identifier (default: None, uses loaded model in LM Studio)
                   Examples: "local-model", "llama-2-7b-chat", etc.

        Note:
            LM Studio must be running with a model loaded for this client to work.
            The model parameter is optional - if None, uses whatever model is loaded.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model or "local-model"  # Default identifier for local model

        # Create OpenAI client pointing to LM Studio
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        self.request_count = 0
        self.total_tokens = 0
        self.failed_requests = 0

        logger.info(
            f"LMStudioClient initialized (endpoint={base_url}, model={self.model})"
        )

    def generate_assessment(self, prompt: str, max_tokens: int = 2000,
                          system_prompt: Optional[str] = None,
                          temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate expert assessment from LM Studio with structured JSON response.

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
            >>> client = LMStudioClient()
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
                "temperature": temperature
            }

            # Note: LM Studio has inconsistent support for response_format across versions
            # We rely on the system prompt requesting JSON and robust parsing instead
            # This ensures compatibility with all LM Studio versions

            # Make API call with retry logic
            response = self._api_call_with_retry(request_params)

            # Extract response text
            response_text = response.choices[0].message.content

            # Update statistics (LM Studio may not return token counts)
            self.request_count += 1
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0
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
                'model': self.model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'finish_reason': response.choices[0].finish_reason if response.choices else None,
                'validated': is_valid,
                'endpoint': self.base_url
            }

            return parsed_response

        except (APIError, APIConnectionError) as e:
            self.failed_requests += 1
            error_response = {
                'error': True,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'endpoint': self.base_url
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

    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LM Studio's response text.

        Local models may produce less consistent formatting than cloud APIs,
        so we use multiple parsing strategies.

        Args:
            response_text: Raw response text from LM Studio

        Returns:
            Parsed JSON as dictionary

        Raises:
            json.JSONDecodeError: If JSON cannot be parsed

        Example:
            >>> client = LMStudioClient()
            >>> text = '{"alternative_rankings": {"A1": 0.7}}'
            >>> result = client.parse_json_response(text)
            >>> print(result)
            {'alternative_rankings': {'A1': 0.7}}
        """
        # Strategy 1: Try direct parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from ```json...``` blocks
        json_block_pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(json_block_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract from generic ```...``` blocks
        code_block_pattern = r'```\s*\n(.*?)\n```'
        match = re.search(code_block_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Find first {...} or [...] pattern
        json_pattern = r'(\{.*\}|\[.*\])'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # If all strategies fail, raise error
        logger.error(f"Failed to parse JSON from response: {response_text[:200]}...")
        raise json.JSONDecodeError(
            f"Could not extract valid JSON from response",
            response_text,
            0
        )

    def validate_response(self, response: Dict[str, Any], expected_keys: List[str]) -> bool:
        """
        Validate that response contains all expected keys.

        Args:
            response: Parsed response dictionary
            expected_keys: List of required keys

        Returns:
            True if all expected keys are present, False otherwise
        """
        if not isinstance(response, dict):
            logger.warning(f"Response is not a dictionary: {type(response)}")
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

        Args:
            request_params: Parameters for API call
            max_retries: Maximum number of retries (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 2.0)

        Returns:
            API response object

        Raises:
            APIError: If all retries fail
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{max_retries}")
                response = self.client.chat.completions.create(**request_params)
                return response

            except APIConnectionError as e:
                last_exception = e
                if attempt == max_retries - 1:
                    logger.error(
                        f"Connection error after {max_retries} attempts. "
                        f"Is LM Studio running at {self.base_url}?"
                    )
                    raise

                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Connection error (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s... Is LM Studio running?"
                )
                time.sleep(delay)

            except APIError as e:
                last_exception = e
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
        Generate a general response from LM Studio (non-structured).

        For structured assessments, use generate_assessment() instead.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Optional system prompt

        Returns:
            Dictionary with response text and metadata
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
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0
            self.total_tokens += input_tokens + output_tokens

            logger.info(f"General response generated successfully (request #{self.request_count})")

            return {
                'response': response_text,
                'model': self.model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'finish_reason': response.choices[0].finish_reason if response.choices else None
            }

        except (APIError, APIConnectionError) as e:
            self.failed_requests += 1
            error_response = {
                'error': True,
                'error_message': str(e),
                'error_type': type(e).__name__
            }
            logger.error(f"API error in generate_response: {error_response}")
            return error_response

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_requests': self.request_count,
            'failed_requests': self.failed_requests,
            'success_rate': (
                self.request_count / (self.request_count + self.failed_requests)
                if (self.request_count + self.failed_requests) > 0
                else 0.0
            ),
            'total_tokens': self.total_tokens,
            'model': self.model,
            'endpoint': self.base_url
        }

    def reset_statistics(self):
        """Reset usage statistics."""
        self.request_count = 0
        self.total_tokens = 0
        self.failed_requests = 0
        logger.info("Statistics reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LMStudioClient(endpoint={self.base_url}, model={self.model}, "
            f"requests={self.request_count}, tokens={self.total_tokens})"
        )
