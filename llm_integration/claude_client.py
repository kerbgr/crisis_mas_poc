"""
Claude API Client
Wrapper for Claude API with error handling, rate limiting, and JSON response parsing

Designed for crisis management multi-agent systems to get structured expert assessments.
"""

import os
import time
import json
import logging
import re
from typing import Dict, Any, List, Optional
from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    Client for interacting with Claude API.

    Provides methods for generating expert assessments with structured JSON responses,
    parsing and validating responses, and handling errors with retry logic.

    Example:
        >>> client = ClaudeClient()
        >>> prompt = "Assess flood risk alternatives..."
        >>> response = client.generate_assessment(prompt)
        >>> print(response['alternative_rankings'])
        {'A1': 0.7, 'A2': 0.2, 'A3': 0.08, 'A4': 0.02}
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            model: Claude model to use (default: claude-sonnet-4-20250514)

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.request_count = 0
        self.total_tokens = 0
        self.failed_requests = 0

        logger.info(f"ClaudeClient initialized with model={model}")

    def generate_assessment(self, prompt: str, max_tokens: int = 2000,
                          system_prompt: Optional[str] = None,
                          temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate expert assessment from Claude with structured JSON response.

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
            >>> client = ClaudeClient()
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
            messages = [{"role": "user", "content": prompt}]

            # Default system prompt emphasizes JSON format
            if system_prompt is None:
                system_prompt = (
                    "You are an expert providing structured assessments for crisis management. "
                    "Always respond with valid JSON format as specified in the prompt."
                )

            # Create request parameters
            request_params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
                "system": system_prompt
            }

            # Make API call with retry logic
            response = self._api_call_with_retry(request_params)

            # Extract response text
            response_text = response.content[0].text

            # Update statistics
            self.request_count += 1
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
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
                'stop_reason': response.stop_reason,
                'validated': is_valid
            }

            return parsed_response

        except (APIError, APIConnectionError) as e:
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

    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from Claude's response text.

        Handles responses that may include markdown code blocks or extra text.
        Attempts multiple parsing strategies:
        1. Direct JSON parsing
        2. Extract from ```json...``` blocks
        3. Extract from ```...``` blocks
        4. Find first {...} or [...] pattern

        Args:
            response_text: Raw response text from Claude

        Returns:
            Parsed JSON as dictionary

        Raises:
            json.JSONDecodeError: If JSON cannot be parsed

        Example:
            >>> client = ClaudeClient()
            >>> text = '```json\\n{"alternative_rankings": {"A1": 0.7}}\\n```'
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

        Example:
            >>> client = ClaudeClient()
            >>> response = {'alternative_rankings': {}, 'reasoning': 'text', 'confidence': 0.8}
            >>> expected = ['alternative_rankings', 'reasoning', 'confidence', 'key_concerns']
            >>> client.validate_response(response, expected)
            False  # Missing 'key_concerns'
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
                response = self.client.messages.create(**request_params)
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
                # Don't retry on non-transient errors (e.g., invalid request)
                if e.status_code and 400 <= e.status_code < 500:
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
        Generate a general response from Claude (non-structured).

        For structured assessments, use generate_assessment() instead.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Optional system prompt

        Returns:
            Dictionary with response text and metadata

        Example:
            >>> client = ClaudeClient()
            >>> result = client.generate_response("Explain flood risk factors")
            >>> print(result['response'])
            "Flood risk factors include..."
        """
        logger.info(f"Generating general response (prompt length: {len(prompt)} chars)")

        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]

            # Create request parameters
            request_params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }

            if system_prompt:
                request_params["system"] = system_prompt

            # Make API call with retry logic
            response = self._api_call_with_retry(request_params)

            # Extract response
            response_text = response.content[0].text

            # Update statistics
            self.request_count += 1
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens

            logger.info(f"General response generated successfully (request #{self.request_count})")

            return {
                'response': response_text,
                'model': response.model,
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'stop_reason': response.stop_reason
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
        """
        Get usage statistics.

        Returns:
            Dictionary with usage stats including:
            - total_requests: Number of successful requests
            - failed_requests: Number of failed requests
            - total_tokens: Cumulative token usage
            - model: Model name being used

        Example:
            >>> client = ClaudeClient()
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
            >>> client = ClaudeClient()
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
            f"ClaudeClient(model={self.model}, "
            f"requests={self.request_count}, "
            f"tokens={self.total_tokens})"
        )
