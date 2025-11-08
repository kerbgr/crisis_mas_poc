"""
LM Studio API Client
Wrapper for LM Studio local API (OpenAI-compatible) with error handling and JSON parsing

LM Studio runs local LLMs with an OpenAI-compatible API endpoint.
Designed for crisis management multi-agent systems to get structured expert assessments.
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
