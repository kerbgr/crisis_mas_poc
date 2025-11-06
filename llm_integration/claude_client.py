"""
Claude API Client
Wrapper for Claude API with error handling and rate limiting
"""

import os
import time
from typing import Dict, Any, List, Optional
import anthropic
from anthropic import Anthropic, APIError, RateLimitError


class ClaudeClient:
    """
    Client for interacting with Claude API.
    Provides methods for generating expert assessments and justifications.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            model: Claude model to use
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

    def generate_response(self, prompt: str,
                         max_tokens: int = 1024,
                         temperature: float = 0.7,
                         system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response from Claude.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system_prompt: Optional system prompt

        Returns:
            Dictionary with response text and metadata
        """
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

            return {
                'response': response_text,
                'model': response.model,
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'stop_reason': response.stop_reason
            }

        except APIError as e:
            return {
                'error': True,
                'error_message': str(e),
                'error_type': type(e).__name__
            }

    def _api_call_with_retry(self, request_params: Dict[str, Any],
                            max_retries: int = 3,
                            base_delay: float = 1.0) -> Any:
        """
        Make API call with exponential backoff retry logic.

        Args:
            request_params: Parameters for API call
            max_retries: Maximum number of retries
            base_delay: Base delay for exponential backoff

        Returns:
            API response

        Raises:
            APIError: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(**request_params)
                return response

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                print(f"Rate limit hit, retrying in {delay}s...")
                time.sleep(delay)

            except APIError as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                print(f"API error: {e}, retrying in {delay}s...")
                time.sleep(delay)

    def get_expert_assessment(self, scenario: Dict[str, Any],
                             expert_profile: Dict[str, Any],
                             prompt_template: str) -> Dict[str, Any]:
        """
        Get expert assessment for a crisis scenario.

        Args:
            scenario: Crisis scenario data
            expert_profile: Expert agent profile
            prompt_template: Template for the prompt

        Returns:
            Dictionary with assessment
        """
        # Format prompt with scenario and profile
        prompt = prompt_template.format(
            expertise_domain=expert_profile.get('expertise_domain', 'general'),
            scenario_type=scenario.get('type', 'unknown'),
            scenario_description=scenario.get('description', ''),
            severity=scenario.get('severity', 0.5),
            affected_population=scenario.get('affected_population', 0)
        )

        system_prompt = (
            f"You are an expert in {expert_profile.get('expertise_domain', 'general')} "
            f"providing professional assessment for crisis management decisions."
        )

        result = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )

        return result

    def get_action_justification(self, scenario: Dict[str, Any],
                                action: Dict[str, Any],
                                expert_profile: Dict[str, Any],
                                prompt_template: str) -> Dict[str, Any]:
        """
        Get detailed justification for a proposed action.

        Args:
            scenario: Crisis scenario data
            action: Proposed action
            expert_profile: Expert agent profile
            prompt_template: Template for the prompt

        Returns:
            Dictionary with justification
        """
        prompt = prompt_template.format(
            expertise_domain=expert_profile.get('expertise_domain', 'general'),
            action_name=action.get('name', action.get('id', 'this action')),
            action_description=action.get('description', ''),
            scenario_description=scenario.get('description', ''),
            scenario_type=scenario.get('type', 'unknown')
        )

        system_prompt = (
            f"You are an expert in {expert_profile.get('expertise_domain', 'general')} "
            f"providing detailed justification for crisis management decisions."
        )

        result = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6
        )

        return result

    def facilitate_consensus(self, expert_proposals: List[Dict[str, Any]],
                           scenario: Dict[str, Any],
                           prompt_template: str) -> Dict[str, Any]:
        """
        Use LLM to facilitate consensus among conflicting expert opinions.

        Args:
            expert_proposals: List of expert proposals
            scenario: Crisis scenario data
            prompt_template: Template for the prompt

        Returns:
            Dictionary with consensus facilitation result
        """
        # Summarize proposals
        proposals_summary = "\n".join([
            f"- {p.get('expertise_domain', 'Expert')}: {p.get('proposed_action', {}).get('name', 'Unknown action')} "
            f"(confidence: {p.get('confidence', 0):.2f})"
            for p in expert_proposals
        ])

        prompt = prompt_template.format(
            scenario_description=scenario.get('description', ''),
            num_experts=len(expert_proposals),
            proposals_summary=proposals_summary
        )

        system_prompt = (
            "You are a crisis management coordinator facilitating consensus "
            "among experts with different perspectives."
        )

        result = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=2048
        )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with usage stats
        """
        return {
            'total_requests': self.request_count,
            'total_tokens': self.total_tokens,
            'model': self.model
        }

    def reset_statistics(self):
        """Reset usage statistics."""
        self.request_count = 0
        self.total_tokens = 0
