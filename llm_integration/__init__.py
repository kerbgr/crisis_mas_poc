"""
LLM Integration Module
Multi-provider LLM integration for agent reasoning enhancement

Supported Providers:
- Claude (Anthropic) - Default, best for complex reasoning
- OpenAI (GPT-4/GPT-3.5) - Alternative cloud provider
- LM Studio - Local models, privacy-focused, no API costs
"""

from .claude_client import ClaudeClient
from .openai_client import OpenAIClient
from .lmstudio_client import LMStudioClient
from .prompt_templates import PromptTemplates

__all__ = ['ClaudeClient', 'OpenAIClient', 'LMStudioClient', 'PromptTemplates']
