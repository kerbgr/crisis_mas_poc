"""
LLM Integration Module - Multi-Provider LLM Support for Expert Agent Reasoning

OBJECTIVE:
This module provides a unified, multi-provider interface for integrating Large Language
Models (LLMs) into the crisis management multi-agent system. It enables expert agents to
leverage advanced reasoning capabilities from cloud-based or local LLMs while maintaining
consistent interfaces, robust error handling, and structured response parsing.

WHY THIS EXISTS:
Crisis management expert agents need sophisticated reasoning to:
1. **Assess Complex Scenarios**: Evaluate multi-dimensional crisis situations
2. **Generate Expert Judgments**: Produce domain-specific assessments (meteorology, operations, medical)
3. **Structured Outputs**: Return consistent JSON responses for automated processing
4. **Provider Flexibility**: Support multiple LLM providers (cloud vs. local, cost vs. performance)
5. **Reliability**: Handle API errors, rate limits, and network failures gracefully
6. **Reproducibility**: Track token usage and maintain statistics for evaluation

This module abstracts LLM complexity, allowing agents to focus on domain expertise while
benefiting from state-of-the-art language model capabilities.

SUPPORTED PROVIDERS:

1. **Claude (Anthropic)** - Default Provider
   - Model: claude-sonnet-4-20250514
   - Best for: Complex reasoning, structured outputs, nuanced analysis
   - Strengths: Superior reasoning quality, instruction following
   - Cost: Moderate (cloud API)
   - Use when: Decision quality is paramount, budget allows
   - See: claude_client.py

2. **OpenAI (GPT-4/GPT-3.5)** - Alternative Cloud Provider
   - Models: gpt-4-turbo-preview, gpt-4, gpt-4o, gpt-3.5-turbo
   - Best for: General-purpose reasoning, good JSON support
   - Strengths: Fast responses, built-in JSON mode, widely adopted
   - Cost: Moderate to High (cloud API)
   - Use when: Need alternative to Claude, fast iteration
   - See: openai_client.py

3. **LM Studio** - Local Model Provider
   - Models: Any locally hosted model (Llama, Mistral, etc.)
   - Best for: Privacy-sensitive scenarios, offline operation, cost reduction
   - Strengths: No API costs, data privacy, offline capability
   - Cost: Free (requires local compute)
   - Use when: Budget constrained, privacy required, or testing
   - See: lmstudio_client.py

PROVIDER COMPARISON:

| Aspect             | Claude          | OpenAI          | LM Studio       |
|--------------------|-----------------|-----------------|-----------------|
| Reasoning Quality  | Excellent       | Very Good       | Variable        |
| Response Speed     | Fast            | Fast            | Slower          |
| Cost per Request   | Moderate        | Moderate-High   | Free            |
| JSON Reliability   | Excellent       | Excellent       | Variable        |
| Setup Complexity   | API Key Only    | API Key Only    | Requires Setup  |
| Privacy            | Cloud (shared)  | Cloud (shared)  | Local (private) |
| Offline Support    | No              | No              | Yes             |
| Error Recovery     | Robust          | Robust          | Basic           |

UNIFIED CLIENT INTERFACE:

All clients implement the same core interface:

```python
class LLMClient:
    def generate_assessment(prompt: str, max_tokens: int, system_prompt: str,
                           temperature: float) -> Dict[str, Any]
    def parse_json_response(response_text: str) -> Dict[str, Any]
    def validate_response(response: Dict, expected_keys: List[str]) -> bool
    def get_statistics() -> Dict[str, Any]
```

This consistency allows agents to switch providers without code changes.

TYPICAL USAGE FLOW:

```python
from llm_integration import ClaudeClient, OpenAIClient, LMStudioClient
from llm_integration import PromptTemplates

# 1. Initialize client (choose one)
client = ClaudeClient()  # Default: best reasoning
# client = OpenAIClient()  # Alternative cloud provider
# client = LMStudioClient()  # Local/offline

# 2. Create specialized prompt
templates = PromptTemplates()
prompt = templates.generate_meteorologist_prompt(scenario, alternatives)

# 3. Generate expert assessment
response = client.generate_assessment(
    prompt=prompt,
    max_tokens=2000,
    temperature=0.7
)

# 4. Check for errors
if response.get('error'):
    print(f"Error: {response['error_message']}")
    # Handle error (retry, fallback, etc.)
else:
    # Use structured response
    rankings = response['alternative_rankings']  # {'A1': 0.7, 'A2': 0.2, ...}
    reasoning = response['reasoning']  # Expert explanation
    confidence = response['confidence']  # 0.0-1.0
    concerns = response['key_concerns']  # List of issues

# 5. Track usage
stats = client.get_statistics()
print(f"Requests: {stats['total_requests']}, Tokens: {stats['total_tokens']}")
```

KEY COMPONENTS:

1. **ClaudeClient** (claude_client.py)
   - Anthropic API wrapper with exponential backoff retry
   - Multi-strategy JSON parsing (handles markdown, code blocks)
   - Response validation (structure, types, ranges)
   - Usage tracking (requests, tokens, failures)

2. **OpenAIClient** (openai_client.py)
   - OpenAI Chat Completions API wrapper
   - Built-in JSON mode support (response_format)
   - Same interface as ClaudeClient (drop-in replacement)
   - Compatible with gpt-4-turbo, gpt-4o, gpt-3.5-turbo

3. **LMStudioClient** (lmstudio_client.py)
   - OpenAI-compatible local API wrapper
   - Connects to localhost:1234 by default
   - Robust parsing (local models less consistent)
   - Helpful error messages for connection issues

4. **PromptTemplates** (prompt_templates.py)
   - Role-specific prompt generation
   - Three expert roles: Meteorologist, Operations, Medical
   - Consistent JSON response format specification
   - Scenario and alternative formatting utilities

EXPECTED RESPONSE FORMAT:

All clients return assessments in this structure:

```json
{
    "alternative_rankings": {
        "A1": 0.7,
        "A2": 0.2,
        "A3": 0.08,
        "A4": 0.02
    },
    "reasoning": "Expert explanation of rankings (2-3 sentences)",
    "confidence": 0.85,
    "key_concerns": [
        "Primary concern",
        "Secondary concern",
        "Additional consideration"
    ],
    "_metadata": {
        "model": "claude-sonnet-4-20250514",
        "input_tokens": 1200,
        "output_tokens": 350,
        "validated": true
    }
}
```

Or on error:

```json
{
    "error": true,
    "error_message": "Rate limit exceeded",
    "error_type": "RateLimitError"
}
```

ERROR HANDLING:

All clients implement robust error handling:

1. **Network Errors**: Exponential backoff retry (3 attempts, 2s/4s/8s delays)
2. **Rate Limits**: Automatic retry with backoff
3. **JSON Parsing Failures**: Multi-strategy parsing with fallbacks
4. **API Errors**: Detailed error messages with error types
5. **Validation Failures**: Warnings logged, metadata indicates validation status
6. **Connection Issues**: Helpful messages (e.g., "Is LM Studio running?")

PROMPT ENGINEERING BEST PRACTICES:

The PromptTemplates class implements proven patterns:

1. **Clear Role Definition**: "You are a SENIOR METEOROLOGIST with 15+ years experience..."
2. **Urgency Framing**: "⚠️ ACTIVE CRISIS SITUATION - Lives depend on your assessment"
3. **Structured Sections**: Clear headers separating context, options, task, format
4. **Explicit JSON Schema**: Show exact response structure expected
5. **Grounding Examples**: Include specific guidance for each field
6. **No Ambiguity**: "Respond ONLY with the JSON object. No preamble."

INTEGRATION WITH AGENTS:

This module integrates with the agents/ module:

```python
# In agents/expert_agent.py
from llm_integration import ClaudeClient, PromptTemplates

class ExpertAgent(BaseAgent):
    def __init__(self, expertise_area: str):
        self.llm_client = ClaudeClient()
        self.prompt_templates = PromptTemplates()

    def assess_scenario(self, scenario, alternatives):
        # Generate specialized prompt
        if self.expertise_area == "meteorologist":
            prompt = self.prompt_templates.generate_meteorologist_prompt(
                scenario, alternatives
            )

        # Get LLM assessment
        response = self.llm_client.generate_assessment(prompt)

        # Return structured assessment
        return response
```

PERFORMANCE CONSIDERATIONS:

- **Latency**: Claude/OpenAI: 2-5s, LM Studio: 5-30s (depends on local hardware)
- **Throughput**: Rate limited by provider (Claude: ~50 req/min, OpenAI: ~60 req/min)
- **Token Costs**: Claude: $3/$15 per 1M tokens, OpenAI: $10/$30 per 1M tokens
- **Memory**: Minimal (clients are stateless, only track counters)
- **Concurrency**: Not thread-safe (create separate clients per thread)

CONFIGURATION:

Set API keys via environment variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # For Claude
export OPENAI_API_KEY="sk-..."        # For OpenAI
# LM Studio requires no API key (local)
```

Or pass explicitly:

```python
client = ClaudeClient(api_key="sk-ant-...")
client = OpenAIClient(api_key="sk-...")
```

TESTING & DEBUGGING:

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check statistics:

```python
stats = client.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Failed requests: {stats['failed_requests']}")
```

CHOOSING A PROVIDER:

**Use Claude (default)** when:
- Decision quality is critical (crisis response)
- Complex reasoning required
- Budget allows moderate API costs

**Use OpenAI** when:
- Need fast iteration or development
- Claude unavailable or rate limited
- JSON reliability paramount (built-in mode)

**Use LM Studio** when:
- Budget constrained (free, local)
- Privacy required (data stays local)
- Offline operation needed
- Development/testing without API costs

LIMITATIONS & KNOWN ISSUES:

1. **Non-Deterministic**: LLMs may produce different outputs for same input
2. **JSON Reliability**: Local models (LM Studio) less consistent than cloud APIs
3. **Rate Limits**: Cloud providers impose request limits
4. **Token Limits**: Max context varies by model (8k-200k tokens)
5. **No Streaming**: Current implementation waits for full response
6. **Single Model**: Each client instance uses one model (no dynamic switching)

RELATED MODULES:

- **agents/**: Expert agents that use these LLM clients
- **scenarios/**: Crisis scenarios passed to LLM prompts
- **decision_framework/**: Aggregates LLM assessments into decisions
- **evaluation/**: Measures quality of LLM-generated assessments

VERSION HISTORY:

- v1.0: Initial implementation (Claude only)
- v1.1: Added OpenAI and LM Studio support
- v1.2: Improved JSON parsing robustness
- v1.3: Added statistics tracking
- v2.0: Comprehensive documentation (Jan 2025)

REFERENCES:

- Anthropic Claude API: https://docs.anthropic.com/claude/reference
- OpenAI API: https://platform.openai.com/docs/api-reference
- LM Studio: https://lmstudio.ai/docs
- Prompt engineering best practices for crisis management
- Multi-agent system LLM integration patterns
"""

from .claude_client import ClaudeClient
from .openai_client import OpenAIClient
from .lmstudio_client import LMStudioClient
from .prompt_templates import PromptTemplates

__all__ = ['ClaudeClient', 'OpenAIClient', 'LMStudioClient', 'PromptTemplates']
