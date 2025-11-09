# Examples and Debug Scripts

This folder contains example scripts and debugging utilities demonstrating various aspects of the Crisis MAS system.

---

## Table of Contents

- [Overview](#overview)
- [Example Scripts](#example-scripts)
- [Debug Scripts](#debug-scripts)
- [Running Examples](#running-examples)
- [Prerequisites](#prerequisites)

---

## Overview

The examples in this folder demonstrate:

1. **LLM Integration** - How to use Claude API and multi-provider LLM clients
2. **Prompt Templates** - How to generate and customize prompts for agent evaluation
3. **Debugging** - How to test and debug specific components

These scripts are **standalone examples** and can be run independently of the main system.

---

## Example Scripts

### example_claude_usage.py

**Purpose:** Demonstrates how to use the Claude API client for expert agent assessments.

**What it shows:**
- Initializing ClaudeClient with API key
- Evaluating alternatives using Claude
- Processing LLM responses
- Error handling for API calls

**Usage:**
```bash
# Requires ANTHROPIC_API_KEY environment variable
export ANTHROPIC_API_KEY="your-api-key-here"
python examples/example_claude_usage.py
```

**Key Features:**
- Shows basic Claude API integration
- Demonstrates response parsing
- Includes error handling patterns
- Example scenario evaluation

**Note:** This requires a valid Anthropic API key. Without it, the script will demonstrate the client setup but cannot make actual API calls.

---

### example_multi_llm_providers.py

**Purpose:** Demonstrates multi-provider LLM support (Claude, OpenAI, etc.)

**What it shows:**
- Configuring multiple LLM providers
- Switching between providers
- Comparing outputs from different models
- Fallback strategies when primary provider fails

**Usage:**
```bash
# Requires API keys for providers you want to use
export ANTHROPIC_API_KEY="your-claude-key"
export OPENAI_API_KEY="your-openai-key"
python examples/example_multi_llm_providers.py
```

**Key Features:**
- Multi-provider architecture
- Provider selection strategies
- Response comparison
- Fallback mechanisms

**Supported Providers:**
- Claude (Anthropic)
- OpenAI GPT models
- (Extensible to other providers)

---

### example_prompt_templates_usage.py

**Purpose:** Demonstrates prompt template generation and customization for agent evaluation.

**What it shows:**
- Creating structured prompts for LLM evaluation
- Customizing templates for different agent roles
- Incorporating scenario context
- Generating prompts for different decision criteria

**Usage:**
```bash
python examples/example_prompt_templates_usage.py
```

**Key Features:**
- Template customization
- Role-specific prompts
- Criteria-based evaluation prompts
- Scenario context integration

**No API Key Required:** This example demonstrates template generation only and doesn't make LLM API calls.

---

## Debug Scripts

### debug_single_agent_dqs.py

**Purpose:** Debug script to test single-agent Decision Quality Score (DQS) calculation.

**What it tests:**
- Single-agent criteria score evaluation
- DQS metric calculation
- Evaluation methodology validation
- Score aggregation from criteria

**Usage:**
```bash
python examples/debug_single_agent_dqs.py
```

**Use Cases:**
- Debugging DQS calculation issues
- Validating evaluation methodology
- Testing metric computation
- Comparing single-agent vs multi-agent DQS

**When to Use:**
- When DQS calculations seem incorrect
- To understand how criteria scores map to DQS
- To validate evaluation fixes (e.g., after commit 09bec4c)
- For educational purposes to understand metrics

---

## Running Examples

### From Project Root

All examples should be run from the project root directory:

```bash
# From project root
python examples/example_claude_usage.py
python examples/example_multi_llm_providers.py
python examples/example_prompt_templates_usage.py
python examples/debug_single_agent_dqs.py
```

### From Examples Directory

You can also run from within the examples directory:

```bash
cd examples
python example_claude_usage.py
```

### With Python Module Syntax

```bash
python -m examples.example_claude_usage
python -m examples.example_multi_llm_providers
python -m examples.example_prompt_templates_usage
python -m examples.debug_single_agent_dqs
```

---

## Prerequisites

### General Requirements

All examples require:

```bash
pip install -r requirements.txt
```

### API Keys

Some examples require API keys:

**Claude API Examples:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**OpenAI Examples:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Alternative: Use .env file**
```bash
# Create .env file in project root
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Quick Check

To verify your environment is set up:

```bash
# Check if API keys are set
python -c "import os; print('Claude API key:', 'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set')"
python -c "import os; print('OpenAI API key:', 'Set' if os.getenv('OPENAI_API_KEY') else 'Not set')"

# Test imports
python -c "from llm_integration.claude_client import ClaudeClient; print('Imports working!')"
```

---

## Example Workflow

### 1. Start with Prompt Templates (No API Key Needed)

```bash
python examples/example_prompt_templates_usage.py
```

This shows how prompts are generated without making API calls.

### 2. Test Claude Integration

```bash
export ANTHROPIC_API_KEY="your-key"
python examples/example_claude_usage.py
```

See how Claude evaluates alternatives in a crisis scenario.

### 3. Compare Multiple Providers

```bash
export ANTHROPIC_API_KEY="your-claude-key"
export OPENAI_API_KEY="your-openai-key"
python examples/example_multi_llm_providers.py
```

Compare outputs from different LLM providers.

### 4. Debug Evaluation Metrics

```bash
python examples/debug_single_agent_dqs.py
```

Understand how Decision Quality Scores are calculated.

---

## Customizing Examples

All examples can be modified to test different scenarios:

```python
# Modify scenario in example files
scenario = {
    'id': 'custom_scenario',
    'title': 'Your Custom Crisis',
    'description': 'Describe your scenario',
    'alternatives': [
        {'id': 'alt_1', 'name': 'Option 1', 'description': 'First option'},
        {'id': 'alt_2', 'name': 'Option 2', 'description': 'Second option'}
    ],
    'criteria': {
        'political': 0.3,
        'economic': 0.5,
        'humanitarian': 0.2
    }
}
```

---

## Troubleshooting

### "Import Error: No module named 'llm_integration'"

**Solution:** Run from project root, not from within examples directory:
```bash
cd /path/to/crisis_mas_poc
python examples/example_claude_usage.py
```

### "Authentication Error" from Claude API

**Causes:**
- Invalid or missing API key
- Expired API key
- API key not set in environment

**Solutions:**
```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY

# Re-export if needed
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use .env file
```

### "Rate Limit Error"

**Cause:** Too many API requests in short time.

**Solution:**
- Wait a few seconds between requests
- Use lower rate limits in examples
- Consider caching responses for development

### Examples Don't Show Expected Output

**Check:**
1. API keys are valid
2. Internet connection is working
3. No firewall blocking API calls
4. LLM provider service is operational

---

## Related Documentation

- **Main System:** See [main README.md](../README.md) for full system documentation
- **LLM Integration:** See [llm_integration/README.md](../llm_integration/README.md) for detailed LLM documentation
- **Testing:** See [tests/README.md](../tests/README.md) for test suite documentation
- **Evaluation:** See [evaluation/EVALUATION_METHODOLOGY.md](../evaluation/EVALUATION_METHODOLOGY.md) for metrics details

---

## Contributing New Examples

When adding new examples:

1. **Naming Convention:** Use `example_<topic>.py` format
2. **Documentation:** Include docstring explaining purpose
3. **Dependencies:** Document any special requirements
4. **Standalone:** Examples should run independently
5. **Error Handling:** Include graceful error handling for missing API keys
6. **Update README:** Add your example to this README

**Example Template:**
```python
#!/usr/bin/env python3
"""
Example: <Brief Description>

Demonstrates:
- Feature 1
- Feature 2
- Feature 3

Usage:
    python examples/example_<name>.py

Requirements:
    - API key (if needed)
    - Special dependencies (if any)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Your example code here
def main():
    print("Example demonstration")
    # ...

if __name__ == "__main__":
    main()
```

---

## Summary

The examples folder provides:

- **Learning Resources**: Understand how to use Crisis MAS components
- **Integration Patterns**: See best practices for LLM integration
- **Debugging Tools**: Scripts to test and validate system behavior
- **Quick Start**: Get started without reading full documentation

**Next Steps:**
1. Run `example_prompt_templates_usage.py` to see how prompts work (no API key needed)
2. Set up API keys and try `example_claude_usage.py`
3. Explore multi-provider support with `example_multi_llm_providers.py`
4. Use debug scripts when troubleshooting metrics calculations
