# Tests Module

**Comprehensive Test Suite for Crisis MAS Quality Assurance**

This directory contains the complete test suite for the Crisis Management Multi-Agent System (Crisis MAS). The tests ensure system reliability, validate functionality, and prevent regressions across all modules.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing New Tests](#writing-new-tests)
- [Mocking Strategies](#mocking-strategies)
- [Best Practices](#best-practices)
- [Continuous Integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

The test suite validates:

1. **Correctness** - All components behave as expected
2. **Reliability** - System handles errors gracefully
3. **Performance** - Critical paths meet performance requirements
4. **Integration** - Components work together correctly
5. **Regression Prevention** - Changes don't break existing functionality

### Test Philosophy

Our testing approach follows these principles:

- **Comprehensive Coverage**: Test all modules and critical paths
- **Fast Execution**: Unit tests should run quickly
- **Isolation**: Tests don't depend on each other
- **Clarity**: Tests document expected behavior
- **Maintainability**: Tests are easy to update and extend

---

## Quick Start

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# From project root
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

### Run Specific Tests

```bash
# Run single test file
pytest tests/test_integration.py -v

# Run specific test function
pytest tests/test_consensus_model.py::test_basic_consensus -v

# Run tests matching pattern
pytest tests/ -k "agent" -v
```

### Check Test Coverage

```bash
# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html

# Open coverage report
open htmlcov/index.html
```

---

## Test Organization

### Directory Structure

```
tests/
├── __init__.py                    # Test suite documentation
├── README.md                      # This file
│
├── Unit Tests (Component-Specific)
├── test_base_agent.py             # BaseAgent tests
├── test_expert_agent.py           # ExpertAgent tests
├── test_coordinator_agent.py      # CoordinatorAgent tests
├── test_claude_client.py          # Claude LLM client tests
├── test_multi_llm_clients.py      # Multi-provider LLM tests
├── test_prompt_templates.py       # Prompt template tests
├── test_consensus_model.py        # Consensus algorithm tests
├── test_mcda_engine.py            # MCDA scoring tests
├── test_evidential_reasoning.py   # Evidential reasoning tests
├── test_baseline.py               # Baseline decision tests
│
├── Integration Tests
├── test_integration.py            # Full system end-to-end
├── test_gat_integration.py        # GAT integration
├── test_evaluation_fix.py         # Evaluation module tests
│
├── Error & Edge Case Tests
├── test_error_scenarios.py        # Robustness tests
│
└── Advanced Tests
    └── test_gat.py                # Graph Attention Network tests
```

### Test Categories

#### 1. Unit Tests

Test individual components in isolation:

| Test File | Component | Purpose |
|-----------|-----------|---------|
| `test_base_agent.py` | `agents/base_agent.py` | Agent base class functionality |
| `test_expert_agent.py` | `agents/expert_agent.py` | Expert reasoning and LLM integration |
| `test_coordinator_agent.py` | `agents/coordinator_agent.py` | Aggregation and coordination |
| `test_claude_client.py` | `llm_integration/claude_client.py` | Claude API integration |
| `test_multi_llm_clients.py` | `llm_integration/*_client.py` | Multi-provider support |
| `test_prompt_templates.py` | `llm_integration/prompt_templates.py` | Template generation |
| `test_consensus_model.py` | `decision_framework/consensus_model.py` | Consensus detection |
| `test_mcda_engine.py` | `decision_framework/mcda_engine.py` | MCDA scoring |
| `test_evidential_reasoning.py` | `decision_framework/evidential_reasoning.py` | Evidence aggregation |
| `test_baseline.py` | Core decision logic | Baseline validation |

#### 2. Integration Tests

Test multiple components working together:

| Test File | Scope | Purpose |
|-----------|-------|---------|
| `test_integration.py` | Full system | End-to-end workflow validation |
| `test_gat_integration.py` | GAT + Agents | Graph attention integration |
| `test_evaluation_fix.py` | Evaluation | Metrics and evaluation |

#### 3. Error Scenario Tests

Test system robustness:

| Test File | Focus | Purpose |
|-----------|-------|---------|
| `test_error_scenarios.py` | Error handling | Invalid inputs, edge cases |

#### 4. Advanced Tests

Test specialized functionality:

| Test File | Component | Purpose |
|-----------|-----------|---------|
| `test_gat.py` | Graph Attention Network | GAT functionality |

---

## Running Tests

### Using pytest (Recommended)

**Basic Usage:**
```bash
# Run all tests
pytest tests/ -v

# Run with detailed output
pytest tests/ -vv

# Run with stdout/stderr
pytest tests/ -s
```

**Coverage Reports:**
```bash
# Generate coverage report
pytest tests/ --cov=. --cov-report=html

# Coverage with missing lines
pytest tests/ --cov=. --cov-report=term-missing

# Coverage for specific module
pytest tests/ --cov=agents --cov-report=html
```

**Test Selection:**
```bash
# Run specific file
pytest tests/test_integration.py -v

# Run specific test
pytest tests/test_consensus_model.py::test_basic_consensus -v

# Run tests matching pattern
pytest tests/ -k "consensus" -v

# Run tests by marker
pytest tests/ -m "integration" -v
```

**Performance:**
```bash
# Run only fast tests (exclude slow ones)
pytest tests/ -m "not slow" -v

# Show slowest 10 tests
pytest tests/ --durations=10
```

**Debugging:**
```bash
# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Show local variables on failure
pytest tests/ -l
```

### Using Python Directly

```bash
# Run individual test file
python tests/test_integration.py

# Run from project root
python -m tests.test_integration
```

### Programmatic Execution

```python
import pytest

# Run all tests
pytest.main(['tests/', '-v'])

# Run specific test with coverage
pytest.main([
    'tests/test_integration.py',
    '--cov=.',
    '--cov-report=html',
    '-v'
])
```

---

## Test Coverage

### Current Coverage

**Module Coverage:**
- ✅ **agents/**: Base agents, expert agents, coordinator - 85%
- ✅ **llm_integration/**: Claude client, multi-provider, prompts - 80%
- ✅ **decision_framework/**: Consensus, MCDA, evidential reasoning - 90%
- ✅ **scenarios/**: Scenario loading and validation - 75%
- ✅ **utils/**: Validation, configuration, safe operations - 95%
- ✅ **evaluation/**: Metrics and evaluation - 70%

**Test Types:**
- ✅ Unit tests: 10 test files
- ✅ Integration tests: 3 test files
- ✅ Error scenario tests: 1 test file
- ✅ Advanced tests: 1 test file

### Coverage Goals

Target coverage levels:
- Critical modules (utils, decision_framework): >90%
- Core modules (agents, llm_integration): >80%
- Support modules (evaluation, scenarios): >70%
- Overall project: >80%

### Running Coverage Analysis

```bash
# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html

# View in browser
open htmlcov/index.html

# Terminal coverage report with missing lines
pytest tests/ --cov=. --cov-report=term-missing

# XML coverage for CI/CD
pytest tests/ --cov=. --cov-report=xml
```

---

## Writing New Tests

### Test File Template

```python
#!/usr/bin/env python3
"""
Test Module for [Component Name]

Tests:
1. Basic functionality
2. Error handling
3. Edge cases
4. Integration with other components

Run with: pytest tests/test_[module].py -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from [module] import [Component]


class TestComponentName:
    """Test suite for [Component]."""

    def setup_method(self):
        """Set up test fixtures."""
        self.component = Component()

    def teardown_method(self):
        """Clean up after tests."""
        pass

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = create_test_data()

        # Act
        result = self.component.process(input_data)

        # Assert
        assert result is not None
        assert result.property == expected_value

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ExpectedException):
            self.component.invalid_operation()

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty input
        result = self.component.process([])
        assert result == expected_empty_result

        # None input
        result = self.component.process(None)
        assert result == expected_none_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Common Test Patterns

**1. Arrange-Act-Assert (AAA)**
```python
def test_consensus_calculation():
    """Test consensus calculation."""
    # Arrange: Set up test data
    beliefs = [
        {'option_a': 0.7, 'option_b': 0.3},
        {'option_a': 0.6, 'option_b': 0.4}
    ]

    # Act: Execute the operation
    consensus = calculate_consensus(beliefs)

    # Assert: Verify results
    assert consensus['option_a'] > consensus['option_b']
    assert 0.6 <= consensus['option_a'] <= 0.7
```

**2. Parametrized Tests**
```python
@pytest.mark.parametrize("input_val,expected", [
    (0.5, 'medium'),
    (0.8, 'high'),
    (0.2, 'low'),
])
def test_confidence_levels(input_val, expected):
    """Test confidence level classification."""
    result = classify_confidence(input_val)
    assert result == expected
```

**3. Fixtures**
```python
@pytest.fixture
def sample_scenario():
    """Provide sample scenario for testing."""
    return {
        'id': 'test_scenario',
        'title': 'Test Scenario',
        'alternatives': [
            {'id': 'alt_1', 'name': 'Option 1'},
            {'id': 'alt_2', 'name': 'Option 2'}
        ],
        'criteria': {'political': 0.5, 'economic': 0.5}
    }

def test_scenario_processing(sample_scenario):
    """Test scenario processing using fixture."""
    result = process_scenario(sample_scenario)
    assert result is not None
```

**4. Mocking**
```python
from unittest.mock import Mock, patch

def test_agent_with_mock_llm():
    """Test agent with mocked LLM."""
    mock_response = {'scores': {'criterion': 0.8}}

    with patch('llm_integration.claude_client.ClaudeClient') as mock:
        mock.return_value.evaluate.return_value = mock_response

        agent = ExpertAgent(profile, config)
        result = agent.evaluate_alternative(alternative, scenario)

        assert result['scores'] == mock_response['scores']
```

---

## Mocking Strategies

### Mock LLM Calls

**Why Mock LLMs?**
- Avoid API costs during testing
- Tests run faster
- Tests are deterministic
- Tests work offline

**Mock Claude Client:**
```python
from unittest.mock import Mock, patch

@pytest.fixture
def mock_claude_client():
    """Mock Claude client for testing."""
    with patch('llm_integration.claude_client.ClaudeClient') as mock:
        mock_instance = mock.return_value
        mock_instance.evaluate_alternative.return_value = {
            'scores': {'political': 0.8, 'economic': 0.6},
            'reasoning': 'Mock reasoning',
            'confidence': 0.9
        }
        yield mock_instance

def test_expert_agent_evaluation(mock_claude_client):
    """Test expert agent with mocked LLM."""
    agent = ExpertAgent(profile, config, llm_client=mock_claude_client)
    result = agent.evaluate_alternative(alternative, scenario)

    assert result['scores']['political'] == 0.8
    assert mock_claude_client.evaluate_alternative.called
```

### Mock File I/O

**Temporary Files:**
```python
import tempfile
import json
from pathlib import Path

def test_scenario_file_loading():
    """Test scenario loading from file."""
    scenario_data = {
        'id': 'test_scenario',
        'title': 'Test',
        'alternatives': [],
        'criteria': {}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(scenario_data, f)
        temp_path = f.name

    try:
        loaded = load_scenario(temp_path)
        assert loaded['id'] == 'test_scenario'
    finally:
        Path(temp_path).unlink()
```

**Mock Path Operations:**
```python
from unittest.mock import patch, mock_open

def test_config_file_reading():
    """Test config file reading with mock."""
    mock_config_data = '{"agents": {"max_agents": 10}}'

    with patch('builtins.open', mock_open(read_data=mock_config_data)):
        config = Config('config.json')
        assert config.get('agents.max_agents') == 10
```

### Mock External Services

**Mock HTTP Requests:**
```python
from unittest.mock import patch
import requests

def test_external_api_call():
    """Test external API call with mock."""
    mock_response = Mock()
    mock_response.json.return_value = {'data': 'test'}
    mock_response.status_code = 200

    with patch('requests.get', return_value=mock_response):
        result = fetch_external_data('http://api.example.com')
        assert result['data'] == 'test'
```

---

## Best Practices

### Test Naming

✅ **Good:**
```python
def test_consensus_model_detects_agreement_when_beliefs_converge():
    """Test that consensus model correctly detects agreement."""
    pass

def test_agent_raises_validation_error_when_profile_invalid():
    """Test agent validation with invalid profile."""
    pass
```

❌ **Bad:**
```python
def test1():
    """Test."""
    pass

def test_agent():
    """Test agent."""
    pass
```

### Test Structure

✅ **Good: Clear AAA structure**
```python
def test_mcda_scoring():
    """Test MCDA scoring calculation."""
    # Arrange
    criteria = {'political': 0.5, 'economic': 0.5}
    scores = {'political': 0.8, 'economic': 0.6}

    # Act
    result = calculate_weighted_score(criteria, scores)

    # Assert
    assert 0.6 <= result <= 0.8
    assert result == pytest.approx(0.7, abs=0.01)
```

❌ **Bad: Mixed setup and assertions**
```python
def test_mcda_scoring():
    """Test MCDA."""
    criteria = {'political': 0.5, 'economic': 0.5}
    assert criteria is not None
    scores = {'political': 0.8, 'economic': 0.6}
    result = calculate_weighted_score(criteria, scores)
    assert result > 0
    assert result < 1
```

### Test Independence

✅ **Good: Each test is independent**
```python
def test_agent_creation():
    """Test agent creation."""
    agent = create_agent(profile)
    assert agent is not None

def test_agent_evaluation():
    """Test agent evaluation."""
    agent = create_agent(profile)
    result = agent.evaluate(alternative)
    assert result is not None
```

❌ **Bad: Tests depend on each other**
```python
agent = None

def test_agent_creation():
    """Test agent creation."""
    global agent
    agent = create_agent(profile)
    assert agent is not None

def test_agent_evaluation():
    """Test agent evaluation - depends on previous test!"""
    global agent
    result = agent.evaluate(alternative)  # Fails if test_agent_creation didn't run
    assert result is not None
```

### Assertions

✅ **Good: Specific, meaningful assertions**
```python
def test_belief_update():
    """Test belief update calculation."""
    old_belief = 0.5
    new_evidence = 0.8
    rate = 0.3

    updated = update_belief(old_belief, new_evidence, rate)

    # Specific assertions with context
    assert isinstance(updated, float)
    assert 0.0 <= updated <= 1.0
    assert updated > old_belief  # Should move toward new evidence
    assert abs(updated - 0.59) < 0.01  # Expected value
```

❌ **Bad: Vague or missing assertions**
```python
def test_belief_update():
    """Test belief update."""
    result = update_belief(0.5, 0.8, 0.3)
    assert result  # What does this even test?
```

### Documentation

✅ **Good: Clear docstrings and comments**
```python
def test_consensus_with_outlier():
    """
    Test consensus detection with outlier agent.

    Scenario:
    - 4 agents with beliefs around 0.7-0.8
    - 1 outlier agent with belief 0.2

    Expected:
    - Consensus should be detected among majority
    - Outlier should be identified
    """
    # Setup majority and outlier
    beliefs = [0.7, 0.75, 0.8, 0.78, 0.2]

    # Calculate consensus
    result = detect_consensus(beliefs, threshold=0.7)

    # Verify majority consensus
    assert result['consensus_reached'] is True
    assert result['consensus_value'] > 0.7
    assert result['outliers'] == [4]  # Index of outlier
```

---

## Continuous Integration

### GitHub Actions Configuration

Create `.github/workflows/tests.yml`:

```yaml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests with coverage
      run: |
        pytest tests/ -v --cov=. --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

    - name: Check coverage threshold
      run: |
        pytest tests/ --cov=. --cov-fail-under=80
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        args: [tests/, -v]
        language: system
        pass_filenames: false
        always_run: true
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:**
```
ImportError: No module named 'agents'
```

**Solution:**
Ensure project root is in `sys.path`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

#### 2. Tests Pass Individually, Fail Together

**Problem:**
Tests work when run alone but fail when run with others.

**Causes:**
- Shared global state
- Test interdependencies
- Resource conflicts (temp files, ports)

**Solutions:**
```python
# Use fixtures for isolation
@pytest.fixture
def clean_state():
    """Provide clean state for each test."""
    state = create_clean_state()
    yield state
    cleanup(state)

# Reset global state in teardown
def teardown_method(self):
    """Clean up after each test."""
    reset_global_state()
```

#### 3. LLM API Test Failures

**Problem:**
Tests fail with API errors or rate limits.

**Solution:**
Mock LLM clients:
```python
@pytest.fixture(autouse=True)
def mock_all_llm_clients(monkeypatch):
    """Automatically mock all LLM clients."""
    def mock_evaluate(*args, **kwargs):
        return {'scores': {}, 'reasoning': 'Mock'}

    monkeypatch.setattr(
        'llm_integration.claude_client.ClaudeClient.evaluate',
        mock_evaluate
    )
```

#### 4. Slow Test Execution

**Problem:**
Test suite takes too long to run.

**Solutions:**
```python
# Mark slow tests
@pytest.mark.slow
def test_full_integration():
    pass

# Run only fast tests
pytest tests/ -m "not slow"

# Parallelize with pytest-xdist
pip install pytest-xdist
pytest tests/ -n auto
```

#### 5. Flaky Tests

**Problem:**
Tests sometimes pass, sometimes fail.

**Causes:**
- Race conditions
- Non-deterministic LLM outputs
- Time-dependent assertions

**Solutions:**
```python
# Use retries for flaky tests
@pytest.mark.flaky(reruns=3)
def test_sometimes_fails():
    pass

# Set random seeds
import random
random.seed(42)

# Use approximate assertions
assert result == pytest.approx(0.7, abs=0.1)
```

---

## Summary

The Crisis MAS test suite provides comprehensive validation across all system components:

- **15 test files** covering unit, integration, and error scenarios
- **Multiple test types** for thorough coverage
- **Best practices** for maintainable, reliable tests
- **Mocking strategies** to isolate components
- **CI/CD integration** for automated quality assurance

**Key Principles:**
1. Test early, test often
2. Mock external dependencies
3. Keep tests fast and isolated
4. Document test intentions
5. Maintain high coverage

**Running Tests:**
```bash
# Quick check
pytest tests/ -v

# Full coverage report
pytest tests/ --cov=. --cov-report=html
```

For questions or issues with tests, refer to test docstrings and the main project documentation.
