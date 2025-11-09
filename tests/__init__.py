"""
═══════════════════════════════════════════════════════════════════════════════
TESTS MODULE
Comprehensive test suite for Crisis MAS system validation and quality assurance
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
═════════
This module provides a comprehensive test suite for the Crisis MAS system,
ensuring:

1. **Unit Testing** - Individual component validation
2. **Integration Testing** - System-wide workflow validation
3. **Error Handling** - Robustness under failure conditions
4. **Performance** - Benchmarking and performance validation
5. **Regression Prevention** - Ensure changes don't break existing functionality


WHY THIS EXISTS
═══════════════
Testing is critical for multi-agent systems because:

1. **Complexity**
   - Multiple agents interacting asynchronously
   - LLM integration with non-deterministic outputs
   - Complex decision aggregation algorithms
   - Many failure modes and edge cases

2. **Reliability**
   - System must handle invalid data gracefully
   - Edge cases (zero agents, empty scenarios) must work
   - LLM failures shouldn't crash the system
   - Consensus must converge correctly

3. **Maintainability**
   - Regression tests ensure refactoring doesn't break features
   - Test suite documents expected behavior
   - Integration tests validate end-to-end workflows

4. **Confidence**
   - Developers can make changes knowing tests will catch issues
   - Stakeholders can trust system reliability
   - CI/CD pipeline ensures quality


TEST ORGANIZATION
═════════════════

The test suite is organized by test type:

1. **Unit Tests** (Component-Specific)
   ────────────────────────────────────
   - test_base_agent.py: BaseAgent class functionality
   - test_expert_agent.py: ExpertAgent reasoning and LLM integration
   - test_coordinator_agent.py: CoordinatorAgent aggregation logic
   - test_claude_client.py: Claude LLM client integration
   - test_multi_llm_clients.py: Multi-provider LLM support
   - test_prompt_templates.py: Prompt template generation
   - test_consensus_model.py: Consensus detection algorithms
   - test_mcda_engine.py: MCDA scoring and ranking
   - test_evidential_reasoning.py: Evidential reasoning aggregation
   - test_baseline.py: Baseline decision-making validation

2. **Integration Tests**
   ─────────────────────
   - test_integration.py: Full system end-to-end workflow
   - test_gat_integration.py: GAT (Graph Attention Network) integration
   - test_evaluation_fix.py: Evaluation module integration

3. **Error Scenario Tests**
   ────────────────────────
   - test_error_scenarios.py: Robustness under failure conditions

4. **Advanced Tests**
   ─────────────────
   - test_gat.py: Graph Attention Network functionality


TEST COVERAGE
═════════════

Module Coverage:
────────────────
✓ agents/ - Base agents, expert agents, coordinator agents
✓ llm_integration/ - Claude client, multi-provider, prompt templates
✓ decision_framework/ - Consensus models, MCDA, evidential reasoning
✓ scenarios/ - Scenario loading and validation
✓ utils/ - Validation, configuration, safe operations
✓ evaluation/ - Metrics calculation and evaluation

Test Types:
───────────
✓ Unit tests: Individual function/class testing
✓ Integration tests: End-to-end workflow validation
✓ Error tests: Robustness and edge case handling
✓ Performance tests: Benchmarking critical paths


RUNNING TESTS
═════════════

Using pytest (Recommended):
───────────────────────────
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_integration.py -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# Run specific test function
pytest tests/test_consensus_model.py::test_basic_consensus -v

# Run with detailed output
pytest tests/ -vv -s
```

Using Python directly:
──────────────────────
```bash
# Run individual test file
python tests/test_integration.py

# Run from project root
python -m tests.test_integration
```

Running from code:
──────────────────
```python
import pytest

# Run all tests programmatically
pytest.main(['tests/', '-v'])

# Run specific test
pytest.main(['tests/test_integration.py', '-v'])
```


WRITING NEW TESTS
═════════════════

Test File Structure:
────────────────────
```python
#!/usr/bin/env python3
\"\"\"
Test Module for [Component Name]

Tests:
1. [Test category 1]
2. [Test category 2]
...

Run with: pytest tests/test_[module].py -v
\"\"\"

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from [module] import [components]


def test_basic_functionality():
    \"\"\"Test basic functionality of [component].\"\"\"
    # Arrange
    component = Component()

    # Act
    result = component.do_something()

    # Assert
    assert result is not None
    assert result.property == expected_value


def test_error_handling():
    \"\"\"Test error handling in [component].\"\"\"
    component = Component()

    with pytest.raises(ExpectedException):
        component.invalid_operation()


def test_edge_cases():
    \"\"\"Test edge cases for [component].\"\"\"
    # Test with empty input
    result = component.process([])
    assert result == expected_empty_result

    # Test with None
    result = component.process(None)
    assert result == expected_none_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

Best Practices:
───────────────
1. **Descriptive Names**: test_function_should_behavior_when_condition()
2. **Arrange-Act-Assert**: Clear test structure
3. **One Assertion Focus**: Test one thing per test function
4. **Isolation**: Tests should not depend on each other
5. **Cleanup**: Use fixtures for setup/teardown
6. **Documentation**: Docstrings explaining what is tested


TEST FIXTURES
═════════════

Common fixtures for reuse:

```python
import pytest
from utils.config import Config
from agents.expert_agent import ExpertAgent


@pytest.fixture
def config():
    \"\"\"Provide test configuration.\"\"\"
    return Config()


@pytest.fixture
def sample_scenario():
    \"\"\"Provide sample scenario for testing.\"\"\"
    return {
        'id': 'test_scenario',
        'title': 'Test Scenario',
        'description': 'Test description',
        'alternatives': [
            {'id': 'alt_1', 'name': 'Option 1', 'description': 'First option'},
            {'id': 'alt_2', 'name': 'Option 2', 'description': 'Second option'}
        ],
        'criteria': {
            'political': 0.5,
            'economic': 0.5
        }
    }


@pytest.fixture
def mock_llm_client(monkeypatch):
    \"\"\"Provide mock LLM client for testing without API calls.\"\"\"
    class MockClient:
        def evaluate_alternative(self, *args, **kwargs):
            return {
                'scores': {'political': 0.8, 'economic': 0.6},
                'reasoning': 'Mock reasoning',
                'confidence': 0.9
            }

    return MockClient()
```


MOCKING STRATEGIES
══════════════════

Mock LLM Calls:
───────────────
```python
import pytest
from unittest.mock import Mock, patch

def test_agent_with_mock_llm(monkeypatch):
    \"\"\"Test agent behavior with mocked LLM.\"\"\"
    mock_response = {
        'scores': {'criterion1': 0.8},
        'reasoning': 'Test reasoning'
    }

    with patch('llm_integration.claude_client.ClaudeClient.evaluate') as mock:
        mock.return_value = mock_response

        agent = ExpertAgent(profile, config)
        result = agent.evaluate_alternative(alternative, scenario)

        assert result['scores'] == mock_response['scores']
```

Mock File I/O:
──────────────
```python
import tempfile
from pathlib import Path

def test_scenario_loading():
    \"\"\"Test scenario loading from file.\"\"\"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(scenario_data, f)
        temp_path = f.name

    try:
        scenario = load_scenario(temp_path)
        assert scenario['id'] == 'test_scenario'
    finally:
        Path(temp_path).unlink()
```


TEST CATEGORIES
═══════════════

1. **Unit Tests**
   Purpose: Test individual functions/classes in isolation
   Example: test_consensus_model.py tests consensus detection

2. **Integration Tests**
   Purpose: Test multiple components working together
   Example: test_integration.py tests full decision workflow

3. **Error Tests**
   Purpose: Test system robustness under failure conditions
   Example: test_error_scenarios.py tests invalid inputs

4. **Performance Tests**
   Purpose: Benchmark performance-critical operations
   Example: Measure consensus convergence time

5. **Regression Tests**
   Purpose: Ensure bugs don't reoccur
   Example: Test specific bug fixes


CONTINUOUS INTEGRATION
═══════════════════════

GitHub Actions Example:
───────────────────────
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```


TROUBLESHOOTING
═══════════════

Common Issues:
──────────────

1. **ImportError: No module named 'crisis_mas'**
   Solution: Ensure sys.path includes project root:
   ```python
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```

2. **Tests pass individually but fail when run together**
   Cause: Tests have shared state or dependencies
   Solution: Use fixtures, ensure proper cleanup, avoid global state

3. **LLM tests fail with API errors**
   Cause: Real API calls in tests
   Solution: Mock LLM clients, use environment variable to skip LLM tests

4. **Slow test execution**
   Cause: Too many integration tests or slow operations
   Solution: Use pytest markers to separate fast/slow tests:
   ```python
   @pytest.mark.slow
   def test_full_integration():
       pass

   # Run only fast tests
   pytest -m "not slow"
   ```

5. **Test failures in CI but pass locally**
   Cause: Environment differences
   Solution: Use Docker for consistent environments, check file paths


BEST PRACTICES
══════════════

1. **Test Naming**
   ✓ test_function_should_behavior_when_condition()
   ✓ test_agent_raises_error_when_invalid_profile()
   ✗ test1(), test2()

2. **Test Independence**
   ✓ Each test can run in isolation
   ✓ No dependencies between tests
   ✗ test2() depends on test1() running first

3. **Mock External Dependencies**
   ✓ Mock LLM API calls
   ✓ Mock file I/O where appropriate
   ✗ Make real API calls in tests

4. **Clear Assertions**
   ✓ assert result.score == 0.8
   ✓ assert len(agents) == 3
   ✗ assert result (too vague)

5. **Test Documentation**
   ✓ Docstrings explaining what is tested
   ✓ Comments for complex setup
   ✗ No documentation


RELATED MODULES
═══════════════

Testing Dependencies:
- pytest: Test framework
- pytest-cov: Coverage reporting
- unittest.mock: Mocking framework
- tempfile: Temporary file creation

Tested Modules:
- agents/: Agent classes and behaviors
- llm_integration/: LLM client integration
- decision_framework/: Decision algorithms
- scenarios/: Scenario management
- utils/: Utilities and validation
- evaluation/: Metrics and evaluation


VERSION HISTORY
═══════════════

Version 1.0.0 (Initial Release)
- Unit tests for all major components
- Integration tests for end-to-end workflows
- Error scenario tests for robustness
- GAT integration tests
- Baseline decision-making tests

Test Coverage Improvements:
- Added test_evaluation_fix.py for evaluation module
- Enhanced error scenario coverage
- Added multi-LLM provider tests
- Added prompt template tests

Future Enhancements:
- Increase code coverage to >90%
- Add performance benchmarking tests
- Add property-based testing with Hypothesis
- Add mutation testing for test quality
- Add visual regression tests for dashboards
"""
