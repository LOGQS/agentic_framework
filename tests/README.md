# Agentic Framework Test Suite

Comprehensive test suite for the agentic framework with high code coverage and thorough validation of all components.

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures and test utilities
├── pytest.ini            # Pytest configuration
├── __init__.py          # Package initialization
├── test_core.py         # Core types, enums, and utilities (475+ lines)
├── test_storage.py      # RocksDB storage layer (432+ lines)
├── test_context.py      # Context management and versioning (545+ lines)
├── test_events.py       # Event system (367+ lines)
├── test_patterns.py     # Pattern extraction (350+ lines)
├── test_tools.py        # Tools system (332+ lines)
├── test_agent.py        # Agent execution (315+ lines)
└── test_logic.py        # Logic flow control (415+ lines)
```

## Installation

Install testing dependencies:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_core.py
pytest tests/test_storage.py
```

### Run specific test class or function
```bash
pytest tests/test_core.py::TestProcessingMode
pytest tests/test_storage.py::TestRocksDBStorageCRUD::test_put_and_get
```

### Run tests with coverage
```bash
pytest --cov=agentic --cov-report=html --cov-report=term-missing
```

### Run tests in parallel
```bash
pytest -n auto
```

### Run only asyncio tests
```bash
pytest -m asyncio
```

### Run tests excluding slow tests
```bash
pytest -m "not slow"
```

### Verbose output with local variables
```bash
pytest -vv --showlocals
```

## Test Coverage

### Module Coverage Summary

| Module | Test File | Coverage Areas |
|--------|-----------|----------------|
| **core.py** | test_core.py | Enums (ProcessingMode, SegmentType, AgentStatus), Data classes (ToolCall, ToolResult, ExtractedSegments, AgentStepResult, AgentConfig), Utility functions (now_timestamp, new_uuid) |
| **storage.py** | test_storage.py | StorageConfig, RocksDBStorage initialization, CRUD operations, Iteration with prefix, Database identification, Path resolution, Error handling |
| **context.py** | test_context.py | IterationManager, ContextManager, Versioning, History tracking, Tombstone deletion, List keys, Clear operations |
| **events.py** | test_events.py | BaseEvent, All 12 event types, Timestamp handling, Event type identifiers |
| **patterns.py** | test_patterns.py | Pattern/PatternSet, PatternRegistry, PatternExtractor (batch), StreamingPatternExtractor, Tool parsing, Malformed patterns, Greedy matching |
| **tools.py** | test_tools.py | ToolDefinition, Tool execution modes, Timeouts, Streaming, ToolRegistry, create_tool helper, Error handling |
| **agent.py** | test_agent.py | Agent configuration, AgentRunner (batch/streaming), Tool execution, Context updates, Pattern extraction, MockLLMProvider, Event emission |
| **logic.py** | test_logic.py | LogicConfig/Condition, LogicRunner loops, Max iterations, Stop/loop-until conditions, Helper functions, Conditional evaluation |

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution (< 1 second per test)
- Examples: test_core.py, test_events.py

### Integration Tests
- Test interactions between components
- Use real RocksDB storage (temporary directories)
- Test agent execution with tools
- Examples: test_agent.py, test_logic.py

### Async Tests
- Tests marked with `@pytest.mark.asyncio`
- Test streaming execution
- Test async tool execution
- Examples: Streaming tests in test_agent.py, test_tools.py

### Performance Tests
- Marked with `@pytest.mark.performance`
- Test batch operations
- Validate timeout behavior
- Examples: TestStoragePerformance, TestToolTimeout

## Key Testing Patterns

### Fixtures
Common fixtures in `conftest.py`:
- `temp_dir`: Temporary directory for storage
- `storage`: Initialized RocksDB storage
- `context_manager`: Context manager with iteration tracking
- `pattern_registry`: Pattern registry with default patterns
- `tool_registry`: Tool registry with sample tools
- `agent`: Fully configured agent instance
- `agent_runner`: Agent runner for execution
- `mock_llm_provider`: Mock LLM for testing

### Test Organization
Each test file follows this structure:
1. **Basic Tests**: Creation, initialization, simple operations
2. **Feature Tests**: Core functionality and behavior
3. **Integration Tests**: Component interactions
4. **Error Handling Tests**: Error cases and edge conditions
5. **Edge Case Tests**: Boundary conditions and unusual inputs

### Naming Conventions
- Test files: `test_<module>.py`
- Test classes: `Test<Component>`
- Test functions: `test_<what_is_being_tested>`
- Fixtures: Descriptive names matching the component

## Coverage Goals

Target coverage levels:
- **Overall**: > 90%
- **Core modules**: > 95%
- **Critical paths**: 100% (storage, context, agent execution)

### Current Coverage Highlights

**Fully Covered Areas**:
- All enums and data classes
- Storage CRUD operations
- Context versioning and history
- Event creation and timestamps
- Pattern extraction (batch and streaming)
- Tool execution modes
- Agent step execution

**Comprehensive Error Testing**:
- Uninitialized storage operations
- Invalid database IDs
- Malformed patterns
- Tool timeouts and failures
- LLM generation errors
- Logic loop conditions

## Running Specific Test Suites

### Storage Layer Tests
```bash
pytest tests/test_storage.py -v
```

### Agent and Logic Tests
```bash
pytest tests/test_agent.py tests/test_logic.py -v
```

### Only Fast Tests (< 1 second)
```bash
pytest -m "not slow"
```

### Only Async Tests
```bash
pytest -m asyncio
```

## Debugging Tests

### Run with print statements visible
```bash
pytest -s
```

### Stop on first failure
```bash
pytest -x
```

### Drop into debugger on failure
```bash
pytest --pdb
```

### Run last failed tests
```bash
pytest --lf
```

### Run failed tests first
```bash
pytest --ff
```

## Test Data and Cleanup

### Temporary Storage
- All storage tests use temporary directories
- Automatically cleaned up after each test
- Fixtures handle initialization and cleanup

### No Test Pollution
- Each test is isolated
- Fixtures ensure clean state
- No shared state between tests

## Continuous Integration

Recommended CI configuration:

```yaml
- name: Run Tests
  run: |
    pip install -r requirements-test.txt
    pytest --cov=agentic --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Writing New Tests

### Template for New Test File

```python
"""
Tests for <module_name>.

Covers:
- Feature 1
- Feature 2
- Error handling
"""
import pytest

from agentic.<module> import ComponentToTest


class TestComponent:
    """Tests for Component."""

    def test_basic_functionality(self):
        """Test basic component behavior."""
        component = ComponentToTest()
        assert component.method() == expected_value

    def test_error_handling(self):
        """Test component error handling."""
        with pytest.raises(ValueError):
            ComponentToTest().invalid_operation()
```

### Guidelines
1. **One test, one assertion focus**: Each test should validate one specific behavior
2. **Descriptive names**: Test names should clearly indicate what is being tested
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use fixtures**: Leverage existing fixtures for common setup
5. **Test edge cases**: Include boundary conditions and unusual inputs
6. **Document intent**: Use docstrings to explain what each test validates

## Troubleshooting

### RocksDB Issues
If you encounter RocksDB-related errors:
```bash
# Reinstall rocksdict
pip uninstall rocksdict
pip install rocksdict --no-cache-dir
```

### Async Test Failures
Ensure pytest-asyncio is installed:
```bash
pip install pytest-asyncio>=0.21.0
```

### Import Errors
Make sure you're running from the project root:
```bash
cd /path/to/agentic_framework
pytest tests/
```

## Test Metrics

### Test Count Summary
- **test_core.py**: 50+ tests
- **test_storage.py**: 45+ tests
- **test_context.py**: 55+ tests
- **test_events.py**: 35+ tests
- **test_patterns.py**: 35+ tests
- **test_tools.py**: 35+ tests
- **test_agent.py**: 30+ tests
- **test_logic.py**: 40+ tests

**Total**: 325+ comprehensive tests

### Execution Time
- Full suite: ~10-30 seconds (depending on hardware)
- Unit tests only: ~5 seconds
- With parallel execution (-n auto): ~5-10 seconds

## Contributing

When adding new features:
1. Write tests first (TDD approach recommended)
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov=agentic`
4. Run linting: `ruff check .` (if configured)
5. Update this README if adding new test files

## Questions and Support

For questions about the test suite:
- Review test file docstrings for detailed coverage information
- Check conftest.py for available fixtures
- Refer to pytest documentation for advanced usage
