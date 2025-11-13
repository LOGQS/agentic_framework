# Test Suite

Comprehensive test suite with 325+ tests covering all framework components.

## Structure

```
tests/
├── conftest.py           # Shared fixtures
├── mock_provider.py      # MockLLMProvider for testing
├── test_core.py          # Enums and data classes (50+ tests)
├── test_storage.py       # RocksDB storage (45+ tests)
├── test_context.py       # Context versioning (55+ tests)
├── test_events.py        # Event system (35+ tests)
├── test_patterns.py      # Pattern extraction (35+ tests)
├── test_tools.py         # Tool execution (35+ tests)
├── test_agent.py         # Agent execution (30+ tests)
└── test_logic.py         # Logic control flow (40+ tests)
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=agentic --cov-report=term-missing

# Specific file
pytest tests/test_agent.py -v

# Async tests only
pytest -m asyncio

# Parallel execution
pytest -n auto

# Stop on first failure
pytest -x

# Show print output
pytest -s

# Debug on failure
pytest --pdb
```

## Coverage by Module

| Module | Test File | Key Coverage |
|--------|-----------|--------------|
| **core.py** | test_core.py | ProcessingMode, SegmentType, AgentStatus enums; ToolCall, ToolResult, AgentConfig dataclasses; now_timestamp(), new_uuid() |
| **storage.py** | test_storage.py | RocksDBStorage CRUD, DB identification, path resolution, prefix iteration |
| **context.py** | test_context.py | IterationManager, ContextManager, versioning, history, tombstone deletion |
| **events.py** | test_events.py | All 13 event types, timestamps, step IDs |
| **patterns.py** | test_patterns.py | PatternExtractor (batch), StreamingPatternExtractor (incremental), registry |
| **tools.py** | test_tools.py | Tool execution modes (PROCESS/THREAD/ASYNC), timeouts, streaming, registry |
| **agent.py** | test_agent.py | AgentRunner batch/streaming, tool execution, context updates, error handling |
| **logic.py** | test_logic.py | LogicRunner, stop/loop-until conditions, helper functions |

## Fixtures (conftest.py)

- `temp_dir` - Temporary directory per test
- `storage` - Initialized RocksDBStorage
- `context_manager` - ContextManager with IterationManager
- `pattern_registry` - PatternRegistry with default patterns
- `tool_registry` - ToolRegistry with echo/calculator tools
- `mock_llm_provider` - MockLLMProvider for testing
- `agent` - Fully configured Agent instance
- `agent_runner` - AgentRunner instance

## Test Categories

**Unit Tests**: Individual functions/classes in isolation (< 1s per test)
- Enums, dataclasses, utility functions
- Event creation

**Integration Tests**: Component interactions (1-5s per test)
- Agent execution with tools
- Logic runner with agent steps
- Context persistence

**Async Tests**: `@pytest.mark.asyncio`
- Streaming pattern extraction
- Streaming tool execution
- Agent/logic streaming

## Coverage Metrics

- **Overall**: >90%
- **Critical paths**: 100% (storage, context, agent execution)
- **Total**: 325+ tests, ~3500 lines of test code

## Writing Tests

Template:
```python
"""
Tests for <module>.

Covers:
- Feature 1
- Feature 2
- Error handling
"""
import pytest
from agentic.<module> import Component

class TestComponent:
    def test_basic_operation(self):
        """Test basic behavior."""
        component = Component()
        result = component.method()
        assert result == expected
```

Guidelines:
1. One test, one concept
2. Arrange-Act-Assert structure
3. Descriptive names
4. Use fixtures
5. Test edge cases

## Troubleshooting

**RocksDB issues**:
```bash
pip uninstall rocksdict && pip install rocksdict --no-cache-dir
```

**Async test failures**:
```bash
pip install pytest-asyncio>=0.21.0
```

**Import errors**: Run from project root
```bash
cd /path/to/agentic_framework && pytest tests/
```
