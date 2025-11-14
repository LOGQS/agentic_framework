# Test Suite

Comprehensive test suite with 550+ tests covering all framework components including validation, resilience, and multi-agent patterns.

## Structure

```
tests/
├── conftest.py                              # Shared fixtures
├── mock_provider.py                         # MockLLMProvider for testing
├── test_core.py                             # Enums and data classes (50+ tests)
├── test_validation.py                       # Validation system (40+ tests)
├── test_storage.py                          # RocksDB storage (45+ tests)
├── test_context.py                          # Context versioning (55+ tests)
├── test_events.py                           # Event system (60+ tests)
├── test_patterns.py                         # Pattern extraction (35+ tests)
├── test_tools.py                            # Tool execution (35+ tests)
├── test_agent.py                            # Agent execution (90+ tests)
├── test_agent_streaming_integration.py      # Agent streaming integration (40+ tests)
├── test_logic.py                            # Logic control flow (55+ tests)
├── test_logic_evaluation_points.py          # Logic evaluation points (15+ tests)
├── test_resilience.py                       # Retry and rate limiting (50+ tests)
├── test_multi_agent.py                      # Multi-agent patterns (50+ tests)
├── test_multi_prompt.py                     # Multi-prompt support (45+ tests)
└── test_security_fixes.py                   # Security enhancements (15+ tests)
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

| Module | Test Files | Key Coverage |
|--------|------------|--------------|
| **core.py** | test_core.py | ProcessingMode, SegmentType, AgentStatus enums; ToolCall, ToolResult, AgentConfig dataclasses; PromptObject support; now_timestamp(), new_uuid() |
| **validation.py** | test_validation.py | ValidatorRegistry, simple_validator, passthrough_validator, custom validators, error reporting |
| **storage.py** | test_storage.py | RocksDBStorage CRUD, DB identification, path resolution, prefix iteration |
| **context.py** | test_context.py | IterationManager, ContextManager, versioning, history, tombstone deletion, string value auto-encoding |
| **events.py** | test_events.py | All 16 event types (LLM, tools, patterns, retry, rate limit, health), timestamps, step IDs |
| **patterns.py** | test_patterns.py | PatternExtractor (batch), StreamingPatternExtractor (incremental), registry, malformed pattern handling |
| **tools.py** | test_tools.py | Tool execution modes (PROCESS/THREAD/ASYNC), timeouts, streaming, registry, validation integration |
| **agent.py** | test_agent.py, test_agent_streaming_integration.py, test_multi_prompt.py | AgentRunner batch/streaming, tool execution, context updates, error handling, PromptObject support, multi-prompt workflows |
| **logic.py** | test_logic.py, test_logic_evaluation_points.py | LogicRunner, stop/loop-until conditions, context health checks, evaluation points, helper functions |
| **resilience.py** | test_resilience.py | RetryConfig with backoff strategies, RateLimiter token bucket, retry_stream, rate_limited_stream, resilient_stream |
| **multi_agent.py** | test_multi_agent.py | AgentChain, SupervisorPattern, ParallelPattern, DebatePattern, configuration options |
| **Security** | test_security_fixes.py | Buffer overflow protection, context history limits, validation enforcement |

## Fixtures (conftest.py)

- `temp_dir` - Temporary directory per test (cleanup automatic)
- `storage` - Initialized RocksDBStorage with unique DB per test
- `iteration_manager` - IterationManager instance
- `context_manager` - ContextManager with IterationManager
- `pattern_registry` - PatternRegistry with default patterns
- `tool_registry` - ToolRegistry with echo/calculator/validation test tools
- `validator_registry` - ValidatorRegistry with built-in validators
- `mock_llm_provider` - MockLLMProvider for testing (configurable responses)
- `agent_config` - Pre-configured AgentConfig for testing
- `agent` - Fully configured Agent instance
- `agent_runner` - AgentRunner instance
- `logic_runner` - LogicRunner instance with test configuration

## Test Categories

**Unit Tests**: Individual functions/classes in isolation (< 1s per test)
- Enums, dataclasses, utility functions
- Event creation and timestamps
- Validator functions
- Retry backoff calculations

**Integration Tests**: Component interactions (1-5s per test)
- Agent execution with tools and validation
- Logic runner with context health monitoring
- Context persistence and versioning
- Multi-agent pattern orchestration
- Resilient streams with retry and rate limiting

**Async Tests**: `@pytest.mark.asyncio`
- Streaming pattern extraction with buffer limits
- Streaming tool execution
- Agent/logic streaming with event handling
- Multi-agent parallel execution
- Rate limited async operations

**Security Tests**: `test_security_fixes.py`
- Buffer overflow protection
- Context history limits
- Input validation enforcement

## Coverage Metrics

- **Overall**: >90% across all modules
- **Critical paths**: 100% (storage, context, agent execution, validation)
- **Total**: 550+ tests, ~9000 lines of test code
- **New modules**: Validation (100%), Resilience (95%), Multi-agent (92%)

## Writing Tests

Template:
```python
"""
Tests for <module>.

Covers:
- Feature 1
- Feature 2
- Error handling
- Edge cases
"""
import pytest
from agentic.<module> import Component

class TestComponent:
    def test_basic_operation(self, fixture_name):
        """Test basic behavior."""
        # Arrange
        component = Component()

        # Act
        result = component.method()

        # Assert
        assert result == expected

    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async behavior."""
        results = []
        async for item in component.stream():
            results.append(item)
        assert len(results) > 0
```

Guidelines:
1. One test, one concept
2. Arrange-Act-Assert structure
3. Descriptive names describing what is tested
4. Use fixtures for common setup
5. Test edge cases and error conditions
6. Use `@pytest.mark.asyncio` for async tests
7. Clean up resources (fixtures handle this automatically)
8. Keep tests independent and idempotent

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
