# Test Suite

Comprehensive test suite with 700+ tests covering all framework components including validation, resilience, multi-agent orchestration, and security features.

## Structure

```
tests/
├── conftest.py                              # Shared fixtures and test utilities
├── mock_provider.py                         # MockLLMProvider for testing
├── test_core.py                             # Core types and utilities (70+ tests)
├── test_validation.py                       # Validation system (40+ tests)
├── test_storage.py                          # RocksDB storage (45+ tests)
├── test_context.py                          # Context versioning (65+ tests)
├── test_events.py                           # Event system (85+ tests)
├── test_patterns.py                         # Pattern extraction (40+ tests)
├── test_tools.py                            # Tool execution (40+ tests)
├── test_agent.py                            # Agent execution (90+ tests)
├── test_agent_streaming_integration.py      # Streaming workflows (20+ tests)
├── test_logic.py                            # Logic control flow (50+ tests)
├── test_logic_evaluation_points.py          # Logic evaluation points (15+ tests)
├── test_resilience.py                       # Retry and rate limiting (60+ tests)
├── test_multi_agent.py                      # Multi-agent coordination (60+ tests)
├── test_multi_prompt.py                     # Multi-prompt and PromptObject (50+ tests)
└── test_security_fixes.py                   # Security features (15+ tests)
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
| **core.py** | test_core.py | ProcessingMode, SegmentType, AgentStatus enums; ToolCall, ToolResult, AgentConfig, PromptObject dataclasses; serialize_tool_output, output_to_string, now_timestamp, new_uuid |
| **validation.py** | test_validation.py | ValidatorRegistry, simple_validator, passthrough_validator, custom validators, required fields, type checking, string/numeric constraints |
| **storage.py** | test_storage.py | RocksDBStorage initialization, CRUD operations, DB identification and validation, path resolution with collision avoidance, prefix iteration, performance characteristics |
| **context.py** | test_context.py | IterationManager, ContextManager, versioning, history tracking with limits, tombstone deletion, update() for incremental writes, list_keys with prefix filtering |
| **events.py** | test_events.py | All 16 event types (LLM, tool, pattern, validation, retry, rate limit, context health), timestamps, step_id tracking, explicit vs auto timestamps |
| **patterns.py** | test_patterns.py | Pattern and PatternSet, PatternRegistry, PatternExtractor (batch), StreamingPatternExtractor (incremental with buffer limits), tool call parsing (JSON/line format), greedy vs non-greedy matching |
| **tools.py** | test_tools.py | Tool execution modes (PROCESS/THREAD/ASYNC), timeouts, streaming support, ToolRegistry operations, create_tool helper, error handling for different exception types |
| **agent.py** | test_agent.py, test_agent_streaming_integration.py | Agent configuration, AgentRunner batch/streaming, tool execution (allowed/not allowed/not found), context updates, pattern extraction, event ordering, lifecycle events |
| **logic.py** | test_logic.py, test_logic_evaluation_points.py | LogicConfig and LogicCondition, max iterations, stop/loop-until conditions, break_on_error, evaluation points (llm_token, tool_detected, tool_finished, any_event, pattern_start, step_complete), helper functions |
| **resilience.py** | test_resilience.py | RetryConfig with backoff strategies (exponential/linear/constant), jitter, RateLimiter token bucket, retry_stream, rate_limited_stream, resilient_stream, event emission |
| **multi_agent.py** | test_multi_agent.py | AgentChain with pass modes (response/full_context/tool_results), custom transform functions, SupervisorPattern with delegation, ParallelPattern with concurrent execution and merge strategies, DebatePattern with consensus |
| **multi_prompt.py** | test_multi_prompt.py | PromptObject dataclass, create_message_prompt_builder, role-based routing (system/user/assistant), multiple system entries, literal prefix, user_input handling |
| **Security** | test_security_fixes.py | Buffer overflow protection in StreamingPatternExtractor (default 10MB limit), context history default limit (100 versions), custom buffer and history sizes |

## Fixtures (conftest.py)

- `temp_dir` - Temporary directory per test with automatic cleanup
- `storage_config` - StorageConfig with temporary directory
- `storage` - Initialized RocksDBStorage with unique DB per test
- `iteration_manager` - IterationManager instance
- `context_manager` - ContextManager with IterationManager
- `pattern_registry` - PatternRegistry with default patterns registered
- `tool_registry` - ToolRegistry with echo and calculator test tools
- `mock_llm_provider` - MockLLMProvider with configurable responses and streaming simulation
- `agent_config` - AgentConfig with tools_allowed, input/output mappings, pattern_set
- `agent` - Fully configured Agent instance with all dependencies
- `agent_runner` - AgentRunner instance for batch and streaming execution
- `sample_pattern_set` - Custom PatternSet for testing pattern variations

## Test Categories

**Unit Tests**: Individual functions/classes in isolation (< 1s per test)
- Enums, dataclasses, utility functions (ProcessingMode, SegmentType, AgentStatus, ToolCall, ToolResult, PromptObject)
- Event creation with timestamps and step IDs
- Validator functions (simple_validator, passthrough_validator)
- Backoff calculations (exponential, linear, constant with jitter)
- Pattern and PatternSet creation

**Integration Tests**: Component interactions (1-5s per test)
- Agent execution with tool calls, validation, and context updates
- Logic runner with stop/loop-until conditions and evaluation points
- Context versioning, history tracking, and tombstone deletion
- Storage CRUD operations with prefix iteration
- Multi-agent patterns (AgentChain, SupervisorPattern, ParallelPattern, DebatePattern)
- Resilient streams combining retry and rate limiting

**Async Tests**: Marked with `@pytest.mark.asyncio`
- Streaming pattern extraction with incremental events and buffer limits
- Tool streaming (run_stream) for tools that support incremental output
- Agent streaming (step_stream) with event ordering and pattern lifecycle
- Logic streaming (run_stream) with evaluation at different points
- Multi-agent parallel execution with timeouts
- Rate limited async operations with token bucket

**Security Tests**: Protection mechanisms in `test_security_fixes.py`
- Buffer overflow protection in StreamingPatternExtractor (default 10MB, configurable)
- Context history limits (default 100 versions, configurable)
- Validation enforcement for tool arguments

## Coverage Metrics

- **Overall**: >90% across all modules
- **Critical paths**: 100% (storage, context, agent execution, validation)
- **Total**: 700+ tests, ~10000 lines of test code
- **Core modules**: Validation (100%), Resilience (95%), Multi-agent (92%), Security (100%)

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
1. One test, one concept - each test should verify a single behavior
2. Arrange-Act-Assert structure for clarity
3. Descriptive names that describe what is tested (not how)
4. Use fixtures for common setup to reduce duplication
5. Test edge cases, boundary conditions, and error paths
6. Use `@pytest.mark.asyncio` for async tests
7. Clean up resources properly (fixtures handle this automatically)
8. Keep tests independent and idempotent - no shared state between tests
9. Verify not just success but correct behavior (check return values, events, state changes)

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
