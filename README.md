# Agentic Framework

An agentic framework with versioned context, persistent storage, streaming-first architecture, and multi-agent orchestration.

## Features

- **Streaming-First**: Real-time event streams for LLM generation, pattern detection, and tool execution
- **16 Event Types**: Comprehensive observability including LLM, tools, patterns, retry, rate limiting, and context health monitoring
- **Persistent Context**: Versioned key-value store in RocksDB with automatic timestamps and iteration tracking
- **Pattern Extraction**: Batch and streaming extraction with incremental detection and malformed pattern handling
- **Multi-Mode Tools**: Execute tools in PROCESS, THREAD, or ASYNC modes with timeout handling and streaming support
- **Validation System**: Format-agnostic validation (simple, JSON Schema, custom) with extensible validator registry
- **Resilience**: Built-in retry logic with exponential backoff and token bucket rate limiting
- **Multi-Agent Patterns**: Chain, Supervisor-Worker, Parallel, and Debate patterns for agent orchestration
- **Flexible Logic**: Conditional loops with pattern-based, regex-based, or context-based conditions and context health monitoring
- **Type-Safe**: Full type hints throughout the codebase

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and rocksdict >= 0.3.0

## Quick Start

```python
from agentic import (
    StorageConfig, RocksDBStorage, IterationManager, ContextManager,
    PatternRegistry, create_default_pattern_set,
    ToolRegistry, create_tool,
    AgentConfig, Agent, AgentRunner,
    ProcessingMode
)

# Initialize storage and context
config = StorageConfig(base_dir="./context", db_name_prefix="my_agent")
storage = RocksDBStorage(config)
storage.initialize()
iteration = IterationManager(storage)
context = ContextManager(storage, iteration)

# Register patterns
patterns = PatternRegistry(storage)
patterns.register_pattern_set(create_default_pattern_set())

# Register tools
tools = ToolRegistry()
def search_web(inputs: dict) -> dict:
    return {"results": f"Search results for: {inputs.get('query', '')}"}

tools.register(create_tool(
    name="search_web",
    func=search_web,
    input_schema={"query": "string"},
    timeout_seconds=30.0,
    processing_mode=ProcessingMode.THREAD
))

# Create LLM provider
class MyLLMProvider:
    def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        # Your LLM API call here
        pass

    async def stream(self, prompt: str, max_tokens: int, temperature: float, **kwargs):
        # Optional: stream tokens for real-time UIs
        text = self.generate(prompt, max_tokens, temperature, **kwargs)
        yield text

provider = MyLLMProvider()

# Configure and run agent (batch mode)
agent_config = AgentConfig(
    agent_id="assistant",
    provider="custom",
    model="gpt-4",
    tools_allowed=["search_web"],
    input_mapping=[("system_prompt", "prepend")],
    output_mapping=[("last_output", "set_latest")],
    pattern_set="default"
)

agent = Agent(agent_config, context, patterns, tools, provider)
runner = AgentRunner(agent)

# Batch execution
result = runner.step("Tell me about agentic systems")
print(f"Status: {result.status}, Response: {result.segments.response}")

# Streaming execution
import asyncio
from agentic import LLMTokenEvent, ToolStartEvent, StepCompleteEvent

async def stream_example():
    async for event in runner.step_stream("Your prompt"):
        if isinstance(event, LLMTokenEvent):
            print(event.token, end="", flush=True)
        elif isinstance(event, ToolStartEvent):
            print(f"\n[Tool: {event.tool_name}]")
        elif isinstance(event, StepCompleteEvent):
            return event.result

asyncio.run(stream_example())
```

## Architecture

```
agentic/
├── core.py         # Enums (ProcessingMode, SegmentType, AgentStatus) & data classes
├── validation.py   # Format-agnostic validation system with extensible validators
├── events.py       # 16 event types (LLM, tools, patterns, retry, rate limit, health)
├── storage.py      # RocksDBStorage with automatic path resolution and DB identification
├── context.py      # IterationManager & ContextManager with versioning and history
├── patterns.py     # PatternExtractor (batch) & StreamingPatternExtractor (incremental)
├── tools.py        # Tool execution with multi-mode support, timeouts, and streaming
├── agent.py        # Agent & AgentRunner with dual-mode execution (step/step_stream)
├── logic.py        # LogicRunner for conditional loops with context health monitoring
├── resilience.py   # Retry logic with backoff and token bucket rate limiting
└── multi_agent.py  # Multi-agent orchestration patterns (Chain, Supervisor, Parallel, Debate)
```

### Key Components

**Storage** (`storage.py`):
- RocksDB backend via `rocksdict`
- Automatic DB ID generation: `agentic_<hash>_<timestamp>_<uuid>`
- CRUD operations: `get()`, `put()`, `delete()`, `iterate()`

**Context** (`context.py`):
- `IterationManager`: Global iteration counter with `get()`, `next()`
- `ContextManager`: Versioned store with `set()`, `get()`, `delete()`, `list_keys()`, `get_history()`
- `ContextRecord`: `{value, iteration, timestamp, version}`
- Tombstone deletion for soft deletes

**Patterns** (`patterns.py`):
- `Pattern`: `{name, start_tag, end_tag, segment_type, greedy}`
- `PatternRegistry`: Persistent pattern set storage
- `PatternExtractor`: Batch extraction
- `StreamingPatternExtractor`: Incremental token processing with malformed pattern handling

**Tools** (`tools.py`):
- `Tool`: Executable with `run()` (batch) and `run_stream()` (streaming)
- `ToolRegistry`: `register()`, `get()`, `exists()`, `list()`, `unregister()`
- Multi-mode execution: PROCESS, THREAD, ASYNC
- Automatic timeout enforcement

**Agent** (`agent.py`):
- `Agent`: Container for config, context, patterns, tools, provider
- `AgentRunner`: `step()` (batch) and `step_stream()` (streaming)
- `AgentConfig`: Configuration with 15+ options including `incremental_context_writes`, `stream_pattern_content`, `on_tool_detected`, `concurrent_tool_execution`

**Logic** (`logic.py`):
- `LogicRunner`: Conditional loops with `run()` (batch) and `run_stream()` (streaming)
- `LogicCondition`: Pattern/regex/context-based with flexible evaluation points
- `ContextHealthCheck`: Monitor context size, version count, and growth rate
- Helpers: `loop_n_times()`, `loop_until_pattern()`, `loop_until_regex()`, `stop_on_error()`

**Validation** (`validation.py`):
- `ValidatorRegistry`: Extensible registry for any validation format
- `ValidationError`: Structured error with field, message, and value
- Built-in validators: `simple_validator` (type checking, constraints), `passthrough_validator`
- Support for JSON Schema, XML Schema, Protocol Buffers, or custom validators

**Events** (`events.py`):
16 event types: `LLMTokenEvent`, `LLMCompleteEvent`, `PatternStartEvent`, `PatternContentEvent`, `PatternEndEvent`, `StatusEvent`, `ToolStartEvent`, `ToolOutputEvent`, `ToolEndEvent`, `ToolValidationEvent`, `ContextWriteEvent`, `ErrorEvent`, `StepCompleteEvent`, `RetryEvent`, `RateLimitEvent`, `ContextHealthEvent`

**Resilience** (`resilience.py`):
- `RetryConfig`: Exponential/linear/constant backoff with jitter
- `RateLimiter`: Token bucket algorithm with per-second/minute/hour limits
- `retry_stream()`: Universal retry wrapper for any async iterator
- `rate_limited_stream()`: Universal rate limiting wrapper
- `resilient_stream()`: Combined retry + rate limiting

**Multi-Agent** (`multi_agent.py`):
- `AgentChain`: Sequential execution with configurable output passing
- `SupervisorPattern`: Supervisor delegates tasks to specialized workers
- `ParallelPattern`: Parallel execution with result merging
- `DebatePattern`: Multi-round debate with consensus detection

## Advanced Usage

### Context Versioning

```python
# Set creates new version
context.set("key", b"value1")  # version 1
context.set("key", b"value2")  # version 2

# Get latest or specific version
latest = context.get("key")
v1 = context.get("key", version=1)

# History
history = context.get_history("key", max_versions=10)

# Delete (creates tombstone)
context.delete("key")
assert context.get("key") is None
```

### Custom Patterns

```python
from agentic import Pattern, PatternSet, SegmentType

custom = PatternSet(
    name="custom",
    patterns=[
        Pattern("thought", "<thought>", "</thought>", SegmentType.REASONING, greedy=False),
        Pattern("action", "<action>", "</action>", SegmentType.TOOL, greedy=False)
    ],
    default_response_behavior="all_remaining"
)
patterns.register_pattern_set(custom)
```

### Logic Control

```python
from agentic import LogicConfig, LogicCondition, LogicRunner

logic_config = LogicConfig(
    logic_id="main_loop",
    max_iterations=10,
    stop_conditions=[
        LogicCondition("default", "DONE", "regex", "response", "llm_complete")
    ],
    break_on_error=True
)

logic = LogicRunner(runner, context, patterns, logic_config)
results = logic.run("Analyze this problem")

# Streaming
async for event in logic.run_stream("Analyze this problem"):
    if isinstance(event, StepCompleteEvent):
        print(f"Iteration {event.result.iteration} complete")
```

### Tool Validation

```python
from agentic import ValidatorRegistry, ValidationError

# Built-in simple validator
tools.register(create_tool(
    name="calculate",
    func=calculate_fn,
    input_schema={
        "validator": "simple",
        "fields": {
            "x": {"type": "number", "required": True},
            "y": {"type": "number", "required": True},
            "op": {"type": "string", "enum": ["+", "-", "*", "/"]}
        }
    }
))

# Custom validator
def my_validator(value: Any, schema: dict) -> tuple[bool, list[ValidationError]]:
    # Your validation logic
    return True, []

registry = ValidatorRegistry()
registry.register("my_format", my_validator)
```

### Retry and Rate Limiting

```python
from agentic import RetryConfig, RateLimitConfig, RateLimiter, resilient_stream

# Retry configuration
retry_config = RetryConfig(
    max_attempts=3,
    backoff="exponential",  # or "linear", "constant"
    base_delay=1.0,
    max_delay=60.0,
    jitter=True,
    retry_on=(TimeoutError, ConnectionError)
)

# Rate limiting
rate_config = RateLimitConfig(
    requests_per_second=10,
    requests_per_minute=100,
    burst_size=20
)
limiter = RateLimiter(rate_config)

# Combined resilient stream
async def my_llm_call():
    async for token in provider.stream(prompt):
        yield token

async for item in resilient_stream(
    my_llm_call,
    retry_config=retry_config,
    rate_limiter=limiter,
    operation_name="gpt-4",
    operation_type="llm"
):
    if isinstance(item, RetryEvent):
        print(f"Retrying after {item.next_delay_seconds}s")
    elif isinstance(item, RateLimitEvent):
        print(f"Rate limit: {item.tokens_remaining} tokens left")
    else:
        print(item, end="")
```

### Multi-Agent Patterns

```python
from agentic import AgentChain, AgentChainConfig, SupervisorPattern, ParallelPattern

# Sequential chain
chain = AgentChain(
    agents=[
        ("researcher", research_agent),
        ("writer", writing_agent),
        ("editor", editing_agent)
    ],
    config=AgentChainConfig(pass_mode="response")
)

async for event in chain.execute("Write article about AI"):
    if isinstance(event, StepCompleteEvent):
        print(f"Agent completed: {event.result.segments.response}")

# Supervisor-Worker pattern
supervisor = SupervisorPattern(
    supervisor=coordinator_agent,
    workers={
        "research": research_agent,
        "coding": coding_agent,
        "testing": testing_agent
    }
)

async for event in supervisor.execute("Build a web scraper"):
    # Supervisor delegates to specialized workers
    pass

# Parallel execution with merging
parallel = ParallelPattern(
    agents={
        "optimist": optimist_agent,
        "pessimist": pessimist_agent,
        "realist": realist_agent
    },
    merger=synthesis_agent,
    config=ParallelConfig(merge_strategy="agent")
)

async for event in parallel.execute_and_merge("Analyze market trends"):
    # All agents run in parallel, results merged
    pass
```

### Context Health Monitoring

```python
from agentic import ContextHealthCheck

logic_config = LogicConfig(
    logic_id="monitored_loop",
    max_iterations=100,
    context_health_checks=[
        ContextHealthCheck(
            check_type="size",
            key_pattern="llm_output:*",
            threshold=1_000_000,  # 1MB
            action="warn"  # or "stop"
        ),
        ContextHealthCheck(
            check_type="version_count",
            key_pattern="*",
            threshold=1000,
            action="stop"
        )
    ]
)

# Health events emitted during execution
async for event in logic.run_stream("Your task"):
    if isinstance(event, ContextHealthEvent):
        print(f"Health issue: {event.check_type} at {event.key}")
        print(f"Current: {event.current_value}, Threshold: {event.threshold}")
```

### Tool Approval Callback

```python
def approve_tool(tool_call) -> bool:
    print(f"Tool '{tool_call.name}' detected")
    return input("Execute? (y/n): ").lower() == 'y'

agent_config = AgentConfig(
    ...
    on_tool_detected=approve_tool  # None = auto-approve
)
```

## Testing

```bash
pytest                    # Run all tests
pytest --cov=agentic      # With coverage
pytest -v                 # Verbose
pytest -m asyncio         # Async tests only
```

550+ tests with >90% coverage across all modules. See `tests/README.md` for details.

## Security Considerations

**Tool Execution:**
- Tools execute with the same permissions as the Python process
- Always review tool implementations before registering them
- Use sandboxing (Docker, separate processes) for untrusted code
- The framework provides APIs but does not restrict tool behavior by design

**Validation:**
- Use input validation on all tools to prevent injection attacks
- Built-in `simple_validator` checks types and constraints
- Custom validators can implement format-specific security checks

**Buffer Limits:**
- `StreamingPatternExtractor` enforces a 10MB buffer limit by default
- Configure via `max_buffer_size` in `AgentConfig` to prevent memory exhaustion
- Partial buffer tracking in `LogicRunner` uses same limit

**Context Health:**
- Monitor context size and version count to prevent resource exhaustion
- Use `ContextHealthCheck` with `action="stop"` to halt execution on threshold breach
- Default history limit of 100 versions prevents unbounded growth

**Rate Limiting:**
- Apply rate limiting to external API calls to prevent quota exhaustion
- Token bucket implementation prevents burst attacks
- Combine with retry logic for resilient operation

**Best Practices:**
- Validate tool inputs before execution using the validation system
- Use timeouts on all tool calls (enforced by framework)
- Review LLM outputs before executing extracted tools
- Use `on_tool_detected` callback for human-in-the-loop approval
- Apply retry logic only to idempotent operations
- Monitor context health in long-running loops

## Design Principles

1. **Streaming First**: Batch mode wraps streaming for consistency
2. **Persistence First**: All state in RocksDB with versioning
3. **Versioning by Default**: Context auto-versioned with full history
4. **Type Safety**: Full type hints throughout
5. **Single Responsibility**: Focused modules with clear boundaries
6. **Extensibility**: Plugin validation, custom patterns, user-defined tools
7. **Resilience**: Built-in retry, rate limiting, and health monitoring
8. **Minimal Dependencies**: Only rocksdict required

## License

MIT
