# Aegeantic

A framework for building LLM agents with versioned context, persistent storage, streaming execution, and multi-agent orchestration.

## Features

- **Streaming-First**: Real-time event streams for LLM generation, pattern detection, and tool execution
- **17 Event Types**: Complete observability across LLM, tools, patterns, validation, retry, rate limiting, and context health
- **Structured Logging**: Built-in logger with contextual metadata for debugging and monitoring
- **Persistent Context**: Versioned key-value store backed by RocksDB with automatic timestamps and iteration tracking
- **Pattern Extraction**: Regex-based extraction with streaming support, incremental detection, and malformed pattern handling
- **Multi-Mode Tools**: Execute tools in PROCESS, THREAD, or ASYNC modes with timeout handling and streaming output
- **Validation System**: Extensible validation registry supporting simple validators, JSON Schema, or custom formats
- **Resilience**: Retry logic with exponential backoff and token bucket rate limiting
- **Tool Verification**: Human-in-the-loop tool approval with timeout handling and rejection tracking
- **Multi-Agent Patterns**: Chain, Supervisor-Worker, Parallel, and Debate patterns for orchestrating multiple agents
- **Flexible Storage**: RocksDB for persistence or in-memory storage for testing and ephemeral use
- **Flexible Logic Flows**: Conditional loops with pattern-based, regex-based, or context-based conditions
- **Type-Safe**: Full type hints throughout

## Installation

### From PyPI (Recommended)

```bash
pip install aegeantic
```

### For Development

```bash
git clone https://github.com/LOGQS/agentic_framework.git
cd agentic_framework
pip install -e .
```

**Requirements:** Python 3.9+ and rocksdict >= 0.3.0

## Quick Start

```python
from agentic import (
    StorageConfig, RocksDBStorage, InMemoryStorage, IterationManager, ContextManager,
    PatternRegistry, create_default_pattern_set,
    ToolRegistry, create_tool,
    AgentConfig, Agent, AgentRunner,
    ProcessingMode, create_message_prompt_builder
)

# Initialize storage and context
# Option 1: RocksDB for persistent storage
config = StorageConfig(base_dir="./context", db_name_prefix="my_agent")
storage = RocksDBStorage(config)
storage.initialize()
# Option 2: In-memory storage for testing or ephemeral use
# storage = InMemoryStorage()
# storage.initialize()

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
    input_schema={
        "validator": "simple",
        "required": ["query"],
        "fields": {
            "query": {"type": "str"}
        }
    },
    timeout_seconds=30.0,
    processing_mode=ProcessingMode.THREAD
))

# Create LLM provider (must implement LLMProvider protocol)
class MyLLMProvider:
    def generate(self, prompt, **kwargs) -> str:
        # Your LLM API call here
        pass

    async def stream(self, prompt, **kwargs):
        # Stream chunks for real-time execution
        text = self.generate(prompt, **kwargs)
        yield text

provider = MyLLMProvider()

# Configure and run agent
agent_config = AgentConfig(
    agent_id="assistant",
    tools_allowed=["search_web"],
    input_mapping=[
        {"context_key": "literal:You are a helpful assistant.", "role": "system", "order": 0}
    ],
    output_mapping=[("last_output", "set_latest")],
    pattern_set="default",
    prompt_builder=create_message_prompt_builder()
)

agent = Agent(agent_config, context, patterns, tools, provider)
runner = AgentRunner(agent)

# Batch execution
result = runner.step("Tell me about agentic systems")
print(f"Status: {result.status}, Response: {result.segments.response}")

# Streaming execution
import asyncio
from agentic import LLMChunkEvent, ToolStartEvent, StepCompleteEvent

async def stream_example():
    async for event in runner.step_stream("Your prompt"):
        if isinstance(event, LLMChunkEvent):
            print(event.chunk, end="", flush=True)
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
├── events.py       # 17 event types (LLM, tools, patterns, retry, rate limit, health)
├── storage.py      # RocksDBStorage & InMemoryStorage with automatic path resolution
├── context.py      # IterationManager & ContextManager with versioning and history
├── patterns.py     # PatternExtractor (batch) & StreamingPatternExtractor (incremental)
├── tools.py        # Tool execution with multi-mode support, timeouts, and streaming
├── agent.py        # Agent & AgentRunner with dual-mode execution (step/step_stream)
├── logic.py        # LogicRunner for conditional loops with context health monitoring
├── resilience.py   # Retry logic with backoff and token bucket rate limiting
├── multi_agent.py  # Multi-agent orchestration patterns (Chain, Supervisor, Parallel, Debate)
└── logging_util.py # Structured logging with contextual metadata for debugging
```

### Key Components

**Storage** (`storage.py`):
- **RocksDBStorage**: Persistent backend via `rocksdict`
  - Automatic DB ID generation: `agentic_<hash>_<timestamp>_<uuid>`
  - CRUD operations: `get()`, `put()`, `delete()`, `iterate()`
- **InMemoryStorage**: Ephemeral storage for testing and development
  - Same interface as RocksDBStorage for drop-in compatibility
  - Faster performance, no disk I/O

**Context** (`context.py`):
- `IterationManager`: Global iteration counter with `get()`, `next()`, `register_event()`
- `ContextManager`: Versioned store with `set()`, `update()`, `get()`, `delete()`, `list_keys()`, `get_history()`, `clear()`
- `ContextRecord`: `{value, iteration, timestamp, version}`
- `set()` creates new version, `update()` overwrites current version
- Tombstone deletion for soft deletes with history preservation

**Patterns** (`patterns.py`):
- `Pattern`: `{name, start_tag, end_tag, segment_type, greedy}`
- `PatternSet`: Collection of patterns with configuration
- `PatternRegistry`: Persistent pattern set storage in RocksDB
- `PatternExtractor`: Batch regex-based extraction
- `StreamingPatternExtractor`: Incremental chunk processing with malformed pattern handling
- Built-in pattern sets: `default`, `json_tools`, `xml_tools`, `backtick_tools`

**Tools** (`tools.py`):
- `Tool`: Executable with `run()` (batch) and `run_stream()` (streaming), supports async callables
- `ToolDefinition`: Metadata with `input_schema`, `output_schema`, `timeout_seconds`, `processing_mode`
- `ToolRegistry`: `register()`, `get()`, `exists()`, `list()`, `get_definitions()`, `unregister()`
- Multi-mode execution: PROCESS, THREAD, ASYNC (inherits from parent if not set)
- Automatic timeout enforcement and validation integration

**Agent** (`agent.py`):
- `LLMProvider`: Protocol defining `generate()` and `stream()` methods accepting `PromptType` (Any)
- `Agent`: Container for config, context, patterns, tools, provider
- `AgentRunner`: `step()` (batch) and `step_stream()` (streaming)
- `AgentConfig`: Configuration with options including `incremental_context_writes`, `stream_pattern_content`, `on_tool_detected`, `concurrent_tool_execution`, `max_partial_buffer_size`, `prompt_builder`, `tool_verification_timeout`, `tool_verification_on_timeout`
- `PromptObject`: Structured prompt with `system`, `messages`, and `metadata` fields
- `create_message_prompt_builder()`: Reference implementation for building PromptObject from input_mapping
- `AgentStepResult`: Contains `tool_decisions` list tracking complete tool lifecycle (verification + execution)
- `ToolExecutionDecision`: Rich decision object with `accepted`, `rejection_reason`, `verification_duration_ms`, `executed`, and `result` fields

**Logic** (`logic.py`):
- `LogicRunner`: Conditional loops with `run()` (batch) and `run_stream()` (streaming)
- `LogicCondition`: Pattern/regex/context-based conditions with evaluation points (`auto`, `llm_chunk`, `llm_complete`, `tool_detected`, `tool_finished`, `step_complete`, `any_event`)
- `LogicConfig`: Loop configuration with `max_iterations`, `stop_conditions`, `loop_until_conditions`, `break_on_error`, `context_health_checks`
- `ContextHealthCheck`: Monitor context size, version count, growth rate with configurable thresholds and actions
- Helper functions: `loop_n_times()`, `loop_until_pattern()`, `loop_until_regex()`, `stop_on_error()`

**Validation** (`validation.py`):
- `ValidatorRegistry`: Extensible registry for any validation format
- `ValidationError`: Structured error with field, message, and value
- Built-in validators: `simple_validator` (type checking, constraints), `passthrough_validator`
- Support for JSON Schema, XML Schema, Protocol Buffers, or custom validators

**Events** (`events.py`):
17 event types: `LLMChunkEvent`, `LLMCompleteEvent`, `PatternStartEvent`, `PatternContentEvent`, `PatternEndEvent`, `StatusEvent`, `ToolStartEvent`, `ToolDecisionEvent`, `ToolOutputEvent`, `ToolEndEvent`, `ToolValidationEvent`, `ContextWriteEvent`, `ErrorEvent`, `StepCompleteEvent`, `RetryEvent`, `RateLimitEvent`, `ContextHealthEvent`

**Logging** (`logging_util.py`):
- Structured logger with contextual metadata
- Integration with Python's standard logging module
- Automatic context enrichment (agent_id, iteration, step_id, etc.)
- Supports JSON-formatted output for log aggregation systems

**Resilience** (`resilience.py`):
- `RetryConfig`: Exponential/linear/constant backoff with jitter and configurable retry exceptions
- `RateLimitConfig`: Per-second/minute/hour limits with burst capacity
- `RateLimiter`: Token bucket implementation with `acquire()`, `try_acquire()`, `tokens_available()`
- `retry_stream()`: Universal retry wrapper for any async iterator with RetryEvent emission
- `rate_limited_stream()`: Universal rate limiting wrapper with RateLimitEvent emission
- `resilient_stream()`: Combined retry + rate limiting for any operation

**Multi-Agent** (`multi_agent.py`):
- `AgentChain`: Sequential execution with configurable output passing (`response`, `full_context`, `tool_results`, `custom`)
- `AgentChainConfig`: Configure pass mode, context templates, and custom transform functions
- `SupervisorPattern`: Supervisor agent delegates tasks to specialized worker agents with delegation detection
- `SupervisorConfig`: Configure delegation pattern name, worker/task keys, max rounds
- `ParallelPattern`: Parallel agent execution with result merging (concat, agent-based, voting)
- `ParallelConfig`: Configure merge strategy, templates, and timeout
- `DebatePattern`: Multi-round debate between agents with consensus detection
- `DebateConfig`: Configure max rounds, consensus detector, and prompt templates

## Advanced Usage

### Context Versioning

```python
# Set creates new version
context.set("key", b"value1")  # version 1
context.set("key", b"value2")  # version 2

# Update overwrites current version (for streaming/incremental writes)
context.update("key", b"partial_value")  # version 2 (overwrites)

# Get latest or specific version
latest = context.get("key")
v1 = context.get("key", version=1)

# History (returns ContextRecord objects)
history = context.get_history("key", max_versions=100)

# Delete (creates tombstone, preserves history)
context.delete("key")
assert context.get("key") is None

# List all keys with optional prefix
all_keys = context.list_keys()
llm_keys = context.list_keys(prefix="llm_output:")
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
        LogicCondition(
            pattern_set="default",
            pattern_name="DONE",
            match_type="regex",
            target="response",
            evaluation_point="llm_complete"
        )
    ],
    loop_until_conditions=[
        LogicCondition(
            pattern_set="default",
            pattern_name="complete",
            match_type="contains",
            target="response",
            evaluation_point="auto"  # auto-detects best evaluation point
        )
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
        "required": ["x", "y"],
        "fields": {
            "x": {"type": "float", "min": 0},
            "y": {"type": "float", "min": 0},
            "op": {"type": "str", "pattern": "^[+\\-*/]$"}
        },
        "allow_extra_fields": False
    }
))

# Custom validator
def my_validator(value: Any, schema: dict) -> tuple[bool, list[ValidationError]]:
    errors = []
    # Your validation logic
    if not isinstance(value, dict):
        errors.append(ValidationError("_root", "Expected dict"))
    return len(errors) == 0, errors

registry = ValidatorRegistry()
registry.register("my_format", my_validator)

# Use in tool registry
tools_with_validation = ToolRegistry(validator_registry=registry)
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
    async for chunk in provider.stream(prompt):
        yield chunk

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
from agentic import (
    AgentChain, AgentChainConfig,
    SupervisorPattern, SupervisorConfig,
    ParallelPattern, ParallelConfig,
    DebatePattern, DebateConfig
)

# Sequential chain
chain = AgentChain(
    agents=[
        ("researcher", research_agent),
        ("writer", writing_agent),
        ("editor", editing_agent)
    ],
    config=AgentChainConfig(
        pass_mode="response",  # or "full_context", "tool_results", "custom"
        prepend_context=True,
        context_template="Previous agent ({agent_id}) output:\n{output}\n\n"
    )
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
    },
    config=SupervisorConfig(
        delegation_pattern_name="delegate",
        worker_key="to",
        task_key="task",
        max_delegation_rounds=10
    )
)

async for event in supervisor.execute("Build a web scraper"):
    # Supervisor delegates to specialized workers
    if isinstance(event, StatusEvent):
        print(event.message)

# Parallel execution with merging
parallel = ParallelPattern(
    agents={
        "optimist": optimist_agent,
        "pessimist": pessimist_agent,
        "realist": realist_agent
    },
    merger=synthesis_agent,
    config=ParallelConfig(
        merge_strategy="agent",  # or "concat", "voting"
        merge_template="Synthesize these perspectives:\n\n{perspectives}",
        timeout_seconds=120.0
    )
)

async for event in parallel.execute_and_merge("Analyze market trends"):
    # All agents run in parallel, results merged
    pass

# Debate pattern
debate = DebatePattern(
    agents={
        "pro": pro_agent,
        "con": con_agent
    },
    moderator=moderator_agent,
    config=DebateConfig(
        max_rounds=5,
        consensus_detector=None  # Uses default similarity check
    )
)

async for event in debate.converge("Should we use microservices?"):
    if isinstance(event, StatusEvent):
        print(f"Debate: {event.message}")
```

### Structured Logging

The framework includes built-in structured logging for debugging and monitoring.

```python
from agentic.logging_util import get_logger
import logging

# Configure logging (optional - framework uses default Python logging)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# The framework automatically logs key events with rich context
# Log entries include: agent_id, iteration, step_id, tool_name, etc.

# Example log output:
# 2025-11-15 10:30:15 - agentic.agent - DEBUG - agent.step.start - {"agent_id": "assistant", "iteration": 1, "step_id": "abc123"}
# 2025-11-15 10:30:16 - agentic.agent - DEBUG - agent.llm.complete - {"agent_id": "assistant", "iteration": 1, "output_length": 250, "tools_detected": 1}
# 2025-11-15 10:30:16 - agentic.agent - DEBUG - tool.verification.start - {"tool_name": "search_web", "call_id": "def456"}
# 2025-11-15 10:30:18 - agentic.agent - DEBUG - agent.step.complete - {"agent_id": "assistant", "status": "TOOL_EXECUTED", "tools_executed": 1}

# Use the logger in custom tools or extensions
logger = get_logger(__name__)

def my_custom_tool(inputs: dict) -> dict:
    logger.info("tool.custom.start", extra={
        "query": inputs.get("query"),
        "user_id": inputs.get("user_id")
    })
    # Your tool logic
    return {"result": "success"}
```

### Context Health Monitoring

```python
from agentic import ContextHealthCheck, ContextHealthEvent

logic_config = LogicConfig(
    logic_id="monitored_loop",
    max_iterations=100,
    context_health_checks=[
        ContextHealthCheck(
            check_type="size",
            key_pattern="llm_output:*",  # Glob pattern
            threshold=1_000_000,  # 1MB
            action="warn",  # or "stop"
            evaluation_point="step_complete",
            max_versions_limit=10000  # Safety limit for version_count checks
        ),
        ContextHealthCheck(
            check_type="version_count",
            key_pattern="*",
            threshold=1000,
            action="stop"
        )
    ]
)

logic = LogicRunner(runner, context, patterns, logic_config)

# Health events emitted during execution
async for event in logic.run_stream("Your task"):
    if isinstance(event, ContextHealthEvent):
        print(f"Health issue: {event.check_type} at {event.key}")
        print(f"Current: {event.current_value}, Threshold: {event.threshold}")
        print(f"Action: {event.recommended_action}")
```

### Tool Verification and Approval

The framework provides comprehensive tool verification with timeout handling and decision tracking.

```python
from agentic import ToolCall, ToolDecisionEvent, AgentStatus, ToolExecutionDecision

def approve_tool(tool_call: ToolCall) -> bool:
    """Callback for human-in-the-loop tool approval."""
    print(f"Tool '{tool_call.name}' detected with args: {tool_call.arguments}")
    return input("Execute? (y/n): ").lower() == 'y'

agent_config = AgentConfig(
    agent_id="assistant",
    tools_allowed=["search_web", "calculate"],
    on_tool_detected=approve_tool,  # None = auto-approve all
    tool_verification_timeout=30.0,  # Timeout in seconds for verification callback
    tool_verification_on_timeout="reject",  # "accept" or "reject" - action when timeout occurs
    concurrent_tool_execution=False  # Set to True to execute tools during LLM streaming
)

# Streaming: Listen for tool decision events in real-time
async for event in runner.step_stream("Your prompt"):
    if isinstance(event, ToolDecisionEvent):
        print(f"Tool: {event.tool_name}")
        print(f"Accepted: {event.accepted}")
        if not event.accepted:
            print(f"Rejection reason: {event.rejection_reason}")
        print(f"Verification took: {event.verification_duration_ms}ms")
    elif isinstance(event, StatusEvent):
        if event.status == AgentStatus.WAITING_FOR_VERIFICATION:
            print(f"Waiting for approval: {event.message}")
        elif event.status == AgentStatus.TOOLS_REJECTED:
            print("All tools were rejected")

# Batch: Access tool decisions from result
result = runner.step("Your prompt")

# Inspect all tool decisions (both accepted and rejected)
for decision in result.tool_decisions:
    print(f"Tool: {decision.tool_call.name}")
    print(f"  Accepted: {decision.accepted}")
    print(f"  Verification required: {decision.verification_required}")
    print(f"  Verification duration: {decision.verification_duration_ms}ms")

    if not decision.accepted:
        print(f"  Rejection reason: {decision.rejection_reason}")

    if decision.executed and decision.result:
        print(f"  Execution success: {decision.result.success}")
        if decision.result.success:
            print(f"  Output: {decision.result.output}")
        else:
            print(f"  Error: {decision.result.error_message}")

# Concurrent tool execution
# When enabled, tools are executed as soon as pattern is detected during LLM streaming
# Otherwise, tools execute after LLM completes
agent_config_concurrent = AgentConfig(
    agent_id="assistant",
    tools_allowed=["search_web"],
    concurrent_tool_execution=True,  # Execute tools during streaming
    stream_pattern_content=True  # Stream pattern content as it's detected
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
2. **Persistence First**: All state stored in RocksDB with versioning
3. **Versioning by Default**: Context auto-versioned with full history and tombstone deletion
4. **Type Safety**: Full type hints throughout codebase
5. **Single Responsibility**: Focused modules with clear boundaries
6. **Extensibility**: Pluggable validation, custom patterns, user-defined tools
7. **Resilience**: Built-in retry, rate limiting, and context health monitoring
8. **Minimal Dependencies**: Only rocksdict required (Python wrapper for RocksDB)

## License

MIT
