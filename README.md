# Agentic Framework

An agentic framework with versioned context, persistent storage, and streaming-first architecture.

## Features

- **Streaming-First**: Real-time event streams for LLM generation, pattern detection, and tool execution
- **13 Event Types**: Comprehensive observability (LLMToken, PatternStart/End, ToolStart/End, Status, Error, etc.)
- **Persistent Context**: Versioned key-value store in RocksDB with automatic timestamps and iteration tracking
- **Pattern Extraction**: Batch (`PatternExtractor`) and streaming (`StreamingPatternExtractor`) with incremental detection
- **Multi-Mode Tools**: Execute tools in PROCESS, THREAD, or ASYNC modes with timeout handling and streaming support
- **Flexible Logic**: Conditional loops with pattern-based, regex-based, or context-based stop conditions
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
├── core.py       # Enums (ProcessingMode, SegmentType, AgentStatus) & Data classes
├── events.py     # 13 event types (LLMToken, PatternStart/End, ToolStart/End, etc.)
├── storage.py    # RocksDBStorage with automatic path resolution and DB identification
├── context.py    # IterationManager & ContextManager with versioning and history
├── patterns.py   # PatternExtractor (batch) & StreamingPatternExtractor (incremental)
├── tools.py      # Tool execution with multi-mode support, timeouts, and streaming
├── agent.py      # Agent & AgentRunner with dual-mode execution (step/step_stream)
└── logic.py      # LogicRunner for conditional loops with flexible evaluation points
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
- Helpers: `loop_n_times()`, `loop_until_pattern()`, `loop_until_regex()`, `stop_on_error()`

**Events** (`events.py`):
13 event types: `LLMTokenEvent`, `LLMCompleteEvent`, `PatternStartEvent`, `PatternContentEvent`, `PatternEndEvent`, `StatusEvent`, `ToolStartEvent`, `ToolOutputEvent`, `ToolEndEvent`, `ContextWriteEvent`, `ErrorEvent`, `StepCompleteEvent`

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

325+ tests with >90% coverage. See `tests/README.md` for details.

## Security Considerations

**Tool Execution:**
- Tools execute with the same permissions as the Python process
- Always review tool implementations before registering them
- Use sandboxing (Docker, separate processes) for untrusted code
- The framework provides APIs but does not restrict tool behavior by design

**Buffer Limits:**
- `StreamingPatternExtractor` enforces a 10MB buffer limit by default
- Configure via `max_buffer_size` parameter if needed

**Context History:**
- `get_history()` defaults to 100 versions to prevent memory exhaustion
- Increase limit explicitly if you need full history access

**Best Practices:**
- Validate tool inputs before execution
- Use timeouts on all tool calls (enforced by framework)
- Review LLM outputs before executing extracted tools
- Use `on_tool_detected` callback for human-in-the-loop approval

## Design Principles

1. **Streaming First**: Batch mode wraps streaming for consistency
2. **Persistence First**: All state in RocksDB
3. **Versioning by Default**: Context auto-versioned
4. **Type Safety**: Full type hints
5. **Single Responsibility**: Focused modules
6. **Minimal Dependencies**: Only rocksdict required

## License

MIT
