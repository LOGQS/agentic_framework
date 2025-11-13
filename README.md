# Agentic Framework

A robust, production-ready agentic framework with versioned context, persistent storage, flexible control flows, and **streaming-first architecture** for real-time observability.

## Features

- **Streaming-First Architecture (V2)**: Real-time event streams for LLM generation, pattern detection, tool execution, and status updates
- **Event System**: 13 event types for comprehensive observability (LLMTokenEvent, PatternStartEvent, ToolStartEvent, StatusEvent, etc.)
- **Incremental Pattern Detection (V2)**: Detect `<tool>`, `<reasoning>`, `<response>` tags as LLM tokens arrive, not just at end
- **Tool Execution Control (V2)**: User callbacks to approve/reject detected tools before execution
- **Tool Streaming (V2)**: Tools can stream partial outputs; automatic fallback for non-streaming tools
- **Batch + Stream Modes**: Use streaming for real-time UIs or batch mode for simple scripts - both produce identical results
- **Persistent Context with Versioning**: All context data is versioned and stored in RocksDB via rocksdict
- **Global Iteration Tracking**: Track execution progress across all agent steps
- **Flexible Pattern Extraction**: Extract tools, reasoning, and responses from LLM output
- **Multi-Mode Execution**: Run agents, tools, and logic flows in process, thread, or async modes
- **Pluggable LLM Providers**: Bring your own LLM provider with optional streaming support
- **Control Flow Logic**: Built-in loops, conditions, and sequence control with streaming events
- **Type-Safe**: Full type hints throughout the codebase
- **100% Backward Compatible**: All V1 code works unchanged in V2

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Initialize Storage and Context

```python
from agentic import (
    StorageConfig,
    RocksDBStorage,
    IterationManager,
    ContextManager
)

# Configure storage
config = StorageConfig(base_dir="./context", db_name_prefix="context")

# Initialize storage (rocksdict)
storage = RocksDBStorage(config)
storage.initialize()

# Create managers
iteration = IterationManager(storage)
context = ContextManager(storage, iteration)
```

### 2. Register Patterns

```python
from agentic import PatternRegistry, create_default_pattern_set

# Create pattern registry
patterns = PatternRegistry(storage)

# Register default patterns (tool, reasoning, response)
default_patterns = create_default_pattern_set()
patterns.register_pattern_set(default_patterns)
```

### 3. Register Tools

```python
from agentic import ToolRegistry, create_tool, ProcessingMode

# Create tool registry
tools = ToolRegistry()

# Define a simple tool
def search_web(inputs: dict) -> dict:
    query = inputs.get("query", "")
    # Your search implementation here
    return {"results": f"Search results for: {query}"}

# Register tool
search_tool = create_tool(
    name="search_web",
    func=search_web,
    input_schema={"query": "string"},
    output_schema={"results": "string"},
    timeout_seconds=30.0,
    processing_mode=ProcessingMode.THREAD,
    description="Search the web for information"
)
tools.register(search_tool)
```

### 4. Create and Run Agent (Batch Mode)

```python
from agentic import (
    AgentConfig,
    Agent,
    AgentRunner,
    MockLLMProvider  # Replace with real provider
)

# Configure agent
agent_config = AgentConfig(
    agent_id="assistant",
    provider="mock",
    model="gpt-4",
    max_tokens=4096,
    temperature=0.7,
    tools_allowed=["search_web"],
    input_mapping=[
        ("conversation_history", "prepend")
    ],
    output_mapping=[
        ("conversation_history", "append_version")
    ],
    pattern_set="default",
    auto_increment_iteration=True
)

# Create provider (replace with real implementation)
provider = MockLLMProvider(
    response="<reasoning>Let me search for that.</reasoning>\n<tool>\nname: search_web\narguments:\n{\"query\": \"agentic systems\"}\n</tool>"
)

# Create agent
agent = Agent(agent_config, context, patterns, tools, provider)

# Create runner
runner = AgentRunner(agent)

# Execute single step (batch mode - blocks until complete)
result = runner.step("Tell me about agentic systems")

print(f"Status: {result.status}")
print(f"Response: {result.segments.response}")
print(f"Tools called: {len(result.tool_results)}")
print(f"Iteration: {result.iteration}")
```

### 4b. Streaming Mode (Real-Time Events)

```python
import asyncio
from agentic import (
    LLMTokenEvent, LLMCompleteEvent, StatusEvent,
    ToolStartEvent, ToolOutputEvent, ToolEndEvent,
    StepCompleteEvent, ErrorEvent
)

# Same setup as above, but use step_stream() for real-time events
async def run_with_streaming():
    async for event in runner.step_stream("Tell me about agentic systems"):
        if isinstance(event, LLMTokenEvent):
            # Stream tokens as they generate
            print(event.token, end="", flush=True)

        elif isinstance(event, LLMCompleteEvent):
            print("\n[LLM Complete]")

        elif isinstance(event, StatusEvent):
            print(f"[Status: {event.status.value}] {event.message}")

        elif isinstance(event, ToolStartEvent):
            print(f"\n[Executing Tool: {event.tool_name}]")
            print(f"Arguments: {event.arguments}")

        elif isinstance(event, ToolOutputEvent):
            print(f"Tool Output: {event.output}")

        elif isinstance(event, ToolEndEvent):
            if event.result.success:
                print(f"[Tool {event.tool_name} completed in {event.result.execution_time:.2f}s]")
            else:
                print(f"[Tool {event.tool_name} failed: {event.result.error_message}]")

        elif isinstance(event, ErrorEvent):
            print(f"[ERROR] {event.error_type}: {event.error_message}")

        elif isinstance(event, StepCompleteEvent):
            result = event.result
            print(f"\n[Step Complete - Status: {result.status.value}]")
            return result

# Run the streaming example
result = asyncio.run(run_with_streaming())
```

### 5. Use Logic Flows (Batch Mode)

```python
from agentic import LogicConfig, LogicCondition, LogicRunner

# Configure logic with stop conditions
logic_config = LogicConfig(
    logic_id="main_loop",
    max_iterations=10,
    stop_conditions=[
        LogicCondition(
            pattern_set="default",
            pattern_name="response",
            match_type="contains",
            target="response"
        )
    ],
    break_on_error=True
)

# Create logic runner
logic = LogicRunner(runner, context, patterns, logic_config)

# Run with logic control (batch mode)
results = logic.run(initial_input="Analyze this problem step by step")

print(f"Completed {len(results)} iterations")
for i, result in enumerate(results):
    print(f"Iteration {i}: {result.status}")
```

### 5b. Logic Flows with Streaming

```python
# Same logic_config as above, use run_stream() for real-time events
async def run_logic_with_streaming():
    async for event in logic.run_stream(initial_input="Analyze this problem step by step"):
        # Handle all events from agent steps plus logic-level StatusEvents
        if isinstance(event, StatusEvent):
            print(f"[Logic: {event.message}]")
        elif isinstance(event, StepCompleteEvent):
            print(f"[Iteration {event.result.iteration} complete]")
        # ... handle other events as needed

asyncio.run(run_logic_with_streaming())
```

## Streaming Architecture (V2)

### Event Types

The framework emits 13 event types during execution:

| Event | Description | Key Fields |
|-------|-------------|------------|
| `LLMTokenEvent` | LLM generates a token | `token: str` |
| `LLMCompleteEvent` | LLM finishes generation | `full_text: str` |
| `PatternStartEvent` | Pattern start tag detected (V2) | `pattern_name: str`, `pattern_type: str` |
| `PatternContentEvent` | Pattern content streaming (V2) | `pattern_name: str`, `content: str`, `is_partial: bool` |
| `PatternEndEvent` | Pattern end tag detected (V2) | `pattern_name: str`, `full_content: str` |
| `StatusEvent` | Agent status changes | `status: AgentStatus`, `message: str` |
| `ToolStartEvent` | Tool execution begins | `tool_name: str`, `arguments: dict` |
| `ToolOutputEvent` | Tool produces output | `tool_name: str`, `output: Any`, `is_partial: bool` |
| `ToolEndEvent` | Tool execution completes | `tool_name: str`, `result: ToolResult` |
| `ContextWriteEvent` | Context updated | `key: str`, `value_preview: str`, `version: int` |
| `ErrorEvent` | Error occurs | `error_type: str`, `error_message: str`, `partial_data: Any` (V2) |
| `StepCompleteEvent` | Agent step finishes | `result: AgentStepResult` |

All events inherit from `BaseEvent` with `type: str` and `timestamp: float`.

### Batch vs Streaming

**Batch Mode** (`step()`, `run()`):
- Blocks until complete
- Returns final result
- Simple API for scripts and CLIs
- Internally wraps streaming and aggregates events

**Streaming Mode** (`step_stream()`, `run_stream()`):
- Yields events as they occur
- Real-time progress updates
- Ideal for UIs and dashboards
- Produces identical final results to batch mode

### Key Principle

**Batch = Degenerate Case of Streaming**

All batch methods internally call streaming methods and aggregate events. This ensures:
- Zero code duplication
- Guaranteed consistency
- Full backward compatibility

### Enable Incremental Context Writes

```python
agent_config = AgentConfig(
    # ... other config ...
    incremental_context_writes=True  # Emit ContextWriteEvent during streaming
)
```

When enabled, context updates happen **during** step execution and emit `ContextWriteEvent`. When disabled (default), context updates happen at the **end** (V1 behavior).

### Incremental Pattern Detection

**New in V2:** Patterns detected as LLM tokens arrive (not just at end):

```python
# Enable pattern content streaming (optional)
agent_config = AgentConfig(
    ...
    stream_pattern_content=True  # Stream <tool> content before </tool> arrives
)

# Handle pattern events
async for event in runner.step_stream("Your prompt"):
    if isinstance(event, PatternStartEvent):
        print(f"[Pattern Started: {event.pattern_name}]")
    elif isinstance(event, PatternContentEvent):
        # Stream content as it arrives (before end tag)
        print(event.content, end="", flush=True)
    elif isinstance(event, PatternEndEvent):
        print(f"\n[Pattern Complete: {event.pattern_name}]")
```

**Use Cases:**
- Display tool arguments as LLM generates them
- Stream reasoning/response content in real-time
- Detect patterns mid-generation for early termination

### Tool Execution Control

**New in V2:** Control tool execution with callbacks:

```python
def approve_tool(tool_call: ToolCall) -> bool:
    """User callback to approve/reject tools."""
    # Example: require user confirmation
    print(f"Tool '{tool_call.name}' detected with args: {tool_call.arguments}")
    response = input("Execute? (y/n): ")
    return response.lower() == 'y'

agent_config = AgentConfig(
    ...
    on_tool_detected=approve_tool  # None = auto-approve (default)
)

# Tools now wait for approval before executing
result = runner.step("Do something with tools")
```

**Status Semantics:**
- `TOOL_EXECUTED` - Tools executed successfully (final status)
- `WAITING_FOR_TOOL` - Tools detected but pending/rejected

### Tool Streaming

**New in V2:** Tools can stream partial outputs:

```python
# Example: Streaming shell tool
async def streaming_shell(inputs: dict):
    """Stream command output line-by-line."""
    async for line in execute_command_async(inputs["command"]):
        yield line  # Framework wraps in ToolOutputEvent

tool = create_tool(
    name="shell",
    func=streaming_shell,  # Callable with run_stream attribute
    ...
)

# Framework automatically uses run_stream() if available,
# otherwise wraps run() as single output event
```

### Flexible Condition Evaluation

**New in V2:** Evaluate conditions at different pipeline stages:

```python
from agentic import LogicCondition, LogicConfig, LogicRunner

logic_config = LogicConfig(
    logic_id="early_termination",
    loop_until_conditions=[
        LogicCondition(
            pattern_set="default",
            pattern_name="tool",
            match_type="contains",
            target="response",
            evaluation_point="llm_complete"  # Stop as soon as LLM finishes
        )
    ]
)

# Options: "auto", "llm_complete", "tool_detected", "step_complete", "any_event"
# "auto" (default) infers from target - context → step_complete, patterns → llm_complete
```

## Architecture

### Event Layer (`events.py`) - NEW in V2

- **13 event types** for comprehensive observability (LLM, Pattern, Tool, Status, Context, Error, Step)
- Real-time streaming of LLM tokens, tool execution, status changes
- Strongly-typed event classes with timestamps
- Used internally by all streaming operations

### Storage Layer (`storage.py`)

- RocksDB-based persistent storage via rocksdict
- Automatic path resolution with collision avoidance
- Database identification and validation

### Context Layer (`context.py`)

- Versioned key-value store
- Global iteration tracking
- Automatic timestamping
- History retrieval
- Optional incremental writes during streaming

### Pattern Layer (`patterns.py`)

- Pattern-based text extraction
- Support for tool calls, reasoning, and responses
- Customizable pattern sets
- Registry for pattern management

### Tool Layer (`tools.py`)

- Multi-mode execution (process, thread, async)
- Timeout handling
- Input/output schema validation
- Centralized tool registry
- Streaming-ready architecture (future: tools can stream partial outputs)

### Agent Layer (`agent.py`)

- Pluggable LLM provider interface with streaming support
- **Dual-mode execution**: `step()` (batch) and `step_stream()` (streaming)
- Configurable input/output mappings
- Automatic pattern extraction
- Tool execution orchestration with real-time events

### Logic Layer (`logic.py`)

- Conditional control flows with streaming events
- **Dual-mode execution**: `run()` (batch) and `run_stream()` (streaming)
- Loop constructs with real-time progress
- Pattern-based stopping conditions
- Context-aware decisions
- Multi-mode execution (process, thread, async)

## Advanced Usage

### Custom Pattern Sets

```python
from agentic import Pattern, PatternSet, SegmentType

custom_patterns = PatternSet(
    name="custom",
    patterns=[
        Pattern(
            name="thought",
            start_tag="<thought>",
            end_tag="</thought>",
            segment_type=SegmentType.REASONING,
            greedy=False
        ),
        Pattern(
            name="action",
            start_tag="<action>",
            end_tag="</action>",
            segment_type=SegmentType.TOOL,
            greedy=False
        )
    ],
    default_response_behavior="all_remaining"
)

patterns.register_pattern_set(custom_patterns)
```

### Context Versioning

```python
# Set context value (creates new version)
context.set("user_profile", b'{"name": "Alice", "age": 30}')

# Get latest version
latest = context.get("user_profile")
print(f"Version: {latest.version}, Iteration: {latest.iteration}")

# Get specific version
old_version = context.get("user_profile", version=1)

# Get version history
history = context.get_history("user_profile", max_versions=5)
for record in history:
    print(f"v{record.version} at iteration {record.iteration}")
```

### Custom LLM Provider

```python
from typing import AsyncIterator
from agentic import LLMProvider

class MyLLMProvider:
    """
    Custom LLM provider with streaming support.

    The generate() method is required. The stream() method is optional -
    if not implemented, framework will simulate streaming using generate().
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize your LLM client here

    def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Batch generation (required)."""
        # Call your LLM API
        # Return the complete generated text
        pass

    async def stream(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> AsyncIterator[str]:
        """
        Stream tokens (optional but recommended).

        Yield tokens as they're generated for real-time UIs.
        """
        # Example with OpenAI-style API:
        # async for chunk in your_api.stream(prompt, max_tokens, temperature):
        #     if chunk.choices[0].delta.content:
        #         yield chunk.choices[0].delta.content

        # If streaming not supported, fall back to generate():
        text = self.generate(prompt, max_tokens, temperature, **kwargs)
        yield text

provider = MyLLMProvider(api_key="your-api-key")
agent = Agent(agent_config, context, patterns, tools, provider)
```

### Convenience Logic Functions

```python
from agentic import loop_n_times, loop_until_pattern, loop_until_regex

# Loop exactly 5 times
logic1 = loop_n_times(runner, context, patterns, n=5)
results1 = logic1.run("Start task")

# Loop until pattern found
logic2 = loop_until_pattern(
    runner, context, patterns,
    pattern_set="default",
    pattern_name="response",
    target="response",
    max_iterations=10
)
results2 = logic2.run("Keep working until done")

# Loop until regex matches
logic3 = loop_until_regex(
    runner, context, patterns,
    regex_pattern=r"COMPLETE|DONE|FINISHED",
    target="response",
    max_iterations=20
)
results3 = logic3.run("Process this task")
```

## Design Principles

1. **Streaming First**: All operations support real-time event streams with batch as a convenience wrapper
2. **Single Responsibility**: Each module has a clear, focused purpose
3. **Persistence First**: All state is persisted to RocksDB via rocksdict
4. **Versioning by Default**: Context changes are automatically versioned
5. **Type Safety**: Comprehensive type hints throughout
6. **Flexibility**: Pluggable providers, patterns, and tools with multi-mode execution
7. **Minimal Dependencies**: Only rocksdict required for core functionality
8. **Backward Compatibility**: V1 code works unchanged in V2

## Project Structure

```
agentic/
├── __init__.py          # Public API exports
├── core.py              # Core types and enums
├── events.py            # Event system (V2)
├── storage.py           # RocksDB storage layer
├── context.py           # Context management with versioning
├── patterns.py          # Pattern extraction
├── tools.py             # Tool execution
├── agent.py             # Agent abstraction with streaming
└── logic.py             # Control flow logic with streaming
```

## V2 Migration Guide

**Good News**: No migration required! All V1 code works unchanged in V2.

**To Use Streaming** (optional):

```python
# V1 code (still works)
result = runner.step(input)
results = logic.run(initial_input)

# V2 streaming (opt-in)
async for event in runner.step_stream(input):
    handle_event(event)

async for event in logic.run_stream(initial_input):
    handle_event(event)
```

**Provider Updates** (optional):

V1 providers (only `generate()`) work unchanged. To enable true streaming:

```python
class MyProvider:
    def generate(self, prompt, max_tokens, temperature, **kwargs) -> str:
        # Required - works in both V1 and V2
        pass

    async def stream(self, prompt, max_tokens, temperature, **kwargs) -> AsyncIterator[str]:
        # Optional - enables true token streaming in V2
        pass
```

For full V2 features and architectural details, see `docs/specification_v2.md`.

## License

MIT License

## Contributing

Contributions welcome! Please ensure:
- Type hints on all functions
- Docstrings for public APIs
- No unnecessary dependencies
- Tests for new features
