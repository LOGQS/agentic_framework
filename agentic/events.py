"""
Event system for streaming agent execution.

All operations emit events that can be consumed in real-time or aggregated for batch processing.
"""
from dataclasses import dataclass
from typing import Any
from .core import AgentStatus, ToolResult, AgentStepResult, now_timestamp


@dataclass
class BaseEvent:
    """Base class for all events."""
    type: str
    timestamp: float
    step_id: str = ""  # Unique identifier for the agent step that generated this event

    def __post_init__(self):
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = now_timestamp()


@dataclass
class LLMTokenEvent(BaseEvent):
    """Emitted when LLM generates a token."""
    token: str

    def __init__(self, token: str, timestamp: float = None, step_id: str = ""):
        self.token = token
        self.type = "llm_token"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class LLMCompleteEvent(BaseEvent):
    """Emitted when LLM generation completes."""
    full_text: str

    def __init__(self, full_text: str, timestamp: float = None, step_id: str = ""):
        self.full_text = full_text
        self.type = "llm_complete"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class StatusEvent(BaseEvent):
    """Emitted when agent status changes."""
    status: AgentStatus
    message: str | None = None

    def __init__(self, status: AgentStatus, message: str | None = None, timestamp: float = None, step_id: str = ""):
        self.status = status
        self.message = message
        self.type = "status"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class ToolStartEvent(BaseEvent):
    """Emitted when tool execution begins."""
    tool_name: str
    arguments: dict[str, Any]
    iteration: int
    call_id: str = ""  # Unique identifier for this tool invocation

    def __init__(self, tool_name: str, arguments: dict[str, Any], iteration: int, call_id: str = "", timestamp: float = None, step_id: str = ""):
        self.tool_name = tool_name
        self.arguments = arguments
        self.iteration = iteration
        self.call_id = call_id
        self.type = "tool_start"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class ToolOutputEvent(BaseEvent):
    """Emitted when tool produces output (may be partial)."""
    tool_name: str
    output: Any
    is_partial: bool = False
    call_id: str = ""  # Unique identifier for this tool invocation

    def __init__(self, tool_name: str, output: Any, is_partial: bool = False, call_id: str = "", timestamp: float = None, step_id: str = ""):
        self.tool_name = tool_name
        self.output = output
        self.is_partial = is_partial
        self.call_id = call_id
        self.type = "tool_output"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class ToolEndEvent(BaseEvent):
    """Emitted when tool execution completes."""
    tool_name: str
    result: ToolResult
    call_id: str = ""  # Unique identifier for this tool invocation

    def __init__(self, tool_name: str, result: ToolResult, call_id: str = "", timestamp: float = None, step_id: str = ""):
        self.tool_name = tool_name
        self.result = result
        self.call_id = call_id
        self.type = "tool_end"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class ContextWriteEvent(BaseEvent):
    """Emitted when context is updated."""
    key: str
    value_preview: str  # First 100 chars or summary
    version: int
    iteration: int

    def __init__(self, key: str, value_preview: str, version: int, iteration: int, timestamp: float = None, step_id: str = ""):
        self.key = key
        self.value_preview = value_preview
        self.version = version
        self.iteration = iteration
        self.type = "context_write"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class ErrorEvent(BaseEvent):
    """Emitted when an error occurs."""
    error_type: str
    error_message: str
    recoverable: bool = False
    partial_data: Any = None  # For malformed patterns, stores partial content separately

    def __init__(self, error_type: str, error_message: str, recoverable: bool = False, partial_data: Any = None, timestamp: float = None, step_id: str = ""):
        self.error_type = error_type
        self.error_message = error_message
        self.recoverable = recoverable
        self.partial_data = partial_data
        self.type = "error"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class PatternStartEvent(BaseEvent):
    """Emitted when a pattern start tag is detected during streaming."""
    pattern_name: str
    pattern_type: str  # "tool" | "reasoning" | "response"

    def __init__(self, pattern_name: str, pattern_type: str, timestamp: float = None, step_id: str = ""):
        self.pattern_name = pattern_name
        self.pattern_type = pattern_type
        self.type = "pattern_start"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class PatternContentEvent(BaseEvent):
    """Emitted when pattern content is streamed (before end tag detected)."""
    pattern_name: str
    content: str
    is_partial: bool = True

    def __init__(self, pattern_name: str, content: str, is_partial: bool = True, timestamp: float = None, step_id: str = ""):
        self.pattern_name = pattern_name
        self.content = content
        self.is_partial = is_partial
        self.type = "pattern_content"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class PatternEndEvent(BaseEvent):
    """Emitted when a pattern end tag is detected during streaming."""
    pattern_name: str
    pattern_type: str  # "tool" | "reasoning" | "response"
    full_content: str  # Complete content between tags

    def __init__(self, pattern_name: str, pattern_type: str, full_content: str, timestamp: float = None, step_id: str = ""):
        self.pattern_name = pattern_name
        self.pattern_type = pattern_type
        self.full_content = full_content
        self.type = "pattern_end"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass
class StepCompleteEvent(BaseEvent):
    """Emitted when agent step completes. Contains final aggregated result."""
    result: AgentStepResult

    def __init__(self, result: AgentStepResult, timestamp: float = None, step_id: str = ""):
        self.result = result
        self.type = "step_complete"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


# Type alias for all possible events
AgentEvent = (
    LLMTokenEvent | LLMCompleteEvent | StatusEvent |
    ToolStartEvent | ToolOutputEvent | ToolEndEvent |
    ContextWriteEvent | ErrorEvent |
    PatternStartEvent | PatternContentEvent | PatternEndEvent |
    StepCompleteEvent
)
