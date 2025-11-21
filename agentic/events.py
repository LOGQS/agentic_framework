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


@dataclass(init=False)
class LLMChunkEvent(BaseEvent):
    chunk: str

    def __init__(self, chunk: str, timestamp: float = None, step_id: str = ""):
        self.chunk = chunk
        self.type = "llm_chunk"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class LLMCompleteEvent(BaseEvent):
    full_text: str

    def __init__(self, full_text: str, timestamp: float = None, step_id: str = ""):
        self.full_text = full_text
        self.type = "llm_complete"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class StatusEvent(BaseEvent):
    status: AgentStatus
    message: str | None = None

    def __init__(self, status: AgentStatus, message: str | None = None, timestamp: float = None, step_id: str = ""):
        self.status = status
        self.message = message
        self.type = "status"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class ToolStartEvent(BaseEvent):
    tool_name: str
    arguments: dict[str, Any]
    iteration: int
    call_id: str = ""

    def __init__(self, tool_name: str, arguments: dict[str, Any], iteration: int, call_id: str = "", timestamp: float = None, step_id: str = ""):
        self.tool_name = tool_name
        self.arguments = arguments
        self.iteration = iteration
        self.call_id = call_id
        self.type = "tool_start"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class ToolDecisionEvent(BaseEvent):
    """Emitted when verification decision is made about a tool (accepted/rejected)."""
    tool_name: str
    call_id: str
    accepted: bool
    rejection_reason: str | None = None
    verification_duration_ms: float = 0.0

    def __init__(self, tool_name: str, call_id: str, accepted: bool, rejection_reason: str | None = None,
                 verification_duration_ms: float = 0.0, timestamp: float = None, step_id: str = ""):
        self.tool_name = tool_name
        self.call_id = call_id
        self.accepted = accepted
        self.rejection_reason = rejection_reason
        self.verification_duration_ms = verification_duration_ms
        self.type = "tool_decision"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class ToolOutputEvent(BaseEvent):
    """May be partial during streaming."""
    tool_name: str
    output: Any
    is_partial: bool = False
    call_id: str = ""

    def __init__(self, tool_name: str, output: Any, is_partial: bool = False, call_id: str = "", timestamp: float = None, step_id: str = ""):
        self.tool_name = tool_name
        self.output = output
        self.is_partial = is_partial
        self.call_id = call_id
        self.type = "tool_output"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class ToolEndEvent(BaseEvent):
    tool_name: str
    result: ToolResult
    call_id: str = ""

    def __init__(self, tool_name: str, result: ToolResult, call_id: str = "", timestamp: float = None, step_id: str = ""):
        self.tool_name = tool_name
        self.result = result
        self.call_id = call_id
        self.type = "tool_end"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class ToolValidationEvent(BaseEvent):
    tool_name: str
    validation_errors: list[dict[str, Any]]

    def __init__(
        self,
        tool_name: str,
        validation_errors: list[dict],
        timestamp: float = None,
        step_id: str = ""
    ):
        self.tool_name = tool_name
        self.validation_errors = validation_errors
        self.type = "tool_validation"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class ContextWriteEvent(BaseEvent):
    key: str
    value_preview: str
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


@dataclass(init=False)
class ErrorEvent(BaseEvent):
    error_type: str
    error_message: str
    recoverable: bool = False
    partial_data: Any = None

    def __init__(self, error_type: str, error_message: str, recoverable: bool = False, partial_data: Any = None, timestamp: float = None, step_id: str = ""):
        self.error_type = error_type
        self.error_message = error_message
        self.recoverable = recoverable
        self.partial_data = partial_data
        self.type = "error"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class PatternStartEvent(BaseEvent):
    """Detected during streaming before content accumulation."""
    pattern_name: str
    pattern_type: str

    def __init__(self, pattern_name: str, pattern_type: str, timestamp: float = None, step_id: str = ""):
        self.pattern_name = pattern_name
        self.pattern_type = pattern_type
        self.type = "pattern_start"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class PatternContentEvent(BaseEvent):
    """Streamed before end tag is detected."""
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


@dataclass(init=False)
class PatternEndEvent(BaseEvent):
    """Contains fully accumulated content after end tag detection."""
    pattern_name: str
    pattern_type: str
    full_content: str

    def __init__(self, pattern_name: str, pattern_type: str, full_content: str, timestamp: float = None, step_id: str = ""):
        self.pattern_name = pattern_name
        self.pattern_type = pattern_type
        self.full_content = full_content
        self.type = "pattern_end"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class StepCompleteEvent(BaseEvent):
    """Contains final aggregated result."""
    result: AgentStepResult

    def __init__(self, result: AgentStepResult, timestamp: float = None, step_id: str = ""):
        self.result = result
        self.type = "step_complete"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class RetryEvent(BaseEvent):
    operation_type: str  # "llm" | "tool" | "custom"
    operation_name: str
    attempt: int
    max_attempts: int
    error: str
    next_delay_seconds: float

    def __init__(
        self,
        operation_type: str,
        operation_name: str,
        attempt: int,
        max_attempts: int,
        error: str,
        next_delay_seconds: float,
        timestamp: float = None,
        step_id: str = ""
    ):
        self.operation_type = operation_type
        self.operation_name = operation_name
        self.attempt = attempt
        self.max_attempts = max_attempts
        self.error = error
        self.next_delay_seconds = next_delay_seconds
        self.type = "retry"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class RateLimitEvent(BaseEvent):
    operation_name: str
    acquired_at: float
    tokens_remaining: float

    def __init__(
        self,
        operation_name: str,
        acquired_at: float,
        tokens_remaining: float,
        timestamp: float = None,
        step_id: str = ""
    ):
        self.operation_name = operation_name
        self.acquired_at = acquired_at
        self.tokens_remaining = tokens_remaining
        self.type = "rate_limit"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class ContextHealthEvent(BaseEvent):
    check_type: str  # "size" | "version_count" | "growth_rate"
    key: str
    current_value: float
    threshold: float
    recommended_action: str

    def __init__(
        self,
        check_type: str,
        key: str,
        current_value: float,
        threshold: float,
        recommended_action: str,
        timestamp: float = None,
        step_id: str = ""
    ):
        self.check_type = check_type
        self.key = key
        self.current_value = current_value
        self.threshold = threshold
        self.recommended_action = recommended_action
        self.type = "context_health"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = step_id


@dataclass(init=False)
class GraphStartEvent(BaseEvent):
    """Emitted when graph execution begins."""
    graph_id: str
    total_nodes: int

    def __init__(self, graph_id: str, total_nodes: int, timestamp: float = None):
        self.graph_id = graph_id
        self.total_nodes = total_nodes
        self.type = "graph_start"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = ""  # Graph-wide, not per agent step


@dataclass(init=False)
class GraphNodeStartEvent(BaseEvent):
    """Emitted when a graph node begins execution."""
    graph_id: str
    node_id: str
    parents: list[str]

    def __init__(self, graph_id: str, node_id: str, parents: list[str], timestamp: float = None):
        self.graph_id = graph_id
        self.node_id = node_id
        self.parents = parents
        self.type = "graph_node_start"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = ""


@dataclass(init=False)
class GraphNodeCompleteEvent(BaseEvent):
    """Emitted when a graph node completes (success or failure)."""
    graph_id: str
    node_id: str
    status: Any  # GraphNodeStatus enum
    error_message: str | None = None

    def __init__(
        self,
        graph_id: str,
        node_id: str,
        status: Any,
        error_message: str | None = None,
        timestamp: float = None
    ):
        self.graph_id = graph_id
        self.node_id = node_id
        self.status = status
        self.error_message = error_message
        self.type = "graph_node_complete"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = ""


@dataclass(init=False)
class GraphCompleteEvent(BaseEvent):
    """Emitted when entire graph execution completes."""
    graph_id: str
    status: str  # "success" | "partial_failure" | "failed"
    stats: dict[str, int]  # {status_name: count}

    def __init__(
        self,
        graph_id: str,
        status: str,
        stats: dict[str, int],
        timestamp: float = None
    ):
        self.graph_id = graph_id
        self.status = status
        self.stats = stats
        self.type = "graph_complete"
        self.timestamp = timestamp or now_timestamp()
        self.step_id = ""


AgentEvent = (
    LLMChunkEvent | LLMCompleteEvent | StatusEvent |
    ToolStartEvent | ToolDecisionEvent | ToolOutputEvent | ToolEndEvent | ToolValidationEvent |
    ContextWriteEvent | ErrorEvent |
    PatternStartEvent | PatternContentEvent | PatternEndEvent |
    StepCompleteEvent | RetryEvent | RateLimitEvent | ContextHealthEvent |
    GraphStartEvent | GraphNodeStartEvent | GraphNodeCompleteEvent | GraphCompleteEvent
)
