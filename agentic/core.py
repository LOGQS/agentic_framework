"""
Core types, enums, and data structures used throughout the framework.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING
import time
import uuid
import json

if TYPE_CHECKING:
    from .context import ContextManager


PromptType = Any


class ProcessingMode(Enum):
    """Execution mode for tools and agents."""
    PROCESS = "process"
    THREAD = "thread"
    ASYNC = "async"


class SegmentType(Enum):
    """Type of extracted segment from LLM output."""
    TOOL = "tool"
    REASONING = "reasoning"
    RESPONSE = "response"


class AgentStatus(Enum):
    """Status of agent execution."""
    OK = "ok"
    WAITING_FOR_VERIFICATION = "waiting_for_verification"  # Tool awaiting accept/reject decision
    WAITING_FOR_TOOL = "waiting_for_tool"
    TOOL_EXECUTED = "tool_executed"
    TOOLS_REJECTED = "tools_rejected"  # All detected tools were rejected
    VALIDATION_ERROR = "validation_error"
    DONE = "done"
    ERROR = "error"

@dataclass
class PromptObject:
    """Structured prompt with system instruction and message list."""
    system: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolCall:
    """Represents a tool invocation extracted from agent output."""
    name: str
    arguments: dict[str, Any]
    raw_segment: str
    iteration: int
    call_id: str = ""  # Unique identifier for this specific tool invocation


@dataclass
class ToolResult:
    """Result of tool execution."""
    name: str
    output: dict[str, Any] | str | bytes | list[Any] | None
    success: bool
    error_message: str | None = None
    execution_time: float = 0.0
    iteration: int = 0
    call_id: str = ""


@dataclass
class ToolExecutionDecision:
    """
    Tracks complete lifecycle of a detected tool call.

    Provides transparency into verification and execution phases.
    """
    tool_call: ToolCall

    # Verification phase (if on_tool_detected callback is set)
    verification_required: bool
    accepted: bool
    rejection_reason: str | None = None
    verification_duration_ms: float = 0.0

    # Execution phase
    executed: bool = False
    result: ToolResult | None = None


def serialize_tool_output(output: Any) -> Any:
    """
    Serialize tool output for JSON storage, preserving native types.
    """
    if output is None:
        return None
    elif isinstance(output, (dict, list, str, int, float, bool)):
        return output
    elif isinstance(output, bytes):
        return {"_type": "bytes", "_hex": output.hex()}
    else:
        return {"_type": "string", "_value": str(output)}


def output_to_string(output: Any) -> str:
    """
    Convert tool output to string for pattern matching and display.
    Used in logic conditions and anywhere we need text representation.
    """
    if output is None:
        return ""
    elif isinstance(output, str):
        return output
    elif isinstance(output, bytes):
        try:
            return output.decode('utf-8')
        except (UnicodeDecodeError, AttributeError):
            return output.hex()
    elif isinstance(output, (dict, list)):
        return json.dumps(output, indent=2)
    else:
        return str(output)


@dataclass
class ExtractedSegments:
    """Segments extracted from agent output via patterns."""
    tools: list[ToolCall] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    response: str | None = None
    parse_errors: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentStepResult:
    """Complete result of a single agent execution step."""
    status: AgentStatus
    raw_output: str
    segments: ExtractedSegments
    tool_results: list[ToolResult]
    iteration: int
    error_message: str | None = None  # Populated when status is ERROR
    error_type: str | None = None  # e.g., "llm_error", "tool_execution_error", "tool_not_found"
    partial_malformed_patterns: dict[str, str] | None = None  # Malformed pattern content (live DB updates reverted, kept in-memory only)
    tool_decisions: list[ToolExecutionDecision] = field(default_factory=list)  # Full tool lifecycle tracking


@dataclass
class AgentConfig:
    """
    Configuration for an agent.
    """
    agent_id: str
    tools_allowed: list[str] = field(default_factory=list)
    tool_name_mapping: dict[str, str] = field(default_factory=dict)
    validate_tool_arguments: bool = True
    input_mapping: list[dict[str, Any]] = field(default_factory=list)
    output_mapping: list[tuple[str, str]] = field(default_factory=list)
    pattern_set: str | None = None
    auto_increment_iteration: bool = True
    processing_mode: ProcessingMode | None = None  # None means inherit from parent
    incremental_context_writes: bool = False
    stream_pattern_content: bool = False
    on_tool_detected: Any = None  # Callable[[ToolCall], bool] - if set, enables verification workflow
    concurrent_tool_execution: bool = False
    max_partial_buffer_size: int = 10_000_000
    prompt_builder: Callable[["ContextManager", "AgentConfig", str | None], PromptType] | None = None
    tool_verification_timeout: float | None = None  # Seconds to wait for on_tool_detected; None = indefinite
    tool_verification_on_timeout: str = "reject"  # "reject" | "accept" - what to do if verification times out

    def __post_init__(self):
        """Validate configuration values."""
        if self.tool_verification_on_timeout not in ("accept", "reject"):
            raise ValueError(
                f"tool_verification_on_timeout must be 'accept' or 'reject', got: {self.tool_verification_on_timeout!r}"
            )


def now_timestamp() -> float:
    return time.time()


def new_uuid() -> str:
    return str(uuid.uuid4())


def create_message_prompt_builder() -> Callable[["ContextManager", "AgentConfig", str | None], PromptObject]:
    """
    Reference prompt builder that constructs PromptObject from input_mapping.

    Routes entries with role="system" to system field, others to messages list.
    Sorts by "order" field. Supports "literal:" prefix for static content.
    """
    def builder(context: "ContextManager", config: "AgentConfig", user_input: str | None) -> PromptObject:
        system_parts = []
        messages = []

        mapping_entries = [m for m in config.input_mapping if isinstance(m, dict)]

        for mapping in sorted(mapping_entries, key=lambda x: x.get("order", 0)):
            context_key = mapping.get("context_key", "")
            role = mapping.get("role", "user")

            if context_key.startswith("literal:"):
                content = context_key[8:]
            else:
                content = context.get(context_key)
                if content is None:
                    continue

            if role == "system":
                system_parts.append(content)
            else:
                messages.append({"role": role, "content": content})

        if user_input:
            messages.append({"role": "user", "content": user_input})

        return PromptObject(
            system="\n\n".join(system_parts) if system_parts else None,
            messages=messages
        )

    return builder
