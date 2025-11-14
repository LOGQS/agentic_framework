"""
Core types, enums, and data structures used throughout the framework.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING
import time
import uuid

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
    WAITING_FOR_TOOL = "waiting_for_tool"
    TOOL_EXECUTED = "tool_executed"
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
    output: dict[str, Any] | str | bytes
    success: bool
    error_message: str | None = None
    execution_time: float = 0.0
    iteration: int = 0


@dataclass
class ExtractedSegments:
    """Segments extracted from agent output via patterns."""
    tools: list[ToolCall] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    response: str | None = None


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


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    provider: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    tools_allowed: list[str] = field(default_factory=list)
    input_mapping: list[dict[str, Any]] = field(default_factory=list)
    output_mapping: list[tuple[str, str]] = field(default_factory=list)
    pattern_set: str | None = None
    auto_increment_iteration: bool = True
    processing_mode: ProcessingMode | None = None  # None means inherit from parent 
    incremental_context_writes: bool = False  # Enable context updates during streaming
    stream_pattern_content: bool = False  # Enable streaming pattern content before end tag (default: wait for complete patterns)
    on_tool_detected: Any = None  # Callable[[ToolCall], bool] - callback to control tool execution (default: auto-execute)
    concurrent_tool_execution: bool = False  # Execute tools concurrently during LLM streaming (default: execute after LLM completes)
    prompt_builder: Callable[["ContextManager", "AgentConfig", str | None], PromptType] | None = None


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
                record = context.get(context_key)
                if record is None:
                    continue
                try:
                    content = record.value.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
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
