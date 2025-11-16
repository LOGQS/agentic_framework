"""
Aegeantic Framework: A robust agentic system with versioned context and RocksDB storage.
"""

# Core types and enums
from .core import (
    ProcessingMode,
    SegmentType,
    AgentStatus,
    ToolCall,
    ToolResult,
    ToolExecutionDecision,
    ExtractedSegments,
    AgentStepResult,
    AgentConfig,
    PromptType,
    PromptObject,
    now_timestamp,
    new_uuid,
    create_message_prompt_builder,
    serialize_tool_output,
    output_to_string
)

# Validation system
from .validation import (
    ValidatorRegistry,
    ValidationError,
    ValidatorFunc,
    simple_validator,
    passthrough_validator
)

# Event system
from .events import (
    BaseEvent,
    AgentEvent,
    LLMChunkEvent,
    LLMCompleteEvent,
    StatusEvent,
    ToolStartEvent,
    ToolDecisionEvent,
    ToolOutputEvent,
    ToolEndEvent,
    ToolValidationEvent,
    ContextWriteEvent,
    ErrorEvent,
    PatternStartEvent,
    PatternContentEvent,
    PatternEndEvent,
    StepCompleteEvent,
    RetryEvent,
    RateLimitEvent,
    ContextHealthEvent
)

# Storage layer
from .storage import (
    StorageConfig,
    RocksDBStorage,
    InMemoryStorage
)

# Context management
from .context import (
    ContextRecord,
    IterationManager,
    ContextManager
)

# Pattern extraction
from .patterns import (
    Pattern,
    PatternSet,
    PatternRegistry,
    PatternExtractor,
    StreamingPatternExtractor,
    create_default_pattern_set,
    create_json_tools_pattern_set,
    create_xml_tools_pattern_set,
    create_backtick_tools_pattern_set
)

# Tools
from .tools import (
    Tool,
    ToolDefinition,
    ToolRegistry,
    create_tool
)

# Agent
from .agent import (
    LLMProvider,
    Agent,
    AgentRunner
)

# Logic flows
from .logic import (
    LogicCondition,
    LogicConfig,
    LogicRunner,
    ContextHealthCheck,
    loop_n_times,
    loop_until_pattern,
    loop_until_regex,
    stop_on_error
)

# Resilience utilities
from .resilience import (
    RetryConfig,
    RateLimitConfig,
    RateLimiter,
    retry_stream,
    rate_limited_stream,
    resilient_stream
)

# Multi-agent patterns
from .multi_agent import (
    AgentChain,
    AgentChainConfig,
    SupervisorPattern,
    SupervisorConfig,
    ParallelPattern,
    ParallelConfig,
    DebatePattern,
    DebateConfig
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "ProcessingMode",
    "SegmentType",
    "AgentStatus",
    "ToolCall",
    "ToolResult",
    "ToolExecutionDecision",
    "ExtractedSegments",
    "AgentStepResult",
    "AgentConfig",
    "PromptType",
    "PromptObject",
    "now_timestamp",
    "new_uuid",
    "create_message_prompt_builder",
    "serialize_tool_output",
    "output_to_string",
    # Validation
    "ValidatorRegistry",
    "ValidationError",
    "ValidatorFunc",
    "simple_validator",
    "passthrough_validator",
    # Events
    "BaseEvent",
    "AgentEvent",
    "LLMChunkEvent",
    "LLMCompleteEvent",
    "StatusEvent",
    "ToolStartEvent",
    "ToolDecisionEvent",
    "ToolOutputEvent",
    "ToolEndEvent",
    "ToolValidationEvent",
    "ContextWriteEvent",
    "ErrorEvent",
    "PatternStartEvent",
    "PatternContentEvent",
    "PatternEndEvent",
    "StepCompleteEvent",
    "RetryEvent",
    "RateLimitEvent",
    "ContextHealthEvent",
    # Storage
    "StorageConfig",
    "RocksDBStorage",
    "InMemoryStorage",
    # Context
    "ContextRecord",
    "IterationManager",
    "ContextManager",
    # Patterns
    "Pattern",
    "PatternSet",
    "PatternRegistry",
    "PatternExtractor",
    "StreamingPatternExtractor",
    "create_default_pattern_set",
    "create_json_tools_pattern_set",
    "create_xml_tools_pattern_set",
    "create_backtick_tools_pattern_set",
    # Tools
    "Tool",
    "ToolDefinition",
    "ToolRegistry",
    "create_tool",
    # Agent
    "LLMProvider",
    "Agent",
    "AgentRunner",
    # Logic
    "LogicCondition",
    "LogicConfig",
    "LogicRunner",
    "ContextHealthCheck",
    "loop_n_times",
    "loop_until_pattern",
    "loop_until_regex",
    "stop_on_error",
    # Resilience
    "RetryConfig",
    "RateLimitConfig",
    "RateLimiter",
    "retry_stream",
    "rate_limited_stream",
    "resilient_stream",
    # Multi-agent
    "AgentChain",
    "AgentChainConfig",
    "SupervisorPattern",
    "SupervisorConfig",
    "ParallelPattern",
    "ParallelConfig",
    "DebatePattern",
    "DebateConfig",
]
