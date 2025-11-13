"""
Agentic Framework: A robust agentic system with versioned context and RocksDB storage.
"""

# Core types and enums
from .core import (
    ProcessingMode,
    SegmentType,
    AgentStatus,
    ToolCall,
    ToolResult,
    ExtractedSegments,
    AgentStepResult,
    AgentConfig,
    now_timestamp,
    new_uuid
)

# Event system
from .events import (
    BaseEvent,
    AgentEvent,
    LLMTokenEvent,
    LLMCompleteEvent,
    StatusEvent,
    ToolStartEvent,
    ToolOutputEvent,
    ToolEndEvent,
    ContextWriteEvent,
    ErrorEvent,
    PatternStartEvent,
    PatternContentEvent,
    PatternEndEvent,
    StepCompleteEvent
)

# Storage layer
from .storage import (
    StorageConfig,
    RocksDBStorage
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
    create_default_pattern_set
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
    AgentRunner,
    MockLLMProvider
)

# Logic flows
from .logic import (
    LogicCondition,
    LogicConfig,
    LogicRunner,
    loop_n_times,
    loop_until_pattern,
    loop_until_regex,
    stop_on_error
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "ProcessingMode",
    "SegmentType",
    "AgentStatus",
    "ToolCall",
    "ToolResult",
    "ExtractedSegments",
    "AgentStepResult",
    "AgentConfig",
    "now_timestamp",
    "new_uuid",
    # Events
    "BaseEvent",
    "AgentEvent",
    "LLMTokenEvent",
    "LLMCompleteEvent",
    "StatusEvent",
    "ToolStartEvent",
    "ToolOutputEvent",
    "ToolEndEvent",
    "ContextWriteEvent",
    "ErrorEvent",
    "PatternStartEvent",
    "PatternContentEvent",
    "PatternEndEvent",
    "StepCompleteEvent",
    # Storage
    "StorageConfig",
    "RocksDBStorage",
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
    # Tools
    "Tool",
    "ToolDefinition",
    "ToolRegistry",
    "create_tool",
    # Agent
    "LLMProvider",
    "Agent",
    "AgentRunner",
    "MockLLMProvider",
    # Logic
    "LogicCondition",
    "LogicConfig",
    "LogicRunner",
    "loop_n_times",
    "loop_until_pattern",
    "loop_until_regex",
    "stop_on_error",
]
