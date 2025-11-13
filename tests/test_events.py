"""
Tests for event system.

Covers:
- BaseEvent and timestamp handling
- LLMTokenEvent
- LLMCompleteEvent
- StatusEvent
- ToolStartEvent, ToolOutputEvent, ToolEndEvent
- ContextWriteEvent
- ErrorEvent
- PatternStartEvent, PatternContentEvent, PatternEndEvent
- StepCompleteEvent
"""
import pytest
import time

from agentic.events import (
    BaseEvent,
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
from agentic.core import (
    AgentStatus,
    ToolResult,
    AgentStepResult,
    ExtractedSegments
)


class TestBaseEvent:
    """Tests for BaseEvent base class."""

    def test_base_event_with_timestamp(self):
        """Test creating BaseEvent with explicit timestamp."""
        ts = 123456.789
        event = BaseEvent(type="test", timestamp=ts)
        assert event.type == "test"
        assert event.timestamp == ts

    def test_base_event_auto_timestamp(self):
        """Test that BaseEvent auto-generates timestamp if not provided."""
        before = time.time()
        event = BaseEvent(type="test", timestamp=None)
        after = time.time()

        assert before <= event.timestamp <= after


class TestLLMTokenEvent:
    """Tests for LLMTokenEvent."""

    def test_llm_token_event_creation(self):
        """Test creating LLMTokenEvent."""
        event = LLMTokenEvent(token="Hello")
        assert event.token == "Hello"
        assert event.type == "llm_token"
        assert isinstance(event.timestamp, float)

    def test_llm_token_event_with_timestamp(self):
        """Test LLMTokenEvent with explicit timestamp."""
        ts = 123456.0
        event = LLMTokenEvent(token="test", timestamp=ts)
        assert event.timestamp == ts

    def test_llm_token_event_empty_token(self):
        """Test LLMTokenEvent with empty token."""
        event = LLMTokenEvent(token="")
        assert event.token == ""

    def test_llm_token_event_multichar(self):
        """Test LLMTokenEvent with multi-character token."""
        event = LLMTokenEvent(token="multiple words")
        assert event.token == "multiple words"


class TestLLMCompleteEvent:
    """Tests for LLMCompleteEvent."""

    def test_llm_complete_event_creation(self):
        """Test creating LLMCompleteEvent."""
        event = LLMCompleteEvent(full_text="Complete response")
        assert event.full_text == "Complete response"
        assert event.type == "llm_complete"
        assert isinstance(event.timestamp, float)

    def test_llm_complete_event_with_timestamp(self):
        """Test LLMCompleteEvent with explicit timestamp."""
        ts = 123456.0
        event = LLMCompleteEvent(full_text="text", timestamp=ts)
        assert event.timestamp == ts

    def test_llm_complete_event_empty_text(self):
        """Test LLMCompleteEvent with empty text."""
        event = LLMCompleteEvent(full_text="")
        assert event.full_text == ""


class TestStatusEvent:
    """Tests for StatusEvent."""

    def test_status_event_creation(self):
        """Test creating StatusEvent."""
        event = StatusEvent(status=AgentStatus.OK)
        assert event.status == AgentStatus.OK
        assert event.message is None
        assert event.type == "status"

    def test_status_event_with_message(self):
        """Test StatusEvent with message."""
        event = StatusEvent(
            status=AgentStatus.WAITING_FOR_TOOL,
            message="Executing tool"
        )
        assert event.status == AgentStatus.WAITING_FOR_TOOL
        assert event.message == "Executing tool"

    def test_status_event_with_timestamp(self):
        """Test StatusEvent with explicit timestamp."""
        ts = 123456.0
        event = StatusEvent(status=AgentStatus.DONE, timestamp=ts)
        assert event.timestamp == ts

    def test_status_event_all_statuses(self):
        """Test StatusEvent with all status types."""
        for status in AgentStatus:
            event = StatusEvent(status=status)
            assert event.status == status


class TestToolStartEvent:
    """Tests for ToolStartEvent."""

    def test_tool_start_event_creation(self):
        """Test creating ToolStartEvent."""
        event = ToolStartEvent(
            tool_name="test_tool",
            arguments={"arg1": "value1"},
            iteration=1
        )
        assert event.tool_name == "test_tool"
        assert event.arguments == {"arg1": "value1"}
        assert event.iteration == 1
        assert event.type == "tool_start"

    def test_tool_start_event_empty_arguments(self):
        """Test ToolStartEvent with empty arguments."""
        event = ToolStartEvent(
            tool_name="tool",
            arguments={},
            iteration=0
        )
        assert event.arguments == {}


class TestToolOutputEvent:
    """Tests for ToolOutputEvent."""

    def test_tool_output_event_creation(self):
        """Test creating ToolOutputEvent."""
        event = ToolOutputEvent(
            tool_name="tool",
            output={"result": "success"}
        )
        assert event.tool_name == "tool"
        assert event.output == {"result": "success"}
        assert event.is_partial is False
        assert event.type == "tool_output"

    def test_tool_output_event_partial(self):
        """Test ToolOutputEvent marked as partial."""
        event = ToolOutputEvent(
            tool_name="tool",
            output="chunk",
            is_partial=True
        )
        assert event.is_partial is True

    def test_tool_output_event_string_output(self):
        """Test ToolOutputEvent with string output."""
        event = ToolOutputEvent(tool_name="tool", output="text output")
        assert event.output == "text output"

    def test_tool_output_event_bytes_output(self):
        """Test ToolOutputEvent with bytes output."""
        event = ToolOutputEvent(tool_name="tool", output=b"binary")
        assert event.output == b"binary"


class TestToolEndEvent:
    """Tests for ToolEndEvent."""

    def test_tool_end_event_success(self):
        """Test ToolEndEvent with successful result."""
        result = ToolResult(
            name="tool",
            output={"result": "ok"},
            success=True,
            execution_time=1.5
        )
        event = ToolEndEvent(tool_name="tool", result=result)
        assert event.tool_name == "tool"
        assert event.result.success is True
        assert event.type == "tool_end"

    def test_tool_end_event_failure(self):
        """Test ToolEndEvent with failed result."""
        result = ToolResult(
            name="tool",
            output={},
            success=False,
            error_message="Failed"
        )
        event = ToolEndEvent(tool_name="tool", result=result)
        assert event.result.success is False
        assert event.result.error_message == "Failed"


class TestContextWriteEvent:
    """Tests for ContextWriteEvent."""

    def test_context_write_event_creation(self):
        """Test creating ContextWriteEvent."""
        event = ContextWriteEvent(
            key="test_key",
            value_preview="preview...",
            version=2,
            iteration=1
        )
        assert event.key == "test_key"
        assert event.value_preview == "preview..."
        assert event.version == 2
        assert event.iteration == 1
        assert event.type == "context_write"

    def test_context_write_event_long_preview(self):
        """Test ContextWriteEvent with long preview."""
        long_text = "x" * 200
        event = ContextWriteEvent(
            key="key",
            value_preview=long_text,
            version=1,
            iteration=0
        )
        assert len(event.value_preview) == 200


class TestErrorEvent:
    """Tests for ErrorEvent."""

    def test_error_event_creation(self):
        """Test creating ErrorEvent."""
        event = ErrorEvent(
            error_type="test_error",
            error_message="Something went wrong"
        )
        assert event.error_type == "test_error"
        assert event.error_message == "Something went wrong"
        assert event.recoverable is False
        assert event.partial_data is None
        assert event.type == "error"

    def test_error_event_recoverable(self):
        """Test ErrorEvent marked as recoverable."""
        event = ErrorEvent(
            error_type="warning",
            error_message="Recoverable error",
            recoverable=True
        )
        assert event.recoverable is True

    def test_error_event_with_partial_data(self):
        """Test ErrorEvent with partial data."""
        partial = {"incomplete": "data"}
        event = ErrorEvent(
            error_type="malformed_pattern",
            error_message="Pattern incomplete",
            recoverable=True,
            partial_data=partial
        )
        assert event.partial_data == partial

    def test_error_event_types(self):
        """Test ErrorEvent with different error types."""
        error_types = [
            "llm_error",
            "tool_execution_error",
            "tool_not_found",
            "tool_timeout",
            "malformed_pattern"
        ]
        for error_type in error_types:
            event = ErrorEvent(error_type=error_type, error_message="error")
            assert event.error_type == error_type


class TestPatternStartEvent:
    """Tests for PatternStartEvent."""

    def test_pattern_start_event_creation(self):
        """Test creating PatternStartEvent."""
        event = PatternStartEvent(
            pattern_name="tool",
            pattern_type="tool"
        )
        assert event.pattern_name == "tool"
        assert event.pattern_type == "tool"
        assert event.type == "pattern_start"

    def test_pattern_start_event_types(self):
        """Test PatternStartEvent with different pattern types."""
        pattern_types = ["tool", "reasoning", "response"]
        for ptype in pattern_types:
            event = PatternStartEvent(pattern_name="test", pattern_type=ptype)
            assert event.pattern_type == ptype


class TestPatternContentEvent:
    """Tests for PatternContentEvent."""

    def test_pattern_content_event_creation(self):
        """Test creating PatternContentEvent."""
        event = PatternContentEvent(
            pattern_name="tool",
            content="partial content"
        )
        assert event.pattern_name == "tool"
        assert event.content == "partial content"
        assert event.is_partial is True
        assert event.type == "pattern_content"

    def test_pattern_content_event_complete(self):
        """Test PatternContentEvent marked as complete."""
        event = PatternContentEvent(
            pattern_name="tool",
            content="complete",
            is_partial=False
        )
        assert event.is_partial is False

    def test_pattern_content_event_empty(self):
        """Test PatternContentEvent with empty content."""
        event = PatternContentEvent(pattern_name="tool", content="")
        assert event.content == ""


class TestPatternEndEvent:
    """Tests for PatternEndEvent."""

    def test_pattern_end_event_creation(self):
        """Test creating PatternEndEvent."""
        event = PatternEndEvent(
            pattern_name="tool",
            pattern_type="tool",
            full_content="complete tool call"
        )
        assert event.pattern_name == "tool"
        assert event.pattern_type == "tool"
        assert event.full_content == "complete tool call"
        assert event.type == "pattern_end"

    def test_pattern_end_event_empty_content(self):
        """Test PatternEndEvent with empty content."""
        event = PatternEndEvent(pattern_name="tool", pattern_type="tool", full_content="")
        assert event.full_content == ""


class TestStepCompleteEvent:
    """Tests for StepCompleteEvent."""

    def test_step_complete_event_creation(self):
        """Test creating StepCompleteEvent."""
        result = AgentStepResult(
            status=AgentStatus.OK,
            raw_output="output",
            segments=ExtractedSegments(),
            tool_results=[],
            iteration=1
        )
        event = StepCompleteEvent(result=result)
        assert event.result == result
        assert event.type == "step_complete"

    def test_step_complete_event_with_error(self):
        """Test StepCompleteEvent with error result."""
        result = AgentStepResult(
            status=AgentStatus.ERROR,
            raw_output="error",
            segments=ExtractedSegments(),
            tool_results=[],
            iteration=1,
            error_message="Failed",
            error_type="test_error"
        )
        event = StepCompleteEvent(result=result)
        assert event.result.status == AgentStatus.ERROR
        assert event.result.error_message == "Failed"


class TestEventTimestamps:
    """Tests for event timestamp behavior."""

    def test_events_have_timestamps(self):
        """Test that all event types have timestamps."""
        events = [
            LLMTokenEvent(token="test"),
            LLMCompleteEvent(full_text="test"),
            StatusEvent(status=AgentStatus.OK),
            ToolStartEvent("tool", {}, 0),
            ToolOutputEvent("tool", "output"),
            ToolEndEvent("tool", ToolResult("tool", "out", True)),
            ContextWriteEvent("key", "preview", 1, 0),
            ErrorEvent("error", "message"),
            PatternStartEvent("pattern", "tool"),
            PatternContentEvent("pattern", "content"),
            PatternEndEvent("pattern", "tool", "content"),
            StepCompleteEvent(AgentStepResult(
                AgentStatus.OK, "out", ExtractedSegments(), [], 0
            ))
        ]

        for event in events:
            assert hasattr(event, 'timestamp')
            assert isinstance(event.timestamp, float)
            assert event.timestamp > 0

    def test_timestamp_precision(self):
        """Test that timestamps have subsecond precision."""
        event1 = LLMTokenEvent(token="a")
        time.sleep(0.001)
        event2 = LLMTokenEvent(token="b")

        assert event2.timestamp > event1.timestamp

    def test_explicit_timestamps(self):
        """Test that explicit timestamps are respected."""
        ts = 999999.123
        events = [
            LLMTokenEvent(token="test", timestamp=ts),
            LLMCompleteEvent(full_text="test", timestamp=ts),
            StatusEvent(status=AgentStatus.OK, timestamp=ts),
            ToolStartEvent("tool", {}, 0, timestamp=ts),
            ToolOutputEvent("tool", "output", timestamp=ts),
            ToolEndEvent("tool", ToolResult("tool", "out", True), timestamp=ts),
            ContextWriteEvent("key", "preview", 1, 0, timestamp=ts),
            ErrorEvent("error", "message", timestamp=ts),
            PatternStartEvent("pattern", "tool", timestamp=ts),
            PatternContentEvent("pattern", "content", timestamp=ts),
            PatternEndEvent("pattern", "tool", "content", timestamp=ts),
            StepCompleteEvent(AgentStepResult(
                AgentStatus.OK, "out", ExtractedSegments(), [], 0
            ), timestamp=ts)
        ]

        for event in events:
            assert event.timestamp == ts


class TestEventTypes:
    """Tests for event type identifiers."""

    def test_event_type_values(self):
        """Test that all events have correct type values."""
        type_mappings = [
            (LLMTokenEvent(token="t"), "llm_token"),
            (LLMCompleteEvent(full_text="t"), "llm_complete"),
            (StatusEvent(status=AgentStatus.OK), "status"),
            (ToolStartEvent("t", {}, 0), "tool_start"),
            (ToolOutputEvent("t", "o"), "tool_output"),
            (ToolEndEvent("t", ToolResult("t", "o", True)), "tool_end"),
            (ContextWriteEvent("k", "p", 1, 0), "context_write"),
            (ErrorEvent("e", "m"), "error"),
            (PatternStartEvent("p", "t"), "pattern_start"),
            (PatternContentEvent("p", "c"), "pattern_content"),
            (PatternEndEvent("p", "t", "c"), "pattern_end"),
            (StepCompleteEvent(AgentStepResult(
                AgentStatus.OK, "o", ExtractedSegments(), [], 0
            )), "step_complete")
        ]

        for event, expected_type in type_mappings:
            assert event.type == expected_type


class TestEventEdgeCases:
    """Tests for edge cases in events."""

    def test_event_with_none_values(self):
        """Test events with None values where allowed."""
        event = StatusEvent(status=AgentStatus.OK, message=None)
        assert event.message is None

        error_event = ErrorEvent("error", "message", partial_data=None)
        assert error_event.partial_data is None

    def test_event_with_large_data(self):
        """Test events with large data."""
        large_text = "x" * 1000000  # 1MB
        event = LLMCompleteEvent(full_text=large_text)
        assert len(event.full_text) == 1000000

    def test_event_with_complex_arguments(self):
        """Test ToolStartEvent with complex nested arguments."""
        complex_args = {
            "nested": {
                "deep": {
                    "value": [1, 2, 3]
                }
            },
            "list": ["a", "b", "c"]
        }
        event = ToolStartEvent("tool", complex_args, 0)
        assert event.arguments == complex_args

    def test_event_with_unicode(self):
        """Test events with unicode content."""
        unicode_text = "Hello \u4e16\u754c \u00e9\u00f1"
        event = LLMTokenEvent(token=unicode_text)
        assert event.token == unicode_text
