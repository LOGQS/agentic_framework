"""
Tests for core types, enums, and utility functions.

Covers:
- ProcessingMode enum
- SegmentType enum
- AgentStatus enum
- ToolCall dataclass
- ToolResult dataclass
- ExtractedSegments dataclass
- AgentStepResult dataclass
- AgentConfig dataclass
- now_timestamp() function
- new_uuid() function
"""
import pytest
import time
import uuid
from dataclasses import fields

from agentic.core import (
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


class TestProcessingMode:
    """Tests for ProcessingMode enum."""

    def test_processing_mode_values(self):
        """Validate that ProcessingMode has expected values."""
        assert ProcessingMode.PROCESS.value == "process"
        assert ProcessingMode.THREAD.value == "thread"
        assert ProcessingMode.ASYNC.value == "async"

    def test_processing_mode_members(self):
        """Validate that all expected members exist."""
        modes = list(ProcessingMode)
        assert len(modes) == 3
        assert ProcessingMode.PROCESS in modes
        assert ProcessingMode.THREAD in modes
        assert ProcessingMode.ASYNC in modes

    def test_processing_mode_from_string(self):
        """Test creating ProcessingMode from string value."""
        assert ProcessingMode("process") == ProcessingMode.PROCESS
        assert ProcessingMode("thread") == ProcessingMode.THREAD
        assert ProcessingMode("async") == ProcessingMode.ASYNC


class TestSegmentType:
    """Tests for SegmentType enum."""

    def test_segment_type_values(self):
        """Validate that SegmentType has expected values."""
        assert SegmentType.TOOL.value == "tool"
        assert SegmentType.REASONING.value == "reasoning"
        assert SegmentType.RESPONSE.value == "response"

    def test_segment_type_members(self):
        """Validate that all expected members exist."""
        types = list(SegmentType)
        assert len(types) == 3
        assert SegmentType.TOOL in types
        assert SegmentType.REASONING in types
        assert SegmentType.RESPONSE in types

    def test_segment_type_from_string(self):
        """Test creating SegmentType from string value."""
        assert SegmentType("tool") == SegmentType.TOOL
        assert SegmentType("reasoning") == SegmentType.REASONING
        assert SegmentType("response") == SegmentType.RESPONSE


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_agent_status_values(self):
        """Validate that AgentStatus has expected values."""
        assert AgentStatus.OK.value == "ok"
        assert AgentStatus.WAITING_FOR_TOOL.value == "waiting_for_tool"
        assert AgentStatus.TOOL_EXECUTED.value == "tool_executed"
        assert AgentStatus.DONE.value == "done"
        assert AgentStatus.ERROR.value == "error"

    def test_agent_status_members(self):
        """Validate that all expected members exist."""
        statuses = list(AgentStatus)
        assert len(statuses) == 5
        assert AgentStatus.OK in statuses
        assert AgentStatus.WAITING_FOR_TOOL in statuses
        assert AgentStatus.TOOL_EXECUTED in statuses
        assert AgentStatus.DONE in statuses
        assert AgentStatus.ERROR in statuses

    def test_agent_status_from_string(self):
        """Test creating AgentStatus from string value."""
        assert AgentStatus("ok") == AgentStatus.OK
        assert AgentStatus("waiting_for_tool") == AgentStatus.WAITING_FOR_TOOL
        assert AgentStatus("tool_executed") == AgentStatus.TOOL_EXECUTED
        assert AgentStatus("done") == AgentStatus.DONE
        assert AgentStatus("error") == AgentStatus.ERROR


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall instance."""
        tool_call = ToolCall(
            name="test_tool",
            arguments={"arg1": "value1", "arg2": 42},
            raw_segment="<tool>test_tool</tool>",
            iteration=1
        )
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"arg1": "value1", "arg2": 42}
        assert tool_call.raw_segment == "<tool>test_tool</tool>"
        assert tool_call.iteration == 1

    def test_tool_call_with_empty_arguments(self):
        """Test ToolCall with empty arguments dict."""
        tool_call = ToolCall(
            name="no_args_tool",
            arguments={},
            raw_segment="<tool>no_args_tool</tool>",
            iteration=0
        )
        assert tool_call.arguments == {}

    def test_tool_call_fields(self):
        """Validate that ToolCall has expected fields."""
        field_names = {f.name for f in fields(ToolCall)}
        expected = {"name", "arguments", "raw_segment", "iteration"}
        assert field_names == expected


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_success(self):
        """Test creating a successful ToolResult."""
        result = ToolResult(
            name="test_tool",
            output={"result": "success"},
            success=True,
            error_message=None,
            execution_time=1.5,
            iteration=1
        )
        assert result.name == "test_tool"
        assert result.output == {"result": "success"}
        assert result.success is True
        assert result.error_message is None
        assert result.execution_time == 1.5
        assert result.iteration == 1

    def test_tool_result_failure(self):
        """Test creating a failed ToolResult."""
        result = ToolResult(
            name="failed_tool",
            output={},
            success=False,
            error_message="Tool execution failed",
            execution_time=0.5,
            iteration=2
        )
        assert result.success is False
        assert result.error_message == "Tool execution failed"

    def test_tool_result_default_values(self):
        """Test that ToolResult has proper default values."""
        result = ToolResult(
            name="tool",
            output="output",
            success=True
        )
        assert result.error_message is None
        assert result.execution_time == 0.0
        assert result.iteration == 0

    def test_tool_result_with_string_output(self):
        """Test ToolResult with string output."""
        result = ToolResult(
            name="string_tool",
            output="text output",
            success=True
        )
        assert result.output == "text output"

    def test_tool_result_with_bytes_output(self):
        """Test ToolResult with bytes output."""
        result = ToolResult(
            name="bytes_tool",
            output=b"binary data",
            success=True
        )
        assert result.output == b"binary data"


class TestExtractedSegments:
    """Tests for ExtractedSegments dataclass."""

    def test_extracted_segments_empty(self):
        """Test creating empty ExtractedSegments."""
        segments = ExtractedSegments()
        assert segments.tools == []
        assert segments.reasoning == []
        assert segments.response is None

    def test_extracted_segments_with_tools(self):
        """Test ExtractedSegments with tool calls."""
        tool_call = ToolCall("tool1", {}, "<tool>tool1</tool>", 1)
        segments = ExtractedSegments(tools=[tool_call])
        assert len(segments.tools) == 1
        assert segments.tools[0].name == "tool1"

    def test_extracted_segments_with_reasoning(self):
        """Test ExtractedSegments with reasoning."""
        segments = ExtractedSegments(reasoning=["step1", "step2"])
        assert segments.reasoning == ["step1", "step2"]

    def test_extracted_segments_with_response(self):
        """Test ExtractedSegments with response."""
        segments = ExtractedSegments(response="Final answer")
        assert segments.response == "Final answer"

    def test_extracted_segments_complete(self):
        """Test ExtractedSegments with all fields populated."""
        tool_call = ToolCall("tool1", {}, "<tool>tool1</tool>", 1)
        segments = ExtractedSegments(
            tools=[tool_call],
            reasoning=["thinking"],
            response="answer"
        )
        assert len(segments.tools) == 1
        assert len(segments.reasoning) == 1
        assert segments.response == "answer"


class TestAgentStepResult:
    """Tests for AgentStepResult dataclass."""

    def test_agent_step_result_ok(self):
        """Test creating a successful AgentStepResult."""
        segments = ExtractedSegments(response="Success")
        result = AgentStepResult(
            status=AgentStatus.OK,
            raw_output="Raw output",
            segments=segments,
            tool_results=[],
            iteration=1
        )
        assert result.status == AgentStatus.OK
        assert result.raw_output == "Raw output"
        assert result.segments.response == "Success"
        assert result.tool_results == []
        assert result.iteration == 1
        assert result.error_message is None
        assert result.error_type is None

    def test_agent_step_result_with_error(self):
        """Test AgentStepResult with error."""
        result = AgentStepResult(
            status=AgentStatus.ERROR,
            raw_output="Error occurred",
            segments=ExtractedSegments(),
            tool_results=[],
            iteration=2,
            error_message="LLM failed",
            error_type="llm_error"
        )
        assert result.status == AgentStatus.ERROR
        assert result.error_message == "LLM failed"
        assert result.error_type == "llm_error"

    def test_agent_step_result_with_tool_execution(self):
        """Test AgentStepResult with tool execution."""
        tool_result = ToolResult("tool", {"result": "ok"}, True)
        result = AgentStepResult(
            status=AgentStatus.TOOL_EXECUTED,
            raw_output="Tool executed",
            segments=ExtractedSegments(),
            tool_results=[tool_result],
            iteration=3
        )
        assert result.status == AgentStatus.TOOL_EXECUTED
        assert len(result.tool_results) == 1
        assert result.tool_results[0].name == "tool"

    def test_agent_step_result_with_malformed_patterns(self):
        """Test AgentStepResult with malformed patterns."""
        result = AgentStepResult(
            status=AgentStatus.OK,
            raw_output="Output with malformed",
            segments=ExtractedSegments(),
            tool_results=[],
            iteration=1,
            partial_malformed_patterns={"tool": "incomplete content"}
        )
        assert result.partial_malformed_patterns == {"tool": "incomplete content"}


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_agent_config_minimal(self):
        """Test creating AgentConfig with required fields only."""
        config = AgentConfig(
            agent_id="agent1",
            provider="openai",
            model="gpt-4"
        )
        assert config.agent_id == "agent1"
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.max_tokens == 4096  # default
        assert config.temperature == 0.7  # default
        assert config.tools_allowed == []  # default
        assert config.auto_increment_iteration is True  # default

    def test_agent_config_full(self):
        """Test creating AgentConfig with all fields."""
        config = AgentConfig(
            agent_id="agent2",
            provider="anthropic",
            model="claude-3",
            max_tokens=8192,
            temperature=0.5,
            tools_allowed=["tool1", "tool2"],
            input_mapping=[("context1", "prepend")],
            output_mapping=[("output1", "set_latest")],
            pattern_set="custom",
            auto_increment_iteration=False,
            processing_mode=ProcessingMode.ASYNC,
            incremental_context_writes=True,
            stream_pattern_content=True,
            on_tool_detected=lambda x: True
        )
        assert config.max_tokens == 8192
        assert config.temperature == 0.5
        assert config.tools_allowed == ["tool1", "tool2"]
        assert config.pattern_set == "custom"
        assert config.auto_increment_iteration is False
        assert config.processing_mode == ProcessingMode.ASYNC
        assert config.incremental_context_writes is True
        assert config.stream_pattern_content is True
        assert config.on_tool_detected is not None

    def test_agent_config_mappings(self):
        """Test AgentConfig input and output mappings."""
        input_map = [("system", "prepend"), ("user", "append")]
        output_map = [("result", "set_latest"), ("history", "append_version")]
        config = AgentConfig(
            agent_id="agent3",
            provider="test",
            model="test-model",
            input_mapping=input_map,
            output_mapping=output_map
        )
        assert config.input_mapping == input_map
        assert config.output_mapping == output_map


class TestUtilityFunctions:
    """Tests for utility functions in core module."""

    def test_now_timestamp_returns_float(self):
        """Test that now_timestamp returns a float."""
        ts = now_timestamp()
        assert isinstance(ts, float)

    def test_now_timestamp_is_current(self):
        """Test that now_timestamp returns current time."""
        before = time.time()
        ts = now_timestamp()
        after = time.time()
        assert before <= ts <= after

    def test_now_timestamp_precision(self):
        """Test that now_timestamp has subsecond precision."""
        ts1 = now_timestamp()
        time.sleep(0.001)  # Sleep 1ms
        ts2 = now_timestamp()
        assert ts2 > ts1

    def test_new_uuid_returns_string(self):
        """Test that new_uuid returns a string."""
        uid = new_uuid()
        assert isinstance(uid, str)

    def test_new_uuid_format(self):
        """Test that new_uuid returns valid UUID format."""
        uid = new_uuid()
        # Should be parseable as UUID
        parsed = uuid.UUID(uid)
        assert str(parsed) == uid

    def test_new_uuid_uniqueness(self):
        """Test that new_uuid generates unique values."""
        uuids = [new_uuid() for _ in range(100)]
        # All should be unique
        assert len(set(uuids)) == 100

    def test_new_uuid_version(self):
        """Test that new_uuid generates UUID4."""
        uid = new_uuid()
        parsed = uuid.UUID(uid)
        # UUID4 has version 4
        assert parsed.version == 4


class TestDataclassImmutability:
    """Tests to ensure dataclasses are properly frozen or mutable as intended."""

    def test_tool_call_is_mutable(self):
        """Test that ToolCall is mutable (can modify fields)."""
        tool_call = ToolCall("tool", {}, "raw", 1)
        tool_call.iteration = 2
        assert tool_call.iteration == 2

    def test_tool_result_is_mutable(self):
        """Test that ToolResult is mutable."""
        result = ToolResult("tool", "output", True)
        result.execution_time = 5.0
        assert result.execution_time == 5.0

    def test_extracted_segments_list_modification(self):
        """Test that ExtractedSegments lists can be modified."""
        segments = ExtractedSegments()
        tool_call = ToolCall("tool", {}, "raw", 1)
        segments.tools.append(tool_call)
        assert len(segments.tools) == 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_tool_call_with_none_in_arguments(self):
        """Test ToolCall with None values in arguments."""
        tool_call = ToolCall(
            name="tool",
            arguments={"key": None},
            raw_segment="raw",
            iteration=0
        )
        assert tool_call.arguments["key"] is None

    def test_tool_result_with_empty_string_error(self):
        """Test ToolResult with empty string error message."""
        result = ToolResult(
            name="tool",
            output="",
            success=False,
            error_message=""
        )
        assert result.error_message == ""

    def test_agent_config_empty_lists(self):
        """Test AgentConfig with explicitly empty lists."""
        config = AgentConfig(
            agent_id="agent",
            provider="test",
            model="test",
            tools_allowed=[],
            input_mapping=[],
            output_mapping=[]
        )
        assert config.tools_allowed == []
        assert config.input_mapping == []
        assert config.output_mapping == []

    def test_extracted_segments_multiple_tools(self):
        """Test ExtractedSegments with multiple tool calls."""
        tools = [
            ToolCall(f"tool{i}", {}, f"raw{i}", i)
            for i in range(10)
        ]
        segments = ExtractedSegments(tools=tools)
        assert len(segments.tools) == 10

    def test_agent_step_result_large_iteration(self):
        """Test AgentStepResult with large iteration number."""
        result = AgentStepResult(
            status=AgentStatus.OK,
            raw_output="output",
            segments=ExtractedSegments(),
            tool_results=[],
            iteration=999999
        )
        assert result.iteration == 999999
