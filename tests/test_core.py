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
    new_uuid,
    serialize_tool_output,
    output_to_string
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
        assert len(statuses) == 8
        assert AgentStatus.OK in statuses
        assert AgentStatus.WAITING_FOR_VERIFICATION in statuses
        assert AgentStatus.WAITING_FOR_TOOL in statuses
        assert AgentStatus.TOOL_EXECUTED in statuses
        assert AgentStatus.TOOLS_REJECTED in statuses
        assert AgentStatus.VALIDATION_ERROR in statuses
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
        expected = {"name", "arguments", "raw_segment", "iteration", "call_id"}
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
        """Test creating a failed ToolResult with output=None."""
        result = ToolResult(
            name="failed_tool",
            output=None,
            success=False,
            error_message="Tool execution failed",
            execution_time=0.5,
            iteration=2
        )
        assert result.success is False
        assert result.error_message == "Tool execution failed"
        assert result.output is None

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

    def test_tool_result_with_none_output(self):
        """Test ToolResult with None output (no output produced)."""
        result = ToolResult(
            name="no_output_tool",
            output=None,
            success=True
        )
        assert result.output is None
        assert result.success is True

    def test_tool_result_with_list_output(self):
        """Test ToolResult with list output (multiple chunks)."""
        result = ToolResult(
            name="multi_chunk_tool",
            output=[{"chunk": 1}, {"chunk": 2}, {"chunk": 3}],
            success=True
        )
        assert isinstance(result.output, list)
        assert len(result.output) == 3
        assert result.output[0] == {"chunk": 1}

    def test_tool_result_empty_dict_vs_none(self):
        """Test that empty dict {} is distinct from None output."""
        result_empty_dict = ToolResult(
            name="tool1",
            output={},
            success=True
        )
        result_none = ToolResult(
            name="tool2",
            output=None,
            success=True
        )

        assert result_empty_dict.output == {}
        assert result_none.output is None
        assert result_empty_dict.output != result_none.output


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
            agent_id="agent1"
        )
        assert config.agent_id == "agent1"
        assert config.tools_allowed == []  # default
        assert config.auto_increment_iteration is True  # default
        assert config.validate_tool_arguments is True  # default

    def test_agent_config_full(self):
        """Test creating AgentConfig with all fields."""
        config = AgentConfig(
            agent_id="agent2",
            tools_allowed=["tool1", "tool2"],
            tool_name_mapping={"public_tool": "internal_tool"},
            validate_tool_arguments=False,
            input_mapping=[{"context_key": "context1", "order": 0}],
            output_mapping=[("output1", "set_latest")],
            pattern_set="custom",
            auto_increment_iteration=False,
            processing_mode=ProcessingMode.ASYNC,
            incremental_context_writes=True,
            stream_pattern_content=True,
            on_tool_detected=lambda x: True,
            concurrent_tool_execution=True,
            max_partial_buffer_size=5_000_000
        )
        assert config.tools_allowed == ["tool1", "tool2"]
        assert config.tool_name_mapping == {"public_tool": "internal_tool"}
        assert config.validate_tool_arguments is False
        assert config.pattern_set == "custom"
        assert config.auto_increment_iteration is False
        assert config.processing_mode == ProcessingMode.ASYNC
        assert config.incremental_context_writes is True
        assert config.stream_pattern_content is True
        assert config.on_tool_detected is not None
        assert config.concurrent_tool_execution is True
        assert config.max_partial_buffer_size == 5_000_000

    def test_agent_config_mappings(self):
        """Test AgentConfig input and output mappings."""
        input_map = [{"context_key": "system", "order": 0}, {"context_key": "user", "order": 1}]
        output_map = [("result", "set_latest"), ("history", "append_version")]
        config = AgentConfig(
            agent_id="agent3",
            input_mapping=input_map,
            output_mapping=output_map
        )
        assert config.input_mapping == input_map
        assert config.output_mapping == output_map

    def test_tool_verification_on_timeout_accept(self):
        """Test AgentConfig accepts 'accept' for tool_verification_on_timeout."""
        config = AgentConfig(
            agent_id="agent",
            tool_verification_on_timeout="accept"
        )
        assert config.tool_verification_on_timeout == "accept"

    def test_tool_verification_on_timeout_reject(self):
        """Test AgentConfig accepts 'reject' for tool_verification_on_timeout."""
        config = AgentConfig(
            agent_id="agent",
            tool_verification_on_timeout="reject"
        )
        assert config.tool_verification_on_timeout == "reject"

    def test_tool_verification_on_timeout_invalid(self):
        """Test AgentConfig rejects invalid tool_verification_on_timeout values."""
        import pytest
        with pytest.raises(ValueError, match="must be 'accept' or 'reject'"):
            AgentConfig(
                agent_id="agent",
                tool_verification_on_timeout="invalid"
            )


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


class TestSerializeToolOutput:
    """Tests for serialize_tool_output() helper function."""

    def test_serialize_none(self):
        """Test serializing None."""
        result = serialize_tool_output(None)
        assert result is None

    def test_serialize_dict(self):
        """Test serializing dict (preserved as-is)."""
        data = {"key": "value", "number": 42}
        result = serialize_tool_output(data)
        assert result == data
        assert isinstance(result, dict)

    def test_serialize_list(self):
        """Test serializing list (preserved as-is)."""
        data = [1, 2, 3, "four"]
        result = serialize_tool_output(data)
        assert result == data
        assert isinstance(result, list)

    def test_serialize_string(self):
        """Test serializing string (preserved as-is)."""
        data = "hello world"
        result = serialize_tool_output(data)
        assert result == data
        assert isinstance(result, str)

    def test_serialize_int(self):
        """Test serializing int (preserved as-is)."""
        data = 42
        result = serialize_tool_output(data)
        assert result == data
        assert isinstance(result, int)

    def test_serialize_float(self):
        """Test serializing float (preserved as-is)."""
        data = 3.14159
        result = serialize_tool_output(data)
        assert result == data
        assert isinstance(result, float)

    def test_serialize_bool(self):
        """Test serializing bool (preserved as-is)."""
        result_true = serialize_tool_output(True)
        result_false = serialize_tool_output(False)
        assert result_true is True
        assert result_false is False

    def test_serialize_bytes(self):
        """Test serializing bytes (converted to hex representation)."""
        data = b"binary data"
        result = serialize_tool_output(data)
        assert isinstance(result, dict)
        assert result["_type"] == "bytes"
        assert result["_hex"] == data.hex()

    def test_serialize_other_type(self):
        """Test serializing unknown type (converted to string)."""
        class CustomObject:
            def __str__(self):
                return "custom object"

        obj = CustomObject()
        result = serialize_tool_output(obj)
        assert isinstance(result, dict)
        assert result["_type"] == "string"
        assert result["_value"] == "custom object"

    def test_serialize_empty_dict(self):
        """Test serializing empty dict (preserved)."""
        result = serialize_tool_output({})
        assert result == {}
        assert isinstance(result, dict)

    def test_serialize_empty_list(self):
        """Test serializing empty list (preserved)."""
        result = serialize_tool_output([])
        assert result == []
        assert isinstance(result, list)

    def test_serialize_empty_string(self):
        """Test serializing empty string (preserved)."""
        result = serialize_tool_output("")
        assert result == ""
        assert isinstance(result, str)


class TestOutputToString:
    """Tests for output_to_string() helper function."""

    def test_output_to_string_none(self):
        """Test converting None to string (returns empty string)."""
        result = output_to_string(None)
        assert result == ""
        assert isinstance(result, str)

    def test_output_to_string_string(self):
        """Test converting string to string (returned as-is)."""
        data = "hello world"
        result = output_to_string(data)
        assert result == "hello world"

    def test_output_to_string_bytes_utf8(self):
        """Test converting UTF-8 bytes to string."""
        data = b"hello bytes"
        result = output_to_string(data)
        assert result == "hello bytes"

    def test_output_to_string_bytes_invalid_utf8(self):
        """Test converting invalid UTF-8 bytes (returns hex)."""
        data = b"\x80\x81\x82"  # Invalid UTF-8
        result = output_to_string(data)
        assert result == data.hex()

    def test_output_to_string_dict(self):
        """Test converting dict to string (JSON formatted)."""
        data = {"key": "value", "number": 42}
        result = output_to_string(data)
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result
        # Should be valid JSON
        import json
        parsed = json.loads(result)
        assert parsed == data

    def test_output_to_string_list(self):
        """Test converting list to string (JSON formatted)."""
        data = [1, 2, 3, "four"]
        result = output_to_string(data)
        assert isinstance(result, str)
        # Should be valid JSON
        import json
        parsed = json.loads(result)
        assert parsed == data

    def test_output_to_string_int(self):
        """Test converting int to string."""
        result = output_to_string(42)
        assert result == "42"

    def test_output_to_string_float(self):
        """Test converting float to string."""
        result = output_to_string(3.14)
        assert result == "3.14"

    def test_output_to_string_bool(self):
        """Test converting bool to string."""
        assert output_to_string(True) == "True"
        assert output_to_string(False) == "False"

    def test_output_to_string_empty_dict(self):
        """Test converting empty dict to string."""
        result = output_to_string({})
        assert result == "{}"

    def test_output_to_string_empty_list(self):
        """Test converting empty list to string."""
        result = output_to_string([])
        assert result == "[]"

    def test_output_to_string_empty_string(self):
        """Test converting empty string (returned as-is)."""
        result = output_to_string("")
        assert result == ""

    def test_output_to_string_empty_bytes(self):
        """Test converting empty bytes (returns empty string)."""
        result = output_to_string(b"")
        assert result == ""

    def test_output_to_string_nested_structure(self):
        """Test converting complex nested structure."""
        data = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ],
            "count": 2
        }
        result = output_to_string(data)
        assert isinstance(result, str)
        import json
        parsed = json.loads(result)
        assert parsed == data
