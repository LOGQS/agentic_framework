"""
Tests for tools system.

Covers:
- ToolDefinition dataclass
- Tool execution in different modes (PROCESS, THREAD, ASYNC)
- Tool timeouts
- Tool streaming support
- ToolRegistry (register, get, exists, list, unregister)
- create_tool helper function
- Error handling
"""
import pytest
import time

from agentic.tools import (
    Tool,
    ToolDefinition,
    create_tool
)
from agentic.core import ProcessingMode


def _picklable_process_func(inputs):
    """Picklable function for testing PROCESS mode."""
    return {"result": "process_result", "input_count": len(inputs)}


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_tool_definition_creation(self):
        """Test creating ToolDefinition."""
        definition = ToolDefinition(
            name="test_tool",
            input_schema={"arg": "string"},
            output_schema={"result": "string"},
            timeout_seconds=10.0,
            description="Test tool"
        )
        assert definition.name == "test_tool"
        assert definition.timeout_seconds == 10.0
        assert definition.description == "Test tool"

    def test_tool_definition_defaults(self):
        """Test ToolDefinition default values."""
        definition = ToolDefinition(
            name="tool",
            input_schema={},
            output_schema={}
        )
        assert definition.timeout_seconds == 30.0
        assert definition.processing_mode is None
        assert definition.description == ""


class TestToolExecution:
    """Tests for basic tool execution."""

    def test_tool_run_success(self):
        """Test successful tool execution."""
        def func(inputs):
            return {"result": inputs.get("value", "default")}

        definition = ToolDefinition("test", {}, {})
        tool = Tool(definition, func)

        result = tool.run({"value": "hello"}, iteration=1)
        assert result.success is True
        assert result.output == {"result": "hello"}
        assert result.name == "test"

    def test_tool_run_with_error(self):
        """Test tool execution with error."""
        def func(inputs):
            raise ValueError("Test error")

        definition = ToolDefinition("error_tool", {}, {})
        tool = Tool(definition, func)

        result = tool.run({}, iteration=1)
        assert result.success is False
        assert "Test error" in result.error_message

    def test_tool_execution_time_tracking(self):
        """Test that execution time is tracked."""
        def slow_func(inputs):
            time.sleep(0.1)
            return {"result": "done"}

        tool = create_tool("slow", slow_func)
        result = tool.run({}, iteration=1)

        assert result.execution_time >= 0.1

    def test_tool_run_returns_string(self):
        """Test tool that returns string."""
        def func(inputs):
            return "string result"

        tool = create_tool("string_tool", func)
        result = tool.run({}, iteration=1)

        assert result.success is True
        assert result.output == "string result"

    def test_tool_run_returns_bytes(self):
        """Test tool that returns bytes."""
        def func(inputs):
            return b"binary data"

        tool = create_tool("bytes_tool", func)
        result = tool.run({}, iteration=1)

        assert result.success is True
        assert result.output == b"binary data"


class TestToolProcessingModes:
    """Tests for different processing modes."""

    def test_tool_thread_mode(self):
        """Test tool execution in THREAD mode."""
        def func(inputs):
            return {"thread_id": "test"}

        tool = create_tool("thread_tool", func, processing_mode=ProcessingMode.THREAD)
        result = tool.run({}, iteration=1, processing_mode=ProcessingMode.THREAD)

        assert result.success is True

    def test_tool_process_mode(self):
        """Test tool execution in PROCESS mode.

        Uses module-level function to satisfy pickling requirements for multiprocessing.
        """
        tool = create_tool("process_tool", _picklable_process_func, processing_mode=ProcessingMode.PROCESS)
        result = tool.run({"key": "value"}, iteration=1, processing_mode=ProcessingMode.PROCESS)

        assert result.success is True
        assert result.output == {"result": "process_result", "input_count": 1}
        assert result.execution_time > 0

    def test_tool_async_mode(self):
        """Test tool execution in ASYNC mode."""
        def func(inputs):
            return {"result": "async_result"}

        tool = create_tool("async_tool", func, processing_mode=ProcessingMode.ASYNC)
        result = tool.run({}, iteration=1, processing_mode=ProcessingMode.ASYNC)

        assert result.success is True

    def test_tool_mode_inheritance(self):
        """Test that tool inherits processing mode from definition."""
        def func(inputs):
            return {"result": "ok"}

        definition = ToolDefinition(
            "inherit_tool",
            {},
            {},
            processing_mode=ProcessingMode.THREAD
        )
        tool = Tool(definition, func)

        # Should use THREAD mode from definition
        result = tool.run({}, iteration=1)
        assert result.success is True


class TestToolTimeout:
    """Tests for tool timeout handling."""

    def test_tool_timeout(self):
        """Test that tool times out correctly."""
        def slow_func(inputs):
            time.sleep(10)
            return {"result": "done"}

        tool = create_tool("timeout_tool", slow_func, timeout_seconds=0.5)
        result = tool.run({}, iteration=1)

        assert result.success is False
        assert "timed out" in result.error_message.lower()

    def test_tool_completes_before_timeout(self):
        """Test that fast tool completes before timeout."""
        def fast_func(inputs):
            return {"result": "quick"}

        tool = create_tool("fast_tool", fast_func, timeout_seconds=5.0)
        result = tool.run({}, iteration=1)

        assert result.success is True
        assert result.execution_time < 5.0


class TestToolStreaming:
    """Tests for tool streaming support."""

    @pytest.mark.asyncio
    async def test_tool_run_stream_non_streaming(self):
        """Test run_stream with non-streaming tool."""
        def func(inputs):
            return {"result": "batch"}

        tool = create_tool("batch_tool", func)

        outputs = []
        async for event in tool.run_stream({}, iteration=1):
            outputs.append(event)

        assert len(outputs) == 1
        assert outputs[0].output == {"result": "batch"}
        assert outputs[0].is_partial is False

    @pytest.mark.asyncio
    async def test_tool_run_stream_with_streaming_tool(self):
        """Test run_stream with tool that supports streaming."""
        class StreamingTool:
            def __call__(self, inputs):
                return {"result": "batch"}

            async def run_stream(self, inputs):
                for i in range(3):
                    yield f"chunk{i}"

        streaming_tool_func = StreamingTool()
        tool = create_tool("streaming_tool", streaming_tool_func)

        outputs = []
        async for event in tool.run_stream({}, iteration=1):
            outputs.append(event)

        assert len(outputs) == 3
        assert all(e.is_partial for e in outputs)

    @pytest.mark.asyncio
    async def test_tool_run_stream_error_handling(self):
        """Test run_stream error handling - exceptions should propagate."""
        class ErrorStreamingTool:
            def __call__(self, inputs):
                return {}

            async def run_stream(self, inputs):
                yield "chunk1"
                raise RuntimeError("Stream failed")

        error_tool_func = ErrorStreamingTool()
        tool = create_tool("error_stream_tool", error_tool_func)

        outputs = []
        with pytest.raises(RuntimeError, match="Stream failed"):
            async for event in tool.run_stream({}, iteration=1):
                outputs.append(event)

        # Should get chunk1 before exception is raised
        assert len(outputs) == 1
        assert outputs[0].output == "chunk1"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_registry_register_and_get(self, tool_registry):
        """Test registering and getting tool from registry."""
        # tool_registry fixture already has some tools
        tool = tool_registry.get("echo")
        assert tool is not None
        assert tool.name == "echo"

    def test_registry_get_nonexistent(self, tool_registry):
        """Test getting tool that doesn't exist."""
        result = tool_registry.get("nonexistent")
        assert result is None

    def test_registry_exists(self, tool_registry):
        """Test checking if tool exists."""
        assert tool_registry.exists("echo") is True
        assert tool_registry.exists("nonexistent") is False

    def test_registry_list(self, tool_registry):
        """Test listing tool names."""
        names = tool_registry.list()
        assert "echo" in names
        assert "calculator" in names
        assert sorted(names) == names  # Should be sorted

    def test_registry_unregister(self, tool_registry):
        """Test unregistering a tool."""
        def func(inputs):
            return {}

        new_tool = create_tool("temp_tool", func)
        tool_registry.register(new_tool)

        assert tool_registry.exists("temp_tool") is True

        removed = tool_registry.unregister("temp_tool")
        assert removed is True
        assert tool_registry.exists("temp_tool") is False

    def test_registry_unregister_nonexistent(self, tool_registry):
        """Test unregistering tool that doesn't exist."""
        removed = tool_registry.unregister("nonexistent")
        assert removed is False

    def test_registry_get_definitions(self, tool_registry):
        """Test getting all tool definitions."""
        definitions = tool_registry.get_definitions()
        assert isinstance(definitions, dict)
        assert "echo" in definitions
        assert isinstance(definitions["echo"], ToolDefinition)


class TestCreateToolHelper:
    """Tests for create_tool helper function."""

    def test_create_tool_minimal(self):
        """Test create_tool with minimal arguments."""
        def func(inputs):
            return {"result": "ok"}

        tool = create_tool("simple", func)
        assert tool.name == "simple"
        assert tool.definition.timeout_seconds == 30.0

    def test_create_tool_full(self):
        """Test create_tool with all arguments."""
        def func(inputs):
            return {"result": "ok"}

        tool = create_tool(
            name="full_tool",
            func=func,
            input_schema={"arg": "string"},
            output_schema={"result": "string"},
            timeout_seconds=15.0,
            processing_mode=ProcessingMode.THREAD,
            description="Full tool"
        )

        assert tool.definition.timeout_seconds == 15.0
        assert tool.definition.processing_mode == ProcessingMode.THREAD
        assert tool.definition.description == "Full tool"


class TestToolEdgeCases:
    """Tests for edge cases in tool execution."""

    def test_tool_with_empty_inputs(self):
        """Test tool execution with empty inputs."""
        def func(inputs):
            return {"result": "no inputs"}

        tool = create_tool("empty_input_tool", func)
        result = tool.run({}, iteration=1)

        assert result.success is True

    def test_tool_with_complex_inputs(self):
        """Test tool with complex nested inputs."""
        def func(inputs):
            return {"received": inputs}

        tool = create_tool("complex_tool", func)
        complex_input = {
            "nested": {"deep": {"value": [1, 2, 3]}},
            "list": ["a", "b", "c"]
        }

        result = tool.run(complex_input, iteration=1)
        assert result.success is True

    def test_tool_exception_types(self):
        """Test tool handling different exception types."""
        exceptions = [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
            Exception("generic error")
        ]

        for exc in exceptions:
            def func(inputs):
                raise exc

            tool = create_tool("error_tool", func)
            result = tool.run({}, iteration=1)
            assert result.success is False

    def test_tool_none_return(self):
        """Test tool that returns None."""
        def func(inputs):
            return None

        tool = create_tool("none_tool", func)
        result = tool.run({}, iteration=1)

        # Should handle None gracefully
        assert result.success is True

    def test_tool_large_output(self):
        """Test tool with large output."""
        def func(inputs):
            return {"data": "x" * 1000000}  # 1MB

        tool = create_tool("large_tool", func)
        result = tool.run({}, iteration=1)

        assert result.success is True
        assert len(result.output["data"]) == 1000000
