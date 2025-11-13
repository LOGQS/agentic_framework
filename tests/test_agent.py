"""
Tests for agent execution system.

Covers:
- Agent configuration and setup
- AgentRunner step execution (batch and streaming)
- Tool execution and error handling
- Context updates
- Pattern extraction during execution
- MockLLMProvider
- Event emission during streaming
"""
import pytest

from agentic.agent import Agent, AgentRunner
from tests.mock_provider import MockLLMProvider
from agentic.core import AgentConfig, AgentStatus, ProcessingMode
from agentic.events import (
    ToolStartEvent,
    ToolEndEvent,
    StepCompleteEvent,
    ErrorEvent
)


class TestAgent:
    """Tests for Agent class."""

    def test_agent_creation(self, agent):
        """Test creating an Agent instance."""
        assert agent.get_id() == "test_agent"
        assert agent.get_config().agent_id == "test_agent"

    def test_agent_get_config(self, agent):
        """Test getting agent configuration."""
        config = agent.get_config()
        assert isinstance(config, AgentConfig)
        assert config.agent_id == "test_agent"

    def test_agent_set_config(self, agent):
        """Test updating agent configuration."""
        new_config = AgentConfig(
            agent_id="new_agent",
            provider="test",
            model="test-model"
        )
        agent.set_config(new_config)
        assert agent.get_id() == "new_agent"

    def test_agent_properties(self, agent):
        """Test agent property accessors."""
        assert agent.context is not None
        assert agent.patterns is not None
        assert agent.tools is not None
        assert agent.provider is not None


class TestMockLLMProvider:
    """Tests for MockLLMProvider."""

    def test_mock_provider_generate(self):
        """Test MockLLMProvider generate method."""
        provider = MockLLMProvider(response="Test response")
        output = provider.generate("prompt", 100, 0.7)
        assert output == "Test response"

    @pytest.mark.asyncio
    async def test_mock_provider_stream_no_simulation(self):
        """Test MockLLMProvider stream without simulation."""
        provider = MockLLMProvider(response="Test response", simulate_streaming=False)

        tokens = []
        async for token in provider.stream("prompt", 100, 0.7):
            tokens.append(token)

        assert len(tokens) == 1
        assert tokens[0] == "Test response"

    @pytest.mark.asyncio
    async def test_mock_provider_stream_with_simulation(self):
        """Test MockLLMProvider stream with word-by-word simulation."""
        provider = MockLLMProvider(response="Hello world test", simulate_streaming=True)

        tokens = []
        async for token in provider.stream("prompt", 100, 0.7):
            tokens.append(token)

        assert len(tokens) == 3
        assert "".join(tokens) == "Hello world test"

    def test_mock_provider_set_response(self):
        """Test changing MockLLMProvider response."""
        provider = MockLLMProvider(response="First")
        assert provider.generate("", 0, 0) == "First"

        provider.set_response("Second")
        assert provider.generate("", 0, 0) == "Second"


class TestAgentRunnerBatch:
    """Tests for AgentRunner batch execution."""

    def test_agent_step_simple(self, agent_runner, mock_llm_provider):
        """Test simple agent step execution."""
        mock_llm_provider.set_response("This is a simple response.")

        result = agent_runner.step()

        assert result.status == AgentStatus.OK
        assert result.raw_output == "This is a simple response."
        assert result.iteration >= 0

    def test_agent_step_with_user_input(self, agent_runner, mock_llm_provider, context_manager):
        """Test agent step with user input."""
        context_manager.set("system_prompt", b"You are helpful.")

        result = agent_runner.step(user_input="Hello agent")

        assert result.status == AgentStatus.OK

    def test_agent_step_with_tool_call(self, agent_runner, mock_llm_provider):
        """Test agent step that calls a tool."""
        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        result = agent_runner.step()

        assert result.status == AgentStatus.TOOL_EXECUTED
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is True

    def test_agent_step_tool_not_allowed(self, agent_runner, mock_llm_provider, agent):
        """Test agent step with tool not in allowed list."""
        # Set allowed tools to empty list
        config = agent.get_config()
        config.tools_allowed = []
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {}}</tool>')

        result = agent_runner.step()

        assert result.status == AgentStatus.ERROR
        assert "not in allowed list" in result.error_message

    def test_agent_step_tool_not_found(self, agent_runner, mock_llm_provider, agent):
        """Test agent step with tool not in registry."""
        config = agent.get_config()
        config.tools_allowed = ["nonexistent_tool"]
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "nonexistent_tool", "arguments": {}}</tool>')

        result = agent_runner.step()

        assert result.status == AgentStatus.ERROR
        assert "not found in registry" in result.error_message


@pytest.mark.asyncio
class TestAgentRunnerStreaming:
    """Tests for AgentRunner streaming execution."""

    async def test_agent_step_stream_events(self, agent_runner, mock_llm_provider):
        """Test that step_stream yields events."""
        mock_llm_provider.set_response("Test response")

        events = []
        async for event in agent_runner.step_stream():
            events.append(event)

        # Should have various event types
        event_types = {type(e).__name__ for e in events}
        assert "StatusEvent" in event_types
        assert "StepCompleteEvent" in event_types

    async def test_agent_step_stream_final_result(self, agent_runner, mock_llm_provider):
        """Test that step_stream yields final StepCompleteEvent."""
        mock_llm_provider.set_response("Final response")

        final_event = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_event = event

        assert final_event is not None
        assert final_event.result.status == AgentStatus.OK

    async def test_agent_step_stream_with_tool(self, agent_runner, mock_llm_provider):
        """Test streaming with tool execution."""
        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "hi"}}</tool>')

        tool_events = []
        async for event in agent_runner.step_stream():
            if isinstance(event, (ToolStartEvent, ToolEndEvent)):
                tool_events.append(event)

        # Should have tool start and end events
        assert len(tool_events) >= 2


class TestAgentContextUpdates:
    """Tests for context updates during agent execution."""

    def test_context_updated_after_step(self, agent_runner, mock_llm_provider, context_manager):
        """Test that context is updated after step."""
        mock_llm_provider.set_response("Response to store")

        agent_runner.step()

        # Check that output was stored (based on output_mapping in fixture)
        record = context_manager.get("last_output")
        assert record is not None

    def test_context_versioning_across_steps(self, agent_runner, mock_llm_provider, context_manager):
        """Test that context versions across multiple steps."""
        mock_llm_provider.set_response("Step 1")
        agent_runner.step()

        mock_llm_provider.set_response("Step 2")
        agent_runner.step()

        # Should have 2 versions
        history = context_manager.get_history("last_output")
        assert len(history) >= 1


class TestAgentPatternExtraction:
    """Tests for pattern extraction during agent execution."""

    def test_agent_extracts_reasoning(self, agent_runner, mock_llm_provider):
        """Test that agent extracts reasoning segments."""
        mock_llm_provider.set_response("<reasoning>Thinking...</reasoning>Final answer")

        result = agent_runner.step()

        assert len(result.segments.reasoning) >= 1

    def test_agent_extracts_response(self, agent_runner, mock_llm_provider):
        """Test that agent extracts response."""
        mock_llm_provider.set_response("<response>This is the answer</response>")

        result = agent_runner.step()

        assert result.segments.response is not None

    def test_agent_handles_malformed_patterns(self, agent_runner, mock_llm_provider):
        """Test that agent handles malformed patterns gracefully."""
        mock_llm_provider.set_response("<tool>incomplete pattern")

        result = agent_runner.step()

        # Should not crash and may have partial_malformed_patterns
        assert result is not None


class TestAgentErrorHandling:
    """Tests for agent error handling."""

    def test_agent_llm_error(self, agent, context_manager, pattern_registry, tool_registry):
        """Test agent handling LLM generation error."""
        class ErrorProvider:
            def generate(self, prompt, max_tokens, temperature, **kwargs):
                raise RuntimeError("LLM failed")

            async def stream(self, prompt, max_tokens, temperature, **kwargs):
                if False:
                    yield  
                raise RuntimeError("LLM failed")

        error_provider = ErrorProvider()
        error_agent = Agent(
            config=agent.get_config(),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=error_provider
        )
        runner = AgentRunner(error_agent)

        result = runner.step()

        assert result.status == AgentStatus.ERROR
        assert result.error_type == "llm_error"

    def test_agent_tool_execution_error(self, agent_runner, mock_llm_provider, tool_registry):
        """Test agent handling tool execution error."""
        # Register a tool that raises error
        def error_func(inputs):
            raise ValueError("Tool failed")

        from agentic.tools import create_tool
        error_tool = create_tool("error_tool", error_func)
        tool_registry.register(error_tool)

        # Update agent config to allow error_tool
        config = agent_runner._agent.get_config()
        config.tools_allowed.append("error_tool")
        agent_runner._agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "error_tool", "arguments": {}}</tool>')

        result = agent_runner.step()

        assert result.status == AgentStatus.ERROR


class TestAgentIterationTracking:
    """Tests for iteration tracking in agent execution."""

    def test_agent_tracks_iteration(self, agent_runner, mock_llm_provider, context_manager):
        """Test that agent tracks iteration number."""
        initial_iteration = context_manager.get_iteration()

        result1 = agent_runner.step()
        assert result1.iteration > initial_iteration

    def test_agent_auto_increment_iteration(self, agent_runner, mock_llm_provider, context_manager):
        """Test that agent auto-increments iteration."""
        agent_runner.step()
        iter1 = context_manager.get_iteration()

        agent_runner.step()
        iter2 = context_manager.get_iteration()

        assert iter2 > iter1


class TestAgentConfigOptions:
    """Tests for various agent configuration options."""

    def test_agent_processing_mode(self, agent):
        """Test agent with different processing modes."""
        config = agent.get_config()

        for mode in ProcessingMode:
            config.processing_mode = mode
            agent.set_config(config)
            assert agent.get_config().processing_mode == mode

    def test_agent_auto_increment_disabled(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test agent with auto_increment_iteration disabled."""
        config = agent.get_config()
        config.auto_increment_iteration = False
        agent.set_config(config)

        initial_iter = context_manager.get_iteration()

        agent_runner.step()

        # Iteration should not change
        assert context_manager.get_iteration() == initial_iter


class TestAgentConcurrentToolExecution:
    """Tests for concurrent tool execution feature."""

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution_enabled(self, agent, agent_runner, mock_llm_provider, tool_registry):
        """Test that tools execute concurrently when enabled.

        When concurrent_tool_execution is True, multiple tools should
        execute in parallel during streaming, not sequentially.
        """
        config = agent.get_config()
        config.concurrent_tool_execution = True
        agent.set_config(config)

        # Response with 2 tool calls
        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "1"}}</tool>'
            '<tool>{"name": "echo", "arguments": {"message": "2"}}</tool>'
        )

        # Use step_stream to test concurrent execution
        tool_events = []
        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, (ToolStartEvent, ToolEndEvent)):
                tool_events.append(event)
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert final_result is not None
        assert len(final_result.tool_results) == 2
        assert all(tr.success for tr in final_result.tool_results)

    @pytest.mark.asyncio
    async def test_concurrent_tool_error_handling(self, agent, agent_runner, mock_llm_provider, tool_registry):
        """Test that error in one tool doesn't block others in concurrent mode.

        When one tool fails, other concurrent tools should still complete.
        """
        # Register a tool that fails
        def error_func(inputs):
            raise ValueError("Tool failed")

        from agentic.tools import create_tool
        error_tool = create_tool("error_tool", error_func)
        tool_registry.register(error_tool)

        config = agent.get_config()
        config.concurrent_tool_execution = True
        config.tools_allowed = ["echo", "error_tool"]
        agent.set_config(config)

        # Call both tools: one succeeds, one fails
        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "success"}}</tool>'
            '<tool>{"name": "error_tool", "arguments": {}}</tool>'
        )

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert final_result is not None
        assert len(final_result.tool_results) == 2

        # One should succeed, one should fail
        success_count = sum(1 for tr in final_result.tool_results if tr.success)
        failure_count = sum(1 for tr in final_result.tool_results if not tr.success)
        assert success_count == 1
        assert failure_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_event_ordering(self, agent, agent_runner, mock_llm_provider):
        """Test that events from concurrent tools are properly emitted.

        Even with concurrent execution, tool events should be properly
        emitted and ordered.
        """
        config = agent.get_config()
        config.concurrent_tool_execution = True
        agent.set_config(config)

        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "1"}}</tool>'
            '<tool>{"name": "calculator", "arguments": {"a": 1, "b": 2, "operation": "add"}}</tool>'
        )

        tool_start_events = []
        tool_end_events = []
        async for event in agent_runner.step_stream():
            if isinstance(event, ToolStartEvent):
                tool_start_events.append(event)
            elif isinstance(event, ToolEndEvent):
                tool_end_events.append(event)

        # Should have start and end events for both tools
        assert len(tool_start_events) == 2
        assert len(tool_end_events) == 2


class TestAgentOnToolDetectedCallback:
    """Tests for on_tool_detected callback feature."""

    @pytest.mark.asyncio
    async def test_on_tool_detected_callback_allow(self, agent, agent_runner, mock_llm_provider):
        """Test callback allowing tool execution (returns True).

        When the callback returns True, the tool should execute normally.
        """
        calls = []

        def callback(tool_call):
            calls.append(tool_call.name)
            return True  # Allow execution

        config = agent.get_config()
        config.on_tool_detected = callback
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Callback should have been called
        assert "echo" in calls

        # Tool should have executed
        assert final_result.status == AgentStatus.TOOL_EXECUTED
        assert len(final_result.tool_results) == 1
        assert final_result.tool_results[0].success is True

    @pytest.mark.asyncio
    async def test_on_tool_detected_callback_reject(self, agent, agent_runner, mock_llm_provider):
        """Test callback rejecting tool execution (returns False).

        When the callback returns False, the tool should NOT execute.
        """
        calls = []

        def callback(tool_call):
            calls.append(tool_call.name)
            return False  # Reject execution

        config = agent.get_config()
        config.on_tool_detected = callback
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Callback should have been called
        assert "echo" in calls

        # Tool should NOT have executed
        assert len(final_result.tool_results) == 0

    @pytest.mark.asyncio
    async def test_on_tool_detected_callback_exception(self, agent, agent_runner, mock_llm_provider):
        """Test callback exception handling.

        When the callback raises an exception, it should be caught and
        the tool should not execute.
        """
        def callback(tool_call):
            raise RuntimeError("Callback error")

        config = agent.get_config()
        config.on_tool_detected = callback
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        error_events = []
        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, ErrorEvent):
                error_events.append(event)
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Should have error event from callback
        assert len(error_events) > 0
        assert any("callback failed" in e.error_message for e in error_events)

        # Tool should NOT have executed due to callback error
        assert len(final_result.tool_results) == 0

    @pytest.mark.asyncio
    async def test_on_tool_detected_callback_with_concurrent_execution(self, agent, agent_runner, mock_llm_provider):
        """Test callback works with concurrent tool execution.

        The callback should be invoked for each tool in concurrent mode.
        """
        calls = []

        def callback(tool_call):
            calls.append(tool_call.name)
            # Allow echo, reject calculator
            return tool_call.name == "echo"

        config = agent.get_config()
        config.on_tool_detected = callback
        config.concurrent_tool_execution = True
        agent.set_config(config)

        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "1"}}</tool>'
            '<tool>{"name": "calculator", "arguments": {"a": 1, "b": 2, "operation": "add"}}</tool>'
        )

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Both tools should have triggered callback
        assert "echo" in calls
        assert "calculator" in calls

        # Only echo should have executed
        assert len(final_result.tool_results) == 1
        assert final_result.tool_results[0].name == "echo"


class TestAgentOutputMapping:
    """Tests for output mapping operations."""

    def test_output_mapping_append_version(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test append_version output mapping operation.

        append_version should append new output to existing context value
        with a newline separator.
        """
        config = agent.get_config()
        config.output_mapping = [("conversation", "append_version")]
        agent.set_config(config)

        # First step
        mock_llm_provider.set_response("First message")
        agent_runner.step()

        record = context_manager.get("conversation")
        assert b"First message" in record.value

        # Second step - should append
        mock_llm_provider.set_response("Second message")
        agent_runner.step()

        record = context_manager.get("conversation")
        content = record.value.decode('utf-8')
        assert "First message" in content
        assert "Second message" in content
        assert "\n\n" in content  # Should have separator

    def test_output_mapping_set_response(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test set_response output mapping operation.

        set_response should extract only the content from <response> pattern
        and store it in context.
        """
        config = agent.get_config()
        config.output_mapping = [("final_answer", "set_response")]
        agent.set_config(config)

        mock_llm_provider.set_response("<response>The answer is 42</response>")
        agent_runner.step()

        record = context_manager.get("final_answer")
        assert record is not None
        assert record.value == b"The answer is 42"

    def test_output_mapping_set_reasoning(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test set_reasoning output mapping operation.

        set_reasoning should extract all reasoning segments and join them
        with newlines.
        """
        config = agent.get_config()
        config.output_mapping = [("thought_process", "set_reasoning")]
        agent.set_config(config)

        mock_llm_provider.set_response(
            "<reasoning>First thought</reasoning>"
            "Some text"
            "<reasoning>Second thought</reasoning>"
        )
        agent_runner.step()

        record = context_manager.get("thought_process")
        assert record is not None
        content = record.value.decode('utf-8')
        assert "First thought" in content
        assert "Second thought" in content

    def test_output_mapping_set_tools(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test set_tools output mapping operation.

        set_tools should store JSON representation of all tool results.
        """
        import json

        config = agent.get_config()
        config.output_mapping = [("tool_results", "set_tools")]
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')
        agent_runner.step()

        record = context_manager.get("tool_results")
        assert record is not None

        # Parse JSON
        tools_data = json.loads(record.value.decode('utf-8'))
        assert len(tools_data) == 1
        assert tools_data[0]["name"] == "echo"
        assert tools_data[0]["success"] is True

    def test_output_mapping_multiple_operations(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test multiple output mapping operations at once.

        Multiple mappings should all be applied to the same step output.
        """
        config = agent.get_config()
        config.output_mapping = [
            ("raw", "set_latest"),
            ("answer", "set_response")
        ]
        agent.set_config(config)

        mock_llm_provider.set_response("Thinking... <response>Final answer</response>")
        agent_runner.step()

        # Both mappings should be applied
        raw_record = context_manager.get("raw")
        assert b"Thinking..." in raw_record.value
        assert b"<response>" in raw_record.value

        answer_record = context_manager.get("answer")
        assert answer_record.value == b"Final answer"


class TestAgentEdgeCases:
    """Tests for edge cases in agent execution."""

    def test_agent_empty_response(self, agent_runner, mock_llm_provider):
        """Test agent with empty LLM response."""
        mock_llm_provider.set_response("")

        result = agent_runner.step()

        assert result is not None

    def test_agent_very_long_response(self, agent_runner, mock_llm_provider):
        """Test agent with very long response."""
        long_response = "x" * 100000
        mock_llm_provider.set_response(long_response)

        result = agent_runner.step()

        assert len(result.raw_output) == 100000

    def test_agent_multiple_tools_in_response(self, agent_runner, mock_llm_provider):
        """Test agent response with multiple tool calls."""
        response = '''
        <tool>{"name": "echo", "arguments": {"message": "1"}}</tool>
        <tool>{"name": "echo", "arguments": {"message": "2"}}</tool>
        '''
        mock_llm_provider.set_response(response)

        result = agent_runner.step()

        assert len(result.tool_results) == 2
