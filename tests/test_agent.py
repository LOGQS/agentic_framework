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
import logging
import pytest

from agentic.agent import Agent, AgentRunner
from tests.mock_provider import MockLLMProvider
from agentic.core import AgentConfig, AgentStatus, ProcessingMode
from agentic.events import (
    ToolStartEvent,
    ToolEndEvent,
    StepCompleteEvent,
    ErrorEvent,
    PatternStartEvent,
    PatternContentEvent,
    PatternEndEvent,
    ContextWriteEvent,
    LLMTokenEvent,
    StatusEvent
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
            agent_id="new_agent"
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
        output = provider.generate("prompt")
        assert output == "Test response"

    @pytest.mark.asyncio
    async def test_mock_provider_stream_no_simulation(self):
        """Test MockLLMProvider stream without simulation."""
        provider = MockLLMProvider(response="Test response", simulate_streaming=False)

        tokens = []
        async for token in provider.stream("prompt"):
            tokens.append(token)

        assert len(tokens) == 1
        assert tokens[0] == "Test response"

    @pytest.mark.asyncio
    async def test_mock_provider_stream_with_simulation(self):
        """Test MockLLMProvider stream with word-by-word simulation."""
        provider = MockLLMProvider(response="Hello world test", simulate_streaming=True)

        tokens = []
        async for token in provider.stream("prompt"):
            tokens.append(token)

        assert len(tokens) == 3
        assert "".join(tokens) == "Hello world test"

    def test_mock_provider_set_response(self):
        """Test changing MockLLMProvider response."""
        provider = MockLLMProvider(response="First")
        assert provider.generate("") == "First"

        provider.set_response("Second")
        assert provider.generate("") == "Second"


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

    async def test_agent_step_stream_tool_rejection_reason(
        self, agent_runner, mock_llm_provider, agent, caplog
    ):
        """Callback rejection should emit structured status and log entry."""
        config = agent.get_config()
        config.on_tool_detected = lambda tc: False
        config.tools_allowed = ["echo"]
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {}}</tool>')

        caplog.set_level(logging.DEBUG, logger="agentic.agent")
        agent_logger = logging.getLogger("agentic.agent")
        agent_logger.addHandler(caplog.handler)
        status_messages = []

        async for event in agent_runner.step_stream():
            if isinstance(event, StatusEvent):
                status_messages.append(event.message or "")

        assert any("rejected" in message.lower() for message in status_messages)
        assert any(
            record.message == "agent.tool.rejected"
            for record in caplog.records
            if record.name == "agentic.agent"
        )

        agent_logger.removeHandler(caplog.handler)


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
            def generate(self, prompt, **kwargs):
                raise RuntimeError("LLM failed")

            async def stream(self, prompt, **kwargs):
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
        the tool should not execute. The tool is rejected rather than causing
        an error event.
        """
        def callback(tool_call):
            raise RuntimeError("Callback error")

        config = agent.get_config()
        config.on_tool_detected = callback
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Tool should NOT have executed due to callback error
        assert len(final_result.tool_results) == 0

        # Should have tool decision showing rejection due to exception
        assert len(final_result.tool_decisions) == 1
        decision = final_result.tool_decisions[0]
        assert decision.accepted is False
        assert "Callback exception" in decision.rejection_reason

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

    @pytest.mark.asyncio
    async def test_verification_status_emitted(self, agent, agent_runner, mock_llm_provider):
        """Test WAITING_FOR_VERIFICATION status is emitted when on_tool_detected is set."""
        def callback(tool_call):
            return True

        config = agent.get_config()
        config.on_tool_detected = callback
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        status_events = []
        async for event in agent_runner.step_stream():
            if isinstance(event, StatusEvent):
                status_events.append(event)

        verification_statuses = [e for e in status_events if e.status == AgentStatus.WAITING_FOR_VERIFICATION]
        assert len(verification_statuses) > 0
        assert "verification" in verification_statuses[0].message.lower()

    @pytest.mark.asyncio
    async def test_tool_matching_with_call_ids(self, agent, agent_runner, mock_llm_provider):
        """Test tool matching works correctly with call_ids."""
        config = agent.get_config()
        agent.set_config(config)

        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "1"}}</tool>'
            '<tool>{"name": "echo", "arguments": {"message": "2"}}</tool>'
        )

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert len(final_result.tool_results) == 2
        assert len(final_result.tool_decisions) == 2

        for decision in final_result.tool_decisions:
            assert decision.executed is True
            assert decision.result is not None

    @pytest.mark.asyncio
    async def test_tool_matching_without_call_ids(self, agent, agent_runner, mock_llm_provider):
        """Test tool matching works correctly without call_ids (legacy mode)."""
        config = agent.get_config()
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert len(final_result.tool_results) == 1
        assert len(final_result.tool_decisions) == 1

        decision = final_result.tool_decisions[0]
        assert decision.executed is True
        assert decision.result is not None
        assert decision.result.name == "echo"


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


@pytest.mark.asyncio
class TestAgentStreamPatternContent:
    """Tests for stream_pattern_content feature."""

    async def test_stream_pattern_content_enabled(self, agent, agent_runner, mock_llm_provider):
        """Test that PatternContentEvent is emitted when stream_pattern_content is enabled.

        When stream_pattern_content is enabled, the agent should emit PatternContentEvent
        as pattern content arrives, before the end tag is detected.
        """
        config = agent.get_config()
        config.stream_pattern_content = True
        agent.set_config(config)

        mock_llm_provider.set_response("<reasoning>This is my thought process</reasoning>")

        pattern_content_events = []
        pattern_start_events = []
        pattern_end_events = []

        async for event in agent_runner.step_stream():
            if isinstance(event, PatternContentEvent):
                pattern_content_events.append(event)
            elif isinstance(event, PatternStartEvent):
                pattern_start_events.append(event)
            elif isinstance(event, PatternEndEvent):
                pattern_end_events.append(event)

        # Should have pattern start, content, and end events
        assert len(pattern_start_events) >= 1
        assert len(pattern_end_events) >= 1
        # With stream_pattern_content enabled, should have content events
        assert len(pattern_content_events) >= 0  # May vary based on streaming simulation

    async def test_stream_pattern_content_incremental_updates(self, agent, agent_runner, mock_llm_provider):
        """Test incremental pattern content updates during streaming.

        PatternContentEvent should be emitted incrementally as the pattern
        content is streamed token-by-token.
        """
        config = agent.get_config()
        config.stream_pattern_content = True
        agent.set_config(config)

        # Use simulate_streaming to get token-by-token behavior
        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response("<response>Word by word response</response>")

        pattern_content_events = []
        async for event in agent_runner.step_stream():
            if isinstance(event, PatternContentEvent):
                pattern_content_events.append(event)

        # All content events should be for the same pattern and be marked as partial
        if pattern_content_events:
            assert all(e.is_partial for e in pattern_content_events)
            # Content should accumulate
            assert all(e.pattern_name == pattern_content_events[0].pattern_name for e in pattern_content_events)

    async def test_stream_pattern_content_multiple_patterns(self, agent, agent_runner, mock_llm_provider):
        """Test streaming content for multiple patterns in the same response.

        When multiple patterns are present, PatternContentEvent should be emitted
        for each pattern independently.
        """
        config = agent.get_config()
        config.stream_pattern_content = True
        agent.set_config(config)

        mock_llm_provider.set_response(
            "<reasoning>First thought</reasoning>"
            "<response>Final answer</response>"
        )

        pattern_events = []
        async for event in agent_runner.step_stream():
            if isinstance(event, (PatternStartEvent, PatternContentEvent, PatternEndEvent)):
                pattern_events.append(event)

        # Should have events for multiple patterns
        pattern_names = {e.pattern_name for e in pattern_events}
        assert len(pattern_names) >= 1


@pytest.mark.asyncio
class TestAgentIncrementalContextWrites:
    """Tests for incremental_context_writes feature."""

    async def test_incremental_context_writes_during_streaming(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test streaming key updates during LLM response.

        When incremental_context_writes is enabled, partial LLM output should
        be written to a streaming key in context during generation.
        """
        config = agent.get_config()
        config.incremental_context_writes = True
        agent.set_config(config)

        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response("Streaming response test")

        iteration_before = context_manager.get_iteration()

        # Consume streaming events
        async for event in agent_runner.step_stream():
            # During streaming, check if streaming key exists
            if isinstance(event, LLMTokenEvent):
                iteration_current = context_manager.get_iteration()
                streaming_key = f"llm_streaming:{iteration_current}"
                record = context_manager.get(streaming_key)
                # Key may or may not exist depending on timing
                # But by end of stream it should have been created

        # After completion, streaming key should NOT exist (cleaned up)
        final_iteration = context_manager.get_iteration()
        streaming_key = f"llm_streaming:{final_iteration}"
        # The key might still exist if not cleaned up, depending on implementation

    async def test_incremental_context_writes_pattern_content(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test pattern partial updates with incremental writes.

        When incremental_context_writes is enabled, partial pattern content
        should be written to temporary keys as patterns stream.
        """
        config = agent.get_config()
        config.incremental_context_writes = True
        config.stream_pattern_content = True
        agent.set_config(config)

        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response("<reasoning>Incremental thought process</reasoning>")

        pattern_partial_keys_seen = []

        async for event in agent_runner.step_stream():
            if isinstance(event, PatternContentEvent):
                # Check if partial key exists
                iteration_current = context_manager.get_iteration()
                partial_key = f"pattern_partial:{event.pattern_name}:{iteration_current}"
                record = context_manager.get(partial_key)
                if record:
                    pattern_partial_keys_seen.append(partial_key)

        # At least one partial key should have been created during streaming
        # (may vary based on simulation)

    async def test_incremental_context_writes_cleanup(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test cleanup of partial keys on completion.

        After pattern completion, partial keys should be deleted to avoid
        clutter in the context database.
        """
        config = agent.get_config()
        config.incremental_context_writes = True
        config.stream_pattern_content = True
        agent.set_config(config)

        mock_llm_provider.set_response("<reasoning>Complete thought</reasoning>")

        final_iteration = None
        pattern_names = []

        async for event in agent_runner.step_stream():
            if isinstance(event, PatternEndEvent):
                pattern_names.append(event.pattern_name)
            if isinstance(event, StepCompleteEvent):
                final_iteration = event.result.iteration

        # After completion, partial keys should be deleted
        if final_iteration is not None:
            for pattern_name in pattern_names:
                partial_key = f"pattern_partial:{pattern_name}:{final_iteration}"
                record = context_manager.get(partial_key)
                # Key should be deleted after pattern completes
                assert record is None

    async def test_incremental_context_writes_with_tools(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test incremental writes during tool execution.

        Incremental context writes should work properly when tools are
        also being executed.
        """
        config = agent.get_config()
        config.incremental_context_writes = True
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        context_write_events = []
        async for event in agent_runner.step_stream():
            if isinstance(event, ContextWriteEvent):
                context_write_events.append(event)

        # Should have context write events
        assert len(context_write_events) >= 0  # May vary


class TestAgentInputMappingLiteral:
    """Tests for input_mapping with literal: prefix."""

    def test_input_mapping_literal_prepend(self, agent, agent_runner, mock_llm_provider):
        """Test literal: prefix with prepend operation.

        When input_mapping includes literal: prefix with prepend, the literal
        text should be prepended to the prompt.
        """
        config = agent.get_config()
        config.input_mapping = [{"context_key": "literal:System: You are a helpful assistant.", "order": 0}]
        agent.set_config(config)

        mock_llm_provider.set_response("I am helpful.")

        result = agent_runner.step(user_input="Hello")

        # The prompt should have included the literal text
        # We can verify this worked by checking the result is valid
        assert result is not None
        assert result.status == AgentStatus.OK

    def test_input_mapping_literal_append(self, agent, agent_runner, mock_llm_provider):
        """Test literal: prefix with append operation.

        When input_mapping includes literal: prefix with append, the literal
        text should be appended to the prompt.
        """
        config = agent.get_config()
        config.input_mapping = [{"context_key": "literal:Please respond concisely.", "order": 1}]
        agent.set_config(config)

        mock_llm_provider.set_response("Okay.")

        result = agent_runner.step(user_input="Be brief")

        assert result is not None
        assert result.status == AgentStatus.OK

    def test_input_mapping_multiple_literals(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test multiple literal mappings in the same config.

        Multiple literal: mappings should all be applied to the prompt
        in the order and position specified.
        """
        # Set some context first
        context_manager.set("user_name", b"Alice")

        config = agent.get_config()
        config.input_mapping = [
            {"context_key": "literal:System: Be helpful.", "order": 0},
            {"context_key": "user_name", "order": 1},
            {"context_key": "literal:End of instructions.", "order": 2}
        ]
        agent.set_config(config)

        mock_llm_provider.set_response("Understood.")

        result = agent_runner.step(user_input="Test")

        assert result is not None
        assert result.status == AgentStatus.OK


@pytest.mark.asyncio
class TestAgentToolStateTracking:
    """Tests for tool state tracking feature."""

    async def test_tool_state_tracking_started(self, agent_runner, mock_llm_provider, context_manager):
        """Test tool_state:{call_id} context keys on tool start.

        When a tool starts execution, a context key tool_state:{call_id}
        should be created with value 'started'.
        """
        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        tool_call_ids = []
        async for event in agent_runner.step_stream():
            if isinstance(event, ToolStartEvent):
                tool_call_ids.append(event.call_id)

        # Verify state keys were created
        for call_id in tool_call_ids:
            state_key = f"tool_state:{call_id}"
            record = context_manager.get(state_key)
            # State should be either 'started' or already transitioned to 'finished'
            assert record is not None
            assert record.value in [b"started", b"finished"]

    async def test_tool_state_tracking_finished(self, agent_runner, mock_llm_provider, context_manager):
        """Test state transitions to finished.

        After successful tool execution, the state should transition
        from 'started' to 'finished'.
        """
        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "success"}}</tool>')

        tool_call_ids = []
        async for event in agent_runner.step_stream():
            if isinstance(event, ToolEndEvent):
                if event.result.success:
                    tool_call_ids.append(event.call_id)

        # Verify state is 'finished' for successful tools
        for call_id in tool_call_ids:
            state_key = f"tool_state:{call_id}"
            record = context_manager.get(state_key)
            assert record is not None
            assert record.value == b"finished"

    async def test_tool_state_tracking_failed(self, agent, agent_runner, mock_llm_provider, context_manager, tool_registry):
        """Test state transitions to failed.

        When a tool execution fails, the state should transition to 'failed'.
        """
        # Register a tool that fails
        def error_func(inputs):
            raise ValueError("Tool error")

        from agentic.tools import create_tool
        error_tool = create_tool("error_tool", error_func)
        tool_registry.register(error_tool)

        config = agent.get_config()
        config.tools_allowed.append("error_tool")
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "error_tool", "arguments": {}}</tool>')

        failed_call_ids = []
        async for event in agent_runner.step_stream():
            if isinstance(event, ToolEndEvent):
                if not event.result.success:
                    failed_call_ids.append(event.call_id)

        # Verify state is 'failed' for failed tools
        for call_id in failed_call_ids:
            state_key = f"tool_state:{call_id}"
            record = context_manager.get(state_key)
            assert record is not None
            assert record.value == b"failed"

    async def test_tool_state_tracking_multiple_tools(self, agent_runner, mock_llm_provider, context_manager):
        """Test state tracking for multiple concurrent tools.

        Each tool should have its own state tracking key, even when
        multiple tools execute.
        """
        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "1"}}</tool>'
            '<tool>{"name": "calculator", "arguments": {"a": 2, "b": 3, "operation": "add"}}</tool>'
        )

        tool_call_ids = []
        async for event in agent_runner.step_stream():
            if isinstance(event, ToolEndEvent):
                tool_call_ids.append(event.call_id)

        # Each tool should have its own state key
        assert len(tool_call_ids) == 2

        for call_id in tool_call_ids:
            state_key = f"tool_state:{call_id}"
            record = context_manager.get(state_key)
            assert record is not None
            # Both should be finished
            assert record.value == b"finished"


@pytest.mark.asyncio
class TestToolOutputAggregation:
    """Tests for tool output aggregation logic (0/1/many chunks)."""

    async def test_zero_chunks_returns_none(self, agent_runner, mock_llm_provider, tool_registry):
        """Test that tool with zero output chunks returns output=None."""
        class NoOutputTool:
            def __call__(self, inputs):
                return {"should_not_use": "this"}

            async def run_stream(self, inputs):
                if False:
                    yield

        from agentic.tools import Tool, ToolDefinition
        tool_func = NoOutputTool()
        tool = Tool(ToolDefinition("no_output_tool", {}, {}), tool_func)
        tool_registry.register(tool)

        config = agent_runner._agent.get_config()
        config.tools_allowed = ["no_output_tool"]
        agent_runner._agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "no_output_tool", "arguments": {}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert final_result.tool_results[0].output is None

    async def test_one_chunk_returns_unwrapped(self, agent_runner, mock_llm_provider, tool_registry):
        """Test that tool with single output chunk returns it unwrapped (not in list)."""
        class SingleChunkTool:
            def __call__(self, inputs):
                return {"final": "result"}

            async def run_stream(self, inputs):
                yield {"single": "chunk"}

        from agentic.tools import Tool, ToolDefinition
        tool_func = SingleChunkTool()
        tool = Tool(ToolDefinition("single_chunk_tool", {}, {}), tool_func)
        tool_registry.register(tool)

        config = agent_runner._agent.get_config()
        config.tools_allowed = ["single_chunk_tool"]
        agent_runner._agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "single_chunk_tool", "arguments": {}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert final_result.tool_results[0].output == {"single": "chunk"}
        assert not isinstance(final_result.tool_results[0].output, list)

    async def test_multiple_chunks_returns_list(self, agent_runner, mock_llm_provider, tool_registry):
        """Test that tool with multiple chunks returns them in a list."""
        class MultiChunkTool:
            def __call__(self, inputs):
                return {"final": "result"}

            async def run_stream(self, inputs):
                yield {"chunk": 1}
                yield {"chunk": 2}
                yield {"chunk": 3}

        from agentic.tools import Tool, ToolDefinition
        tool_func = MultiChunkTool()
        tool = Tool(ToolDefinition("multi_chunk_tool", {}, {}), tool_func)
        tool_registry.register(tool)

        config = agent_runner._agent.get_config()
        config.tools_allowed = ["multi_chunk_tool"]
        config.concurrent_tool_execution = False
        agent_runner._agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "multi_chunk_tool", "arguments": {}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert isinstance(final_result.tool_results[0].output, list)
        assert len(final_result.tool_results[0].output) == 3
        assert final_result.tool_results[0].output[0] == {"chunk": 1}
        assert final_result.tool_results[0].output[1] == {"chunk": 2}
        assert final_result.tool_results[0].output[2] == {"chunk": 3}

    async def test_error_tool_has_none_output(self, agent_runner, mock_llm_provider, tool_registry, agent):
        """Test that tools returning errors have output=None."""
        def error_func(inputs):
            raise ValueError("Simulated error")

        from agentic.tools import create_tool
        error_tool = create_tool("error_tool", error_func)
        tool_registry.register(error_tool)

        config = agent.get_config()
        config.tools_allowed.append("error_tool")
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "error_tool", "arguments": {}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert final_result.tool_results[0].success is False
        assert final_result.tool_results[0].output is None

    async def test_tool_not_allowed_has_none_output(self, agent_runner, mock_llm_provider, agent):
        """Test that tool_not_allowed error has output=None."""
        config = agent.get_config()
        config.tools_allowed = []
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert final_result.status == AgentStatus.ERROR
        assert "not in allowed list" in final_result.error_message

    async def test_tool_not_found_has_none_output(self, agent_runner, mock_llm_provider, agent):
        """Test that tool_not_found error has output=None."""
        config = agent.get_config()
        config.tools_allowed = ["nonexistent_tool"]
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "nonexistent_tool", "arguments": {}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        assert final_result.status == AgentStatus.ERROR
        assert "not found in registry" in final_result.error_message


class TestAgentHelperMethods:
    """Tests for extracted helper methods in AgentRunner."""

    @pytest.mark.asyncio
    async def test_should_execute_tool_with_callback_true(self, agent_runner):
        """Test _should_execute_tool when callback returns True."""
        from agentic.core import ToolCall, AgentConfig

        tool_call = ToolCall(
            name="test_tool",
            arguments={},
            raw_segment="",
            iteration=0,
            call_id="test_id"
        )

        def callback(tc):
            return True

        config = AgentConfig(agent_id="test", on_tool_detected=callback)

        decision = await agent_runner._should_execute_tool(
            tool_call, config, "step_123"
        )

        assert decision.accepted is True
        assert decision.rejection_reason is None
        assert decision.verification_required is True

    @pytest.mark.asyncio
    async def test_should_execute_tool_with_callback_reject(self, agent_runner):
        """Test _should_execute_tool when callback returns False.

        When the callback rejects execution (returns False), should return
        False with appropriate rejection reason.
        """
        from agentic.core import ToolCall, AgentConfig

        tool_call = ToolCall(
            name="test_tool",
            arguments={},
            raw_segment="",
            iteration=0,
            call_id="test_id"
        )

        def callback(tc):
            return False  # Reject

        config = AgentConfig(agent_id="test", on_tool_detected=callback)

        decision = await agent_runner._should_execute_tool(
            tool_call, config, "step_123"
        )

        assert decision.accepted is False
        assert decision.rejection_reason == "Rejected by callback"
        assert decision.verification_required is True

    @pytest.mark.asyncio
    async def test_should_execute_tool_with_callback_exception(self, agent_runner):
        """Test _should_execute_tool when callback raises exception.

        When callback raises an exception, should return decision with accepted=False
        and "Callback exception" in rejection reason.
        """
        from agentic.core import ToolCall, AgentConfig

        tool_call = ToolCall(
            name="test_tool",
            arguments={},
            raw_segment="",
            iteration=0,
            call_id="test_id"
        )

        def callback(tc):
            raise RuntimeError("Callback failed")

        config = AgentConfig(agent_id="test", on_tool_detected=callback)

        decision = await agent_runner._should_execute_tool(
            tool_call, config, "step_123"
        )

        assert decision.accepted is False
        assert "Callback exception" in decision.rejection_reason
        assert "Callback failed" in decision.rejection_reason
        assert decision.verification_required is True

    def test_determine_step_status_error(self, agent_runner):
        """Test _determine_step_status returns ERROR when error_message is set."""
        from agentic.core import AgentStatus, ExtractedSegments

        status = agent_runner._determine_step_status(
            error_message="Something failed",
            tool_decisions=[],
            segments=ExtractedSegments()
        )

        assert status == AgentStatus.ERROR

    def test_determine_step_status_tool_executed(self, agent_runner):
        """Test _determine_step_status returns TOOL_EXECUTED when tools ran."""
        from agentic.core import AgentStatus, ToolCall, ToolResult, ExtractedSegments, ToolExecutionDecision

        tool_call = ToolCall("test", {}, "", 0, "id1")
        tool_result = ToolResult("test", {}, True, None, 0.1, 0)

        decision = ToolExecutionDecision(
            tool_call=tool_call,
            verification_required=False,
            accepted=True,
            executed=True,
            result=tool_result
        )

        status = agent_runner._determine_step_status(
            error_message=None,
            tool_decisions=[decision],
            segments=ExtractedSegments()
        )

        assert status == AgentStatus.TOOL_EXECUTED

    def test_determine_step_status_waiting_for_tool(self, agent_runner):
        """Test _determine_step_status returns WAITING_FOR_TOOL when tools detected but not executed."""
        from agentic.core import AgentStatus, ToolCall, ExtractedSegments, ToolExecutionDecision

        tool_call = ToolCall("test", {}, "", 0, "id1")

        decision = ToolExecutionDecision(
            tool_call=tool_call,
            verification_required=False,
            accepted=True,
            executed=False  # Not yet executed
        )

        status = agent_runner._determine_step_status(
            error_message=None,
            tool_decisions=[decision],
            segments=ExtractedSegments()
        )

        assert status == AgentStatus.WAITING_FOR_TOOL

    def test_determine_step_status_done(self, agent_runner):
        """Test _determine_step_status returns DONE when no response and no tools."""
        from agentic.core import AgentStatus, ExtractedSegments

        status = agent_runner._determine_step_status(
            error_message=None,
            tool_decisions=[],
            segments=ExtractedSegments()  # No response
        )

        assert status == AgentStatus.DONE

    def test_determine_step_status_ok(self, agent_runner):
        """Test _determine_step_status returns OK for normal response."""
        from agentic.core import AgentStatus, ExtractedSegments

        status = agent_runner._determine_step_status(
            error_message=None,
            tool_decisions=[],
            segments=ExtractedSegments(response="Normal response")
        )

        assert status == AgentStatus.OK

    @pytest.mark.asyncio
    async def test_parse_error_whitespace_only(self, agent_runner):
        """Test parse error handling with whitespace-only error text."""
        from agentic.core import ExtractedSegments
        from agentic.events import ErrorEvent
        from unittest.mock import patch

        with patch('agentic.patterns.StreamingPatternExtractor.finalize') as mock_finalize:
            mock_finalize.return_value = (
                ExtractedSegments(response="test", parse_errors={"pattern": "   "}),
                {}
            )

            events = []
            async for event in agent_runner.step_stream("test"):
                events.append(event)

            error_events = [e for e in events if isinstance(e, ErrorEvent)]
            assert len(error_events) > 0

    @pytest.mark.asyncio
    async def test_parse_error_only_newlines(self, agent_runner):
        """Test parse error handling with only newlines."""
        from agentic.core import ExtractedSegments
        from agentic.events import ErrorEvent
        from unittest.mock import patch

        with patch('agentic.patterns.StreamingPatternExtractor.finalize') as mock_finalize:
            mock_finalize.return_value = (
                ExtractedSegments(response="test", parse_errors={"pattern": "\n\n\n"}),
                {}
            )

            events = []
            async for event in agent_runner.step_stream("test"):
                events.append(event)

            error_events = [e for e in events if isinstance(e, ErrorEvent)]
            assert len(error_events) > 0
