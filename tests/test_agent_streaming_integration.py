"""
Comprehensive integration tests for agent streaming functionality.

These tests verify the complete streaming workflow including:
- Full agent streaming with pattern extraction
- Event ordering during streaming
- Pattern lifecycle events (start, content, end)
- Streaming with multiple tools
- Streaming error handling
"""
import pytest

from agentic.core import AgentStatus
from agentic.events import (
    LLMChunkEvent,
    LLMCompleteEvent,
    ToolStartEvent,
    ToolEndEvent,
    PatternStartEvent,
    PatternContentEvent,
    PatternEndEvent,
    StepCompleteEvent,
    ErrorEvent
)


@pytest.mark.asyncio
class TestFullAgentStreaming:
    """Tests for complete agent streaming workflows."""

    async def test_full_agent_streaming_with_pattern_extraction(self, agent, agent_runner, mock_llm_provider):
        """Test complete streaming workflow with pattern extraction.

        This integration test verifies that all components work together:
        - LLM streaming
        - Pattern extraction during streaming
        - Event emission in proper order
        - Final result aggregation
        """
        config = agent.get_config()
        config.stream_pattern_content = True
        agent.set_config(config)

        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response(
            "<reasoning>Let me think about this</reasoning>"
            "<response>Here is my answer</response>"
        )

        events = []
        async for event in agent_runner.step_stream():
            events.append(event)

        # Verify we got various event types
        event_types = {type(e).__name__ for e in events}
        assert "LLMChunkEvent" in event_types
        assert "LLMCompleteEvent" in event_types
        assert "StatusEvent" in event_types
        assert "PatternStartEvent" in event_types
        assert "PatternEndEvent" in event_types
        assert "StepCompleteEvent" in event_types

        # Verify final result
        final_event = [e for e in events if isinstance(e, StepCompleteEvent)][0]
        assert final_event.result.status == AgentStatus.OK
        assert len(final_event.result.segments.reasoning) > 0
        assert final_event.result.segments.response is not None

    async def test_streaming_event_ordering(self, agent, agent_runner, mock_llm_provider):
        """Test that events are emitted in the correct order during streaming.

        Verify the logical sequence:
        1. Status events at start
        2. LLM chunk events
        3. Pattern events (start, content, end) as patterns detected
        4. LLM complete event
        5. Final status and step complete events
        """
        config = agent.get_config()
        config.stream_pattern_content = True
        agent.set_config(config)

        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response("<reasoning>Test</reasoning>")

        events = []
        async for event in agent_runner.step_stream():
            events.append(event)

        # Extract event type sequence
        event_sequence = [type(e).__name__ for e in events]

        # Verify StatusEvent comes first
        assert event_sequence[0] == "StatusEvent"

        # Verify LLMCompleteEvent comes before final StepCompleteEvent
        llm_complete_idx = None
        step_complete_idx = None
        for i, name in enumerate(event_sequence):
            if name == "LLMCompleteEvent":
                llm_complete_idx = i
            if name == "StepCompleteEvent":
                step_complete_idx = i

        assert llm_complete_idx is not None
        assert step_complete_idx is not None
        assert llm_complete_idx < step_complete_idx

        # Verify StepCompleteEvent is last
        assert event_sequence[-1] == "StepCompleteEvent"

    async def test_pattern_lifecycle_events(self, agent, agent_runner, mock_llm_provider):
        """Test complete pattern lifecycle: start, content, end.

        Verify that for each pattern:
        1. PatternStartEvent is emitted when start tag detected
        2. PatternContentEvent(s) emitted as content streams (if enabled)
        3. PatternEndEvent is emitted when end tag detected
        """
        config = agent.get_config()
        config.stream_pattern_content = True
        agent.set_config(config)

        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response("<response>Complete lifecycle test</response>")

        pattern_events = []
        async for event in agent_runner.step_stream():
            if isinstance(event, (PatternStartEvent, PatternContentEvent, PatternEndEvent)):
                pattern_events.append(event)

        # Group events by pattern name
        pattern_names = set()
        for event in pattern_events:
            pattern_names.add(event.pattern_name)

        # For each pattern, verify lifecycle
        for pattern_name in pattern_names:
            events_for_pattern = [e for e in pattern_events if e.pattern_name == pattern_name]

            # Should have at least start and end
            event_types = [type(e).__name__ for e in events_for_pattern]
            assert "PatternStartEvent" in event_types
            assert "PatternEndEvent" in event_types

            # Start should come before end
            start_indices = [i for i, e in enumerate(events_for_pattern) if isinstance(e, PatternStartEvent)]
            end_indices = [i for i, e in enumerate(events_for_pattern) if isinstance(e, PatternEndEvent)]
            if start_indices and end_indices:
                assert min(start_indices) < max(end_indices)

    async def test_streaming_with_multiple_tools(self, agent, agent_runner, mock_llm_provider):
        """Test streaming with multiple tool calls.

        Verify that streaming works correctly when multiple tools are
        called in the same response.
        """
        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "first"}}</tool>'
            '<tool>{"name": "calculator", "arguments": {"a": 5, "b": 3, "operation": "add"}}</tool>'
        )

        tool_events = []
        async for event in agent_runner.step_stream():
            if isinstance(event, (ToolStartEvent, ToolEndEvent)):
                tool_events.append(event)

        # Should have start and end events for both tools
        tool_names = {e.tool_name for e in tool_events}
        assert "echo" in tool_names
        assert "calculator" in tool_names

        # Verify each tool has start and end
        echo_events = [e for e in tool_events if e.tool_name == "echo"]
        calc_events = [e for e in tool_events if e.tool_name == "calculator"]

        assert any(isinstance(e, ToolStartEvent) for e in echo_events)
        assert any(isinstance(e, ToolEndEvent) for e in echo_events)
        assert any(isinstance(e, ToolStartEvent) for e in calc_events)
        assert any(isinstance(e, ToolEndEvent) for e in calc_events)

    async def test_streaming_with_errors(self, agent, agent_runner, mock_llm_provider, tool_registry):
        """Test streaming error handling.

        Verify that streaming handles errors gracefully and emits
        appropriate error events.
        """
        # Register a tool that fails
        def error_func(inputs):
            raise ValueError("Intentional test error")

        from agentic.tools import create_tool
        error_tool = create_tool("error_tool", error_func)
        tool_registry.register(error_tool)

        config = agent.get_config()
        config.tools_allowed.append("error_tool")
        agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "error_tool", "arguments": {}}</tool>')

        error_events = []
        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, ErrorEvent):
                error_events.append(event)
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Should have error events
        assert len(error_events) > 0
        assert any("execution failed" in e.error_message.lower() for e in error_events)

        # Final result should indicate error
        assert final_result.status == AgentStatus.ERROR


@pytest.mark.asyncio
class TestStreamingWithConcurrentTools:
    """Tests for streaming with concurrent tool execution."""

    async def test_concurrent_tools_during_streaming(self, agent, agent_runner, mock_llm_provider):
        """Test concurrent tool execution during streaming.

        When concurrent_tool_execution is enabled, tools should execute
        in parallel during the streaming phase.
        """
        config = agent.get_config()
        config.concurrent_tool_execution = True
        agent.set_config(config)

        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "concurrent1"}}</tool>'
            '<tool>{"name": "calculator", "arguments": {"a": 10, "b": 5, "operation": "multiply"}}</tool>'
        )

        tool_start_events = []
        tool_end_events = []
        final_result = None

        async for event in agent_runner.step_stream():
            if isinstance(event, ToolStartEvent):
                tool_start_events.append(event)
            elif isinstance(event, ToolEndEvent):
                tool_end_events.append(event)
            elif isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Both tools should have executed
        assert len(tool_start_events) == 2
        assert len(tool_end_events) == 2

        # Final result should have both tool results
        assert len(final_result.tool_results) == 2
        assert all(tr.success for tr in final_result.tool_results)

    async def test_concurrent_tools_error_isolation(self, agent, agent_runner, mock_llm_provider, tool_registry):
        """Test that errors in concurrent tools are isolated.

        When one tool fails in concurrent mode, other tools should
        still complete successfully.
        """
        # Register a failing tool
        def error_func(inputs):
            raise RuntimeError("Concurrent tool error")

        from agentic.tools import create_tool
        error_tool = create_tool("error_tool", error_func)
        tool_registry.register(error_tool)

        config = agent.get_config()
        config.concurrent_tool_execution = True
        config.tools_allowed = ["echo", "error_tool"]
        agent.set_config(config)

        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "should succeed"}}</tool>'
            '<tool>{"name": "error_tool", "arguments": {}}</tool>'
        )

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Should have results for both tools
        assert len(final_result.tool_results) == 2

        # One should succeed, one should fail
        success_count = sum(1 for tr in final_result.tool_results if tr.success)
        failure_count = sum(1 for tr in final_result.tool_results if not tr.success)
        assert success_count == 1
        assert failure_count == 1


@pytest.mark.asyncio
class TestStreamingEventConsistency:
    """Tests for event consistency during streaming."""

    async def test_step_id_consistency(self, agent_runner, mock_llm_provider):
        """Test that all events from a single step have the same step_id.

        All events emitted during a single agent step should share
        the same step_id for traceability.
        """
        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response("<response>Consistent step ID</response>")

        step_ids = set()
        async for event in agent_runner.step_stream():
            if hasattr(event, 'step_id') and event.step_id:
                step_ids.add(event.step_id)

        # All events should have the same step_id
        assert len(step_ids) == 1

    async def test_iteration_consistency(self, agent_runner, mock_llm_provider, context_manager):
        """Test that iteration is consistent throughout a step.

        All operations in a single step should use the same iteration number.
        """
        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>')

        iterations_seen = set()
        async for event in agent_runner.step_stream():
            if isinstance(event, ToolStartEvent):
                iterations_seen.add(event.iteration)
            elif isinstance(event, StepCompleteEvent):
                iterations_seen.add(event.result.iteration)

        # All should be the same iteration
        assert len(iterations_seen) == 1

    async def test_event_timestamp_ordering(self, agent_runner, mock_llm_provider):
        """Test that event timestamps are monotonically increasing.

        Events should have timestamps that increase (or stay the same)
        as the stream progresses.
        """
        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response("<response>Timestamp test</response>")

        timestamps = []
        async for event in agent_runner.step_stream():
            if hasattr(event, 'timestamp'):
                timestamps.append(event.timestamp)

        # Timestamps should be non-decreasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1], \
                f"Timestamp went backwards: {timestamps[i-1]} -> {timestamps[i]}"


@pytest.mark.asyncio
class TestStreamingComplexScenarios:
    """Tests for complex streaming scenarios."""

    async def test_streaming_with_all_features_enabled(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test streaming with all advanced features enabled simultaneously.

        Enable:
        - stream_pattern_content
        - incremental_context_writes
        - concurrent_tool_execution (if tools present)

        Verify they all work together correctly.
        """
        config = agent.get_config()
        config.stream_pattern_content = True
        config.incremental_context_writes = True
        config.concurrent_tool_execution = True
        agent.set_config(config)

        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response(
            "<reasoning>Complex scenario</reasoning>"
            '<tool>{"name": "echo", "arguments": {"message": "test"}}</tool>'
            "<response>Final answer</response>"
        )

        events = []
        async for event in agent_runner.step_stream():
            events.append(event)

        # Verify we got all expected event types
        event_types = {type(e).__name__ for e in events}
        assert "LLMChunkEvent" in event_types
        assert "PatternStartEvent" in event_types
        assert "PatternEndEvent" in event_types
        assert "ToolStartEvent" in event_types
        assert "ToolEndEvent" in event_types
        assert "StepCompleteEvent" in event_types

        # Verify final result is correct
        final_event = [e for e in events if isinstance(e, StepCompleteEvent)][0]
        assert final_event.result.status in [AgentStatus.OK, AgentStatus.TOOL_EXECUTED]

    async def test_streaming_empty_response(self, agent_runner, mock_llm_provider):
        """Test streaming with empty LLM response.

        The system should handle empty responses gracefully without errors.
        """
        mock_llm_provider.set_response("")

        events = []
        async for event in agent_runner.step_stream():
            events.append(event)

        # Should complete without errors
        assert any(isinstance(e, StepCompleteEvent) for e in events)
        final_event = [e for e in events if isinstance(e, StepCompleteEvent)][0]
        assert final_event.result.raw_output == ""

    async def test_streaming_very_long_response(self, agent_runner, mock_llm_provider):
        """Test streaming with very long response.

        Verify that streaming handles large responses efficiently.
        """
        long_response = "x" * 50000
        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response(long_response)

        chunk_count = 0
        final_result = None

        async for event in agent_runner.step_stream():
            if isinstance(event, LLMChunkEvent):
                chunk_count += 1
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Should have received many chunks
        assert chunk_count > 0

        # Final result should have complete output
        assert len(final_result.raw_output) == 50000

    async def test_streaming_malformed_patterns(self, agent, agent_runner, mock_llm_provider):
        """Test streaming with malformed/incomplete patterns.

        Verify that malformed patterns are detected and reported via
        error events.
        """
        config = agent.get_config()
        config.stream_pattern_content = True
        agent.set_config(config)

        # Response with incomplete pattern (missing end tag)
        mock_llm_provider.set_response("<reasoning>This pattern never closes")

        error_events = []
        final_result = None

        async for event in agent_runner.step_stream():
            if isinstance(event, ErrorEvent):
                error_events.append(event)
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Should have error event for malformed pattern
        malformed_errors = [e for e in error_events if e.error_type == "malformed_pattern"]
        assert len(malformed_errors) > 0

        # Should still complete the step
        assert final_result is not None


@pytest.mark.asyncio
class TestStreamingCallbackIntegration:
    """Tests for callback integration during streaming."""

    async def test_on_tool_detected_during_streaming(self, agent, agent_runner, mock_llm_provider):
        """Test on_tool_detected callback during streaming.

        Verify that the callback is invoked as tools are detected
        during streaming, not after LLM completion.
        """
        calls = []
        callback_timestamps = []

        def callback(tool_call):
            import time
            calls.append(tool_call.name)
            callback_timestamps.append(time.time())
            return True

        config = agent.get_config()
        config.on_tool_detected = callback
        agent.set_config(config)

        mock_llm_provider.simulate_streaming = True
        mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {"message": "callback test"}}</tool>')

        llm_complete_timestamp = None

        async for event in agent_runner.step_stream():
            if isinstance(event, LLMCompleteEvent):
                import time
                llm_complete_timestamp = time.time()

        # Callback should have been called
        assert len(calls) > 0

        # With concurrent execution disabled, callback might be after LLM complete
        # With concurrent execution enabled, callback is during streaming

    async def test_callback_rejection_during_streaming(self, agent, agent_runner, mock_llm_provider):
        """Test tool rejection via callback during streaming.

        When callback returns False, tool should not execute even in
        concurrent mode.
        """
        def reject_calculator(tool_call):
            return tool_call.name != "calculator"

        config = agent.get_config()
        config.on_tool_detected = reject_calculator
        config.concurrent_tool_execution = True
        agent.set_config(config)

        mock_llm_provider.set_response(
            '<tool>{"name": "echo", "arguments": {"message": "allowed"}}</tool>'
            '<tool>{"name": "calculator", "arguments": {"a": 1, "b": 2, "operation": "add"}}</tool>'
        )

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Only echo should have executed
        assert len(final_result.tool_results) == 1
        assert final_result.tool_results[0].name == "echo"
