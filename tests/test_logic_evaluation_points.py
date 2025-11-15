"""
Tests for logic evaluation points.

These tests verify that different evaluation points can be configured
and that conditions are evaluated at the appropriate times.
"""
import pytest

from agentic.logic import LogicRunner, LogicConfig, LogicCondition
from agentic.events import StepCompleteEvent


class TestLogicEvaluationPoints:
    """Tests for different evaluation points in logic conditions."""

    def test_evaluation_point_llm_chunk_configuration(self):
        """Test that llm_chunk evaluation point can be configured.

        evaluation_point="llm_chunk" allows conditions to be checked
        on every LLM chunk as it streams.
        """
        condition = LogicCondition(
            pattern_set="default",
            pattern_name="STOP",
            match_type="regex",
            target="response",
            evaluation_point="llm_chunk"
        )

        assert condition.evaluation_point == "llm_chunk"

    def test_evaluation_point_tool_detected_configuration(self):
        """Test that tool_detected evaluation point can be configured.

        evaluation_point="tool_detected" allows conditions to be checked
        immediately after a tool pattern is extracted, before tool execution.
        """
        condition = LogicCondition(
            pattern_set="default",
            pattern_name="tool",
            match_type="contains",
            target="response",
            evaluation_point="tool_detected"
        )

        assert condition.evaluation_point == "tool_detected"

    def test_evaluation_point_tool_finished_configuration(self):
        """Test that tool_finished evaluation point can be configured.

        evaluation_point="tool_finished" allows conditions to be checked
        after tool execution is done, allowing inspection of tool results.
        """
        condition = LogicCondition(
            pattern_set="default",
            pattern_name="result",
            match_type="regex",
            target="tool_output",
            evaluation_point="tool_finished"
        )

        assert condition.evaluation_point == "tool_finished"

    def test_evaluation_point_any_event_configuration(self):
        """Test that any_event evaluation point can be configured.

        evaluation_point="any_event" allows conditions to be checked
        on every single event emitted during execution.
        """
        condition = LogicCondition(
            pattern_set="default",
            pattern_name="ready",
            match_type="regex",
            target="context:trigger",
            evaluation_point="any_event"
        )

        assert condition.evaluation_point == "any_event"

    def test_evaluation_point_pattern_start_configuration(self):
        """Test that pattern_start evaluation point can be configured.

        evaluation_point="pattern_start" allows conditions to be checked
        when a pattern start tag is detected.
        """
        condition = LogicCondition(
            pattern_set="default",
            pattern_name="reasoning",
            match_type="contains",
            target="response",
            evaluation_point="pattern_start"
        )

        assert condition.evaluation_point == "pattern_start"

    def test_evaluation_point_step_complete(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test condition evaluation at step completion.

        evaluation_point="step_complete" (the default) should check
        conditions only after the entire agent step completes.
        """
        mock_llm_provider.set_response("Final result: DONE")

        condition = LogicCondition(
            pattern_set="default",
            pattern_name="DONE",
            match_type="regex",
            target="response",
            evaluation_point="step_complete"
        )

        config = LogicConfig(
            logic_id="step_complete_eval",
            max_iterations=10,
            stop_conditions=[condition]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()

        # Should stop after first complete step
        assert len(results) == 1

    def test_evaluation_point_auto_inference(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test 'auto' evaluation_point inference.

        When evaluation_point='auto', the system should infer the appropriate
        evaluation point based on the target:
        - context:* targets should use step_complete
        - Other targets default to step_complete
        """
        # Test with context target (should use step_complete)
        context_manager.set("status", b"complete")
        mock_llm_provider.set_response("Processing")

        condition = LogicCondition(
            pattern_set="default",
            pattern_name="complete",
            match_type="regex",
            target="context:status",
            evaluation_point="auto"  # Should infer step_complete
        )

        config = LogicConfig(
            logic_id="auto_eval",
            max_iterations=5,
            stop_conditions=[condition]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()

        # Should stop due to context match
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_evaluation_point_with_streaming(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test that evaluation points work with streaming execution.

        This test verifies that evaluation points can be used in streaming
        mode and that conditions are properly evaluated.
        """
        mock_llm_provider.set_response("Test response for streaming")

        # Use a condition that won't match, so we run a full iteration
        condition = LogicCondition(
            pattern_set="default",
            pattern_name="NEVER_MATCH_THIS_PATTERN",
            match_type="regex",
            target="response",
            evaluation_point="llm_chunk"  # Evaluated during streaming
        )

        config = LogicConfig(
            logic_id="streaming_eval",
            max_iterations=1,
            stop_conditions=[condition]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = []
        async for event in runner.run_stream():
            if isinstance(event, StepCompleteEvent):
                results.append(event.result)

        # Should complete one iteration without stopping
        assert len(results) == 1

    def test_multiple_evaluation_points(self):
        """Test that multiple conditions can have different evaluation points.

        Different conditions can use different evaluation points in the
        same LogicConfig.
        """
        token_condition = LogicCondition(
            pattern_set="default",
            pattern_name="STOP",
            match_type="regex",
            target="response",
            evaluation_point="llm_chunk"
        )

        complete_condition = LogicCondition(
            pattern_set="default",
            pattern_name="COMPLETE",
            match_type="regex",
            target="response",
            evaluation_point="step_complete"
        )

        # Verify both evaluation_points are set correctly
        assert token_condition.evaluation_point == "llm_chunk"
        assert complete_condition.evaluation_point == "step_complete"

        # Can be used together in config
        config = LogicConfig(
            logic_id="multi_eval",
            max_iterations=10,
            stop_conditions=[token_condition, complete_condition]
        )

        assert len(config.stop_conditions) == 2
        assert config.stop_conditions[0].evaluation_point == "llm_chunk"
        assert config.stop_conditions[1].evaluation_point == "step_complete"
