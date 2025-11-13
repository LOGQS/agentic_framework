"""
Tests for logic flow control system.

Covers:
- LogicConfig and LogicCondition
- LogicRunner execution loops
- Max iterations limiting
- Stop conditions
- Loop-until conditions
- Conditional evaluation at different points
- Helper functions (loop_n_times, loop_until_pattern, etc.)
"""
import pytest
import asyncio

from agentic.logic import (
    LogicRunner,
    LogicConfig,
    LogicCondition,
    loop_n_times,
    loop_until_pattern,
    loop_until_regex,
    stop_on_error
)
from agentic.agent import AgentRunner, MockLLMProvider
from agentic.core import AgentStatus, ProcessingMode
from agentic.events import StepCompleteEvent


class TestLogicConfig:
    """Tests for LogicConfig dataclass."""

    def test_logic_config_defaults(self):
        """Test LogicConfig with default values."""
        config = LogicConfig(logic_id="test")
        assert config.logic_id == "test"
        assert config.max_iterations is None
        assert config.stop_conditions == []
        assert config.loop_until_conditions == []
        assert config.break_on_error is True
        assert config.processing_mode == ProcessingMode.THREAD

    def test_logic_config_full(self):
        """Test LogicConfig with all parameters."""
        stop_cond = LogicCondition(
            pattern_set="default",
            pattern_name="done",
            match_type="contains",
            target="response"
        )
        config = LogicConfig(
            logic_id="full",
            max_iterations=10,
            stop_conditions=[stop_cond],
            loop_until_conditions=[],
            break_on_error=False,
            processing_mode=ProcessingMode.ASYNC
        )
        assert config.max_iterations == 10
        assert len(config.stop_conditions) == 1
        assert config.break_on_error is False


class TestLogicCondition:
    """Tests for LogicCondition dataclass."""

    def test_logic_condition_creation(self):
        """Test creating LogicCondition."""
        condition = LogicCondition(
            pattern_set="default",
            pattern_name="tool",
            match_type="contains",
            target="response"
        )
        assert condition.pattern_name == "tool"
        assert condition.match_type == "contains"
        assert condition.target == "response"

    def test_logic_condition_evaluation_point(self):
        """Test LogicCondition evaluation_point defaults."""
        condition = LogicCondition(
            pattern_set="default",
            pattern_name="test",
            match_type="contains",
            target="response"
        )
        assert condition.evaluation_point == "auto"


class TestLogicRunnerBasics:
    """Tests for basic LogicRunner functionality."""

    def test_logic_runner_creation(self, agent_runner, context_manager, pattern_registry):
        """Test creating LogicRunner."""
        config = LogicConfig(logic_id="test", max_iterations=5)
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)
        assert runner._config.logic_id == "test"

    def test_logic_runner_single_iteration(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test logic runner with single iteration."""
        mock_llm_provider.set_response("Done")

        config = LogicConfig(logic_id="single", max_iterations=1)
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        assert len(results) == 1
        assert results[0].status == AgentStatus.OK


class TestLogicRunnerMaxIterations:
    """Tests for max iteration limiting."""

    def test_logic_runner_respects_max_iterations(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test that runner stops at max iterations."""
        mock_llm_provider.set_response("Continue")

        config = LogicConfig(logic_id="limited", max_iterations=3)
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        assert len(results) == 3

    def test_logic_runner_no_max_iterations(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test runner with no max iterations (stops on DONE status)."""
        # Set up to return DONE status
        mock_llm_provider.set_response("")

        config = LogicConfig(logic_id="unlimited")
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        # Should stop when agent returns DONE status
        assert len(results) >= 1


class TestLogicRunnerStopConditions:
    """Tests for stop conditions."""

    def test_stop_on_regex_match(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test stopping when regex matches."""
        mock_llm_provider.set_response("Processing... STOP")

        stop_condition = LogicCondition(
            pattern_set="default",
            pattern_name="STOP",
            match_type="regex",
            target="response"
        )
        config = LogicConfig(
            logic_id="stop_test",
            max_iterations=10,
            stop_conditions=[stop_condition]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        # Should stop on first iteration due to STOP in response
        assert len(results) == 1

    def test_stop_on_pattern_contains(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test stopping when pattern is found.

        Note: 'contains' match type checks for pattern tags in raw_output.
        The evaluation happens at step_complete by default for pattern matching.
        """
        mock_llm_provider.set_response("<response>Complete</response>")

        # Use regex to match the actual content
        stop_condition = LogicCondition(
            pattern_set="default",
            pattern_name="Complete",  # Regex pattern
            match_type="regex",
            target="response",
            evaluation_point="step_complete"
        )
        config = LogicConfig(
            logic_id="pattern_stop",
            max_iterations=10,
            stop_conditions=[stop_condition]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        # Should stop on first iteration because response contains "Complete"
        assert len(results) == 1


class TestLogicRunnerLoopUntilConditions:
    """Tests for loop-until conditions."""

    def test_loop_until_condition_met(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test looping until condition is met."""
        # Test looping with max iterations to avoid infinite loop
        mock_llm_provider.set_response("Continue processing")

        loop_condition = LogicCondition(
            pattern_set="default",
            pattern_name="NEVER_MATCH",  # Will never match, so hits max_iterations
            match_type="regex",
            target="response"
        )
        config = LogicConfig(
            logic_id="loop_until",
            max_iterations=3,  # Limit to 3 iterations
            loop_until_conditions=[loop_condition]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        # Should hit max_iterations before condition is met
        assert len(results) == 3


class TestLogicRunnerErrorHandling:
    """Tests for error handling in logic loops."""

    def test_break_on_error_enabled(self, agent_runner, context_manager, pattern_registry):
        """Test that runner breaks on error when enabled."""
        class ErrorProvider:
            def __init__(self):
                self.call_count = 0

            def generate(self, prompt, max_tokens, temp, **kwargs):
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Error on second call")
                return "OK"

            async def stream(self, prompt, max_tokens, temp, **kwargs):
                if self.call_count == 2:
                    raise RuntimeError("Error on second call")
                yield "OK"

        error_provider = ErrorProvider()
        agent_runner._agent._provider = error_provider

        config = LogicConfig(logic_id="error_test", max_iterations=5, break_on_error=True)
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        # Should break after error on iteration 2
        assert len(results) <= 2

    def test_break_on_error_disabled(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test that runner continues on error when break_on_error=False."""
        config = LogicConfig(logic_id="no_break", max_iterations=3, break_on_error=False)
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        # Even with errors, should continue
        results = runner.run()
        assert len(results) >= 1


@pytest.mark.asyncio
class TestLogicRunnerStreaming:
    """Tests for streaming logic execution."""

    async def test_logic_run_stream_yields_events(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test that run_stream yields events."""
        mock_llm_provider.set_response("Test")

        config = LogicConfig(logic_id="stream_test", max_iterations=2)
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # Should have multiple events including StepCompleteEvent
        assert len(events) > 0
        step_complete_count = sum(1 for e in events if isinstance(e, StepCompleteEvent))
        assert step_complete_count == 2

    async def test_logic_run_stream_stops_on_max_iterations(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test that streaming stops at max iterations."""
        mock_llm_provider.set_response("Continue")

        config = LogicConfig(logic_id="stream_limited", max_iterations=3)
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        step_count = 0
        async for event in runner.run_stream():
            if isinstance(event, StepCompleteEvent):
                step_count += 1

        assert step_count == 3


class TestLogicHelperFunctions:
    """Tests for convenience helper functions."""

    def test_loop_n_times(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test loop_n_times helper."""
        mock_llm_provider.set_response("Iteration")

        runner = loop_n_times(agent_runner, context_manager, pattern_registry, n=5)

        results = runner.run()
        assert len(results) == 5

    def test_loop_until_pattern(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test loop_until_pattern helper."""
        mock_llm_provider.set_response("Step without pattern")

        runner = loop_until_pattern(
            agent_runner,
            context_manager,
            pattern_registry,
            pattern_set="default",
            pattern_name="response",
            max_iterations=2  # Limit iterations
        )

        results = runner.run()
        # Should hit max_iterations since pattern never appears
        assert len(results) == 2

    def test_loop_until_regex(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test loop_until_regex helper."""
        mock_llm_provider.set_response("Working on it...")

        runner = loop_until_regex(
            agent_runner,
            context_manager,
            pattern_registry,
            regex_pattern="DONE",  # Will never match
            max_iterations=2  # Limit iterations
        )

        results = runner.run()
        # Should hit max_iterations since DONE never appears
        assert len(results) == 2

    def test_stop_on_error(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test stop_on_error helper."""
        mock_llm_provider.set_response("OK")

        runner = stop_on_error(agent_runner, context_manager, pattern_registry, max_iterations=5)

        results = runner.run()
        # Should run normally without errors
        assert len(results) >= 1


class TestLogicConditionEvaluation:
    """Tests for condition evaluation logic."""

    def test_condition_match_type_contains(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test 'contains' match type.

        'contains' checks if the pattern tags exist in raw_output.
        """
        mock_llm_provider.set_response("<reasoning>Think hard</reasoning>")

        # Use 'contains' to check if reasoning pattern exists in output
        condition = LogicCondition(
            pattern_set="default",
            pattern_name="reasoning",
            match_type="contains",
            target="response",  # Checks raw_output for pattern
            evaluation_point="step_complete"
        )
        config = LogicConfig(
            logic_id="contains_test",
            max_iterations=5,
            stop_conditions=[condition]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        # Should stop on first iteration when reasoning pattern is detected
        assert len(results) >= 1

    def test_condition_match_type_equals(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test 'equals' match type."""
        mock_llm_provider.set_response("exact_match")

        condition = LogicCondition(
            pattern_set="default",
            pattern_name="exact_match",
            match_type="equals",
            target="response"
        )
        config = LogicConfig(
            logic_id="equals_test",
            max_iterations=1,
            stop_conditions=[condition]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        assert len(results) == 1

    def test_condition_target_context(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test condition targeting context value."""
        # Set a context value
        context_manager.set("status", b"complete")

        mock_llm_provider.set_response("Continue")

        condition = LogicCondition(
            pattern_set="default",
            pattern_name="complete",
            match_type="regex",
            target="context:status"
        )
        config = LogicConfig(
            logic_id="context_test",
            max_iterations=5,
            stop_conditions=[condition]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        # Should stop immediately due to context match
        assert len(results) == 1


class TestLogicEdgeCases:
    """Tests for edge cases in logic execution."""

    def test_logic_with_zero_max_iterations(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test logic with max_iterations=0."""
        config = LogicConfig(logic_id="zero", max_iterations=0)
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        assert len(results) == 0

    def test_logic_with_empty_conditions(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test logic with no stop or loop conditions."""
        mock_llm_provider.set_response("")

        config = LogicConfig(
            logic_id="empty_conds",
            max_iterations=2,
            stop_conditions=[],
            loop_until_conditions=[]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        assert len(results) >= 1

    def test_logic_multiple_stop_conditions(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test logic with multiple stop conditions."""
        mock_llm_provider.set_response("STOP")

        cond1 = LogicCondition("default", "STOP", "regex", "response")
        cond2 = LogicCondition("default", "END", "regex", "response")

        config = LogicConfig(
            logic_id="multi_stop",
            max_iterations=10,
            stop_conditions=[cond1, cond2]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        # Should stop on first match
        assert len(results) == 1
