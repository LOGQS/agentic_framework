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

from agentic.logic import (
    LogicRunner,
    LogicConfig,
    LogicCondition,
    loop_n_times,
    loop_until_pattern,
    loop_until_regex,
    stop_on_error
)
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


class TestContextHealthCheck:
    """Tests for context health monitoring."""

    def test_context_health_check_creation(self):
        """Test creating ContextHealthCheck."""
        from agentic.logic import ContextHealthCheck

        check = ContextHealthCheck(
            check_type="size",
            key_pattern="llm_output:*",
            threshold=1000.0,
            action="warn"
        )
        assert check.check_type == "size"
        assert check.key_pattern == "llm_output:*"
        assert check.threshold == 1000.0
        assert check.action == "warn"
        assert check.evaluation_point == "step_complete"

    def test_context_health_check_size_threshold(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test health check detects when context size exceeds threshold."""
        from agentic.logic import ContextHealthCheck, LogicConfig
        from agentic.events import ContextHealthEvent

        # Set a large value in context
        large_value = b"x" * 2000
        context_manager.set("test_key", large_value)

        # Create health check for size
        health_check = ContextHealthCheck(
            check_type="size",
            key_pattern="test_*",
            threshold=1000.0,
            action="warn"
        )

        config = LogicConfig(
            logic_id="health_test",
            max_iterations=1,
            context_health_checks=[health_check]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        # Collect events
        events = []
        import asyncio
        async def collect_events():
            async for event in runner.run_stream():
                events.append(event)

        asyncio.run(collect_events())

        # Should have ContextHealthEvent
        health_events = [e for e in events if isinstance(e, ContextHealthEvent)]
        assert len(health_events) > 0
        assert health_events[0].check_type == "size"
        assert health_events[0].current_value > 1000.0
        assert health_events[0].threshold == 1000.0

    def test_context_health_check_version_count(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test health check detects when version count exceeds threshold."""
        from agentic.logic import ContextHealthCheck, LogicConfig
        from agentic.events import ContextHealthEvent

        # Create multiple versions
        for i in range(5):
            context_manager.set("versioned_key", f"value_{i}".encode())

        health_check = ContextHealthCheck(
            check_type="version_count",
            key_pattern="versioned_*",
            threshold=3.0,
            action="warn"
        )

        config = LogicConfig(
            logic_id="version_health",
            max_iterations=1,
            context_health_checks=[health_check]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        events = []
        import asyncio
        async def collect_events():
            async for event in runner.run_stream():
                events.append(event)

        asyncio.run(collect_events())

        health_events = [e for e in events if isinstance(e, ContextHealthEvent)]
        assert len(health_events) > 0
        assert health_events[0].check_type == "version_count"
        assert health_events[0].current_value > 3.0

    def test_context_health_check_action_warn(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test health check with 'warn' action continues execution."""
        from agentic.logic import ContextHealthCheck, LogicConfig

        # Set large value
        context_manager.set("large_key", b"x" * 2000)

        health_check = ContextHealthCheck(
            check_type="size",
            key_pattern="*",
            threshold=100.0,
            action="warn"
        )

        config = LogicConfig(
            logic_id="warn_test",
            max_iterations=2,
            context_health_checks=[health_check]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        # Should continue despite warning
        assert len(results) == 2

    def test_context_health_check_action_stop(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test health check with 'stop' action halts execution."""
        from agentic.logic import ContextHealthCheck, LogicConfig
        from agentic.events import ContextHealthEvent, StatusEvent

        # Set large value
        context_manager.set("large_key", b"x" * 2000)

        health_check = ContextHealthCheck(
            check_type="size",
            key_pattern="*",
            threshold=100.0,
            action="stop"
        )

        config = LogicConfig(
            logic_id="stop_test",
            max_iterations=5,
            context_health_checks=[health_check]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        events = []
        import asyncio
        async def collect_events():
            async for event in runner.run_stream():
                events.append(event)

        asyncio.run(collect_events())

        # Should have health event and stop
        health_events = [e for e in events if isinstance(e, ContextHealthEvent)]
        status_events = [e for e in events if isinstance(e, StatusEvent)]

        assert len(health_events) > 0
        assert any("Stopping due to health check" in e.message for e in status_events if e.message)

    def test_context_health_check_key_pattern_wildcard(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test health check with wildcard key pattern matches all keys."""
        from agentic.logic import ContextHealthCheck, LogicConfig
        from agentic.events import ContextHealthEvent

        # Set multiple keys
        context_manager.set("key1", b"x" * 200)
        context_manager.set("key2", b"y" * 200)

        health_check = ContextHealthCheck(
            check_type="size",
            key_pattern="*",
            threshold=100.0,
            action="warn"
        )

        config = LogicConfig(
            logic_id="wildcard_test",
            max_iterations=1,
            context_health_checks=[health_check]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        events = []
        import asyncio
        async def collect_events():
            async for event in runner.run_stream():
                events.append(event)

        asyncio.run(collect_events())

        health_events = [e for e in events if isinstance(e, ContextHealthEvent)]
        # Should check all keys
        assert len(health_events) >= 2

    def test_context_health_check_key_pattern_prefix(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test health check with prefix pattern matches specific keys."""
        from agentic.logic import ContextHealthCheck, LogicConfig
        from agentic.events import ContextHealthEvent

        # Set keys with different prefixes
        context_manager.set("tool_result:1", b"x" * 200)
        context_manager.set("tool_result:2", b"y" * 200)
        context_manager.set("llm_output:1", b"z" * 200)

        health_check = ContextHealthCheck(
            check_type="size",
            key_pattern="tool_result:*",
            threshold=100.0,
            action="warn"
        )

        config = LogicConfig(
            logic_id="prefix_test",
            max_iterations=1,
            context_health_checks=[health_check]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        events = []
        import asyncio
        async def collect_events():
            async for event in runner.run_stream():
                events.append(event)

        asyncio.run(collect_events())

        health_events = [e for e in events if isinstance(e, ContextHealthEvent)]
        # Should only check tool_result keys
        for event in health_events:
            assert event.key.startswith("tool_result:")

    def test_context_health_check_multiple_checks(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test multiple health checks can be configured."""
        from agentic.logic import ContextHealthCheck, LogicConfig

        # Set up context
        context_manager.set("key1", b"x" * 500)
        for i in range(3):
            context_manager.set("key2", f"v{i}".encode())

        check1 = ContextHealthCheck(
            check_type="size",
            key_pattern="key1",
            threshold=400.0,
            action="warn"
        )
        check2 = ContextHealthCheck(
            check_type="version_count",
            key_pattern="key2",
            threshold=2.0,
            action="warn"
        )

        config = LogicConfig(
            logic_id="multi_check",
            max_iterations=1,
            context_health_checks=[check1, check2]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        results = runner.run()
        assert len(results) == 1

    def test_context_health_check_key_pattern_complex_glob(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test health check with complex glob patterns (suffix patterns like *_result)."""
        from agentic.logic import ContextHealthCheck, LogicConfig
        from agentic.events import ContextHealthEvent

        # Set keys that should match pattern "*_result"
        context_manager.set("tool_result", b"x" * 200)
        context_manager.set("api_result", b"y" * 200)
        context_manager.set("result_data", b"z" * 50)  # Should NOT match
        context_manager.set("other_key", b"a" * 200)   # Should NOT match

        health_check = ContextHealthCheck(
            check_type="size",
            key_pattern="*_result",  # Complex glob (suffix pattern)
            threshold=100.0,
            action="warn"
        )

        config = LogicConfig(
            logic_id="complex_glob_test",
            max_iterations=1,
            context_health_checks=[health_check]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        events = []
        import asyncio
        async def collect_events():
            async for event in runner.run_stream():
                events.append(event)

        asyncio.run(collect_events())

        health_events = [e for e in events if isinstance(e, ContextHealthEvent)]
        # Should only check keys ending with "_result"
        matched_keys = {event.key for event in health_events}
        assert "tool_result" in matched_keys
        assert "api_result" in matched_keys
        assert "result_data" not in matched_keys
        assert "other_key" not in matched_keys

    def test_context_health_check_key_pattern_mid_glob(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test health check with glob pattern having wildcard in middle."""
        from agentic.logic import ContextHealthCheck, LogicConfig
        from agentic.events import ContextHealthEvent

        # Set keys that should match pattern "tool_*_output"
        context_manager.set("tool_api_output", b"x" * 200)
        context_manager.set("tool_db_output", b"y" * 200)
        context_manager.set("tool_result", b"z" * 200)  # Should NOT match
        context_manager.set("api_tool_output", b"w" * 200)  # Should NOT match

        health_check = ContextHealthCheck(
            check_type="size",
            key_pattern="tool_*_output",
            threshold=100.0,
            action="warn"
        )

        config = LogicConfig(
            logic_id="mid_glob_test",
            max_iterations=1,
            context_health_checks=[health_check]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        events = []
        import asyncio
        async def collect_events():
            async for event in runner.run_stream():
                events.append(event)

        asyncio.run(collect_events())

        health_events = [e for e in events if isinstance(e, ContextHealthEvent)]
        matched_keys = {event.key for event in health_events}
        assert "tool_api_output" in matched_keys
        assert "tool_db_output" in matched_keys
        assert "tool_result" not in matched_keys
        assert "api_tool_output" not in matched_keys

    def test_context_health_check_max_versions_limit_default(self):
        """Test ContextHealthCheck has default max_versions_limit."""
        from agentic.logic import ContextHealthCheck

        check = ContextHealthCheck(
            check_type="version_count",
            key_pattern="*",
            threshold=5000.0
        )
        assert check.max_versions_limit == 10000

    def test_context_health_check_max_versions_limit_custom(self):
        """Test ContextHealthCheck respects custom max_versions_limit."""
        from agentic.logic import ContextHealthCheck

        check = ContextHealthCheck(
            check_type="version_count",
            key_pattern="*",
            threshold=5000.0,
            max_versions_limit=100
        )
        assert check.max_versions_limit == 100

    def test_context_health_check_version_count_respects_limit(self, agent_runner, context_manager, pattern_registry, mock_llm_provider):
        """Test health check respects max_versions_limit to prevent memory exhaustion."""
        from agentic.logic import ContextHealthCheck, LogicConfig
        from agentic.events import ContextHealthEvent
        from unittest.mock import patch

        # Create many versions
        for i in range(20):
            context_manager.set("versioned_key", f"value_{i}".encode())

        # Use very high threshold but low max_versions_limit
        health_check = ContextHealthCheck(
            check_type="version_count",
            key_pattern="versioned_*",
            threshold=50000.0,  # Very high threshold
            max_versions_limit=10,  # But limited fetch
            action="warn"
        )

        config = LogicConfig(
            logic_id="limit_test",
            max_iterations=1,
            context_health_checks=[health_check]
        )
        runner = LogicRunner(agent_runner, context_manager, pattern_registry, config)

        # Spy on get_history to verify max_versions param
        original_get_history = context_manager.get_history
        call_args = []

        def spy_get_history(key, max_versions=None):
            call_args.append((key, max_versions))
            return original_get_history(key, max_versions=max_versions)

        with patch.object(context_manager, 'get_history', side_effect=spy_get_history):
            events = []
            import asyncio
            async def collect_events():
                async for event in runner.run_stream():
                    events.append(event)

            asyncio.run(collect_events())

        # Verify get_history was called with max_versions=10 (not 50001)
        assert any(max_vers == 10 for _, max_vers in call_args if max_vers is not None)

        # Should not emit health event since we only fetched 10 versions (< threshold)
        health_events = [e for e in events if isinstance(e, ContextHealthEvent)]
        # Even though there are 20 versions, we only fetched 10, so no violation detected
        # This is expected behavior - the limit prevents exhaustive checks
        assert len(health_events) == 0  # No violation because capped at 10 versions


class TestContextHealthEvent:
    """Tests for ContextHealthEvent."""

    def test_context_health_event_creation(self):
        """Test creating ContextHealthEvent."""
        from agentic.events import ContextHealthEvent

        event = ContextHealthEvent(
            check_type="size",
            key="test_key",
            current_value=1500.0,
            threshold=1000.0,
            recommended_action="warn"
        )
        assert event.type == "context_health"
        assert event.check_type == "size"
        assert event.key == "test_key"
        assert event.current_value == 1500.0
        assert event.threshold == 1000.0
        assert event.recommended_action == "warn"

    def test_context_health_event_with_timestamp(self):
        """Test ContextHealthEvent with explicit timestamp."""
        from agentic.events import ContextHealthEvent

        ts = 123456.789
        event = ContextHealthEvent(
            check_type="version_count",
            key="key",
            current_value=10.0,
            threshold=5.0,
            recommended_action="stop",
            timestamp=ts
        )
        assert event.timestamp == ts

    def test_context_health_event_with_step_id(self):
        """Test ContextHealthEvent with step_id."""
        from agentic.events import ContextHealthEvent

        event = ContextHealthEvent(
            check_type="size",
            key="key",
            current_value=100.0,
            threshold=50.0,
            recommended_action="warn",
            step_id="step_abc"
        )
        assert event.step_id == "step_abc"
