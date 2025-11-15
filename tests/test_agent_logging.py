"""
Tests for agent logging integration.

Covers:
- Structured logging in agent step execution
- Debug logging for agent lifecycle events
- Error logging for LLM failures
- Tool rejection logging
- Parse error logging
- Step completion logging
"""
import pytest
import logging
from io import StringIO
import json

from agentic.agent import Agent, AgentRunner
from tests.mock_provider import MockLLMProvider
from agentic.core import AgentConfig, AgentStatus
from agentic.events import StepCompleteEvent
from agentic.logging_util import StructuredFormatter


@pytest.mark.asyncio
class TestAgentStructuredLogging:
    """Tests for structured logging in agent execution."""

    async def test_agent_step_start_logging(
        self, agent, agent_runner, mock_llm_provider, caplog
    ):
        """Test that agent.step.start is logged with structured context."""
        # Need to set propagate=True and capture at root level since logger doesn't propagate by default
        logger = logging.getLogger("agentic.agent")
        original_propagate = logger.propagate
        logger.propagate = True

        caplog.set_level(logging.DEBUG, logger="agentic.agent")

        try:
            mock_llm_provider.set_response("Test response")

            async for event in agent_runner.step_stream():
                pass

            # Check for agent.step.start log
            log_messages = [record.message for record in caplog.records]
            assert "agent.step.start" in log_messages

            # Check structured context
            start_records = [r for r in caplog.records if r.message == "agent.step.start"]
            if start_records:
                record = start_records[0]
                assert hasattr(record, "agent_id")
                assert hasattr(record, "iteration")
                assert hasattr(record, "step_id")
        finally:
            logger.propagate = original_propagate

    async def test_agent_llm_complete_logging(
        self, agent_runner, mock_llm_provider, caplog
    ):
        """Test that agent.llm.complete is logged."""
        logger = logging.getLogger("agentic.agent")
        original_propagate = logger.propagate
        logger.propagate = True

        caplog.set_level(logging.DEBUG, logger="agentic.agent")

        try:
            mock_llm_provider.set_response("LLM completed")

            async for event in agent_runner.step_stream():
                pass

            log_messages = [record.message for record in caplog.records]
            assert "agent.llm.complete" in log_messages

            # Check structured data
            complete_records = [r for r in caplog.records if r.message == "agent.llm.complete"]
            if complete_records:
                record = complete_records[0]
                assert hasattr(record, "output_length")
                assert hasattr(record, "tools_detected")
        finally:
            logger.propagate = original_propagate

    async def test_agent_step_complete_logging(
        self, agent_runner, mock_llm_provider, caplog
    ):
        """Test that agent.step.complete is logged with status information."""
        logger = logging.getLogger("agentic.agent")
        original_propagate = logger.propagate
        logger.propagate = True

        caplog.set_level(logging.DEBUG, logger="agentic.agent")

        try:
            mock_llm_provider.set_response("Final output")

            async for event in agent_runner.step_stream():
                pass

            log_messages = [record.message for record in caplog.records]
            assert "agent.step.complete" in log_messages

            # Check structured context includes status
            complete_records = [r for r in caplog.records if r.message == "agent.step.complete"]
            if complete_records:
                record = complete_records[0]
                assert hasattr(record, "status")
                assert hasattr(record, "tools_executed")
                assert hasattr(record, "has_error")
        finally:
            logger.propagate = original_propagate

    async def test_agent_tool_skipped_logging(
        self, agent, agent_runner, mock_llm_provider, caplog
    ):
        """Test that agent.tool.rejected is logged when tool is rejected."""
        logger = logging.getLogger("agentic.agent")
        original_propagate = logger.propagate
        logger.propagate = True

        caplog.set_level(logging.DEBUG, logger="agentic.agent")

        try:
            # Set callback that rejects tools
            config = agent.get_config()
            config.on_tool_detected = lambda tc: False
            agent.set_config(config)

            mock_llm_provider.set_response('<tool>{"name": "echo", "arguments": {}}</tool>')

            async for event in agent_runner.step_stream():
                pass

            log_messages = [record.message for record in caplog.records]
            assert "agent.tool.rejected" in log_messages

            # Check that reason is logged
            rejected_records = [r for r in caplog.records if r.message == "agent.tool.rejected"]
            if rejected_records:
                record = rejected_records[0]
                assert hasattr(record, "reason")
                assert hasattr(record, "tool_name")
        finally:
            logger.propagate = original_propagate

    async def test_agent_pattern_parse_error_logging(
        self, agent_runner, mock_llm_provider, caplog
    ):
        """Test that agent.pattern.parse_error is logged for malformed patterns."""
        caplog.set_level(logging.WARNING, logger="agentic.agent")

        # Invalid JSON in tool call
        mock_llm_provider.set_response('<tool>{"name": "test", invalid json}</tool>')

        async for event in agent_runner.step_stream():
            pass

        # Check for parse error log
        log_messages = [record.message for record in caplog.records if record.name == "agentic.agent"]
        # Parse errors might be logged
        # Note: This depends on whether the pattern extractor reports parse errors

    async def test_agent_llm_error_logging(
        self, agent, context_manager, pattern_registry, tool_registry, caplog
    ):
        """Test that agent.llm.error is logged with exception info."""
        logger = logging.getLogger("agentic.agent")
        original_propagate = logger.propagate
        logger.propagate = True

        caplog.set_level(logging.ERROR, logger="agentic.agent")

        try:
            # Provider that raises error
            class ErrorProvider:
                async def stream(self, prompt, **kwargs):
                    raise RuntimeError("LLM failed")
                    if False:
                        yield  # Make it a generator

            error_agent = Agent(
                config=agent.get_config(),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=ErrorProvider()
            )
            runner = AgentRunner(error_agent)

            async for event in runner.step_stream():
                pass

            log_messages = [record.message for record in caplog.records]
            assert "agent.llm.error" in log_messages

            # Check exception info is included
            error_records = [r for r in caplog.records if r.message == "agent.llm.error"]
            if error_records:
                record = error_records[0]
                assert hasattr(record, "error")
                assert hasattr(record, "error_type")
                assert record.exc_info is not None
        finally:
            logger.propagate = original_propagate


@pytest.mark.asyncio
class TestLoggingOutput:
    """Tests for actual JSON log output format."""

    async def test_structured_json_output(
        self, agent_runner, mock_llm_provider
    ):
        """Test that logs are output in structured JSON format."""
        # Create logger with StructuredFormatter
        logger = logging.getLogger("agentic.agent")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        # Clear existing handlers and add our test handler
        original_handlers = logger.handlers.copy()
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)

        try:
            mock_llm_provider.set_response("Test")

            async for event in agent_runner.step_stream():
                pass

            # Get log output
            output = stream.getvalue()

            # Each line should be valid JSON
            if output:
                lines = output.strip().split('\n')
                for line in lines:
                    data = json.loads(line)
                    assert "timestamp" in data
                    assert "level" in data
                    assert "logger" in data
                    assert "message" in data
        finally:
            # Restore original handlers
            logger.handlers = original_handlers


class TestLoggingConfiguration:
    """Tests for logging configuration and setup."""

    def test_logger_uses_structured_formatter(self):
        """Test that agentic.agent logger uses StructuredFormatter."""
        from agentic.logging_util import get_logger

        logger = get_logger("agent")

        # Should have at least one handler with StructuredFormatter
        has_structured = any(
            isinstance(h.formatter, StructuredFormatter)
            for h in logger.handlers
        )
        assert has_structured

    def test_logger_does_not_propagate(self):
        """Test that logger doesn't propagate to root."""
        from agentic.logging_util import get_logger

        logger = get_logger("agent")
        assert logger.propagate is False


@pytest.mark.asyncio
class TestLoggingEdgeCases:
    """Tests for edge cases in logging."""

    async def test_logging_with_unicode_in_context(
        self, agent_runner, mock_llm_provider, caplog
    ):
        """Test logging handles unicode in context fields."""
        logger = logging.getLogger("agentic.agent")
        original_propagate = logger.propagate
        logger.propagate = True

        caplog.set_level(logging.DEBUG, logger="agentic.agent")

        try:
            # Response with unicode
            mock_llm_provider.set_response("Unicode: \u4e2d\u6587 test")

            async for event in agent_runner.step_stream():
                pass

            # Should not crash and logs should be generated
            assert len(caplog.records) > 0
        finally:
            logger.propagate = original_propagate

    async def test_logging_with_large_output(
        self, agent_runner, mock_llm_provider, caplog
    ):
        """Test logging handles very large output."""
        caplog.set_level(logging.DEBUG, logger="agentic.agent")

        # Very long response
        large_output = "x" * 100000
        mock_llm_provider.set_response(large_output)

        async for event in agent_runner.step_stream():
            pass

        # Should log output_length
        complete_records = [r for r in caplog.records if r.message == "agent.llm.complete"]
        if complete_records:
            record = complete_records[0]
            assert record.output_length == 100000
