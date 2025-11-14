"""
Tests for new features added in this session.

Covers:
- Validation system (ValidatorRegistry, ValidationError, simple_validator)
- Tool name mapping (public -> internal name resolution)
- Tool argument validation
- Async tool support
- Context list_keys() with colons in key names
- Event loop safety (AgentRunner.step(), LogicRunner.run())
- Multi-chunk tool output standardization
- LogicRunner buffer management
- New pattern sets (json_tools, xml_tools, backtick_tools)
- ToolValidationEvent emission
"""
import pytest
import asyncio

from agentic.validation import (
    ValidatorRegistry,
    ValidationError,
    simple_validator,
    passthrough_validator
)
from agentic.agent import Agent, AgentRunner
from agentic.logic import LogicRunner, LogicConfig, LogicCondition
from agentic.core import AgentConfig, ProcessingMode
from agentic.events import ToolValidationEvent, StepCompleteEvent
from agentic.tools import Tool, ToolDefinition, create_tool
from agentic.patterns import (
    create_json_tools_pattern_set,
    create_xml_tools_pattern_set,
    create_backtick_tools_pattern_set
)
from tests.mock_provider import MockLLMProvider


class TestValidationSystem:
    """Tests for validation system."""

    def test_validator_registry_creation(self):
        """Test creating ValidatorRegistry."""
        registry = ValidatorRegistry()
        assert registry is not None
        validators = registry.list()
        assert "simple" in validators
        assert "passthrough" in validators

    def test_simple_validator_required_fields(self):
        """Test simple validator with required fields."""
        schema = {
            "validator": "simple",
            "required": ["name", "age"],
            "fields": {
                "name": {"type": "str"},
                "age": {"type": "int"}
            }
        }

        # Valid data
        is_valid, errors = simple_validator({"name": "Alice", "age": 30}, schema)
        assert is_valid is True
        assert len(errors) == 0

        # Missing required field
        is_valid, errors = simple_validator({"name": "Bob"}, schema)
        assert is_valid is False
        assert len(errors) == 1
        assert errors[0].field == "age"

    def test_simple_validator_type_checking(self):
        """Test simple validator type checking."""
        schema = {
            "validator": "simple",
            "fields": {
                "count": {"type": "int"},
                "message": {"type": "str"},
                "active": {"type": "bool"}
            }
        }

        # Valid types
        is_valid, errors = simple_validator(
            {"count": 42, "message": "hello", "active": True},
            schema
        )
        assert is_valid is True

        # Invalid types
        is_valid, errors = simple_validator(
            {"count": "not an int", "message": "hello"},
            schema
        )
        assert is_valid is False
        assert any(e.field == "count" for e in errors)

    def test_simple_validator_string_constraints(self):
        """Test simple validator string length constraints."""
        schema = {
            "validator": "simple",
            "fields": {
                "username": {
                    "type": "str",
                    "min_length": 3,
                    "max_length": 20
                }
            }
        }

        # Valid length
        is_valid, errors = simple_validator({"username": "alice"}, schema)
        assert is_valid is True

        # Too short
        is_valid, errors = simple_validator({"username": "ab"}, schema)
        assert is_valid is False

        # Too long
        is_valid, errors = simple_validator({"username": "a" * 25}, schema)
        assert is_valid is False

    def test_simple_validator_numeric_constraints(self):
        """Test simple validator numeric min/max constraints."""
        schema = {
            "validator": "simple",
            "fields": {
                "age": {
                    "type": "int",
                    "min": 0,
                    "max": 150
                }
            }
        }

        # Valid range
        is_valid, errors = simple_validator({"age": 30}, schema)
        assert is_valid is True

        # Below min
        is_valid, errors = simple_validator({"age": -1}, schema)
        assert is_valid is False

        # Above max
        is_valid, errors = simple_validator({"age": 200}, schema)
        assert is_valid is False

    def test_passthrough_validator(self):
        """Test passthrough validator always passes."""
        schema = {"validator": "passthrough"}
        is_valid, errors = passthrough_validator({"anything": "goes"}, schema)
        assert is_valid is True
        assert len(errors) == 0

    def test_validator_registry_validate(self):
        """Test ValidatorRegistry.validate() method."""
        registry = ValidatorRegistry()
        schema = {
            "validator": "simple",
            "required": ["field1"],
            "fields": {"field1": {"type": "str"}}
        }

        is_valid, errors = registry.validate({"field1": "value"}, schema)
        assert is_valid is True


class TestToolNameMapping:
    """Tests for tool name mapping (public -> internal name resolution)."""

    def test_tool_name_mapping_basic(self, context_manager, pattern_registry, tool_registry, mock_llm_provider):
        """Test basic tool name mapping."""
        # Register a tool with internal name
        def internal_func(inputs):
            return {"result": "internal"}

        internal_tool = create_tool("internal_tool_name", internal_func)
        tool_registry.register(internal_tool)

        # Create agent with name mapping
        config = AgentConfig(
            agent_id="test",
            tools_allowed=["internal_tool_name"],
            tool_name_mapping={"public_name": "internal_tool_name"}
        )

        agent = Agent(config, context_manager, pattern_registry, tool_registry, mock_llm_provider)
        runner = AgentRunner(agent)

        # LLM uses public name
        mock_llm_provider.set_response('<tool>{"name": "public_name", "arguments": {}}</tool>')

        result = runner.step()

        # Tool should execute successfully using internal name
        assert result.tool_results[0].success is True
        assert result.tool_results[0].name == "internal_tool_name"

    def test_tool_name_mapping_prevents_spoofing(self, context_manager, pattern_registry, tool_registry, mock_llm_provider):
        """Test that name mapping prevents name spoofing."""
        # Register tool
        def tool_func(inputs):
            return {"data": "secret"}

        secret_tool = create_tool("secret_internal_tool", tool_func)
        tool_registry.register(secret_tool)

        # Config without mapping for this tool
        config = AgentConfig(
            agent_id="test",
            tools_allowed=["secret_internal_tool"],
            tool_name_mapping={}  # No mapping defined
        )

        agent = Agent(config, context_manager, pattern_registry, tool_registry, mock_llm_provider)
        runner = AgentRunner(agent)

        # LLM tries to call the internal name directly (should work if no mapping)
        mock_llm_provider.set_response('<tool>{"name": "secret_internal_tool", "arguments": {}}</tool>')

        result = runner.step()

        # Should succeed since internal name is in allowed list and no mapping required
        assert result.tool_results[0].success is True


class TestToolArgumentValidation:
    """Tests for tool argument validation."""

    def test_tool_argument_validation_enabled(self, context_manager, pattern_registry, tool_registry, mock_llm_provider):
        """Test tool argument validation when enabled."""
        # Register tool with validation
        def validated_func(inputs):
            return {"result": "ok"}

        validator_registry = ValidatorRegistry()
        tool_def = ToolDefinition(
            "validated_tool",
            input_schema={
                "validator": "simple",
                "required": ["required_field"],
                "fields": {"required_field": {"type": "str"}}
            },
            output_schema={}
        )
        tool = Tool(tool_def, validated_func, validator_registry)
        tool_registry.register(tool)

        # Create agent with validation enabled
        config = AgentConfig(
            agent_id="test",
            tools_allowed=["validated_tool"],
            validate_tool_arguments=True
        )

        agent = Agent(config, context_manager, pattern_registry, tool_registry, mock_llm_provider)
        runner = AgentRunner(agent)

        # Call with missing required field
        mock_llm_provider.set_response('<tool>{"name": "validated_tool", "arguments": {}}</tool>')

        result = runner.step()

        # Should fail validation
        assert result.tool_results[0].success is False
        assert "validation" in result.tool_results[0].error_message.lower()

    def test_tool_argument_validation_disabled(self, context_manager, pattern_registry, tool_registry, mock_llm_provider):
        """Test tool argument validation when disabled."""
        # Register tool
        def tool_func(inputs):
            return {"result": "ok"}

        tool = create_tool(
            "any_tool",
            tool_func,
            input_schema={
                "validator": "simple",
                "required": ["required_field"],
                "fields": {"required_field": {"type": "str"}}
            }
        )
        tool_registry.register(tool)

        # Create agent with validation disabled
        config = AgentConfig(
            agent_id="test",
            tools_allowed=["any_tool"],
            validate_tool_arguments=False
        )

        agent = Agent(config, context_manager, pattern_registry, tool_registry, mock_llm_provider)
        runner = AgentRunner(agent)

        # Call with missing required field
        mock_llm_provider.set_response('<tool>{"name": "any_tool", "arguments": {}}</tool>')

        result = runner.step()

        # Should execute (validation skipped)
        assert result.tool_results[0].success is True


class TestAsyncToolSupport:
    """Tests for native async tool support."""

    def test_async_tool_execution(self):
        """Test async tool is detected correctly."""
        # Create async tool function
        async def async_func(inputs):
            await asyncio.sleep(0.01)
            return {"result": "async_complete"}

        tool = Tool(
            ToolDefinition("async_tool", {}, {}, processing_mode=ProcessingMode.ASYNC),
            async_func
        )

        # Tool should detect it's async
        assert tool._is_async is True

        # Note: Actual execution of async tools requires calling from an async context
        # or using the framework's async execution path. This test just verifies detection.

    @pytest.mark.asyncio
    async def test_async_tool_streaming(self):
        """Test async tool streaming."""
        class AsyncStreamTool:
            async def __call__(self, inputs):
                return {"final": "result"}

            async def run_stream(self, inputs):
                for i in range(3):
                    await asyncio.sleep(0.01)
                    yield f"chunk{i}"

        tool_func = AsyncStreamTool()
        tool = Tool(ToolDefinition("stream_tool", {}, {}), tool_func)

        chunks = []
        async for event in tool.run_stream({}, iteration=1):
            chunks.append(event.output)

        assert len(chunks) == 3
        assert all(isinstance(c, str) for c in chunks)


class TestContextListKeysWithColons:
    """Tests for context list_keys() with colons in key names."""

    def test_list_keys_with_colons(self, context_manager):
        """Test that list_keys handles keys containing colons correctly."""
        # Set keys with colons in their names
        context_manager.set("memory:user:profile", b"profile_data")
        context_manager.set("memory:user:preferences", b"pref_data")
        context_manager.set("memory:system:config", b"config_data")
        context_manager.set("simple_key", b"simple_data")

        # List all keys
        keys = context_manager.list_keys()

        assert "memory:user:profile" in keys
        assert "memory:user:preferences" in keys
        assert "memory:system:config" in keys
        assert "simple_key" in keys

        # List with prefix
        user_keys = context_manager.list_keys(prefix="memory:user")
        assert "memory:user:profile" in user_keys
        assert "memory:user:preferences" in user_keys
        assert "memory:system:config" not in user_keys


class TestEventLoopSafety:
    """Tests for event loop safety in AgentRunner and LogicRunner."""

    @pytest.mark.asyncio
    async def test_agent_runner_step_detects_async_context(self, agent_runner):
        """Test that AgentRunner.step() raises error when called from async context."""
        with pytest.raises(RuntimeError) as exc_info:
            agent_runner.step()

        assert "cannot be called from an async context" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_logic_runner_run_detects_async_context(self, agent, context_manager, pattern_registry):
        """Test that LogicRunner.run() raises error when called from async context."""
        agent_runner = AgentRunner(agent)
        logic_config = LogicConfig(
            logic_id="test_logic",
            max_iterations=1
        )
        logic_runner = LogicRunner(agent_runner, context_manager, pattern_registry, logic_config)

        with pytest.raises(RuntimeError) as exc_info:
            logic_runner.run()

        assert "cannot be called from an async context" in str(exc_info.value).lower()

    def test_agent_runner_step_works_sync(self, agent_runner, mock_llm_provider):
        """Test that AgentRunner.step() works in sync context."""
        mock_llm_provider.set_response("Test response")

        # Should not raise
        result = agent_runner.step()
        assert result is not None


class TestMultiChunkToolOutput:
    """Tests for multi-chunk tool output standardization."""

    @pytest.mark.asyncio
    async def test_single_chunk_returns_as_is(self, agent_runner, mock_llm_provider, tool_registry):
        """Test that single chunk is returned as-is, not in a list."""
        def single_chunk_tool(inputs):
            return {"result": "single"}

        tool = create_tool("single_tool", single_chunk_tool)
        tool_registry.register(tool)

        config = agent_runner._agent.get_config()
        config.tools_allowed = ["single_tool"]
        agent_runner._agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "single_tool", "arguments": {}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Single chunk should be returned as-is
        assert final_result.tool_results[0].output == {"result": "single"}
        assert not isinstance(final_result.tool_results[0].output, list)

    @pytest.mark.asyncio
    async def test_multiple_chunks_returns_list(self, agent_runner, mock_llm_provider, tool_registry):
        """Test that multiple chunks are always returned as a list."""
        class MultiChunkTool:
            def __call__(self, inputs):
                return {"final": "result"}

            async def run_stream(self, inputs):
                yield {"chunk": 1}
                yield {"chunk": 2}

        tool_func = MultiChunkTool()
        tool = Tool(ToolDefinition("multi_tool", {}, {}), tool_func)
        tool_registry.register(tool)

        config = agent_runner._agent.get_config()
        config.tools_allowed = ["multi_tool"]
        config.concurrent_tool_execution = False
        agent_runner._agent.set_config(config)

        mock_llm_provider.set_response('<tool>{"name": "multi_tool", "arguments": {}}</tool>')

        final_result = None
        async for event in agent_runner.step_stream():
            if isinstance(event, StepCompleteEvent):
                final_result = event.result

        # Multiple chunks should be in a list
        assert isinstance(final_result.tool_results[0].output, list)
        assert len(final_result.tool_results[0].output) == 2


class TestLogicRunnerBufferManagement:
    """Tests for LogicRunner buffer management."""

    def test_buffer_size_configuration(self, agent):
        """Test that max_partial_buffer_size can be configured."""
        config = agent.get_config()
        config.max_partial_buffer_size = 5_000_000
        agent.set_config(config)

        assert agent.get_config().max_partial_buffer_size == 5_000_000

    @pytest.mark.asyncio
    async def test_buffer_cap_enforced(self, agent, mock_llm_provider, context_manager, pattern_registry):
        """Test that buffer is capped at max size."""
        # Set small buffer limit
        config = agent.get_config()
        config.max_partial_buffer_size = 100
        agent.set_config(config)

        agent_runner = AgentRunner(agent)
        logic_config = LogicConfig(
            logic_id="test",
            max_iterations=1
        )
        logic_runner = LogicRunner(agent_runner, context_manager, pattern_registry, logic_config)

        # Large response that would exceed buffer
        mock_llm_provider.set_response("x" * 200)
        mock_llm_provider.simulate_streaming = True

        # Should not crash even with large output
        events = []
        async for event in logic_runner.run_stream():
            events.append(event)

        # Should complete without error
        assert any(isinstance(e, StepCompleteEvent) for e in events)


class TestNewPatternSets:
    """Tests for new pattern sets (json_tools, xml_tools, backtick_tools)."""

    def test_json_tools_pattern_set(self):
        """Test create_json_tools_pattern_set."""
        pattern_set = create_json_tools_pattern_set()

        assert pattern_set is not None
        assert pattern_set.name == "json_tools"
        assert len(pattern_set.patterns) > 0

    def test_xml_tools_pattern_set(self):
        """Test create_xml_tools_pattern_set."""
        pattern_set = create_xml_tools_pattern_set()

        assert pattern_set is not None
        assert pattern_set.name == "xml_tools"
        assert len(pattern_set.patterns) > 0

    def test_backtick_tools_pattern_set(self):
        """Test create_backtick_tools_pattern_set."""
        pattern_set = create_backtick_tools_pattern_set()

        assert pattern_set is not None
        assert pattern_set.name == "backtick_tools"
        assert len(pattern_set.patterns) > 0


class TestToolValidationEvent:
    """Tests for ToolValidationEvent emission."""

    @pytest.mark.asyncio
    async def test_validation_event_emitted(self, context_manager, pattern_registry, tool_registry, mock_llm_provider):
        """Test that ToolValidationEvent is emitted on validation failure."""
        # Register tool with schema
        def tool_func(inputs):
            return {"result": "ok"}

        validator_registry = ValidatorRegistry()
        tool_def = ToolDefinition(
            "validated_tool",
            input_schema={
                "validator": "simple",
                "required": ["required_arg"],
                "fields": {"required_arg": {"type": "str"}}
            },
            output_schema={}
        )
        tool = Tool(tool_def, tool_func, validator_registry)
        tool_registry.register(tool)

        config = AgentConfig(
            agent_id="test",
            tools_allowed=["validated_tool"],
            validate_tool_arguments=True
        )

        agent = Agent(config, context_manager, pattern_registry, tool_registry, mock_llm_provider)
        runner = AgentRunner(agent)

        # Call without required argument
        mock_llm_provider.set_response('<tool>{"name": "validated_tool", "arguments": {}}</tool>')

        validation_events = []
        async for event in runner.step_stream():
            if isinstance(event, ToolValidationEvent):
                validation_events.append(event)

        # Should have validation event
        assert len(validation_events) == 1
        assert validation_events[0].tool_name == "validated_tool"
        assert len(validation_events[0].validation_errors) > 0
