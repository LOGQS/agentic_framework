"""
Tests for multi-prompt feature.

Covers:
- PromptObject dataclass
- create_message_prompt_builder() function
- AgentRunner._build_prompt() with custom builders
- AgentRunner._build_prompt() default behavior with dict format
- LLMProvider with PromptType (str and PromptObject)
"""
import pytest
from agentic.core import (
    PromptObject,
    AgentConfig,
    create_message_prompt_builder
)
from agentic.agent import AgentRunner
from tests.mock_provider import MockLLMProvider


class TestPromptObject:
    """Tests for PromptObject dataclass."""

    def test_prompt_object_creation_empty(self):
        """Test creating empty PromptObject with defaults."""
        prompt = PromptObject()
        assert prompt.system is None
        assert prompt.messages == []
        assert prompt.metadata == {}

    def test_prompt_object_creation_with_system(self):
        """Test creating PromptObject with system instruction."""
        prompt = PromptObject(system="You are a helpful assistant.")
        assert prompt.system == "You are a helpful assistant."
        assert prompt.messages == []
        assert prompt.metadata == {}

    def test_prompt_object_creation_with_messages(self):
        """Test creating PromptObject with messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        prompt = PromptObject(messages=messages)
        assert prompt.system is None
        assert prompt.messages == messages
        assert prompt.metadata == {}

    def test_prompt_object_creation_full(self):
        """Test creating PromptObject with all fields."""
        messages = [{"role": "user", "content": "Test"}]
        metadata = {"temperature": 0.7, "model": "test-model"}
        prompt = PromptObject(
            system="System instruction",
            messages=messages,
            metadata=metadata
        )
        assert prompt.system == "System instruction"
        assert prompt.messages == messages
        assert prompt.metadata == metadata

    def test_prompt_object_messages_mutable(self):
        """Test that PromptObject messages list is mutable."""
        prompt = PromptObject()
        prompt.messages.append({"role": "user", "content": "Hello"})
        assert len(prompt.messages) == 1
        assert prompt.messages[0]["content"] == "Hello"

    def test_prompt_object_metadata_mutable(self):
        """Test that PromptObject metadata dict is mutable."""
        prompt = PromptObject()
        prompt.metadata["key"] = "value"
        assert prompt.metadata["key"] == "value"


class TestCreateMessagePromptBuilder:
    """Tests for create_message_prompt_builder() function."""

    def test_builder_returns_callable(self):
        """Test that create_message_prompt_builder returns a callable."""
        builder = create_message_prompt_builder()
        assert callable(builder)

    def test_builder_empty_input_mapping(self, context_manager):
        """Test builder with empty input_mapping."""
        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[]
        )
        result = builder(context_manager, config, None)
        assert isinstance(result, PromptObject)
        assert result.system is None
        assert result.messages == []

    def test_builder_with_system_role(self, context_manager):
        """Test builder routes role='system' to system field."""
        context_manager.set("system_instruction", b"You are helpful.")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "system_instruction", "role": "system", "order": 0}
            ]
        )
        result = builder(context_manager, config, None)
        assert result.system == "You are helpful."
        assert result.messages == []

    def test_builder_with_user_role(self, context_manager):
        """Test builder adds role='user' to messages list."""
        context_manager.set("user_prompt", b"Hello agent")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "user_prompt", "role": "user", "order": 0}
            ]
        )
        result = builder(context_manager, config, None)
        assert result.system is None
        assert len(result.messages) == 1
        assert result.messages[0] == {"role": "user", "content": "Hello agent"}

    def test_builder_with_assistant_role(self, context_manager):
        """Test builder adds role='assistant' to messages list."""
        context_manager.set("assistant_msg", b"I'm here to help.")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "assistant_msg", "role": "assistant", "order": 0}
            ]
        )
        result = builder(context_manager, config, None)
        assert result.system is None
        assert len(result.messages) == 1
        assert result.messages[0] == {"role": "assistant", "content": "I'm here to help."}

    def test_builder_multiple_system_entries(self, context_manager):
        """Test builder concatenates multiple system entries."""
        context_manager.set("sys1", b"First instruction.")
        context_manager.set("sys2", b"Second instruction.")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "sys1", "role": "system", "order": 0},
                {"context_key": "sys2", "role": "system", "order": 1}
            ]
        )
        result = builder(context_manager, config, None)
        assert result.system == "First instruction.\n\nSecond instruction."
        assert result.messages == []

    def test_builder_mixed_roles(self, context_manager):
        """Test builder with mixed system and message roles."""
        context_manager.set("system", b"Be helpful.")
        context_manager.set("user_msg", b"Hello")
        context_manager.set("assistant_msg", b"Hi there!")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "system", "role": "system", "order": 0},
                {"context_key": "user_msg", "role": "user", "order": 1},
                {"context_key": "assistant_msg", "role": "assistant", "order": 2}
            ]
        )
        result = builder(context_manager, config, None)
        assert result.system == "Be helpful."
        assert len(result.messages) == 2
        assert result.messages[0] == {"role": "user", "content": "Hello"}
        assert result.messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_builder_respects_order(self, context_manager):
        """Test builder respects 'order' field for sorting."""
        context_manager.set("msg1", b"First")
        context_manager.set("msg2", b"Second")
        context_manager.set("msg3", b"Third")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "msg3", "role": "user", "order": 2},
                {"context_key": "msg1", "role": "user", "order": 0},
                {"context_key": "msg2", "role": "user", "order": 1}
            ]
        )
        result = builder(context_manager, config, None)
        assert len(result.messages) == 3
        assert result.messages[0]["content"] == "First"
        assert result.messages[1]["content"] == "Second"
        assert result.messages[2]["content"] == "Third"

    def test_builder_with_literal_prefix(self, context_manager):
        """Test builder handles 'literal:' prefix for static content."""
        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "literal:Static system instruction", "role": "system", "order": 0},
                {"context_key": "literal:Static user message", "role": "user", "order": 1}
            ]
        )
        result = builder(context_manager, config, None)
        assert result.system == "Static system instruction"
        assert len(result.messages) == 1
        assert result.messages[0] == {"role": "user", "content": "Static user message"}

    def test_builder_with_user_input(self, context_manager):
        """Test builder appends user_input as final user message."""
        context_manager.set("system", b"Be helpful.")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "system", "role": "system", "order": 0}
            ]
        )
        result = builder(context_manager, config, "User input text")
        assert result.system == "Be helpful."
        assert len(result.messages) == 1
        assert result.messages[0] == {"role": "user", "content": "User input text"}

    def test_builder_skips_missing_context_keys(self, context_manager):
        """Test builder skips entries with missing context keys."""
        context_manager.set("exists", b"This exists")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "exists", "role": "user", "order": 0},
                {"context_key": "missing", "role": "user", "order": 1}
            ]
        )
        result = builder(context_manager, config, None)
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "This exists"

    def test_builder_handles_decode_errors(self, context_manager):
        """Test builder skips entries that can't be decoded."""
        # Set invalid UTF-8 bytes
        context_manager.set("invalid", b'\x80\x81\x82')
        context_manager.set("valid", b"Valid text")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "invalid", "role": "user", "order": 0},
                {"context_key": "valid", "role": "user", "order": 1}
            ]
        )
        result = builder(context_manager, config, None)
        # Should only include the valid entry
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "Valid text"

    def test_builder_default_role_is_user(self, context_manager):
        """Test builder defaults to 'user' role if not specified."""
        context_manager.set("msg", b"Test message")

        builder = create_message_prompt_builder()
        config = AgentConfig(
            agent_id="test",
            input_mapping=[
                {"context_key": "msg", "order": 0}  # No role specified
            ]
        )
        result = builder(context_manager, config, None)
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"
        assert result.messages[0]["content"] == "Test message"


class TestBuildPromptWithCustomBuilder:
    """Tests for AgentRunner._build_prompt() with custom prompt_builder."""

    def test_custom_builder_is_used(self, agent, context_manager):
        """Test that custom prompt_builder is called when configured."""
        custom_called = []

        def custom_builder(ctx, cfg, user_input):
            custom_called.append(True)
            return PromptObject(
                system="Custom system",
                messages=[{"role": "user", "content": "Custom message"}]
            )

        config = agent.get_config()
        config.prompt_builder = custom_builder
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt(None)

        assert len(custom_called) == 1
        assert isinstance(prompt, PromptObject)
        assert prompt.system == "Custom system"
        assert prompt.messages[0]["content"] == "Custom message"

    def test_custom_builder_receives_context(self, agent, context_manager):
        """Test that custom builder receives ContextManager."""
        received_context = []

        def custom_builder(ctx, cfg, user_input):
            received_context.append(ctx)
            return PromptObject()

        config = agent.get_config()
        config.prompt_builder = custom_builder
        agent.set_config(config)

        runner = AgentRunner(agent)
        runner._build_prompt(None)

        assert len(received_context) == 1
        assert received_context[0] is context_manager

    def test_custom_builder_receives_config(self, agent, context_manager):
        """Test that custom builder receives AgentConfig."""
        received_config = []

        def custom_builder(ctx, cfg, user_input):
            received_config.append(cfg)
            return PromptObject()

        config = agent.get_config()
        config.prompt_builder = custom_builder
        agent.set_config(config)

        runner = AgentRunner(agent)
        runner._build_prompt(None)

        assert len(received_config) == 1
        assert received_config[0].agent_id == "test_agent"

    def test_custom_builder_receives_user_input(self, agent, context_manager):
        """Test that custom builder receives user_input parameter."""
        received_input = []

        def custom_builder(ctx, cfg, user_input):
            received_input.append(user_input)
            return PromptObject()

        config = agent.get_config()
        config.prompt_builder = custom_builder
        agent.set_config(config)

        runner = AgentRunner(agent)
        runner._build_prompt("Test user input")

        assert len(received_input) == 1
        assert received_input[0] == "Test user input"

    def test_custom_builder_can_return_string(self, agent, context_manager):
        """Test that custom builder can return a string instead of PromptObject."""
        def custom_builder(ctx, cfg, user_input):
            return "Simple string prompt"

        config = agent.get_config()
        config.prompt_builder = custom_builder
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt(None)

        assert prompt == "Simple string prompt"

    def test_message_prompt_builder_integration(self, agent, context_manager):
        """Test using create_message_prompt_builder() in agent config."""
        context_manager.set("system", b"You are helpful.")
        context_manager.set("history", b"Previous conversation")

        config = agent.get_config()
        config.prompt_builder = create_message_prompt_builder()
        config.input_mapping = [
            {"context_key": "system", "role": "system", "order": 0},
            {"context_key": "history", "role": "user", "order": 1}
        ]
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt("New message")

        assert isinstance(prompt, PromptObject)
        assert prompt.system == "You are helpful."
        assert len(prompt.messages) == 2
        assert prompt.messages[0] == {"role": "user", "content": "Previous conversation"}
        assert prompt.messages[1] == {"role": "user", "content": "New message"}


class TestBuildPromptDefaultBehavior:
    """Tests for AgentRunner._build_prompt() default behavior with dict format."""

    def test_default_builder_concatenates_context_keys(self, agent, context_manager):
        """Test default builder concatenates entries from input_mapping."""
        context_manager.set("part1", b"First part")
        context_manager.set("part2", b"Second part")

        config = agent.get_config()
        config.input_mapping = [
            {"context_key": "part1", "order": 0},
            {"context_key": "part2", "order": 1}
        ]
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt(None)

        assert isinstance(prompt, str)
        assert "First part" in prompt
        assert "Second part" in prompt

    def test_default_builder_with_literal_prefix(self, agent, context_manager):
        """Test default builder handles 'literal:' prefix."""
        config = agent.get_config()
        config.input_mapping = [
            {"context_key": "literal:Static text here", "order": 0}
        ]
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt(None)

        assert "Static text here" in prompt

    def test_default_builder_appends_user_input(self, agent, context_manager):
        """Test default builder appends user_input to prompt."""
        context_manager.set("context", b"Context text")

        config = agent.get_config()
        config.input_mapping = [
            {"context_key": "context", "order": 0}
        ]
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt("User input")

        assert "Context text" in prompt
        assert "User input" in prompt

    def test_default_builder_skips_missing_keys(self, agent, context_manager):
        """Test default builder skips missing context keys."""
        context_manager.set("exists", b"Exists")

        config = agent.get_config()
        config.input_mapping = [
            {"context_key": "exists", "order": 0},
            {"context_key": "missing", "order": 1}
        ]
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt(None)

        assert "Exists" in prompt
        assert prompt.count("\n\n") >= 0  # Should still work

    def test_default_builder_handles_decode_errors(self, agent, context_manager):
        """Test default builder skips entries with decode errors."""
        context_manager.set("invalid", b'\x80\x81\x82')
        context_manager.set("valid", b"Valid")

        config = agent.get_config()
        config.input_mapping = [
            {"context_key": "invalid", "order": 0},
            {"context_key": "valid", "order": 1}
        ]
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt(None)

        assert "Valid" in prompt

    def test_default_builder_empty_input_mapping(self, agent, context_manager):
        """Test default builder with empty input_mapping."""
        config = agent.get_config()
        config.input_mapping = []
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt("Just user input")

        assert prompt == "Just user input"

    def test_default_builder_no_user_input(self, agent, context_manager):
        """Test default builder with no user input."""
        context_manager.set("context", b"Context only")

        config = agent.get_config()
        config.input_mapping = [
            {"context_key": "context", "order": 0}
        ]
        agent.set_config(config)

        runner = AgentRunner(agent)
        prompt = runner._build_prompt(None)

        assert prompt == "Context only"


class TestLLMProviderWithPromptType:
    """Tests for LLMProvider accepting both str and PromptObject."""

    def test_mock_provider_accepts_string(self):
        """Test MockLLMProvider.generate() accepts string prompt."""
        provider = MockLLMProvider(response="Test response")
        output = provider.generate("String prompt")
        assert output == "Test response"

    def test_mock_provider_accepts_prompt_object(self):
        """Test MockLLMProvider.generate() accepts PromptObject."""
        provider = MockLLMProvider(response="Test response")
        prompt_obj = PromptObject(
            system="System instruction",
            messages=[{"role": "user", "content": "Hello"}]
        )
        output = provider.generate(prompt_obj)
        assert output == "Test response"

    @pytest.mark.asyncio
    async def test_mock_provider_stream_accepts_string(self):
        """Test MockLLMProvider.stream() accepts string prompt."""
        provider = MockLLMProvider(response="Test response", simulate_streaming=False)
        chunks = []
        async for chunk in provider.stream("String prompt"):
            chunks.append(chunk)
        assert "".join(chunks) == "Test response"

    @pytest.mark.asyncio
    async def test_mock_provider_stream_accepts_prompt_object(self):
        """Test MockLLMProvider.stream() accepts PromptObject."""
        provider = MockLLMProvider(response="Test response", simulate_streaming=False)
        prompt_obj = PromptObject(
            system="System",
            messages=[{"role": "user", "content": "Hi"}]
        )
        chunks = []
        async for chunk in provider.stream(prompt_obj):
            chunks.append(chunk)
        assert "".join(chunks) == "Test response"

    def test_agent_runner_with_string_prompt(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test AgentRunner.step() works with default string prompt."""
        context_manager.set("system", b"Test system")
        mock_llm_provider.set_response("Response text")

        result = agent_runner.step()

        assert result.raw_output == "Response text"

    def test_agent_runner_with_prompt_object(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test AgentRunner.step() works with PromptObject from custom builder."""
        context_manager.set("system", b"Test system")

        config = agent.get_config()
        config.prompt_builder = create_message_prompt_builder()
        config.input_mapping = [
            {"context_key": "system", "role": "system", "order": 0}
        ]
        agent.set_config(config)

        mock_llm_provider.set_response("Response text")

        result = agent_runner.step("User message")

        assert result.raw_output == "Response text"

    @pytest.mark.asyncio
    async def test_agent_runner_stream_with_prompt_object(self, agent, agent_runner, mock_llm_provider, context_manager):
        """Test AgentRunner.step_stream() works with PromptObject."""
        context_manager.set("system", b"Test system")

        config = agent.get_config()
        config.prompt_builder = create_message_prompt_builder()
        config.input_mapping = [
            {"context_key": "system", "role": "system", "order": 0}
        ]
        agent.set_config(config)

        mock_llm_provider.set_response("Response text")

        final_result = None
        async for event in agent_runner.step_stream("User message"):
            if hasattr(event, 'result'):
                final_result = event.result

        assert final_result is not None
        assert final_result.raw_output == "Response text"


class TestPromptTypeIntegration:
    """Integration tests for PromptType across the system."""

    def test_full_flow_with_message_builder(self, agent, context_manager, mock_llm_provider):
        """Test complete flow: message builder -> PromptObject -> LLM -> result."""
        # Setup context
        context_manager.set("system_prompt", b"You are a test assistant.")
        context_manager.set("conversation", b"User: Hello")

        # Configure agent with message prompt builder
        config = agent.get_config()
        config.prompt_builder = create_message_prompt_builder()
        config.input_mapping = [
            {"context_key": "system_prompt", "role": "system", "order": 0},
            {"context_key": "conversation", "role": "user", "order": 1}
        ]
        agent.set_config(config)

        # Set mock response
        mock_llm_provider.set_response("Assistant: I'm here to help!")

        # Execute
        runner = AgentRunner(agent)
        result = runner.step("What can you do?")

        # Verify
        assert result.raw_output == "Assistant: I'm here to help!"

    def test_switching_between_builders(self, agent, context_manager, mock_llm_provider):
        """Test switching between default and custom builders."""
        context_manager.set("context", b"Context text")

        runner = AgentRunner(agent)

        # First with default builder (returns string)
        config = agent.get_config()
        config.input_mapping = [{"context_key": "context", "order": 0}]
        config.prompt_builder = None
        agent.set_config(config)

        mock_llm_provider.set_response("Response 1")
        result1 = runner.step()
        assert result1.raw_output == "Response 1"

        # Then with message builder (returns PromptObject)
        config.prompt_builder = create_message_prompt_builder()
        config.input_mapping = [{"context_key": "context", "role": "user", "order": 0}]
        agent.set_config(config)

        mock_llm_provider.set_response("Response 2")
        result2 = runner.step()
        assert result2.raw_output == "Response 2"

    def test_literal_content_in_message_builder(self, agent, context_manager, mock_llm_provider):
        """Test literal content works with message prompt builder."""
        config = agent.get_config()
        config.prompt_builder = create_message_prompt_builder()
        config.input_mapping = [
            {"context_key": "literal:You are a helpful AI assistant.", "role": "system", "order": 0},
            {"context_key": "literal:Remember to be concise.", "role": "system", "order": 1}
        ]
        agent.set_config(config)

        mock_llm_provider.set_response("Understood.")

        runner = AgentRunner(agent)
        result = runner.step("Test")

        assert result.raw_output == "Understood."
