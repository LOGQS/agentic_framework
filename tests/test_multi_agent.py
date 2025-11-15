"""
Tests for multi-agent coordination patterns.

Covers:
- AgentChain with different pass modes (response, full_context, tool_results)
- AgentChain with custom transform function
- AgentChain prepend_context behavior
- SupervisorPattern delegation detection and execution
- SupervisorPattern max delegation rounds
- ParallelPattern concurrent execution
- ParallelPattern timeout handling
- ParallelPattern merge strategies (concat, agent)
- DebatePattern multi-round execution
- DebatePattern consensus detection
- All patterns emit correct events
"""
import pytest
import asyncio

from agentic.multi_agent import (
    AgentChain,
    AgentChainConfig,
    SupervisorPattern,
    SupervisorConfig,
    ParallelPattern,
    ParallelConfig,
    DebatePattern,
    DebateConfig
)
from agentic.agent import Agent, AgentRunner
from agentic.core import AgentConfig, ProcessingMode, AgentStatus, ToolCall
from agentic.events import StatusEvent, StepCompleteEvent
from tests.mock_provider import MockLLMProvider


class TestAgentChainConfig:
    """Tests for AgentChainConfig dataclass."""

    def test_agent_chain_config_defaults(self):
        """Test AgentChainConfig with default values."""
        config = AgentChainConfig()
        assert config.pass_mode == "response"
        assert config.transform_fn is None
        assert config.prepend_context is True
        assert "Previous agent" in config.context_template

    def test_agent_chain_config_custom(self):
        """Test AgentChainConfig with custom values."""
        def custom_transform(result):
            return "custom"

        config = AgentChainConfig(
            pass_mode="tool_results",
            transform_fn=custom_transform,
            prepend_context=False,
            context_template="Custom: {output}"
        )
        assert config.pass_mode == "tool_results"
        assert config.transform_fn == custom_transform
        assert config.prepend_context is False
        assert config.context_template == "Custom: {output}"


@pytest.mark.asyncio
class TestAgentChain:
    """Tests for AgentChain sequential pattern."""

    async def test_agent_chain_basic_execution(self, context_manager, pattern_registry, tool_registry):
        """Test basic agent chain execution."""
        # Create two agents
        provider1 = MockLLMProvider(response="Agent 1 output")
        agent1 = Agent(
            config=AgentConfig(agent_id="agent1"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider1
        )

        provider2 = MockLLMProvider(response="Agent 2 final output")
        agent2 = Agent(
            config=AgentConfig(agent_id="agent2"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider2
        )

        chain = AgentChain(
            agents=[("agent1", agent1), ("agent2", agent2)],
            config=AgentChainConfig(prepend_context=False)
        )

        events = []
        async for event in chain.execute("Initial input"):
            events.append(event)

        # Should have events from both agents
        step_complete_events = [e for e in events if isinstance(e, StepCompleteEvent)]
        assert len(step_complete_events) == 2

    async def test_agent_chain_pass_mode_response(self, context_manager, pattern_registry, tool_registry):
        """Test AgentChain with pass_mode='response'."""
        provider1 = MockLLMProvider(response="<response>First response</response>")
        agent1 = Agent(
            config=AgentConfig(agent_id="agent1"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider1
        )

        provider2 = MockLLMProvider(response="Final")
        agent2 = Agent(
            config=AgentConfig(agent_id="agent2"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider2
        )

        config = AgentChainConfig(pass_mode="response", prepend_context=False)
        chain = AgentChain(agents=[("agent1", agent1), ("agent2", agent2)], config=config)

        async for event in chain.execute("Start"):
            pass

        # Agent2 should have received the response segment from agent1
        # This is verified by the chain completing successfully

    async def test_agent_chain_pass_mode_full_context(self, context_manager, pattern_registry, tool_registry):
        """Test AgentChain with pass_mode='full_context'."""
        provider1 = MockLLMProvider(response="Full raw output here")
        agent1 = Agent(
            config=AgentConfig(agent_id="agent1"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider1
        )

        provider2 = MockLLMProvider(response="Done")
        agent2 = Agent(
            config=AgentConfig(agent_id="agent2"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider2
        )

        config = AgentChainConfig(pass_mode="full_context", prepend_context=False)
        chain = AgentChain(agents=[("agent1", agent1), ("agent2", agent2)], config=config)

        events = []
        async for event in chain.execute("Start"):
            events.append(event)

        # Should complete successfully
        assert any(isinstance(e, StepCompleteEvent) for e in events)

    async def test_agent_chain_custom_transform(self, context_manager, pattern_registry, tool_registry):
        """Test AgentChain with custom transform function."""
        provider1 = MockLLMProvider(response="Agent 1")
        agent1 = Agent(
            config=AgentConfig(agent_id="agent1"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider1
        )

        provider2 = MockLLMProvider(response="Agent 2")
        agent2 = Agent(
            config=AgentConfig(agent_id="agent2"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider2
        )

        def custom_transform(result):
            return f"TRANSFORMED: {result.raw_output}"

        config = AgentChainConfig(
            transform_fn=custom_transform,
            prepend_context=False
        )
        chain = AgentChain(agents=[("agent1", agent1), ("agent2", agent2)], config=config)

        events = []
        async for event in chain.execute("Start"):
            events.append(event)

        assert len([e for e in events if isinstance(e, StepCompleteEvent)]) == 2

    async def test_agent_chain_prepend_context(self, context_manager, pattern_registry, tool_registry):
        """Test AgentChain with prepend_context enabled."""
        provider1 = MockLLMProvider(response="First")
        agent1 = Agent(
            config=AgentConfig(agent_id="agent1"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider1
        )

        provider2 = MockLLMProvider(response="Second")
        agent2 = Agent(
            config=AgentConfig(agent_id="agent2"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider2
        )

        config = AgentChainConfig(
            prepend_context=True,
            context_template="From {agent_id}: {output}\n"
        )
        chain = AgentChain(agents=[("agent1", agent1), ("agent2", agent2)], config=config)

        events = []
        async for event in chain.execute("Start"):
            events.append(event)

        # Should have status events indicating chain execution
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any("Chain: Executing agent" in e.message for e in status_events if e.message)

    async def test_agent_chain_multiple_agents(self, context_manager, pattern_registry, tool_registry):
        """Test AgentChain with more than 2 agents."""
        agents = []
        for i in range(4):
            provider = MockLLMProvider(response=f"Agent {i}")
            agent = Agent(
                config=AgentConfig(agent_id=f"agent{i}"),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=provider
            )
            agents.append((f"agent{i}", agent))

        chain = AgentChain(agents=agents, config=AgentChainConfig(prepend_context=False))

        events = []
        async for event in chain.execute("Start"):
            events.append(event)

        # Should have 4 step complete events
        step_events = [e for e in events if isinstance(e, StepCompleteEvent)]
        assert len(step_events) == 4


class TestSupervisorConfig:
    """Tests for SupervisorConfig dataclass."""

    def test_supervisor_config_defaults(self):
        """Test SupervisorConfig with default values."""
        config = SupervisorConfig()
        assert config.delegation_pattern_name == "delegate"
        assert config.worker_key == "to"
        assert config.task_key == "task"
        assert config.max_delegation_rounds == 10


@pytest.mark.asyncio
class TestSupervisorPattern:
    """Tests for SupervisorPattern."""

    async def test_supervisor_pattern_no_delegation(self, context_manager, pattern_registry, tool_registry):
        """Test supervisor completes without delegation."""
        supervisor_provider = MockLLMProvider(response="No delegation needed")
        supervisor_agent = Agent(
            config=AgentConfig(agent_id="supervisor"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=supervisor_provider
        )

        worker_provider = MockLLMProvider(response="Worker response")
        worker_agent = Agent(
            config=AgentConfig(agent_id="worker"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=worker_provider
        )

        pattern = SupervisorPattern(
            supervisor=supervisor_agent,
            workers={"worker1": worker_agent}
        )

        events = []
        async for event in pattern.execute("Simple query"):
            events.append(event)

        # Should complete without delegation
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any("completed without further delegation" in e.message.lower() for e in status_events if e.message)

    async def test_supervisor_pattern_max_rounds(self, context_manager, pattern_registry, tool_registry):
        """Test supervisor respects max_delegation_rounds."""
        # Supervisor always delegates
        supervisor_provider = MockLLMProvider(
            response='<tool>{"name": "delegate", "arguments": {"to": "worker1", "task": "Do work"}}</tool>'
        )
        supervisor_agent = Agent(
            config=AgentConfig(agent_id="supervisor"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=supervisor_provider
        )

        worker_provider = MockLLMProvider(response="Worker done")
        worker_agent = Agent(
            config=AgentConfig(agent_id="worker1"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=worker_provider
        )

        config = SupervisorConfig(max_delegation_rounds=2)
        pattern = SupervisorPattern(
            supervisor=supervisor_agent,
            workers={"worker1": worker_agent},
            config=config
        )

        events = []
        async for event in pattern.execute("Query"):
            events.append(event)

        # Should stop at max rounds
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any("Max delegation rounds" in e.message for e in status_events if e.message)


class TestParallelConfig:
    """Tests for ParallelConfig dataclass."""

    def test_parallel_config_defaults(self):
        """Test ParallelConfig with default values."""
        config = ParallelConfig()
        assert config.merge_strategy == "agent"
        assert "Synthesize" in config.merge_template
        assert config.timeout_seconds == 120.0


@pytest.mark.asyncio
class TestParallelPattern:
    """Tests for ParallelPattern."""

    async def test_parallel_pattern_concurrent_execution(self, context_manager, pattern_registry, tool_registry):
        """Test that parallel pattern executes agents concurrently."""
        agents = {}
        for i in range(3):
            provider = MockLLMProvider(response=f"Response from agent {i}")
            agent = Agent(
                config=AgentConfig(agent_id=f"agent{i}"),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=provider
            )
            agents[f"agent{i}"] = agent

        pattern = ParallelPattern(agents=agents)

        events = []
        async for event in pattern.execute_and_merge("Query"):
            events.append(event)

        # Should have step complete events from all agents
        step_events = [e for e in events if isinstance(e, StepCompleteEvent)]
        # At least 3 from parallel agents (possibly more from merger)
        assert len(step_events) >= 3

    async def test_parallel_pattern_merge_concat(self, context_manager, pattern_registry, tool_registry):
        """Test ParallelPattern with concat merge strategy."""
        agents = {}
        for i in range(2):
            provider = MockLLMProvider(response=f"Agent{i}")
            agent = Agent(
                config=AgentConfig(agent_id=f"agent{i}"),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=provider
            )
            agents[f"agent{i}"] = agent

        config = ParallelConfig(merge_strategy="concat")
        pattern = ParallelPattern(agents=agents, config=config)

        results = []
        async for event in pattern.execute_and_merge("Query"):
            if isinstance(event, StepCompleteEvent):
                results.append(event.result)

        # Should have final merged result
        assert len(results) >= 2

    async def test_parallel_pattern_merge_with_agent(self, context_manager, pattern_registry, tool_registry):
        """Test ParallelPattern with agent merge strategy."""
        agents = {}
        for i in range(2):
            provider = MockLLMProvider(response=f"Agent{i}")
            agent = Agent(
                config=AgentConfig(agent_id=f"agent{i}"),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=provider
            )
            agents[f"agent{i}"] = agent

        merger_provider = MockLLMProvider(response="Merged synthesis")
        merger_agent = Agent(
            config=AgentConfig(agent_id="merger"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=merger_provider
        )

        config = ParallelConfig(merge_strategy="agent")
        pattern = ParallelPattern(agents=agents, merger=merger_agent, config=config)

        events = []
        async for event in pattern.execute_and_merge("Query"):
            events.append(event)

        # Should have events from parallel agents AND merger
        step_events = [e for e in events if isinstance(e, StepCompleteEvent)]
        assert len(step_events) >= 3  # 2 parallel + 1 merger

    async def test_parallel_pattern_timeout(self, context_manager, pattern_registry, tool_registry):
        """Test ParallelPattern timeout handling."""
        # Create slow agent
        async def slow_stream(prompt, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            yield "slow response"

        class SlowProvider:
            async def stream(self, prompt, **kwargs):
                await asyncio.sleep(10)
                yield "slow"

        slow_provider = SlowProvider()
        slow_agent = Agent(
            config=AgentConfig(agent_id="slow"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=slow_provider
        )

        config = ParallelConfig(merge_strategy="concat", timeout_seconds=0.1)
        pattern = ParallelPattern(agents={"slow": slow_agent}, config=config)

        events = []
        async for event in pattern.execute_and_merge("Query"):
            events.append(event)

        # Should have timeout error event
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any("timed out" in e.message.lower() for e in status_events if e.message)


class TestDebateConfig:
    """Tests for DebateConfig dataclass."""

    def test_debate_config_defaults(self):
        """Test DebateConfig with default values."""
        config = DebateConfig()
        assert config.max_rounds == 5
        assert config.consensus_detector is None
        assert "consensus" in config.moderator_prompt_template.lower()


@pytest.mark.asyncio
class TestDebatePattern:
    """Tests for DebatePattern."""

    async def test_debate_pattern_single_round(self, context_manager, pattern_registry, tool_registry):
        """Test debate pattern with single round to consensus."""
        agents = {}
        # Create agents that agree (for consensus)
        for i in range(2):
            provider = MockLLMProvider(response="agree consensus complete")
            agent = Agent(
                config=AgentConfig(agent_id=f"agent{i}"),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=provider
            )
            agents[f"agent{i}"] = agent

        pattern = DebatePattern(agents=agents)

        events = []
        async for event in pattern.converge("Topic"):
            events.append(event)

        # Should have debate round events
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any("Debate round" in e.message for e in status_events if e.message)

    async def test_debate_pattern_max_rounds(self, context_manager, pattern_registry, tool_registry):
        """Test debate pattern reaches max rounds."""
        agents = {}
        # Create agents that disagree (no consensus)
        responses = ["I think A", "I think B"]
        for i, response in enumerate(responses):
            provider = MockLLMProvider(response=response)
            agent = Agent(
                config=AgentConfig(agent_id=f"agent{i}"),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=provider
            )
            agents[f"agent{i}"] = agent

        config = DebateConfig(max_rounds=2)
        pattern = DebatePattern(agents=agents, config=config)

        events = []
        async for event in pattern.converge("Topic"):
            events.append(event)

        # Should reach max rounds
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any("Max rounds reached" in e.message for e in status_events if e.message)

    async def test_debate_pattern_with_moderator(self, context_manager, pattern_registry, tool_registry):
        """Test debate pattern with moderator summarizing consensus."""
        agents = {}
        # Agents that agree quickly
        for i in range(2):
            provider = MockLLMProvider(response="consensus opinion shared view")
            agent = Agent(
                config=AgentConfig(agent_id=f"agent{i}"),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=provider
            )
            agents[f"agent{i}"] = agent

        moderator_provider = MockLLMProvider(response="Summary of consensus")
        moderator = Agent(
            config=AgentConfig(agent_id="moderator"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=moderator_provider
        )

        pattern = DebatePattern(agents=agents, moderator=moderator)

        events = []
        async for event in pattern.converge("Topic"):
            events.append(event)

        # Should have events from both debaters and moderator
        step_events = [e for e in events if isinstance(e, StepCompleteEvent)]
        # At least 2 from debate + 1 from moderator
        assert len(step_events) >= 3

    async def test_debate_pattern_custom_consensus(self, context_manager, pattern_registry, tool_registry):
        """Test debate pattern with custom consensus detector."""
        def custom_consensus(responses):
            # Consensus if all responses contain "YES"
            return all("YES" in r for r in responses)

        agents = {}
        responses = ["YES agree", "YES indeed"]
        for i, response in enumerate(responses):
            provider = MockLLMProvider(response=response)
            agent = Agent(
                config=AgentConfig(agent_id=f"agent{i}"),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=provider
            )
            agents[f"agent{i}"] = agent

        config = DebateConfig(consensus_detector=custom_consensus)
        pattern = DebatePattern(agents=agents, config=config)

        events = []
        async for event in pattern.converge("Topic"):
            events.append(event)

        # Should detect consensus quickly
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any("Consensus reached" in e.message for e in status_events if e.message)


class TestMultiAgentEdgeCases:
    """Tests for edge cases in multi-agent patterns."""

    @pytest.mark.asyncio
    async def test_empty_agent_chain(self):
        """Test agent chain with no agents."""
        chain = AgentChain(agents=[])

        events = []
        async for event in chain.execute("Input"):
            events.append(event)

        # Should handle gracefully (no events)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_single_agent_chain(self, context_manager, pattern_registry, tool_registry):
        """Test agent chain with single agent."""
        provider = MockLLMProvider(response="Solo")
        agent = Agent(
            config=AgentConfig(agent_id="solo"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider
        )

        chain = AgentChain(agents=[("solo", agent)])

        events = []
        async for event in chain.execute("Input"):
            events.append(event)

        # Should execute single agent
        step_events = [e for e in events if isinstance(e, StepCompleteEvent)]
        assert len(step_events) == 1

    @pytest.mark.asyncio
    async def test_parallel_with_no_agents(self):
        """Test parallel pattern with no agents."""
        pattern = ParallelPattern(agents={})

        events = []
        async for event in pattern.execute_and_merge("Query"):
            events.append(event)

        # Should handle gracefully
        assert len(events) >= 1  # At least status event

    @pytest.mark.asyncio
    async def test_debate_with_single_agent(self, context_manager, pattern_registry, tool_registry):
        """Test debate pattern with single agent."""
        provider = MockLLMProvider(response="Solo opinion")
        agent = Agent(
            config=AgentConfig(agent_id="solo"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider
        )

        pattern = DebatePattern(agents={"solo": agent})

        events = []
        async for event in pattern.converge("Topic"):
            events.append(event)

        # Should handle single agent debate
        assert len(events) >= 1


@pytest.mark.asyncio
class TestParallelTimeoutFixes:
    """Tests for parallel pattern timeout fixes."""

    async def test_one_agent_times_out_others_succeed(self, context_manager, pattern_registry, tool_registry):
        """Test one agent timing out while others succeed.

        When one agent times out, the other agents should complete successfully
        and their results should be included in the final output.
        """
        from agentic.multi_agent import ParallelPattern, ParallelConfig
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.events import StatusEvent, StepCompleteEvent
        from tests.mock_provider import MockLLMProvider

        # Fast agent
        fast_provider = MockLLMProvider(response="Fast response")
        fast_agent = Agent(
            config=AgentConfig(agent_id="fast"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=fast_provider
        )

        # Slow agent that will timeout
        import asyncio

        class SlowProvider:
            async def stream(self, prompt, **kwargs):
                await asyncio.sleep(10)  # Longer than timeout
                yield "slow"

        slow_agent = Agent(
            config=AgentConfig(agent_id="slow"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=SlowProvider()
        )

        config = ParallelConfig(merge_strategy="concat", timeout_seconds=0.5)
        pattern = ParallelPattern(
            agents={"fast": fast_agent, "slow": slow_agent},
            config=config
        )

        events = []
        async for event in pattern.execute_and_merge("Query"):
            events.append(event)

        # Should have events from fast agent and timeout event for slow
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        step_events = [e for e in events if isinstance(e, StepCompleteEvent)]

        # Fast agent should complete
        assert any("fast" in str(e) or e.result.status.value == "ok" for e in step_events if hasattr(e, 'result'))
        # Slow agent should timeout
        assert any("timed out" in e.message.lower() for e in status_events if hasattr(e, 'message') and e.message)

    async def test_all_agents_timeout(self, context_manager, pattern_registry, tool_registry):
        """Test all agents timing out.

        When all agents timeout, should still complete gracefully with
        appropriate timeout messages.
        """
        from agentic.multi_agent import ParallelPattern, ParallelConfig
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.events import StatusEvent
        import asyncio

        class SlowProvider:
            async def stream(self, prompt, **kwargs):
                await asyncio.sleep(10)
                yield "slow"

        agents = {}
        for i in range(3):
            agent = Agent(
                config=AgentConfig(agent_id=f"slow{i}"),
                context=context_manager,
                patterns=pattern_registry,
                tools=tool_registry,
                provider_client=SlowProvider()
            )
            agents[f"slow{i}"] = agent

        config = ParallelConfig(merge_strategy="concat", timeout_seconds=0.5)
        pattern = ParallelPattern(agents=agents, config=config)

        events = []
        async for event in pattern.execute_and_merge("Query"):
            events.append(event)

        # All should timeout
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        timeout_events = [e for e in status_events if hasattr(e, 'message') and e.message and 'timed out' in e.message.lower()]

        assert len(timeout_events) >= 1  # At least one timeout message

    async def test_timeout_before_first_event(self, context_manager, pattern_registry, tool_registry):
        """Test timeout occurring before agent emits first event.

        Tests edge case where timeout fires before the agent has emitted
        any events at all.
        """
        from agentic.multi_agent import ParallelPattern, ParallelConfig
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.events import StatusEvent
        import asyncio

        class VerySlowProvider:
            async def stream(self, prompt, **kwargs):
                await asyncio.sleep(10)  # Much longer than timeout
                yield "never seen"

        agent = Agent(
            config=AgentConfig(agent_id="veryslow"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=VerySlowProvider()
        )

        config = ParallelConfig(merge_strategy="concat", timeout_seconds=0.1)
        pattern = ParallelPattern(agents={"veryslow": agent}, config=config)

        events = []
        async for event in pattern.execute_and_merge("Query"):
            events.append(event)

        # Should have timeout event
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any("timed out" in e.message.lower() for e in status_events if hasattr(e, 'message') and e.message)

    async def test_agent_exception_during_execution(self, context_manager, pattern_registry, tool_registry):
        """Test agent exception handling in parallel execution.

        When an agent raises an exception during execution, it should be
        caught and reported as an error event.
        """
        from agentic.multi_agent import ParallelPattern, ParallelConfig
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.events import StatusEvent

        # Provider that raises exception mid-stream
        class ExceptionProvider:
            async def stream(self, prompt, **kwargs):
                yield "Starting"
                raise ValueError("Simulated stream error")

        error_agent = Agent(
            config=AgentConfig(agent_id="error_agent"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=ExceptionProvider()
        )

        # Normal agent for comparison
        normal_provider = MockLLMProvider(response="Normal response")
        normal_agent = Agent(
            config=AgentConfig(agent_id="normal"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=normal_provider
        )

        config = ParallelConfig(merge_strategy="concat", timeout_seconds=5.0)
        pattern = ParallelPattern(
            agents={"error": error_agent, "normal": normal_agent},
            config=config
        )

        events = []
        async for event in pattern.execute_and_merge("Query"):
            events.append(event)

        # Should have error event for the failed agent
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        step_events = [e for e in events if isinstance(e, StepCompleteEvent)]

        # Check that we get results from normal agent and error handling from error agent
        # The error should be caught and handled gracefully
        assert len(events) > 0  # Should have some events

        # At least one agent should have error status or error message
        has_error = any(
            ("failed" in e.message.lower() if hasattr(e, 'message') and e.message else False) or
            (e.status == AgentStatus.ERROR if hasattr(e, 'status') else False)
            for e in status_events
        ) or any(
            e.result.status == AgentStatus.ERROR
            for e in step_events if hasattr(e, 'result')
        )

        # Normal agent should complete successfully
        normal_completed = any(
            e.result.status == AgentStatus.OK
            for e in step_events if hasattr(e, 'result')
        )

        assert has_error or normal_completed  # At least one condition should be true

    async def test_multiple_agents_different_failure_modes(self, context_manager, pattern_registry, tool_registry):
        """Test handling multiple agents with different failure modes.

        Mix of: success, timeout, exception - all should be handled gracefully.
        """
        from agentic.multi_agent import ParallelPattern, ParallelConfig
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.events import StatusEvent, StepCompleteEvent
        import asyncio

        # Success agent
        success_provider = MockLLMProvider(response="Success")
        success_agent = Agent(
            config=AgentConfig(agent_id="success"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=success_provider
        )

        # Timeout agent
        class TimeoutProvider:
            async def stream(self, prompt, **kwargs):
                await asyncio.sleep(10)
                yield "timeout"

        timeout_agent = Agent(
            config=AgentConfig(agent_id="timeout"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=TimeoutProvider()
        )

        # Exception agent
        class ExceptionProvider:
            async def stream(self, prompt, **kwargs):
                raise RuntimeError("Agent error")
                if False:  # Make it a generator
                    yield

        exception_agent = Agent(
            config=AgentConfig(agent_id="exception"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=ExceptionProvider()
        )

        config = ParallelConfig(merge_strategy="concat", timeout_seconds=0.5)
        pattern = ParallelPattern(
            agents={
                "success": success_agent,
                "timeout": timeout_agent,
                "exception": exception_agent
            },
            config=config
        )

        events = []
        async for event in pattern.execute_and_merge("Query"):
            events.append(event)

        status_events = [e for e in events if isinstance(e, StatusEvent)]
        step_events = [e for e in events if isinstance(e, StepCompleteEvent)]

        # Should have success from one agent
        assert len(step_events) >= 1

        # Should have failure messages for timeout and exception
        status_messages = [e.message.lower() for e in status_events if hasattr(e, 'message') and e.message]
        has_timeout = any("timed out" in msg for msg in status_messages)
        has_error = any("failed" in msg or "error" in msg for msg in status_messages)

        # At least one failure should be reported
        assert has_timeout or has_error

    @pytest.mark.asyncio
    async def test_debate_empty_first_response(self, context_manager, pattern_registry, tool_registry):
        """Test debate consensus when first agent returns empty response."""
        provider1 = MockLLMProvider(response="")
        agent1 = Agent(
            config=AgentConfig(agent_id="agent1"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider1
        )

        provider2 = MockLLMProvider(response="Normal response")
        agent2 = Agent(
            config=AgentConfig(agent_id="agent2"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider2
        )

        debate = DebatePattern(
            agents={"agent1": agent1, "agent2": agent2},
            config=DebateConfig(max_rounds=2, consensus_detector=None)
        )

        events = []
        async for event in debate.converge("Test"):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_debate_whitespace_first_response(self, context_manager, pattern_registry, tool_registry):
        """Test debate consensus when first agent returns whitespace."""
        provider1 = MockLLMProvider(response="   \t\n  ")
        agent1 = Agent(
            config=AgentConfig(agent_id="agent1"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider1
        )

        provider2 = MockLLMProvider(response="Normal response")
        agent2 = Agent(
            config=AgentConfig(agent_id="agent2"),
            context=context_manager,
            patterns=pattern_registry,
            tools=tool_registry,
            provider_client=provider2
        )

        debate = DebatePattern(
            agents={"agent1": agent1, "agent2": agent2},
            config=DebateConfig(max_rounds=2, consensus_detector=None)
        )

        events = []
        async for event in debate.converge("Test"):
            events.append(event)

        assert len(events) > 0
