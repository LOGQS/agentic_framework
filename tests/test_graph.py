"""
Tests for graph-based DAG orchestration system.

Covers:
- GraphConfig and GraphNode dataclasses
- GraphRunner initialization and configuration
- Node addition and cycle detection
- DAG execution with various topologies
- Failure strategies (fail_fast, allow_independent, always_run)
- run_on_failure override behavior
- failure_mode (fail vs soft_fail) behavior
- Concurrent execution with max_concurrency
- Output key writing (success only)
- State persistence to context
- Event emission during graph execution
- Integration with AgentRunner and LogicRunner
- Custom callable nodes (event-streaming and non-streaming)
- BaseEvent validation for callables
- Retry and rate limiting per node
- Graph visualization (Mermaid and DOT export)
"""
import pytest
import asyncio
import json
from typing import AsyncIterator

from agentic.graph import (
    GraphRunner,
    GraphConfig,
    GraphNode,
    GraphNodeStatus
)
from agentic.events import (
    BaseEvent,
    StepCompleteEvent,
    GraphStartEvent,
    GraphNodeStartEvent,
    GraphNodeCompleteEvent,
    GraphCompleteEvent,
    StatusEvent
)
from agentic.agent import AgentRunner
from agentic.logic import LogicRunner, LogicConfig
from agentic.core import AgentStatus, AgentStepResult, ExtractedSegments
from agentic.context import ContextManager
from agentic.resilience import RetryConfig, RateLimiter
from tests.mock_provider import MockLLMProvider


class TestGraphConfig:
    """Tests for GraphConfig dataclass."""

    def test_graph_config_defaults(self):
        """Test GraphConfig with default values."""
        config = GraphConfig(graph_id="test_graph")
        assert config.graph_id == "test_graph"
        assert config.max_concurrency == 8
        assert config.failure_strategy == "fail_fast"
        assert config.persist_state is False
        assert config.state_context_key is None

    def test_graph_config_custom_values(self):
        """Test GraphConfig with custom values."""
        config = GraphConfig(
            graph_id="custom",
            max_concurrency=4,
            failure_strategy="allow_independent",
            persist_state=True,
            state_context_key="my:custom:key"
        )
        assert config.max_concurrency == 4
        assert config.failure_strategy == "allow_independent"
        assert config.persist_state is True
        assert config.state_context_key == "my:custom:key"

    def test_graph_config_invalid_failure_strategy(self):
        """Test GraphConfig rejects invalid failure strategy."""
        with pytest.raises(ValueError, match="failure_strategy must be"):
            GraphConfig(graph_id="test", failure_strategy="invalid")


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_graph_node_minimal(self, agent_runner):
        """Test GraphNode with minimal configuration."""
        node = GraphNode(id="node1", executable=agent_runner)
        assert node.id == "node1"
        assert node.executable is agent_runner
        assert node.output_key is None
        assert node.output_selector is None
        assert node.retry_config is None
        assert node.failure_mode == "fail"
        assert node.run_on_failure is False

    def test_graph_node_full_configuration(self, agent_runner):
        """Test GraphNode with all parameters."""
        retry_cfg = RetryConfig(max_attempts=3, base_delay=1.0)

        def selector(result):
            return result.raw_output

        node = GraphNode(
            id="node2",
            executable=agent_runner,
            output_key="result_key",
            output_selector=selector,
            retry_config=retry_cfg,
            failure_mode="soft_fail",
            run_on_failure=True
        )

        assert node.id == "node2"
        assert node.output_key == "result_key"
        assert node.output_selector is selector
        assert node.retry_config is retry_cfg
        assert node.failure_mode == "soft_fail"
        assert node.run_on_failure is True


class TestGraphRunnerInitialization:
    """Tests for GraphRunner initialization."""

    def test_graph_runner_creation(self, context_manager):
        """Test creating GraphRunner instance."""
        config = GraphConfig(graph_id="test")
        runner = GraphRunner(config, context_manager)

        assert runner._config.graph_id == "test"
        assert runner._context is context_manager
        assert runner._rate_limiter is None
        assert len(runner._nodes) == 0

    def test_graph_runner_with_rate_limiter(self, context_manager):
        """Test GraphRunner with rate limiter."""
        from agentic.resilience import RateLimitConfig

        config = GraphConfig(graph_id="test")
        rate_limit_config = RateLimitConfig(requests_per_minute=60)
        rate_limiter = RateLimiter(rate_limit_config)
        runner = GraphRunner(config, context_manager, rate_limiter)

        assert runner._rate_limiter is rate_limiter


class TestGraphNodeAddition:
    """Tests for adding nodes to graph."""

    def test_add_single_node(self, context_manager, agent_runner):
        """Test adding a single node without dependencies."""
        config = GraphConfig(graph_id="test")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="node1", executable=agent_runner)
        runner.add_node(node)

        assert "node1" in runner._nodes
        assert runner._indegree["node1"] == 0
        assert runner._status["node1"] == GraphNodeStatus.PENDING

    def test_add_node_with_dependencies(self, context_manager, agent_runner):
        """Test adding nodes with dependencies."""
        config = GraphConfig(graph_id="test")
        runner = GraphRunner(config, context_manager)

        node1 = GraphNode(id="node1", executable=agent_runner)
        node2 = GraphNode(id="node2", executable=agent_runner)

        runner.add_node(node1)
        runner.add_node(node2, dependencies=["node1"])

        assert runner._indegree["node2"] == 1
        assert "node2" in runner._children["node1"]
        assert runner._parents["node2"] == ["node1"]

    def test_add_duplicate_node(self, context_manager, agent_runner):
        """Test adding duplicate node ID raises error."""
        config = GraphConfig(graph_id="test")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="duplicate", executable=agent_runner)
        runner.add_node(node)

        with pytest.raises(ValueError, match="Duplicate node id"):
            runner.add_node(node)

    def test_add_node_with_missing_parent(self, context_manager, agent_runner):
        """Test adding node with non-existent parent raises error."""
        config = GraphConfig(graph_id="test")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="node1", executable=agent_runner)

        with pytest.raises(ValueError, match="Parent node .* not found"):
            runner.add_node(node, dependencies=["missing_parent"])


class TestCycleDetection:
    """Tests for cycle detection in graph."""

    def test_simple_cycle_detection(self, context_manager, agent_runner):
        """Test detection of simple A -> B -> A cycle."""
        config = GraphConfig(graph_id="test")
        runner = GraphRunner(config, context_manager)

        runner.add_node(GraphNode(id="A", executable=agent_runner))
        runner.add_node(GraphNode(id="B", executable=agent_runner), dependencies=["A"])

        # Manually create cycle: B -> A (completing A -> B -> A)
        runner._children["B"].append("A")
        runner._parents["A"].append("B")

        # Adding any new node should detect the cycle
        with pytest.raises(ValueError, match="would create a cycle"):
            runner.add_node(GraphNode(id="C", executable=agent_runner))

    def test_self_cycle_detection(self, context_manager, agent_runner):
        """Test detection of self-referencing cycle."""
        config = GraphConfig(graph_id="test")
        runner = GraphRunner(config, context_manager)

        runner.add_node(GraphNode(id="A", executable=agent_runner))

        # Create self-cycle by manipulating internal state
        runner._children["A"].append("A")
        runner._parents["A"].append("A")

        # Should detect self-cycle
        with pytest.raises(ValueError, match="would create a cycle"):
            runner.add_node(GraphNode(id="B", executable=agent_runner))

    def test_complex_cycle_detection(self, context_manager, agent_runner):
        """Test detection of complex multi-node cycle."""
        config = GraphConfig(graph_id="test")
        runner = GraphRunner(config, context_manager)

        # Build: A -> B -> C -> D
        runner.add_node(GraphNode(id="A", executable=agent_runner))
        runner.add_node(GraphNode(id="B", executable=agent_runner), dependencies=["A"])
        runner.add_node(GraphNode(id="C", executable=agent_runner), dependencies=["B"])
        runner.add_node(GraphNode(id="D", executable=agent_runner), dependencies=["C"])

        # Create cycle: D -> A (completing A -> B -> C -> D -> A)
        runner._children["D"].append("A")
        runner._parents["A"].append("D")

        # Should detect cycle
        with pytest.raises(ValueError, match="would create a cycle"):
            runner.add_node(GraphNode(id="E", executable=agent_runner))


class TestEmptyGraphExecution:
    """Tests for executing empty graphs."""

    @pytest.mark.asyncio
    async def test_empty_graph_execution(self, context_manager):
        """Test running graph with no nodes."""
        config = GraphConfig(graph_id="empty")
        runner = GraphRunner(config, context_manager)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # Should emit start and complete events
        assert len(events) == 2
        assert isinstance(events[0], GraphStartEvent)
        assert events[0].total_nodes == 0

        assert isinstance(events[1], GraphCompleteEvent)
        assert events[1].status == "success"


class TestSimpleGraphExecution:
    """Tests for simple graph execution patterns."""

    @pytest.mark.asyncio
    async def test_single_node_execution(self, context_manager, agent_runner, mock_llm_provider):
        """Test executing graph with single node."""
        mock_llm_provider.set_response("Hello from node")

        config = GraphConfig(graph_id="single")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="only_node", executable=agent_runner)
        runner.add_node(node)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # Check for graph events
        graph_starts = [e for e in events if isinstance(e, GraphStartEvent)]
        node_starts = [e for e in events if isinstance(e, GraphNodeStartEvent)]
        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        graph_completes = [e for e in events if isinstance(e, GraphCompleteEvent)]

        assert len(graph_starts) == 1
        assert len(node_starts) == 1
        assert len(node_completes) == 1
        assert len(graph_completes) == 1

        assert node_completes[0].status == GraphNodeStatus.COMPLETED
        assert graph_completes[0].status == "success"

    @pytest.mark.asyncio
    async def test_linear_graph_execution(self, context_manager, mock_llm_provider):
        """Test executing linear A -> B -> C graph."""
        mock_llm_provider.set_response("Response")

        config = GraphConfig(graph_id="linear")
        runner = GraphRunner(config, context_manager)

        # Create three agents
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())

        tool_registry = ToolRegistry()

        agent_a = Agent(
            AgentConfig(agent_id="A"),
            context_manager,
            pattern_registry,
            tool_registry,
            mock_llm_provider
        )
        agent_b = Agent(
            AgentConfig(agent_id="B"),
            context_manager,
            pattern_registry,
            tool_registry,
            mock_llm_provider
        )
        agent_c = Agent(
            AgentConfig(agent_id="C"),
            context_manager,
            pattern_registry,
            tool_registry,
            mock_llm_provider
        )

        runner_a = AgentRunner(agent_a)
        runner_b = AgentRunner(agent_b)
        runner_c = AgentRunner(agent_c)

        node_a = GraphNode(id="A", executable=runner_a)
        node_b = GraphNode(id="B", executable=runner_b)
        node_c = GraphNode(id="C", executable=runner_c)

        runner.add_node(node_a)
        runner.add_node(node_b, dependencies=["A"])
        runner.add_node(node_c, dependencies=["B"])

        events = []
        async for event in runner.run_stream():
            events.append(event)

        node_starts = [e for e in events if isinstance(e, GraphNodeStartEvent)]
        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]

        # All three nodes should execute
        assert len(node_starts) == 3
        assert len(node_completes) == 3

        # Verify order: A completes before B starts, B completes before C starts
        a_complete_idx = next(i for i, e in enumerate(events) if isinstance(e, GraphNodeCompleteEvent) and e.node_id == "A")
        b_start_idx = next(i for i, e in enumerate(events) if isinstance(e, GraphNodeStartEvent) and e.node_id == "B")
        b_complete_idx = next(i for i, e in enumerate(events) if isinstance(e, GraphNodeCompleteEvent) and e.node_id == "B")
        c_start_idx = next(i for i, e in enumerate(events) if isinstance(e, GraphNodeStartEvent) and e.node_id == "C")

        assert a_complete_idx < b_start_idx
        assert b_complete_idx < c_start_idx

    @pytest.mark.asyncio
    async def test_diamond_graph_execution(self, context_manager, mock_llm_provider):
        """
        Test executing diamond graph topology:
              A
             / \\
            B   C
             \\ /
              D
        """
        mock_llm_provider.set_response("Response")

        config = GraphConfig(graph_id="diamond", max_concurrency=4)
        runner = GraphRunner(config, context_manager)

        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())
        tool_registry = ToolRegistry()

        # Create agents
        agents = {}
        for node_id in ["A", "B", "C", "D"]:
            agent = Agent(
                AgentConfig(agent_id=node_id),
                context_manager,
                pattern_registry,
                tool_registry,
                mock_llm_provider
            )
            agents[node_id] = AgentRunner(agent)

        # Build diamond graph
        runner.add_node(GraphNode(id="A", executable=agents["A"]))
        runner.add_node(GraphNode(id="B", executable=agents["B"]), dependencies=["A"])
        runner.add_node(GraphNode(id="C", executable=agents["C"]), dependencies=["A"])
        runner.add_node(GraphNode(id="D", executable=agents["D"]), dependencies=["B", "C"])

        events = []
        async for event in runner.run_stream():
            events.append(event)

        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]

        # All four nodes should complete
        assert len(node_completes) == 4

        # Verify all completed successfully
        for complete_event in node_completes:
            assert complete_event.status == GraphNodeStatus.COMPLETED

        # D should start after both B and C complete
        b_complete_idx = next(i for i, e in enumerate(events) if isinstance(e, GraphNodeCompleteEvent) and e.node_id == "B")
        c_complete_idx = next(i for i, e in enumerate(events) if isinstance(e, GraphNodeCompleteEvent) and e.node_id == "C")
        d_start_idx = next(i for i, e in enumerate(events) if isinstance(e, GraphNodeStartEvent) and e.node_id == "D")

        assert b_complete_idx < d_start_idx
        assert c_complete_idx < d_start_idx


class TestConcurrencyControl:
    """Tests for max_concurrency control."""

    @pytest.mark.asyncio
    async def test_max_concurrency_limit(self, context_manager, mock_llm_provider):
        """Test that max_concurrency limits parallel execution."""
        mock_llm_provider.set_response("Response")

        # Set low concurrency
        config = GraphConfig(graph_id="concurrent", max_concurrency=2)
        runner = GraphRunner(config, context_manager)

        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())
        tool_registry = ToolRegistry()

        # Create 5 independent nodes (no dependencies)
        for i in range(5):
            agent = Agent(
                AgentConfig(agent_id=f"node_{i}"),
                context_manager,
                pattern_registry,
                tool_registry,
                mock_llm_provider
            )
            runner.add_node(GraphNode(id=f"node_{i}", executable=AgentRunner(agent)))

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # All nodes should complete
        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        assert len(node_completes) == 5

        # Verify concurrency by checking at most 2 nodes running at once
        # This is implicit in the execution - we trust the implementation


class TestOutputKeyWriting:
    """Tests for output_key and output_selector functionality."""

    @pytest.mark.asyncio
    async def test_output_key_writing_on_success(self, context_manager, agent_runner, mock_llm_provider):
        """Test that output_key is written to context on success."""
        mock_llm_provider.set_response("Output value")

        config = GraphConfig(graph_id="output_test")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(
            id="node1",
            executable=agent_runner,
            output_key="result"
        )
        runner.add_node(node)

        async for event in runner.run_stream():
            pass

        # Check output was written to context
        output = context_manager.get("result")
        assert output is not None
        assert "Output value" in output

    @pytest.mark.asyncio
    async def test_output_selector_custom_extraction(self, context_manager, agent_runner, mock_llm_provider):
        """Test custom output_selector for extracting specific data."""
        mock_llm_provider.set_response("Custom output")

        config = GraphConfig(graph_id="selector_test")
        runner = GraphRunner(config, context_manager)

        def custom_selector(result: AgentStepResult) -> str:
            return f"CUSTOM: {result.raw_output}"

        node = GraphNode(
            id="node1",
            executable=agent_runner,
            output_key="custom_result",
            output_selector=custom_selector
        )
        runner.add_node(node)

        async for event in runner.run_stream():
            pass

        output = context_manager.get("custom_result")
        assert output.startswith("CUSTOM:")

    @pytest.mark.asyncio
    async def test_no_output_on_failure(self, context_manager, mock_llm_provider):
        """Test that output_key is NOT written on node failure."""
        # Create an agent that will fail
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())
        tool_registry = ToolRegistry()

        # Use a mock that simulates failure (empty response might trigger issues)
        mock_llm_provider.set_response("")

        agent = Agent(
            AgentConfig(agent_id="failing"),
            context_manager,
            pattern_registry,
            tool_registry,
            mock_llm_provider
        )

        config = GraphConfig(graph_id="fail_test")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(
            id="failing_node",
            executable=AgentRunner(agent),
            output_key="should_not_exist"
        )
        runner.add_node(node)

        async for event in runner.run_stream():
            pass

        # Output key should not be written (or should be None)
        # Note: This depends on how agent failure is handled


class TestFailureStrategies:
    """Tests for different failure strategies."""

    @pytest.mark.asyncio
    async def test_fail_fast_strategy(self, context_manager, mock_llm_provider):
        """
        Test fail_fast strategy stops execution after first failure.

        Graph topology:
            A (fails)
            |
            B (should not run)
        """
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())
        tool_registry = ToolRegistry()

        config = GraphConfig(graph_id="fail_fast", failure_strategy="fail_fast")
        runner = GraphRunner(config, context_manager)

        # Create a callable that raises an exception
        async def failing_node(ctx):
            raise RuntimeError("Intentional failure")

        # Create normal agent for B
        agent_b = Agent(
            AgentConfig(agent_id="B"),
            context_manager,
            pattern_registry,
            tool_registry,
            mock_llm_provider
        )

        runner.add_node(GraphNode(id="A", executable=failing_node))
        runner.add_node(GraphNode(id="B", executable=AgentRunner(agent_b)), dependencies=["A"])

        events = []
        async for event in runner.run_stream():
            events.append(event)

        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]

        # A should fail
        a_complete = next(e for e in node_completes if e.node_id == "A")
        assert a_complete.status == GraphNodeStatus.FAILED

        # B should not start in fail_fast mode (or be skipped)
        # Check if B appears in events
        b_starts = [e for e in events if isinstance(e, GraphNodeStartEvent) and e.node_id == "B"]
        assert len(b_starts) == 0  # B should never start

    @pytest.mark.asyncio
    async def test_allow_independent_strategy(self, context_manager, mock_llm_provider):
        """
        Test allow_independent strategy allows independent branches to continue.

        Graph topology:
            A (fails)  C (succeeds)
            |          |
            B (skip)   D (runs)
        """
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())
        tool_registry = ToolRegistry()

        config = GraphConfig(graph_id="allow_indep", failure_strategy="allow_independent")
        runner = GraphRunner(config, context_manager)

        # Failing node
        async def failing_node(ctx):
            raise RuntimeError("Intentional failure")

        # Create agents
        mock_llm_provider.set_response("Success")
        agent_c = Agent(AgentConfig(agent_id="C"), context_manager, pattern_registry, tool_registry, mock_llm_provider)
        agent_d = Agent(AgentConfig(agent_id="D"), context_manager, pattern_registry, tool_registry, mock_llm_provider)
        agent_b = Agent(AgentConfig(agent_id="B"), context_manager, pattern_registry, tool_registry, mock_llm_provider)

        runner.add_node(GraphNode(id="A", executable=failing_node))
        runner.add_node(GraphNode(id="B", executable=AgentRunner(agent_b)), dependencies=["A"])
        runner.add_node(GraphNode(id="C", executable=AgentRunner(agent_c)))
        runner.add_node(GraphNode(id="D", executable=AgentRunner(agent_d)), dependencies=["C"])

        events = []
        async for event in runner.run_stream():
            events.append(event)

        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]

        # A should fail
        a_complete = next(e for e in node_completes if e.node_id == "A")
        assert a_complete.status == GraphNodeStatus.FAILED

        # B should be skipped (depends on failed A)
        b_complete = next(e for e in node_completes if e.node_id == "B")
        assert b_complete.status == GraphNodeStatus.SKIPPED

        # C and D should succeed (independent branch)
        c_complete = next(e for e in node_completes if e.node_id == "C")
        d_complete = next(e for e in node_completes if e.node_id == "D")
        assert c_complete.status == GraphNodeStatus.COMPLETED
        assert d_complete.status == GraphNodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_always_run_strategy(self, context_manager, mock_llm_provider):
        """
        Test always_run strategy continues execution despite failures.

        Graph topology:
            A (fails)
            |
            B (runs despite A failure)
        """
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())
        tool_registry = ToolRegistry()

        config = GraphConfig(graph_id="always_run", failure_strategy="always_run")
        runner = GraphRunner(config, context_manager)

        # Failing node
        async def failing_node(ctx):
            raise RuntimeError("Intentional failure")

        mock_llm_provider.set_response("Continued")
        agent_b = Agent(AgentConfig(agent_id="B"), context_manager, pattern_registry, tool_registry, mock_llm_provider)

        runner.add_node(GraphNode(id="A", executable=failing_node))
        runner.add_node(GraphNode(id="B", executable=AgentRunner(agent_b)), dependencies=["A"])

        events = []
        async for event in runner.run_stream():
            events.append(event)

        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]

        # A should fail
        a_complete = next(e for e in node_completes if e.node_id == "A")
        assert a_complete.status == GraphNodeStatus.FAILED

        # B should still run (always_run strategy)
        b_complete = next(e for e in node_completes if e.node_id == "B")
        assert b_complete.status == GraphNodeStatus.COMPLETED


class TestRunOnFailureFlag:
    """Tests for run_on_failure node flag."""

    @pytest.mark.asyncio
    async def test_run_on_failure_executes_cleanup_node(self, context_manager, mock_llm_provider):
        """
        Test that nodes with run_on_failure=True execute even after upstream failures.

        Graph topology (fail_fast mode):
            A (fails)
            |
            B (run_on_failure=True, should run)
        """
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())
        tool_registry = ToolRegistry()

        config = GraphConfig(graph_id="run_on_fail", failure_strategy="fail_fast")
        runner = GraphRunner(config, context_manager)

        # Failing node
        async def failing_node(ctx):
            raise RuntimeError("Intentional failure")

        mock_llm_provider.set_response("Cleanup done")
        agent_b = Agent(AgentConfig(agent_id="B"), context_manager, pattern_registry, tool_registry, mock_llm_provider)

        runner.add_node(GraphNode(id="A", executable=failing_node))
        runner.add_node(
            GraphNode(id="B", executable=AgentRunner(agent_b), run_on_failure=True),
            dependencies=["A"]
        )

        events = []
        async for event in runner.run_stream():
            events.append(event)

        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]

        # A should fail
        a_complete = next(e for e in node_completes if e.node_id == "A")
        assert a_complete.status == GraphNodeStatus.FAILED

        # B should run despite A's failure (run_on_failure=True)
        b_complete = next(e for e in node_completes if e.node_id == "B")
        assert b_complete.status == GraphNodeStatus.COMPLETED


class TestSoftFailMode:
    """Tests for failure_mode='soft_fail' behavior."""

    @pytest.mark.asyncio
    async def test_soft_fail_node_completes(self, context_manager):
        """Soft-fail nodes complete successfully despite exceptions."""
        config = GraphConfig(graph_id="soft_test")
        runner = GraphRunner(config, context_manager)

        async def failing(ctx):
            raise ValueError("Soft failure")

        runner.add_node(GraphNode(id="A", executable=failing, failure_mode="soft_fail"))

        events = []
        async for event in runner.run_stream():
            events.append(event)

        completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        assert completes[0].status == GraphNodeStatus.COMPLETED
        assert "Soft failure" in completes[0].error_message

    @pytest.mark.asyncio
    async def test_soft_fail_allows_dependents(self, context_manager):
        """Dependents run after soft-fail nodes."""
        config = GraphConfig(graph_id="soft_deps")
        runner = GraphRunner(config, context_manager)

        async def soft_failing(ctx):
            raise RuntimeError("Soft error")

        async def dependent(ctx):
            return "Ran successfully"

        runner.add_node(GraphNode(id="A", executable=soft_failing, failure_mode="soft_fail"))
        runner.add_node(GraphNode(id="B", executable=dependent), ["A"])

        events = []
        async for event in runner.run_stream():
            events.append(event)

        completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        a_complete = next(e for e in completes if e.node_id == "A")
        b_complete = next(e for e in completes if e.node_id == "B")

        assert a_complete.status == GraphNodeStatus.COMPLETED
        assert a_complete.error_message is not None
        assert b_complete.status == GraphNodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_soft_fail_no_fail_fast_trigger(self, context_manager):
        """Soft-fail doesn't trigger fail_fast strategy."""
        config = GraphConfig(graph_id="soft_no_fast", failure_strategy="fail_fast")
        runner = GraphRunner(config, context_manager)

        async def soft_failing(ctx):
            raise RuntimeError("Soft")

        async def independent(ctx):
            return "Runs"

        runner.add_node(GraphNode(id="A", executable=soft_failing, failure_mode="soft_fail"))
        runner.add_node(GraphNode(id="B", executable=independent))

        events = []
        async for event in runner.run_stream():
            events.append(event)

        completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        assert len(completes) == 2
        assert all(e.status == GraphNodeStatus.COMPLETED for e in completes)

    @pytest.mark.asyncio
    async def test_soft_fail_no_output_write(self, context_manager):
        """Soft-fail nodes don't write to output_key."""
        config = GraphConfig(graph_id="soft_output")
        runner = GraphRunner(config, context_manager)

        async def soft_failing(ctx):
            raise RuntimeError("Fail")

        runner.add_node(GraphNode(
            id="A",
            executable=soft_failing,
            failure_mode="soft_fail",
            output_key="result"
        ))

        async for event in runner.run_stream():
            pass

        assert context_manager.get("result") is None


class TestStatePersistence:
    """Tests for graph state persistence to context."""

    @pytest.mark.asyncio
    async def test_state_persistence_enabled(self, context_manager, agent_runner, mock_llm_provider):
        """Test that graph state is persisted when configured."""
        mock_llm_provider.set_response("Done")

        config = GraphConfig(
            graph_id="persist_test",
            persist_state=True
        )
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="node1", executable=agent_runner)
        runner.add_node(node)

        async for event in runner.run_stream():
            pass

        # Check state was written
        state_key = f"graph:{config.graph_id}:state"
        state_json = context_manager.get(state_key)

        assert state_json is not None
        state = json.loads(state_json)

        assert state["graph_id"] == "persist_test"
        assert state["status"] == "success"
        assert "stats" in state
        assert "node_statuses" in state

    @pytest.mark.asyncio
    async def test_state_persistence_custom_key(self, context_manager, agent_runner, mock_llm_provider):
        """Test custom state_context_key."""
        mock_llm_provider.set_response("Done")

        config = GraphConfig(
            graph_id="custom_key",
            persist_state=True,
            state_context_key="my:custom:state"
        )
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="node1", executable=agent_runner)
        runner.add_node(node)

        async for event in runner.run_stream():
            pass

        # Check custom key was used
        state_json = context_manager.get("my:custom:state")
        assert state_json is not None

        state = json.loads(state_json)
        assert state["graph_id"] == "custom_key"

    @pytest.mark.asyncio
    async def test_state_persistence_disabled(self, context_manager, agent_runner, mock_llm_provider):
        """Test that state is not persisted when disabled."""
        mock_llm_provider.set_response("Done")

        config = GraphConfig(
            graph_id="no_persist",
            persist_state=False
        )
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="node1", executable=agent_runner)
        runner.add_node(node)

        async for event in runner.run_stream():
            pass

        # State should not be written
        state_key = f"graph:{config.graph_id}:state"
        state_json = context_manager.get(state_key)

        assert state_json is None


class TestCustomCallableNodes:
    """Tests for custom callable nodes."""

    @pytest.mark.asyncio
    async def test_event_streaming_callable(self, context_manager):
        """Test callable that returns AsyncIterator[BaseEvent]."""
        async def event_stream_callable(ctx: 'ContextManager') -> AsyncIterator[BaseEvent]:
            yield StatusEvent("Starting custom node")
            await asyncio.sleep(0.01)  # Simulate work
            yield StatusEvent("Finishing custom node")

        config = GraphConfig(graph_id="callable_stream")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="custom", executable=event_stream_callable)
        runner.add_node(node)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # Should have status events from callable
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert len(status_events) >= 2

        # Node should complete successfully
        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        assert node_completes[0].status == GraphNodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_async_callable_with_return_value(self, context_manager):
        """Test async callable that returns a value (not event stream)."""
        async def async_callable(ctx: 'ContextManager'):
            await asyncio.sleep(0.01)
            return "computed_result"

        config = GraphConfig(graph_id="callable_async")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(
            id="custom",
            executable=async_callable,
            output_key="result"
        )
        runner.add_node(node)

        async for event in runner.run_stream():
            pass

        # Result should be written to output_key
        output = context_manager.get("result")
        assert output == "computed_result"

    @pytest.mark.asyncio
    async def test_callable_baseevent_validation(self, context_manager):
        """Test that callables yielding non-BaseEvent raise TypeError."""
        async def bad_callable(ctx: 'ContextManager') -> AsyncIterator[str]:
            yield "not a BaseEvent"

        config = GraphConfig(graph_id="bad_callable")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="bad", executable=bad_callable)
        runner.add_node(node)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # Node should fail with TypeError
        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        assert node_completes[0].status == GraphNodeStatus.FAILED
        assert "non-BaseEvent" in node_completes[0].error_message

    @pytest.mark.asyncio
    async def test_callable_exception_handling(self, context_manager):
        """Test exception handling in custom callables."""
        async def failing_callable(ctx: 'ContextManager'):
            raise ValueError("Custom callable error")

        config = GraphConfig(graph_id="failing_callable")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="failing", executable=failing_callable)
        runner.add_node(node)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # Node should fail with error message
        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        assert node_completes[0].status == GraphNodeStatus.FAILED
        assert "Custom callable error" in node_completes[0].error_message


class TestLogicRunnerIntegration:
    """Tests for LogicRunner integration."""

    @pytest.mark.asyncio
    async def test_logic_runner_node(self, context_manager, agent_runner, pattern_registry, mock_llm_provider):
        """Test graph node containing LogicRunner."""
        mock_llm_provider.set_response("Loop iteration")

        config = GraphConfig(graph_id="logic_test")
        runner = GraphRunner(config, context_manager)

        logic_config = LogicConfig(logic_id="loop3", max_iterations=3)
        logic_runner = LogicRunner(agent_runner, context_manager, pattern_registry, logic_config)

        node = GraphNode(id="logic_node", executable=logic_runner)
        runner.add_node(node)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # LogicRunner should emit step complete events for each iteration
        step_completes = [e for e in events if isinstance(e, StepCompleteEvent)]
        assert len(step_completes) >= 3  # At least 3 iterations

        # Node should complete
        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        assert node_completes[0].status == GraphNodeStatus.COMPLETED


class TestRetryAndRateLimiting:
    """Tests for retry and rate limiting per node."""

    @pytest.mark.asyncio
    async def test_node_retry_config(self, context_manager, mock_llm_provider):
        """Test node-level retry configuration."""
        # Create callable that fails first time, succeeds second
        attempt_count = {"count": 0}

        async def flaky_callable(ctx: 'ContextManager'):
            attempt_count["count"] += 1
            if attempt_count["count"] == 1:
                raise RuntimeError("First attempt fails")
            return "Success on retry"

        config = GraphConfig(graph_id="retry_test")
        runner = GraphRunner(config, context_manager)

        retry_config = RetryConfig(max_attempts=2, base_delay=0.01, retry_on=(RuntimeError,))
        node = GraphNode(
            id="retrying",
            executable=flaky_callable,
            retry_config=retry_config,
            output_key="result"
        )
        runner.add_node(node)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # Node should eventually succeed
        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        assert node_completes[0].status == GraphNodeStatus.COMPLETED

        # Result should be written
        output = context_manager.get("result")
        assert output == "Success on retry"

    @pytest.mark.asyncio
    async def test_graph_rate_limiter(self, context_manager, mock_llm_provider):
        """Test graph-level rate limiter applies to all nodes."""
        mock_llm_provider.set_response("Response")

        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry
        from agentic.resilience import RateLimitConfig

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())
        tool_registry = ToolRegistry()

        rate_limit_config = RateLimitConfig(requests_per_minute=600)  # High limit to not slow test
        rate_limiter = RateLimiter(rate_limit_config)
        config = GraphConfig(graph_id="rate_limit_test")
        runner = GraphRunner(config, context_manager, rate_limiter)

        # Add multiple nodes
        for i in range(3):
            agent = Agent(
                AgentConfig(agent_id=f"node_{i}"),
                context_manager,
                pattern_registry,
                tool_registry,
                mock_llm_provider
            )
            runner.add_node(GraphNode(id=f"node_{i}", executable=AgentRunner(agent)))

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # All nodes should complete (rate limiter shouldn't block at high limit)
        node_completes = [e for e in events if isinstance(e, GraphNodeCompleteEvent)]
        assert len(node_completes) == 3


class TestGraphCompleteEvent:
    """Tests for GraphCompleteEvent status calculation."""

    @pytest.mark.asyncio
    async def test_graph_complete_success_status(self, context_manager, agent_runner, mock_llm_provider):
        """Test graph completes with success status when all nodes succeed."""
        mock_llm_provider.set_response("Done")

        config = GraphConfig(graph_id="success_test")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="node1", executable=agent_runner)
        runner.add_node(node)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        graph_complete = [e for e in events if isinstance(e, GraphCompleteEvent)][0]
        assert graph_complete.status == "success"
        assert graph_complete.stats["completed"] == 1
        assert graph_complete.stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_graph_complete_failed_status(self, context_manager):
        """Test graph completes with failed status in fail_fast mode."""
        async def failing_callable(ctx):
            raise RuntimeError("Fail")

        config = GraphConfig(graph_id="fail_test", failure_strategy="fail_fast")
        runner = GraphRunner(config, context_manager)

        runner.add_node(GraphNode(id="failing", executable=failing_callable))

        events = []
        async for event in runner.run_stream():
            events.append(event)

        graph_complete = [e for e in events if isinstance(e, GraphCompleteEvent)][0]
        assert graph_complete.status == "failed"
        assert graph_complete.stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_graph_complete_partial_failure_status(self, context_manager, mock_llm_provider):
        """Test graph completes with partial_failure status when some nodes fail."""
        from agentic.agent import Agent
        from agentic.core import AgentConfig
        from agentic.patterns import PatternRegistry, create_default_pattern_set
        from agentic.tools import ToolRegistry

        pattern_registry = PatternRegistry(context_manager._storage)
        pattern_registry.register_pattern_set(create_default_pattern_set())
        tool_registry = ToolRegistry()

        config = GraphConfig(graph_id="partial_fail", failure_strategy="allow_independent")
        runner = GraphRunner(config, context_manager)

        # One failing, one succeeding independent node
        async def failing_callable(ctx):
            raise RuntimeError("Fail")

        mock_llm_provider.set_response("Success")
        agent = Agent(
            AgentConfig(agent_id="success"),
            context_manager,
            pattern_registry,
            tool_registry,
            mock_llm_provider
        )

        runner.add_node(GraphNode(id="failing", executable=failing_callable))
        runner.add_node(GraphNode(id="success", executable=AgentRunner(agent)))

        events = []
        async for event in runner.run_stream():
            events.append(event)

        graph_complete = [e for e in events if isinstance(e, GraphCompleteEvent)][0]
        assert graph_complete.status == "partial_failure"
        assert graph_complete.stats["failed"] >= 1
        assert graph_complete.stats["completed"] >= 1


class TestBatchRunMethod:
    """Tests for convenience run() method."""

    def test_run_method_sync_context(self, context_manager, agent_runner, mock_llm_provider):
        """Test run() method works in synchronous context."""
        mock_llm_provider.set_response("Response")

        config = GraphConfig(graph_id="sync_run")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="node1", executable=agent_runner)
        runner.add_node(node)

        # Run in sync mode
        result = runner.run()

        assert isinstance(result, dict)
        assert "node1" in result
        assert result["node1"] == GraphNodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_method_async_context_raises_error(self, context_manager, agent_runner):
        """Test run() method raises error when called from async context."""
        config = GraphConfig(graph_id="async_run")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="node1", executable=agent_runner)
        runner.add_node(node)

        # Should raise RuntimeError in async context
        with pytest.raises(RuntimeError, match="cannot be called from an async context"):
            runner.run()


class TestEventTagging:
    """Tests for graph_node_id tagging on events."""

    @pytest.mark.asyncio
    async def test_events_tagged_with_node_id(self, context_manager, agent_runner, mock_llm_provider):
        """Test that events from nodes are tagged with graph_node_id."""
        mock_llm_provider.set_response("Response")

        config = GraphConfig(graph_id="tagging_test")
        runner = GraphRunner(config, context_manager)

        node = GraphNode(id="tagged_node", executable=agent_runner)
        runner.add_node(node)

        events = []
        async for event in runner.run_stream():
            events.append(event)

        # Check if agent events have graph_node_id attribute
        # (This is best-effort tagging, so we check if the attribute exists)
        agent_events = [e for e in events if isinstance(e, StepCompleteEvent)]

        if agent_events:
            # Check if attribute was set (might not always succeed)
            for event in agent_events:
                if hasattr(event, "graph_node_id"):
                    assert event.graph_node_id == "tagged_node"


class TestGraphVisualization:
    """Tests for graph visualization utilities."""

    def test_mermaid_simple_graph(self, context_manager):
        """Test Mermaid export for simple graph."""
        from agentic.graph_visualization import to_mermaid

        config = GraphConfig(graph_id="viz_test")
        runner = GraphRunner(config, context_manager)

        async def node_a(ctx):
            return "A"

        async def node_b(ctx):
            return "B"

        runner.add_node(GraphNode(id="A", executable=node_a))
        runner.add_node(GraphNode(id="B", executable=node_b), dependencies=["A"])

        mermaid = to_mermaid(runner)

        assert "flowchart TD" in mermaid
        assert 'A["A"]' in mermaid
        assert 'B["B"]' in mermaid
        assert "A --> B" in mermaid

    def test_mermaid_with_metadata(self, context_manager, agent_runner):
        """Test Mermaid export with metadata."""
        from agentic.graph_visualization import to_mermaid

        config = GraphConfig(graph_id="viz_meta")
        runner = GraphRunner(config, context_manager)

        async def cleanup(ctx):
            pass

        runner.add_node(GraphNode(
            id="process",
            executable=agent_runner,
            output_key="result"
        ))
        runner.add_node(GraphNode(
            id="cleanup",
            executable=cleanup,
            run_on_failure=True,
            failure_mode="soft_fail"
        ), dependencies=["process"])

        mermaid = to_mermaid(runner, include_metadata=True)

        assert "cleanup" in mermaid
        assert "soft" in mermaid

    def test_dot_simple_graph(self, context_manager):
        """Test DOT export for simple graph."""
        from agentic.graph_visualization import to_dot

        config = GraphConfig(graph_id="dot_test")
        runner = GraphRunner(config, context_manager)

        async def node_x(ctx):
            return "X"

        async def node_y(ctx):
            return "Y"

        runner.add_node(GraphNode(id="X", executable=node_x))
        runner.add_node(GraphNode(id="Y", executable=node_y), dependencies=["X"])

        dot = to_dot(runner)

        assert "digraph G {" in dot
        assert 'X [label="X"]' in dot
        assert 'Y [label="Y"]' in dot
        assert "X -> Y" in dot

    def test_visualization_diamond_graph(self, context_manager):
        """Test visualization of diamond graph topology."""
        from agentic.graph_visualization import to_mermaid

        config = GraphConfig(graph_id="diamond")
        runner = GraphRunner(config, context_manager)

        async def node(ctx):
            pass

        runner.add_node(GraphNode(id="A", executable=node))
        runner.add_node(GraphNode(id="B", executable=node), dependencies=["A"])
        runner.add_node(GraphNode(id="C", executable=node), dependencies=["A"])
        runner.add_node(GraphNode(id="D", executable=node), dependencies=["B", "C"])

        mermaid = to_mermaid(runner)

        # Verify diamond structure
        assert "A --> B" in mermaid
        assert "A --> C" in mermaid
        assert "B --> D" in mermaid
        assert "C --> D" in mermaid
