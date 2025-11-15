"""
Multi-agent patterns for coordinating multiple agents.

Provides common patterns: Chain, Supervisor-Worker, Parallel, Debate.
Leverages existing Agent/AgentRunner/events infrastructure.
"""
from dataclasses import dataclass, field
from typing import Callable, AsyncIterator, Any
import asyncio
import json

from .agent import Agent, AgentRunner
from .core import AgentStatus, AgentStepResult, ExtractedSegments
from .events import AgentEvent, StatusEvent, StepCompleteEvent


@dataclass
class AgentChainConfig:
    pass_mode: str = "response"  # "response" | "full_context" | "tool_results" | "custom"
    transform_fn: Callable[[AgentStepResult], str] | None = None  # Custom transform between agents
    prepend_context: bool = True  # Prepend previous agent output to prompt
    context_template: str = "Previous agent ({agent_id}) output:\n{output}\n\n"


class AgentChain:
    """
    Sequential agent chain: output of agent N becomes input of agent N+1.

    Flexible configuration for how outputs pass between agents.
    Each agent keeps its own prompt builder and configuration.
    """

    def __init__(
        self,
        agents: list[tuple[str, Agent]],
        config: AgentChainConfig = None
    ):
        """
        Initialize agent chain.

        Args:
            agents: List of (agent_id, agent) tuples in execution order
            config: Chain configuration
        """
        self._agents = agents
        self._runners = {aid: AgentRunner(agent) for aid, agent in agents}
        self._config = config or AgentChainConfig()

    async def execute(
        self,
        initial_input: str,
        processing_mode=None
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute chain, passing output from each agent to next.

        Args:
            initial_input: Initial input to first agent
            processing_mode: Optional processing mode override

        Yields:
            All events from all agents in the chain
        """
        current_input = initial_input
        previous_agent_id = None

        for agent_id, agent in self._agents:
            yield StatusEvent(
                AgentStatus.OK,
                f"Chain: Executing agent '{agent_id}'"
            )

            agent_input = current_input
            if previous_agent_id and self._config.prepend_context:
                context = self._config.context_template.format(
                    agent_id=previous_agent_id,
                    output=current_input
                )
                agent_input = context + current_input

            result = None
            async for event in self._runners[agent_id].step_stream(agent_input, processing_mode):
                yield event
                if isinstance(event, StepCompleteEvent):
                    result = event.result

            if result:
                current_input = self._transform_output(result, agent_id)
                previous_agent_id = agent_id

    def _transform_output(self, result: AgentStepResult, agent_id: str) -> str:
        if self._config.transform_fn:
            return self._config.transform_fn(result)

        if self._config.pass_mode == "response":
            return result.segments.response or ""

        elif self._config.pass_mode == "tool_results":
            return json.dumps([tr.output for tr in result.tool_results], indent=2)

        elif self._config.pass_mode == "full_context":
            return result.raw_output

        return result.segments.response or result.raw_output


@dataclass
class SupervisorConfig:
    delegation_pattern_name: str = "delegate"  # Pattern name for delegation
    worker_key: str = "to"  # Key in delegation pattern content to extract worker name
    task_key: str = "task"  # Key to extract task description
    max_delegation_rounds: int = 10  # Prevent infinite delegation loops


class SupervisorPattern:
    """
    Supervisor-worker pattern: One supervisor delegates tasks to specialized workers.

    Supervisor agent detects delegation requests via pattern extraction.
    Workers execute delegated tasks and return results to supervisor.
    """

    def __init__(
        self,
        supervisor: Agent,
        workers: dict[str, Agent],
        config: SupervisorConfig = None
    ):
        """
        Initialize supervisor-worker pattern.

        Args:
            supervisor: Supervisor agent that makes delegation decisions
            workers: Dict of {worker_name: agent} for specialized workers
            config: Supervisor configuration
        """
        self._supervisor = AgentRunner(supervisor)
        self._workers = {name: AgentRunner(agent) for name, agent in workers.items()}
        self._config = config or SupervisorConfig()

    async def execute(
        self,
        query: str,
        processing_mode=None
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute supervisor-worker pattern.

        Supervisor analyzes query and delegates to workers as needed.
        Workers return results that are fed back to supervisor.

        Args:
            query: Initial query for supervisor
            processing_mode: Optional processing mode

        Yields:
            All events from supervisor and workers
        """
        yield StatusEvent(AgentStatus.OK, "Supervisor analyzing task")

        delegation_count = 0
        supervisor_input = query

        while delegation_count < self._config.max_delegation_rounds:
            # Execute supervisor
            supervisor_result = None
            delegations = []

            async for event in self._supervisor.step_stream(supervisor_input, processing_mode):
                yield event

                if isinstance(event, StepCompleteEvent):
                    supervisor_result = event.result

                    for tool_call in supervisor_result.segments.tools:
                        if tool_call.name == self._config.delegation_pattern_name:
                            delegations.append(tool_call.arguments)

            if not delegations:
                yield StatusEvent(AgentStatus.DONE, "Supervisor completed without further delegation")
                return

            worker_results = []
            for delegation in delegations:
                worker_name = delegation.get(self._config.worker_key)
                task = delegation.get(self._config.task_key)

                if worker_name in self._workers:
                    yield StatusEvent(
                        AgentStatus.OK,
                        f"Delegating to worker '{worker_name}': {task[:50]}..."
                    )

                    worker_result = None
                    async for worker_event in self._workers[worker_name].step_stream(task, processing_mode):
                        yield worker_event
                        if isinstance(worker_event, StepCompleteEvent):
                            worker_result = worker_event.result

                    if worker_result:
                        worker_results.append({
                            "worker": worker_name,
                            "task": task,
                            "result": worker_result.segments.response
                        })

            if worker_results:
                results_text = json.dumps(worker_results, indent=2)
                supervisor_input = f"Worker results:\n{results_text}\n\nWhat's next?"
                delegation_count += 1
            else:
                break

        if delegation_count >= self._config.max_delegation_rounds:
            yield StatusEvent(
                AgentStatus.DONE,
                f"Max delegation rounds ({self._config.max_delegation_rounds}) reached"
            )


@dataclass
class ParallelConfig:
    merge_strategy: str = "agent"  # "agent" | "concat" | "voting"
    merge_template: str = "Synthesize these perspectives:\n\n{perspectives}"
    timeout_seconds: float = 120.0  # Timeout for parallel execution


class ParallelPattern:
    """
    Parallel analysis pattern: Multiple agents analyze same input, results merged.

    Agents execute in parallel, results merged using configured strategy.
    """

    def __init__(
        self,
        agents: dict[str, Agent],
        merger: Agent | None = None,
        config: ParallelConfig = None
    ):
        """
        Initialize parallel pattern.

        Args:
            agents: Dict of {agent_name: agent} for parallel execution
            merger: Optional merger agent to synthesize results
            config: Parallel configuration
        """
        self._agents = {name: AgentRunner(agent) for name, agent in agents.items()}
        self._merger = AgentRunner(merger) if merger else None
        self._config = config or ParallelConfig()

    async def execute_and_merge(
        self,
        query: str,
        processing_mode=None
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute on all agents in parallel, merge results.

        Args:
            query: Input query for all agents
            processing_mode: Optional processing mode

        Yields:
            All events from parallel agents and merger
        """
        yield StatusEvent(
            AgentStatus.OK,
            f"Parallel execution on {len(self._agents)} agents"
        )

        # Create tasks for all agents
        results: dict[str, AgentStepResult] = {}
        event_queue = asyncio.Queue()

        async def run_agent_with_timeout(name: str, runner: AgentRunner):
            """
            Run agent with timeout and proper cleanup.

            Ensures tasks are fully cancelled and awaited to prevent resource leaks.
            """
            async def agent_task():
                async for event in runner.step_stream(query, processing_mode):
                    await event_queue.put((name, event))
                    if isinstance(event, StepCompleteEvent):
                        results[name] = event.result

            task = asyncio.create_task(agent_task())

            try:
                await asyncio.wait_for(task, timeout=self._config.timeout_seconds)
                await event_queue.put((name, None))
            except asyncio.TimeoutError:
                # Explicit cancellation
                task.cancel()

                # Wait for task to finish cancelling (up to 1 second)
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # Task cancelled or didn't finish within 1s - continue anyway
                    pass

                await event_queue.put((name, StatusEvent(
                    AgentStatus.ERROR,
                    f"Agent '{name}' timed out after {self._config.timeout_seconds}s"
                )))
                await event_queue.put((name, None))
            except Exception as exc:
                await event_queue.put((name, StatusEvent(
                    AgentStatus.ERROR,
                    f"Agent '{name}' failed: {type(exc).__name__}: {exc}"
                )))
                await event_queue.put((name, None))

        tasks = [
            asyncio.create_task(run_agent_with_timeout(name, runner))
            for name, runner in self._agents.items()
        ]

        finished = 0
        try:
            while finished < len(self._agents):
                name, event = await event_queue.get()
                if event is None:
                    finished += 1
                else:
                    yield event

            await asyncio.gather(*tasks, return_exceptions=True)

        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        if self._config.merge_strategy == "concat":
            merged = self._merge_concat(results)
            yield StatusEvent(AgentStatus.OK, "Results concatenated")
            yield StepCompleteEvent(AgentStepResult(
                status=AgentStatus.DONE,
                raw_output=merged,
                segments=ExtractedSegments(response=merged),
                tool_results=[],
                iteration=0
            ))

        elif self._config.merge_strategy == "agent" and self._merger:
            yield StatusEvent(AgentStatus.OK, "Merging with merger agent")
            merger_input = self._prepare_merger_input(results)
            async for event in self._merger.step_stream(merger_input, processing_mode):
                yield event

    def _merge_concat(self, results: dict[str, AgentStepResult]) -> str:
        parts = []
        for name, result in results.items():
            parts.append(f"## {name}:\n{result.segments.response or result.raw_output}")
        return "\n\n".join(parts)

    def _prepare_merger_input(self, results: dict[str, AgentStepResult]) -> str:
        perspectives = []
        for name, result in results.items():
            perspectives.append(f"{name}: {result.segments.response or result.raw_output}")

        return self._config.merge_template.format(
            perspectives="\n\n".join(perspectives)
        )


@dataclass
class DebateConfig:
    max_rounds: int = 5
    consensus_detector: Callable[[list[str]], bool] | None = None
    moderator_prompt_template: str = "Summarize the consensus from this debate:\n{history}"
    round_prompt_template: str = "Topic: {topic}\n\nPrevious rounds:\n{history}\n\nYour turn:"


class DebatePattern:
    """
    Debate pattern: Multiple agents debate until consensus or max rounds.

    Agents take turns responding, building on previous rounds.
    Optional moderator summarizes final consensus.
    """

    def __init__(
        self,
        agents: dict[str, Agent],
        moderator: Agent | None = None,
        config: DebateConfig = None
    ):
        """
        Initialize debate pattern.

        Args:
            agents: Dict of {role: agent} for debate participants
            moderator: Optional moderator agent to summarize
            config: Debate configuration
        """
        self._agents = {name: AgentRunner(agent) for name, agent in agents.items()}
        self._moderator = AgentRunner(moderator) if moderator else None
        self._config = config or DebateConfig()

    async def converge(
        self,
        topic: str,
        processing_mode=None
    ) -> AsyncIterator[AgentEvent]:
        """
        Run debate rounds until consensus.

        Args:
            topic: Debate topic
            processing_mode: Optional processing mode

        Yields:
            All events from debate rounds and moderator
        """
        history: list[dict[str, str]] = []

        for round_num in range(self._config.max_rounds):
            yield StatusEvent(
                AgentStatus.OK,
                f"Debate round {round_num + 1}/{self._config.max_rounds}"
            )

            round_responses = {}

            for name, runner in self._agents.items():
                prompt = self._build_round_prompt(topic, history, name)

                result = None
                async for event in runner.step_stream(prompt, processing_mode):
                    yield event
                    if isinstance(event, StepCompleteEvent):
                        result = event.result

                if result:
                    round_responses[name] = result.segments.response or result.raw_output

            history.append(round_responses)

            responses = list(round_responses.values())
            if self._check_consensus(responses):
                yield StatusEvent(
                    AgentStatus.DONE,
                    f"Consensus reached in round {round_num + 1}"
                )

                if self._moderator:
                    summary_input = self._config.moderator_prompt_template.format(
                        history=self._format_history(history)
                    )
                    async for event in self._moderator.step_stream(summary_input, processing_mode):
                        yield event
                return

        yield StatusEvent(
            AgentStatus.DONE,
            "Max rounds reached without full consensus"
        )

    def _build_round_prompt(self, topic: str, history: list[dict[str, str]], agent_name: str) -> str:
        return self._config.round_prompt_template.format(
            topic=topic,
            history=self._format_history(history)
        )

    def _format_history(self, history: list[dict[str, str]]) -> str:
        parts = []
        for i, round_responses in enumerate(history):
            parts.append(f"Round {i + 1}:")
            for name, response in round_responses.items():
                parts.append(f"  {name}: {response}")
        return "\n".join(parts)

    def _check_consensus(self, responses: list[str]) -> bool:
        if self._config.consensus_detector:
            return self._config.consensus_detector(responses)

        # Default: similarity check based on keyword overlap
        if not responses:
            return False

        first_words = set(responses[0].lower().split())
        if not first_words:
            return False

        for response in responses[1:]:
            response_words = set(response.lower().split())
            overlap = len(first_words & response_words) / len(first_words)
            if overlap < 0.7:
                return False

        return True
