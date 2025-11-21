"""
Graph-based DAG orchestration for multi-agent workflows.

Provides declarative graph execution with:
- Dynamic indegree scheduling (max concurrency)
- Failure strategies (fail_fast, allow_independent, always_run)
- Retry and rate limiting per node
- Graph-level observability via events
- Context-based data flow (no return value passing)
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, AsyncIterator
import asyncio
import json
from collections import deque

from .context import ContextManager
from .agent import AgentRunner
from .logic import LogicRunner
from .core import AgentStatus, AgentStepResult
from .events import (
    BaseEvent, AgentEvent, StepCompleteEvent,
    GraphStartEvent, GraphNodeStartEvent, GraphNodeCompleteEvent, GraphCompleteEvent
)
from .resilience import RetryConfig, RateLimiter, rate_limited_stream, retry_stream
from .logging_util import get_logger


logger = get_logger(__name__)


class GraphNodeStatus(Enum):
    """Status of a graph node during execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GraphNode:
    """
    Represents an executable node in the graph.

    Nodes can be:
    - AgentRunner: Single agent execution
    - LogicRunner: Multi-step logic flow
    - Callable: Custom async function returning AsyncIterator[AgentEvent] or Awaitable[Any]
    """
    id: str
    executable: Any  # AgentRunner | LogicRunner | Callable[[ContextManager], AsyncIterator[AgentEvent]] | Callable[[ContextManager], Awaitable[Any]]
    output_key: str | None = None  # Context key to store primary output
    output_selector: Callable[[AgentStepResult], str] | None = None  # How to extract string output from result
    retry_config: RetryConfig | None = None  # Node-level retry override
    failure_mode: str = "fail"  # "fail" | "soft_fail"
    run_on_failure: bool = False  # If True, run even if upstream nodes failed (for cleanup/logging)


@dataclass
class GraphConfig:
    """Configuration for graph execution."""
    graph_id: str
    max_concurrency: int = 8
    failure_strategy: str = "fail_fast"  # "fail_fast" | "allow_independent" | "always_run"
    persist_state: bool = False  # If True, write graph status summary to context
    state_context_key: str | None = None  # Override key; default "graph:{graph_id}:state"

    def __post_init__(self):
        """Validate configuration."""
        if self.failure_strategy not in ("fail_fast", "allow_independent", "always_run"):
            raise ValueError(
                f"failure_strategy must be 'fail_fast', 'allow_independent', or 'always_run', "
                f"got: {self.failure_strategy!r}"
            )


@dataclass
class _NodeFinishedInternalEvent:
    """Internal event for node completion signaling."""
    node_id: str
    status: GraphNodeStatus
    error_message: str | None


class GraphRunner:
    """
    Executes a DAG of nodes with dynamic scheduling and failure handling.

    Data flow is via ContextManager (no return value passing).
    Yields all underlying events plus graph-level events for observability.
    """

    def __init__(
        self,
        config: GraphConfig,
        context: ContextManager,
        rate_limiter: RateLimiter | None = None
    ):
        """
        Initialize graph runner.

        Args:
            config: Graph configuration
            context: Shared context manager for data flow
            rate_limiter: Optional rate limiter for all nodes
        """
        self._config = config
        self._context = context
        self._rate_limiter = rate_limiter

        # Graph structure
        self._nodes: dict[str, GraphNode] = {}
        self._parents: dict[str, list[str]] = {}
        self._children: dict[str, list[str]] = {}
        self._indegree: dict[str, int] = {}

        # Execution state
        self._status: dict[str, GraphNodeStatus] = {}
        self._error_messages: dict[str, str | None] = {}
        self._failed_nodes: set[str] = set()

    def add_node(self, node: GraphNode, dependencies: list[str] | None = None) -> None:
        """
        Add node to graph with optional dependencies.

        Args:
            node: Node to add
            dependencies: List of parent node IDs (must exist)

        Raises:
            ValueError: If node ID is duplicate, parent doesn't exist, or cycle detected
        """
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.id!r}")

        deps = dependencies or []

        for parent_id in deps:
            if parent_id not in self._nodes:
                raise ValueError(
                    f"Parent node {parent_id!r} not found. "
                    f"Add parent nodes before their children."
                )

        self._nodes[node.id] = node
        self._parents[node.id] = list(deps)
        self._children.setdefault(node.id, [])
        self._status[node.id] = GraphNodeStatus.PENDING
        self._error_messages[node.id] = None
        self._indegree[node.id] = len(deps)

        for parent_id in deps:
            self._children[parent_id].append(node.id)

        # DFS cycle detection
        if self._has_cycle():
            del self._nodes[node.id]
            del self._parents[node.id]
            del self._children[node.id]
            del self._status[node.id]
            del self._error_messages[node.id]
            del self._indegree[node.id]
            for parent_id in deps:
                self._children[parent_id].remove(node.id)

            raise ValueError(
                f"Adding node {node.id!r} would create a cycle. "
                f"Graph must be a DAG. Use LogicRunner for iterative loops."
            )

        logger.debug("graph.node.added", extra={
            "graph_id": self._config.graph_id,
            "node_id": node.id,
            "dependencies": deps
        })

    def _has_cycle(self) -> bool:
        """Detect cycles using DFS. Returns True if cycle exists."""
        visited = set()
        rec_stack = set()

        def visit(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for child_id in self._children.get(node_id, []):
                if child_id not in visited:
                    if visit(child_id):
                        return True
                elif child_id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self._nodes:
            if node_id not in visited:
                if visit(node_id):
                    return True

        return False

    def run(self) -> dict[str, GraphNodeStatus]:
        """
        Execute graph in batch mode (convenience wrapper).

        Returns:
            Dict mapping node_id to final status
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "GraphRunner.run() cannot be called from an async context. "
                "Use 'await run_stream()' instead, or call from a synchronous context."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                raise

        return asyncio.run(self._collect_run_events())

    async def _collect_run_events(self) -> dict[str, GraphNodeStatus]:
        """Helper to collect all events and return final status map."""
        async for event in self.run_stream():
            pass
        return dict(self._status)

    async def run_stream(self) -> AsyncIterator[AgentEvent]:
        """
        Execute graph with streaming events.

        Yields:
            AgentEvent: All events from nodes plus graph-level events

        Algorithm:
            1. Initialize ready queue with nodes having indegree=0
            2. Schedule nodes up to max_concurrency
            3. As nodes complete, update children indegrees and ready queue
            4. Apply failure strategy to skip/stop nodes
            5. Persist state if configured
        """
        if not self._nodes:
            logger.warning("graph.empty", extra={"graph_id": self._config.graph_id})
            yield GraphStartEvent(self._config.graph_id, 0)
            yield GraphCompleteEvent(
                self._config.graph_id,
                "success",
                {}
            )
            return

        logger.info("graph.start", extra={
            "graph_id": self._config.graph_id,
            "total_nodes": len(self._nodes),
            "max_concurrency": self._config.max_concurrency,
            "failure_strategy": self._config.failure_strategy
        })

        yield GraphStartEvent(self._config.graph_id, len(self._nodes))

        ready_queue: asyncio.Queue[str] = asyncio.Queue()
        events_queue: asyncio.Queue[BaseEvent | _NodeFinishedInternalEvent] = asyncio.Queue()

        for node_id, indegree in self._indegree.items():
            if indegree == 0:
                await ready_queue.put(node_id)

        active_tasks: dict[str, asyncio.Task] = {}
        graph_failed = False

        try:
            while active_tasks or not ready_queue.empty() or not events_queue.empty():
                while (
                    len(active_tasks) < self._config.max_concurrency
                    and not ready_queue.empty()
                ):
                    node_id = await ready_queue.get()

                    if graph_failed and self._config.failure_strategy == "fail_fast":
                        node = self._nodes[node_id]
                        if not node.run_on_failure:
                            self._status[node_id] = GraphNodeStatus.SKIPPED
                            yield GraphNodeCompleteEvent(
                                self._config.graph_id,
                                node_id,
                                GraphNodeStatus.SKIPPED,
                                "Skipped due to fail_fast after upstream failure"
                            )
                            await self._handle_node_completion(node_id, ready_queue, events_queue)
                            continue

                    if self._should_skip_node(node_id):
                        self._status[node_id] = GraphNodeStatus.SKIPPED
                        yield GraphNodeCompleteEvent(
                            self._config.graph_id,
                            node_id,
                            GraphNodeStatus.SKIPPED,
                            "Skipped due to upstream failure"
                        )
                        await self._handle_node_completion(node_id, ready_queue, events_queue)
                        continue

                    self._status[node_id] = GraphNodeStatus.RUNNING
                    yield GraphNodeStartEvent(
                        self._config.graph_id,
                        node_id,
                        self._parents[node_id]
                    )

                    logger.debug("graph.node.start", extra={
                        "graph_id": self._config.graph_id,
                        "node_id": node_id,
                        "active_count": len(active_tasks) + 1
                    })

                    task = asyncio.create_task(
                        self._execute_node(node_id, events_queue),
                        name=f"graph_node_{node_id}"
                    )
                    active_tasks[node_id] = task

                if not events_queue.empty() or active_tasks:
                    try:
                        event = await asyncio.wait_for(events_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    if isinstance(event, _NodeFinishedInternalEvent):
                        node_id = event.node_id
                        self._status[node_id] = event.status
                        self._error_messages[node_id] = event.error_message

                        if event.status == GraphNodeStatus.FAILED:
                            self._failed_nodes.add(node_id)
                            graph_failed = True

                        logger.debug("graph.node.complete", extra={
                            "graph_id": self._config.graph_id,
                            "node_id": node_id,
                            "status": event.status.value,
                            "error": event.error_message
                        })

                        yield GraphNodeCompleteEvent(
                            self._config.graph_id,
                            node_id,
                            event.status,
                            event.error_message
                        )

                        if node_id in active_tasks:
                            del active_tasks[node_id]

                        await self._handle_node_completion(node_id, ready_queue, events_queue)

                        if event.status == GraphNodeStatus.FAILED and self._config.failure_strategy == "fail_fast":
                            await self._skip_all_pending_nodes(ready_queue, events_queue)

                    else:
                        yield event

        finally:
            for task in active_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        stats = {
            "completed": sum(1 for s in self._status.values() if s == GraphNodeStatus.COMPLETED),
            "failed": sum(1 for s in self._status.values() if s == GraphNodeStatus.FAILED),
            "skipped": sum(1 for s in self._status.values() if s == GraphNodeStatus.SKIPPED),
            "pending": sum(1 for s in self._status.values() if s == GraphNodeStatus.PENDING),
        }

        if stats["failed"] == 0 and stats["pending"] == 0:
            graph_status = "success"
        elif stats["failed"] > 0 and self._config.failure_strategy == "fail_fast":
            graph_status = "failed"
        elif stats["failed"] > 0:
            graph_status = "partial_failure"
        else:
            graph_status = "failed"  # Shouldn't happen, but safe default

        logger.info("graph.complete", extra={
            "graph_id": self._config.graph_id,
            "status": graph_status,
            "stats": stats
        })

        yield GraphCompleteEvent(self._config.graph_id, graph_status, stats)

        if self._config.persist_state:
            state_key = self._config.state_context_key or f"graph:{self._config.graph_id}:state"
            state_data = {
                "graph_id": self._config.graph_id,
                "status": graph_status,
                "stats": stats,
                "node_statuses": {
                    node_id: status.value
                    for node_id, status in self._status.items()
                },
                "errors": {
                    node_id: msg
                    for node_id, msg in self._error_messages.items()
                    if msg is not None
                }
            }
            self._context.set(state_key, json.dumps(state_data, indent=2))

            logger.debug("graph.state.persisted", extra={
                "graph_id": self._config.graph_id,
                "state_key": state_key
            })

    def _should_skip_node(self, node_id: str) -> bool:
        """
        Determine if node should be skipped based on failure strategy.

        Args:
            node_id: Node to check

        Returns:
            True if node should be skipped
        """
        node = self._nodes[node_id]

        if node.run_on_failure:
            return False

        if self._config.failure_strategy == "always_run":
            return False

        if self._config.failure_strategy == "allow_independent":
            return self._has_failed_ancestor(node_id)

        return False

    def _has_failed_ancestor(self, node_id: str) -> bool:
        """
        Check if node has any failed ancestor (transitive).

        Uses BFS to traverse all ancestors and check for failures.
        """
        visited = set()
        queue = deque(self._parents[node_id])

        while queue:
            ancestor_id = queue.popleft()
            if ancestor_id in visited:
                continue
            visited.add(ancestor_id)

            if self._status[ancestor_id] == GraphNodeStatus.FAILED:
                return True

            queue.extend(self._parents[ancestor_id])

        return False

    async def _skip_all_pending_nodes(
        self,
        ready_queue: asyncio.Queue,
        events_queue: asyncio.Queue
    ) -> None:
        """
        Skip all pending nodes that don't have run_on_failure=True (for fail_fast strategy).

        Nodes with run_on_failure=True will continue to be scheduled despite failures.
        This prevents infinite loop when fail_fast is triggered.

        Args:
            ready_queue: Queue containing nodes ready to run
            events_queue: Queue to emit skip events into
        """
        for node_id, status in self._status.items():
            if status == GraphNodeStatus.PENDING:
                node = self._nodes[node_id]
                if not node.run_on_failure:
                    self._status[node_id] = GraphNodeStatus.SKIPPED
                    await events_queue.put(GraphNodeCompleteEvent(
                        self._config.graph_id,
                        node_id,
                        GraphNodeStatus.SKIPPED,
                        "Skipped due to fail_fast after upstream failure"
                    ))

    async def _handle_node_completion(
        self,
        node_id: str,
        ready_queue: asyncio.Queue,
        events_queue: asyncio.Queue
    ) -> None:
        """
        Handle node completion: update children indegrees and ready queue.

        Args:
            node_id: Completed node ID
            ready_queue: Queue of nodes ready to execute
            events_queue: Event queue (unused, for future extension)
        """
        for child_id in self._children.get(node_id, []):
            self._indegree[child_id] -= 1

            if self._indegree[child_id] == 0:
                await ready_queue.put(child_id)

    async def _execute_node(
        self,
        node_id: str,
        events_q: asyncio.Queue
    ) -> None:
        """
        Execute a single node and emit events.

        Handles:
        - AgentRunner: Streams AgentEvents
        - LogicRunner: Streams LogicEvents
        - Callable: Executes custom function
        - Retry and rate limiting
        - Output key writing on success

        Args:
            node_id: Node to execute
            events_q: Queue to put events into
        """
        node = self._nodes[node_id]

        async def node_stream() -> AsyncIterator[BaseEvent]:
            """Create event stream from node executable."""
            if isinstance(node.executable, AgentRunner):
                async for ev in node.executable.step_stream():
                    yield ev

            elif isinstance(node.executable, LogicRunner):
                async for ev in node.executable.run_stream():
                    yield ev

            elif callable(node.executable):
                try:
                    res = node.executable(self._context)

                    if hasattr(res, "__aiter__"):
                        async for ev in res:
                            if not isinstance(ev, BaseEvent):
                                raise TypeError(
                                    f"Callable for node {node_id!r} yielded non-BaseEvent: {type(ev).__name__}. "
                                    f"Must yield BaseEvent instances."
                                )
                            yield ev
                    else:
                        if asyncio.iscoroutine(res):
                            res = await res

                        if node.output_key is not None:
                            output_value = str(res) if res is not None else ""
                            self._context.set(node.output_key, output_value)

                except Exception as e:
                    raise

            else:
                raise TypeError(
                    f"Node {node_id!r} executable must be AgentRunner, LogicRunner, or Callable, "
                    f"got: {type(node.executable).__name__}"
                )

        stream_fn = node_stream
        if node.retry_config is not None:
            base_stream_fn = stream_fn
            async def retry_wrapped() -> AsyncIterator[BaseEvent]:
                async for item in retry_stream(
                    base_stream_fn,
                    node.retry_config,
                    operation_name=node_id,
                    operation_type="graph_node"
                ):
                    yield item
            stream_fn = retry_wrapped

        if self._rate_limiter is not None:
            base_stream_fn = stream_fn 
            async def rl_wrapped() -> AsyncIterator[BaseEvent]:
                async for item in rate_limited_stream(
                    base_stream_fn,
                    self._rate_limiter,
                    operation_name=f"graph_node:{node_id}"
                ):
                    yield item
            stream_fn = rl_wrapped

        last_result: AgentStepResult | None = None
        error_message: str | None = None
        status = GraphNodeStatus.COMPLETED

        try:
            async for event in stream_fn():
                try:
                    setattr(event, "graph_node_id", node_id)
                except Exception:
                    pass

                await events_q.put(event)

                if isinstance(event, StepCompleteEvent):
                    last_result = event.result

            if isinstance(node.executable, (AgentRunner, LogicRunner)) and last_result is not None:
                if last_result.status in (AgentStatus.OK, AgentStatus.DONE, AgentStatus.TOOL_EXECUTED):
                    status = GraphNodeStatus.COMPLETED

                    if node.output_key is not None:
                        selector = node.output_selector or (
                            lambda r: (r.segments.response if r.segments.response else r.raw_output)
                        )
                        output_value = selector(last_result)
                        self._context.set(node.output_key, output_value)

                else:
                    error_message = last_result.error_message or f"Agent failed with status: {last_result.status.value}"

                    if node.failure_mode == "soft_fail":
                        status = GraphNodeStatus.COMPLETED
                        logger.warning("graph.node.soft_fail", extra={
                            "graph_id": self._config.graph_id,
                            "node_id": node_id,
                            "error": error_message
                        })
                    else:
                        status = GraphNodeStatus.FAILED

            elif not isinstance(node.executable, (AgentRunner, LogicRunner)):
                status = GraphNodeStatus.COMPLETED

        except Exception as e:
            error_message = f"Node execution failed: {str(e)}"

            if node.failure_mode == "soft_fail":
                status = GraphNodeStatus.COMPLETED
                logger.warning("graph.node.soft_fail", extra={
                    "graph_id": self._config.graph_id,
                    "node_id": node_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            else:
                status = GraphNodeStatus.FAILED
                logger.error("graph.node.error", extra={
                    "graph_id": self._config.graph_id,
                    "node_id": node_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }, exc_info=True)

        await events_q.put(_NodeFinishedInternalEvent(node_id, status, error_message))
