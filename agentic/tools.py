"""
Tool abstraction with multi-mode execution support.
"""
from dataclasses import dataclass
from typing import Callable, Any, AsyncIterator, TYPE_CHECKING
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError

from .core import ProcessingMode, ToolResult
from .events import ToolOutputEvent
from .logging_util import get_logger

if TYPE_CHECKING:
    from .validation import ValidatorRegistry, ValidationError


@dataclass
class ToolDefinition:
    """Metadata and configuration for a tool."""
    name: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    timeout_seconds: float = 30.0
    processing_mode: ProcessingMode | None = None  # None means inherit from parent
    description: str = ""


logger = get_logger(__name__)


class Tool:
    """
    Executable tool with support for multiple processing modes and native async callables.
    """

    def __init__(
        self,
        definition: ToolDefinition,
        callable_func: Callable[[dict], dict | str | bytes],
        validator_registry: "ValidatorRegistry | None" = None
    ):
        self._definition = definition
        self._callable = callable_func
        self._validator_registry = validator_registry
        self._is_async = asyncio.iscoroutinefunction(callable_func)

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def validate_arguments(self, arguments: dict) -> tuple[bool, list["ValidationError"]]:
        """
        Validate tool arguments against input schema.

        Returns:
            (is_valid, errors) - is_valid is True if no errors
        """
        if not self._validator_registry or not self._definition.input_schema:
            return True, []

        return self._validator_registry.validate(arguments, self._definition.input_schema)

    def run(self, inputs: dict[str, Any], iteration: int, processing_mode: ProcessingMode | None = None) -> ToolResult:
        """Execute tool with inputs. Handles timeout and execution mode automatically."""
        start_time = time.time()

        effective_mode = self._definition.processing_mode if self._definition.processing_mode else processing_mode
        if effective_mode is None:
            effective_mode = ProcessingMode.THREAD

        logger.debug("tool.execute.start", extra={
            "tool_name": self._definition.name,
            "iteration": iteration,
            "processing_mode": effective_mode.value
        })

        try:
            if effective_mode == ProcessingMode.PROCESS:
                output = self._run_in_process(inputs)
            elif effective_mode == ProcessingMode.THREAD:
                output = self._run_in_thread(inputs)
            elif effective_mode == ProcessingMode.ASYNC:
                output = self._run_async(inputs)
            else:
                output = self._run_sync(inputs)

            execution_time = time.time() - start_time

            logger.debug("tool.execute.complete", extra={
                "tool_name": self._definition.name,
                "iteration": iteration,
                "execution_time_ms": round(execution_time * 1000, 2),
                "success": True
            })

            return ToolResult(
                name=self._definition.name,
                output=output,
                success=True,
                error_message=None,
                execution_time=execution_time,
                iteration=iteration
            )

        except (TimeoutError, asyncio.TimeoutError) as e:
            execution_time = time.time() - start_time

            logger.warning("tool.execute.timeout", extra={
                "tool_name": self._definition.name,
                "iteration": iteration,
                "timeout_seconds": self._definition.timeout_seconds,
                "execution_time_ms": round(execution_time * 1000, 2)
            })

            return ToolResult(
                name=self._definition.name,
                output=None,
                success=False,
                error_message=f"Tool execution timed out after {self._definition.timeout_seconds}s",
                execution_time=execution_time,
                iteration=iteration
            )

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error("tool.execute.error", extra={
                "tool_name": self._definition.name,
                "iteration": iteration,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": round(execution_time * 1000, 2)
            }, exc_info=True)

            return ToolResult(
                name=self._definition.name,
                output=None,
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                execution_time=execution_time,
                iteration=iteration
            )

    async def run_stream(
        self,
        inputs: dict[str, Any],
        iteration: int,
        processing_mode: ProcessingMode | None = None
    ) -> AsyncIterator[ToolOutputEvent]:
        """
        Execute tool with streaming output. Supports native async callables.

        If the tool callable has a run_stream method, uses it for true streaming.
        If the callable is async, calls it directly.
        Otherwise wraps run() and yields a single output event.

        Yields:
            ToolOutputEvent - Tool output events (partial or complete)
        """
        if hasattr(self._callable, 'run_stream') and callable(getattr(self._callable, 'run_stream')):
            async for output_chunk in self._callable.run_stream(inputs):
                yield ToolOutputEvent(
                    tool_name=self._definition.name,
                    output=output_chunk,
                    is_partial=True
                )
        elif self._is_async:
            result = await self._callable(inputs)
            yield ToolOutputEvent(
                tool_name=self._definition.name,
                output=result,
                is_partial=False
            )
        else:
            # Sync callable - run in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.run(inputs, iteration, processing_mode)
            )

            if not result.success:
                raise RuntimeError(result.error_message or "Tool execution failed")

            yield ToolOutputEvent(
                tool_name=self._definition.name,
                output=result.output,
                is_partial=False
            )

    def _run_sync(self, inputs: dict[str, Any]) -> dict | str | bytes:
        return self._callable(inputs)

    def _run_in_thread(self, inputs: dict[str, Any]) -> dict | str | bytes:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._callable, inputs)
            try:
                return future.result(timeout=self._definition.timeout_seconds)
            except Exception as e:
                future.cancel()
                raise e

    def _run_in_process(self, inputs: dict[str, Any]) -> dict | str | bytes:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._callable, inputs)
            try:
                return future.result(timeout=self._definition.timeout_seconds)
            except Exception as e:
                future.cancel()
                raise e

    def _run_async(self, inputs: dict[str, Any]) -> dict | str | bytes:
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call sync _run_async from within an async context. "
                "Use async/await pattern instead."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e) or "no current event loop" in str(e):
                return asyncio.run(
                    asyncio.wait_for(
                        self._async_wrapper(inputs),
                        timeout=self._definition.timeout_seconds
                    )
                )
            else:
                raise

    async def _async_wrapper(self, inputs: dict[str, Any]) -> dict | str | bytes:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._callable, inputs)


class ToolRegistry:
    """Registry for all available tools with optional validator injection."""

    def __init__(self, validator_registry: "ValidatorRegistry | None" = None):
        self._tools: dict[str, Tool] = {}
        self._validator_registry = validator_registry

    def register(self, tool: Tool) -> None:
        """Register tool and inject validator if tool doesn't have one."""
        if self._validator_registry and not tool._validator_registry:
            tool._validator_registry = self._validator_registry
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def exists(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> list[str]:
        return sorted(self._tools.keys())

    def get_definitions(self) -> dict[str, ToolDefinition]:
        """Get all tool definitions for schema generation."""
        return {name: tool.definition for name, tool in self._tools.items()}

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False


def create_tool(
    name: str,
    func: Callable[[dict], dict | str | bytes],
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    timeout_seconds: float = 30.0,
    processing_mode: ProcessingMode = ProcessingMode.THREAD,
    description: str = ""
) -> Tool:
    definition = ToolDefinition(
        name=name,
        input_schema=input_schema or {},
        output_schema=output_schema or {},
        timeout_seconds=timeout_seconds,
        processing_mode=processing_mode,
        description=description
    )
    return Tool(definition, func)
