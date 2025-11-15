"""
Agent abstraction and execution runner.
"""
from typing import Protocol, AsyncIterator, TYPE_CHECKING
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .core import AgentConfig, AgentStatus, AgentStepResult, ExtractedSegments, ToolResult, ToolCall, ToolExecutionDecision, ProcessingMode, new_uuid, PromptType, serialize_tool_output
from .context import ContextManager
from .patterns import PatternRegistry, PatternExtractor, StreamingPatternExtractor
from .tools import ToolRegistry
from .events import (
    AgentEvent, LLMChunkEvent, LLMCompleteEvent, StatusEvent,
    ToolStartEvent, ToolEndEvent, ToolValidationEvent, ToolDecisionEvent,
    ContextWriteEvent, ErrorEvent, StepCompleteEvent,
    PatternStartEvent, PatternContentEvent, PatternEndEvent
)
from .logging_util import get_logger

if TYPE_CHECKING:
    from .validation import ValidationError


class LLMProvider(Protocol):
    """
    Protocol for LLM provider implementations.

    Providers can implement streaming or non-streaming generation.
    If stream() is not implemented, framework will simulate streaming
    by emitting the full generate() output as a single chunk.

    The prompt parameter accepts PromptType (Any). Providers interpret structure.
    """

    def generate(self, prompt: PromptType, **kwargs) -> str:
        """
        Generate complete text from prompt (blocking).

        Args:
            prompt: Prompt in any format the provider supports
            **kwargs: Provider-specific options (model, temperature, max_tokens, etc.)

        Returns:
            Generated text
        """
        ...

    async def stream(self, prompt: PromptType, **kwargs) -> AsyncIterator[str]:
        """
        Stream chunks from prompt (optional).

        If not implemented, framework falls back to generate()
        and simulates streaming.

        Args:
            prompt: Prompt in any format the provider supports
            **kwargs: Provider-specific options (model, temperature, max_tokens, etc.)

        Yields:
            Text chunks
        """
        text = self.generate(prompt, **kwargs)
        yield text


class Agent:
    """Manages agent configuration, context, patterns, tools, and LLM provider."""

    def __init__(
        self,
        config: AgentConfig,
        context: ContextManager,
        patterns: PatternRegistry,
        tools: ToolRegistry,
        provider_client: LLMProvider
    ):
        self._config = config
        self._context = context
        self._patterns = patterns
        self._tools = tools
        self._provider = provider_client

    def get_id(self) -> str:
        return self._config.agent_id

    def get_config(self) -> AgentConfig:
        return self._config

    def set_config(self, config: AgentConfig) -> None:
        self._config = config

    @property
    def context(self) -> ContextManager:
        return self._context

    @property
    def patterns(self) -> PatternRegistry:
        return self._patterns

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def provider(self) -> LLMProvider:
        return self._provider


logger = get_logger(__name__)


class AgentRunner:
    """
    Executes agent steps: prompt building, LLM generation, tool execution, context updates.
    """

    def __init__(self, agent: Agent):
        self._agent = agent

    def _create_tool_not_allowed_error(self, tool_name: str, iteration: int, call_id: str = "") -> ToolResult:
        """Create error result for tool not in allowed list."""
        return ToolResult(
            name=tool_name,
            output=None,
            success=False,
            error_message=f"Tool '{tool_name}' not in allowed list",
            execution_time=0.0,
            iteration=iteration,
            call_id=call_id
        )

    def _create_tool_not_found_error(self, tool_name: str, iteration: int, call_id: str = "") -> ToolResult:
        """Create error result for tool not found in registry."""
        return ToolResult(
            name=tool_name,
            output=None,
            success=False,
            error_message=f"Tool '{tool_name}' not found in registry",
            execution_time=0.0,
            iteration=iteration,
            call_id=call_id
        )

    def _create_tool_validation_error(
        self,
        tool_name: str,
        errors: list["ValidationError"],
        iteration: int,
        call_id: str = ""
    ) -> ToolResult:
        """Create error result for failed validation."""
        error_msg = "; ".join([f"{e.field}: {e.message}" for e in errors])
        return ToolResult(
            name=tool_name,
            output={"validation_errors": [{"field": e.field, "message": e.message, "value": e.value} for e in errors]},
            success=False,
            error_message=f"Argument validation failed: {error_msg}",
            execution_time=0.0,
            iteration=iteration,
            call_id=call_id
        )

    def _resolve_tool_name(self, public_name: str) -> str:
        """
        Resolve public tool name to internal registry name.

        Uses tool_name_mapping from config to map public names (that LLMs see)
        to internal registry names.

        Args:
            public_name: Tool name from LLM output

        Returns:
            Internal tool name for registry lookup
        """
        config = self._agent.get_config()
        return config.tool_name_mapping.get(public_name, public_name)

    def step(self, user_input: str | None = None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        """
        Execute a single agent step (batch mode).

        This is a convenience wrapper around step_stream() that aggregates
        all events and returns the final result.
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "AgentRunner.step() cannot be called from an async context. "
                "Use 'await step_stream()' instead, or call from a synchronous context."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                raise

        return asyncio.run(self._collect_step_events(user_input, processing_mode))

    async def _collect_step_events(
        self,
        user_input: str | None,
        processing_mode: ProcessingMode | None
    ) -> AgentStepResult:
        """Helper to collect all events into final result."""
        final_result = None
        async for event in self.step_stream(user_input, processing_mode):
            if isinstance(event, StepCompleteEvent):
                final_result = event.result
        return final_result

    async def _should_execute_tool(
        self,
        tool_call: ToolCall,
        config: AgentConfig,
        step_id: str
    ) -> ToolExecutionDecision:
        """
        Determine if a tool should be executed and create decision record.

        Handles verification workflow if on_tool_detected callback is configured.
        Tracks verification duration and creates complete decision object.
        Supports timeout via tool_verification_timeout config.

        Returns:
            ToolExecutionDecision with acceptance status and metadata
        """
        verification_required = config.on_tool_detected is not None

        # Create decision object
        decision = ToolExecutionDecision(
            tool_call=tool_call,
            verification_required=verification_required,
            accepted=True,  # Default to accepted if no callback
            rejection_reason=None,
            verification_duration_ms=0.0,
            executed=False,
            result=None
        )

        if verification_required:
            start_time = time.time()

            logger.debug("tool.verification.start", extra={
                "tool_name": tool_call.name,
                "call_id": tool_call.call_id,
                "step_id": step_id
            })

            try:
                loop = asyncio.get_event_loop()
                callback_coro = loop.run_in_executor(None, config.on_tool_detected, tool_call)

                if config.tool_verification_timeout is not None:
                    accepted = await asyncio.wait_for(callback_coro, timeout=config.tool_verification_timeout)
                else:
                    accepted = await callback_coro

                decision.verification_duration_ms = (time.time() - start_time) * 1000
                decision.accepted = accepted

                if not accepted:
                    decision.rejection_reason = "Rejected by callback"

            except asyncio.TimeoutError:
                decision.verification_duration_ms = (time.time() - start_time) * 1000
                decision.accepted = config.tool_verification_on_timeout == "accept"
                decision.rejection_reason = f"Verification timeout after {config.tool_verification_timeout}s"

                logger.warning("tool.verification.timeout", extra={
                    "tool_name": tool_call.name,
                    "call_id": tool_call.call_id,
                    "timeout_seconds": config.tool_verification_timeout,
                    "on_timeout_action": config.tool_verification_on_timeout
                })

            except Exception as e:
                decision.verification_duration_ms = (time.time() - start_time) * 1000
                decision.accepted = False
                decision.rejection_reason = f"Callback exception: {str(e)}"

                logger.error("tool.verification.error", extra={
                    "tool_name": tool_call.name,
                    "call_id": tool_call.call_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }, exc_info=True)

        return decision

    def _determine_step_status(
        self,
        error_message: str | None,
        tool_decisions: list[ToolExecutionDecision],
        segments: ExtractedSegments
    ) -> AgentStatus:
        """
        Determine final agent status based on tool decisions.

        Uses rich decision data instead of simple counts for accurate status.

        Args:
            error_message: Error message if any error occurred
            tool_decisions: All tool execution decisions
            segments: Extracted segments from LLM output

        Returns:
            AgentStatus enum value
        """
        if error_message:
            return AgentStatus.ERROR

        # Analyze decisions
        accepted_count = sum(1 for d in tool_decisions if d.accepted)
        rejected_count = sum(1 for d in tool_decisions if not d.accepted)
        executed_count = sum(1 for d in tool_decisions if d.executed)
        pending_count = sum(1 for d in tool_decisions if d.accepted and not d.executed)

        # Status priority order
        if executed_count > 0:
            return AgentStatus.TOOL_EXECUTED
        elif pending_count > 0:
            # Tools accepted but not yet executed (concurrent mode)
            return AgentStatus.WAITING_FOR_TOOL
        elif rejected_count > 0 and accepted_count == 0:
            # All tools rejected
            return AgentStatus.TOOLS_REJECTED
        elif not segments.response and not tool_decisions:
            return AgentStatus.DONE
        else:
            return AgentStatus.OK

    async def step_stream(
        self,
        user_input: str | None = None,
        processing_mode: ProcessingMode | None = None
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute agent step with streaming events.

        Yields events as execution progresses:
        - LLMChunkEvent: As LLM generates chunks
        - LLMCompleteEvent: When LLM completes
        - PatternStartEvent, PatternContentEvent, PatternEndEvent: As patterns detected
        - StatusEvent: When status changes
        - ToolStartEvent, ToolOutputEvent, ToolEndEvent: During tool execution
        - ContextWriteEvent: When context is updated (if incremental_context_writes=True)
        - ErrorEvent: If errors occur
        - StepCompleteEvent: Final result with aggregated data
        """
        config = self._agent.get_config()
        effective_mode = processing_mode if processing_mode is not None else config.processing_mode

        if config.auto_increment_iteration:
            current_iteration = self._agent.context.next_iteration()
        else:
            current_iteration = self._agent.context.get_iteration()

        step_id = new_uuid()
        prompt = self._build_prompt(user_input)

        logger.debug("agent.step.start", extra={
            "agent_id": config.agent_id,
            "iteration": current_iteration,
            "step_id": step_id,
            "processing_mode": effective_mode.value if effective_mode else None
        })

        yield StatusEvent(AgentStatus.OK, "Starting agent step", step_id=step_id)

        pattern_set_name = config.pattern_set or "default"
        pattern_set = self._agent.patterns.get_pattern_set(pattern_set_name)

        if pattern_set is None:
            pattern_extractor = None
        else:
            pattern_extractor = StreamingPatternExtractor(
                pattern_set=pattern_set,
                stream_content=config.stream_pattern_content
            )

        raw_output_buffer = []
        detected_tools: list[ToolCall] = []
        tool_decisions: list[ToolExecutionDecision] = []  # Track full tool lifecycle
        tool_execution_tasks: list[asyncio.Task] = []
        tool_results: list[ToolResult] = []
        pattern_counters: dict[str, int] = {}
        tool_event_queue: asyncio.Queue = asyncio.Queue()

        try:
            async for chunk in self._agent.provider.stream(prompt=prompt):
                raw_output_buffer.append(chunk)
                yield LLMChunkEvent(chunk, step_id=step_id)

                while not tool_event_queue.empty():
                    try:
                        event = tool_event_queue.get_nowait()
                        yield event
                    except asyncio.QueueEmpty:
                        break

                if config.incremental_context_writes:
                    partial_output = "".join(raw_output_buffer)
                    streaming_key = f"llm_streaming:{current_iteration}"
                    self._agent.context.update(streaming_key, partial_output, iteration=current_iteration)

                if pattern_extractor:
                    for event_data in pattern_extractor.feed_chunk(chunk):
                        event_type = event_data[0]

                        if event_type == "pattern_start":
                            _, pattern_name, pattern_type = event_data
                            yield PatternStartEvent(pattern_name, pattern_type, step_id=step_id)

                            if pattern_type == "tool":
                                yield StatusEvent(AgentStatus.WAITING_FOR_TOOL, f"Tool pattern detected: {pattern_name}", step_id=step_id)

                        elif event_type == "pattern_content":
                            _, pattern_name, content = event_data
                            yield PatternContentEvent(pattern_name, content, is_partial=True, step_id=step_id)

                            if config.incremental_context_writes:
                                partial_key = f"pattern_partial:{pattern_name}:{current_iteration}"
                                existing = self._agent.context.get(partial_key)
                                if existing:
                                    accumulated = existing + content
                                else:
                                    accumulated = content
                                self._agent.context.update(partial_key, accumulated, iteration=current_iteration)

                        elif event_type == "pattern_end":
                            _, pattern_name, pattern_type, full_content, tool_call = event_data
                            yield PatternEndEvent(pattern_name, pattern_type, full_content, step_id=step_id)

                            if pattern_type not in pattern_counters:
                                pattern_counters[pattern_type] = 0
                            pattern_key = f"pattern:{pattern_type}:{current_iteration}:{pattern_counters[pattern_type]}"
                            self._agent.context.set(pattern_key, full_content, iteration=current_iteration)
                            pattern_counters[pattern_type] += 1

                            if config.incremental_context_writes:
                                partial_key = f"pattern_partial:{pattern_name}:{current_iteration}"
                                self._agent.context.delete(partial_key)

                            if tool_call:
                                detected_tools.append(tool_call)

                                if config.concurrent_tool_execution:
                                    if config.on_tool_detected is not None:
                                        yield StatusEvent(
                                            AgentStatus.WAITING_FOR_VERIFICATION,
                                            f"Awaiting verification for tool '{tool_call.name}'",
                                            step_id=step_id
                                        )

                                    # Create and track decision
                                    decision = await self._should_execute_tool(tool_call, config, step_id)
                                    tool_decisions.append(decision)

                                    # Emit decision event
                                    yield ToolDecisionEvent(
                                        tool_name=tool_call.name,
                                        call_id=tool_call.call_id,
                                        accepted=decision.accepted,
                                        rejection_reason=decision.rejection_reason,
                                        verification_duration_ms=decision.verification_duration_ms,
                                        step_id=step_id
                                    )

                                    if decision.accepted:
                                        yield StatusEvent(AgentStatus.WAITING_FOR_TOOL, f"Starting concurrent execution of tool '{tool_call.name}'", step_id=step_id)
                                        task = asyncio.create_task(
                                            self._execute_single_tool_concurrent(
                                                tool_call, current_iteration, effective_mode,
                                                tool_results, tool_event_queue, step_id
                                            )
                                        )
                                        tool_execution_tasks.append(task)
                                    else:
                                        reason_text = decision.rejection_reason or "No reason provided"
                                        logger.debug("agent.tool.rejected", extra={
                                            "agent_id": config.agent_id,
                                            "iteration": current_iteration,
                                            "step_id": step_id,
                                            "tool_name": tool_call.name,
                                            "reason": reason_text,
                                            "verification_duration_ms": decision.verification_duration_ms
                                        })
                                        yield StatusEvent(
                                            AgentStatus.OK,
                                            f"Tool '{tool_call.name}' rejected: {reason_text}",
                                            step_id=step_id
                                        )

            raw_output = "".join(raw_output_buffer)

            logger.debug("agent.llm.complete", extra={
                "agent_id": config.agent_id,
                "iteration": current_iteration,
                "step_id": step_id,
                "output_length": len(raw_output),
                "tools_detected": len(detected_tools)
            })

            yield LLMCompleteEvent(raw_output, step_id=step_id)

            if tool_execution_tasks:
                yield StatusEvent(AgentStatus.WAITING_FOR_TOOL, f"Waiting for {len(tool_execution_tasks)} concurrent tool(s) to complete", step_id=step_id)
                await asyncio.gather(*tool_execution_tasks, return_exceptions=True)

                while not tool_event_queue.empty():
                    try:
                        event = tool_event_queue.get_nowait()
                        yield event
                    except asyncio.QueueEmpty:
                        break

        except Exception as e:
            for task in tool_execution_tasks:
                if not task.done():
                    task.cancel()

            logger.error("agent.llm.error", extra={
                "agent_id": config.agent_id,
                "iteration": current_iteration,
                "step_id": step_id,
                "error": str(e),
                "error_type": type(e).__name__
            }, exc_info=True)

            yield ErrorEvent("llm_error", str(e), recoverable=False, step_id=step_id)
            yield StepCompleteEvent(AgentStepResult(
                status=AgentStatus.ERROR,
                raw_output=f"LLM Error: {str(e)}",
                segments=ExtractedSegments(),
                tool_results=[],
                iteration=current_iteration,
                error_message=str(e),
                error_type="llm_error"
            ), step_id=step_id)
            return

        if pattern_extractor:
            segments, malformed_patterns = pattern_extractor.finalize(iteration=current_iteration)
            parse_error_keys = set(segments.parse_errors.keys()) if segments.parse_errors else set()

            if malformed_patterns:
                for pattern_name, partial_content in malformed_patterns.items():
                    if pattern_name in parse_error_keys:
                        continue

                    yield ErrorEvent(
                        error_type="malformed_pattern",
                        error_message=f"Pattern '{pattern_name}' missing end tag",
                        recoverable=True,
                        partial_data=partial_content,
                        step_id=step_id
                    )

                    if config.incremental_context_writes:
                        partial_key = f"pattern_partial:{pattern_name}:{current_iteration}"
                        self._agent.context.delete(partial_key)
        else:
            segments = ExtractedSegments(response=raw_output)
            malformed_patterns = None

        if segments.parse_errors:
            for error_key, error_text in segments.parse_errors.items():
                preview_line = ""
                if error_text:
                    lines = error_text.strip().splitlines()
                    if lines:
                        preview_line = lines[0]
                        if len(preview_line) > 200:
                            preview_line = f"{preview_line[:197]}..."
                    else:
                        preview_line = "Unknown parse error"
                else:
                    preview_line = "Unknown parse error"

                logger.warning("agent.pattern.parse_error", extra={
                    "agent_id": config.agent_id,
                    "iteration": current_iteration,
                    "step_id": step_id,
                    "pattern_key": error_key
                })

                yield ErrorEvent(
                    error_type="pattern_parse_error",
                    error_message=f"Pattern '{error_key}' parse error: {preview_line}",
                    recoverable=True,
                    partial_data=error_text,
                    step_id=step_id
                )

        if not config.concurrent_tool_execution and detected_tools:
            tools_to_execute = []
            for tool_call in detected_tools:
                # Emit verification status if verification required
                if config.on_tool_detected is not None:
                    yield StatusEvent(
                        AgentStatus.WAITING_FOR_VERIFICATION,
                        f"Awaiting verification for tool '{tool_call.name}'",
                        step_id=step_id
                    )

                # Create and track decision
                decision = await self._should_execute_tool(tool_call, config, step_id)
                tool_decisions.append(decision)

                # Emit decision event
                yield ToolDecisionEvent(
                    tool_name=tool_call.name,
                    call_id=tool_call.call_id,
                    accepted=decision.accepted,
                    rejection_reason=decision.rejection_reason,
                    verification_duration_ms=decision.verification_duration_ms,
                    step_id=step_id
                )

                if decision.accepted:
                    tools_to_execute.append(tool_call)
                else:
                    reason_text = decision.rejection_reason or "No reason provided"
                    logger.debug("agent.tool.rejected", extra={
                        "agent_id": config.agent_id,
                        "iteration": current_iteration,
                        "step_id": step_id,
                        "tool_name": tool_call.name,
                        "reason": reason_text,
                        "verification_duration_ms": decision.verification_duration_ms
                    })
                    yield StatusEvent(
                        AgentStatus.OK,
                        f"Tool '{tool_call.name}' rejected: {reason_text}",
                        step_id=step_id
                    )

            if tools_to_execute:
                yield StatusEvent(AgentStatus.WAITING_FOR_TOOL, f"Executing {len(tools_to_execute)} approved tool(s)", step_id=step_id)

                async for event in self._execute_tools_stream(tools_to_execute, current_iteration, effective_mode, step_id):
                    yield event
                    if isinstance(event, ToolEndEvent):
                        tool_results.append(event.result)

        tool_execution_failed = any(not tr.success for tr in tool_results)

        self._update_context_from_output(raw_output, segments, tool_results, current_iteration)

        if config.incremental_context_writes:
            for context_key, _ in config.output_mapping:
                record = self._agent.context.get_record(context_key)
                if record:
                    try:
                        value = record.value.decode('utf-8')
                        preview = value[:100] if len(value) < 100 else value[:97] + "..."
                        yield ContextWriteEvent(
                            key=context_key,
                            value_preview=preview,
                            version=record.version,
                            iteration=record.iteration,
                            step_id=step_id
                        )
                    except UnicodeDecodeError:
                        pass

        # Match tool results back to decisions and mark as executed
        for tool_result in tool_results:
            for decision in tool_decisions:
                if tool_result.call_id and decision.tool_call.call_id:
                    if tool_result.call_id == decision.tool_call.call_id:
                        decision.executed = True
                        decision.result = tool_result
                        break
                elif not tool_result.call_id and not decision.tool_call.call_id:
                    if decision.tool_call.name == tool_result.name and not decision.executed:
                        decision.executed = True
                        decision.result = tool_result
                        break

        error_message = None
        error_type = None

        if tool_execution_failed:
            failed_tools = [tr for tr in tool_results if not tr.success]
            if failed_tools:
                first_failure = failed_tools[0]
                error_message = first_failure.error_message
                if "not in allowed list" in (error_message or ""):
                    error_type = "tool_not_allowed"
                elif "not found in registry" in (error_message or ""):
                    error_type = "tool_not_found"
                elif "timed out" in (error_message or ""):
                    error_type = "tool_timeout"
                else:
                    error_type = "tool_execution_error"
                if len(failed_tools) > 1:
                    error_message = f"{error_message} (and {len(failed_tools) - 1} other tool(s) failed)"
            yield ErrorEvent(error_type, error_message, recoverable=False, step_id=step_id)

        status = self._determine_step_status(error_message, tool_decisions, segments)

        logger.debug("agent.step.complete", extra={
            "agent_id": config.agent_id,
            "iteration": current_iteration,
            "step_id": step_id,
            "status": status.value,
            "tools_executed": len(tool_results),
            "tools_succeeded": sum(1 for tr in tool_results if tr.success),
            "has_error": error_message is not None
        })

        yield StatusEvent(status, "Agent step complete", step_id=step_id)

        final_result = AgentStepResult(
            status=status,
            raw_output=raw_output,
            segments=segments,
            tool_results=tool_results,
            iteration=current_iteration,
            error_message=error_message,
            error_type=error_type,
            partial_malformed_patterns=malformed_patterns,
            tool_decisions=tool_decisions
        )
        yield StepCompleteEvent(final_result, step_id=step_id)

    def _step_impl(self, user_input: str | None = None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        """Internal synchronous implementation of agent step."""
        config = self._agent.get_config()
        effective_mode = processing_mode if processing_mode is not None else config.processing_mode

        if config.auto_increment_iteration:
            current_iteration = self._agent.context.next_iteration()
        else:
            current_iteration = self._agent.context.get_iteration()

        logger.debug("agent.step.start", extra={
            "agent_id": config.agent_id,
            "iteration": current_iteration,
            "processing_mode": effective_mode.value if effective_mode else None
        })

        prompt = self._build_prompt(user_input)

        try:
            raw_output = self._agent.provider.generate(prompt=prompt)
        except Exception as e:
            return AgentStepResult(
                status=AgentStatus.ERROR,
                raw_output=f"LLM Error: {str(e)}",
                segments=ExtractedSegments(),
                tool_results=[],
                iteration=current_iteration,
                error_message=str(e),
                error_type="llm_error"
            )

        segments = self._extract_segments(raw_output, current_iteration)

        logger.debug("agent.llm.complete", extra={
            "agent_id": config.agent_id,
            "iteration": current_iteration,
            "output_length": len(raw_output),
            "tools_detected": len(segments.tools)
        })

        tool_results = []
        tool_execution_failed = False

        if segments.tools:
            tool_results = self._execute_tools(segments.tools, current_iteration, processing_mode=effective_mode)
            tool_execution_failed = any(not tr.success for tr in tool_results)

        self._update_context_from_output(raw_output, segments, tool_results, current_iteration)

        error_message = None
        error_type = None

        if tool_execution_failed:
            status = AgentStatus.ERROR
            failed_tools = [tr for tr in tool_results if not tr.success]
            if failed_tools:
                first_failure = failed_tools[0]
                error_message = first_failure.error_message
                if "not in allowed list" in (error_message or ""):
                    error_type = "tool_not_allowed"
                elif "not found in registry" in (error_message or ""):
                    error_type = "tool_not_found"
                elif "timed out" in (error_message or ""):
                    error_type = "tool_timeout"
                else:
                    error_type = "tool_execution_error"

                if len(failed_tools) > 1:
                    error_message = f"{error_message} (and {len(failed_tools) - 1} other tool(s) failed)"
        elif segments.tools and not tool_execution_failed:
            status = AgentStatus.TOOL_EXECUTED
        elif not segments.response and not segments.tools:
            status = AgentStatus.DONE
        else:
            status = AgentStatus.OK

        logger.debug("agent.step.complete", extra={
            "agent_id": config.agent_id,
            "iteration": current_iteration,
            "status": status.value,
            "tools_executed": len([tr for tr in tool_results if tr.success]),
            "has_error": error_message is not None
        })

        return AgentStepResult(
            status=status,
            raw_output=raw_output,
            segments=segments,
            tool_results=tool_results,
            iteration=current_iteration,
            error_message=error_message,
            error_type=error_type
        )

    def _build_prompt(self, user_input: str | None) -> PromptType:
        """Build prompt from context. Delegates to prompt_builder if configured, else concatenates input_mapping entries."""
        config = self._agent.get_config()

        if config.prompt_builder is not None:
            return config.prompt_builder(self._agent.context, config, user_input)

        parts = []
        for entry in config.input_mapping:
            context_key = entry.get("context_key", "")
            if context_key.startswith("literal:"):
                parts.append(context_key[8:])
            else:
                value = self._agent.context.get(context_key)
                if value is not None:
                    parts.append(value)

        if user_input:
            parts.append(user_input)

        return "\n\n".join(parts)

    def _extract_segments(self, output: str, iteration: int) -> ExtractedSegments:
        """Extract structured segments from LLM output using agent's pattern set."""
        config = self._agent.get_config()
        pattern_set_name = config.pattern_set or "default"
        pattern_set = self._agent.patterns.get_pattern_set(pattern_set_name)

        if pattern_set is None:
            return ExtractedSegments(response=output)

        extractor = PatternExtractor(pattern_set)
        return extractor.extract(output, iteration)

    async def _execute_tools_stream(
        self,
        tool_calls: list[ToolCall],
        iteration: int,
        processing_mode: ProcessingMode | None = None,
        step_id: str = ""
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute tools and yield events for each stage.

        Yields:
        - ToolStartEvent when tool begins
        - ToolOutputEvent for tool output (full or partial)
        - ToolEndEvent when tool completes
        - ErrorEvent if tool fails
        """
        config = self._agent.get_config()
        effective_mode = processing_mode if processing_mode is not None else config.processing_mode

        for tool_index, tool_call in enumerate(tool_calls):
            tool_state_key = f"tool_state:{tool_call.call_id}"
            self._agent.context.set(tool_state_key, b"started", iteration=iteration)

            yield ToolStartEvent(tool_call.name, tool_call.arguments, iteration, tool_call.call_id, step_id=step_id)

            internal_name = self._resolve_tool_name(tool_call.name)

            if internal_name not in config.tools_allowed:
                result = self._create_tool_not_allowed_error(internal_name, iteration, tool_call.call_id)
                yield ErrorEvent("tool_not_allowed", result.error_message, recoverable=True, step_id=step_id)
                yield ToolEndEvent(internal_name, result, tool_call.call_id, step_id=step_id)
                self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
                self._store_tool_result(tool_call.call_id, result, iteration)
                continue

            tool = self._agent.tools.get(internal_name)
            if tool is None:
                result = self._create_tool_not_found_error(internal_name, iteration, tool_call.call_id)
                yield ErrorEvent("tool_not_found", result.error_message, recoverable=True, step_id=step_id)
                yield ToolEndEvent(internal_name, result, tool_call.call_id, step_id=step_id)
                self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
                self._store_tool_result(tool_call.call_id, result, iteration)
                continue

            if config.validate_tool_arguments:
                is_valid, validation_errors = tool.validate_arguments(tool_call.arguments)
                if not is_valid:
                    result = self._create_tool_validation_error(internal_name, validation_errors, iteration, tool_call.call_id)
                    yield ErrorEvent("tool_validation_error", result.error_message, recoverable=True, step_id=step_id)
                    yield ToolValidationEvent(internal_name,
                        [{"field": e.field, "message": e.message, "value": e.value} for e in validation_errors],
                        step_id=step_id)
                    yield ToolEndEvent(internal_name, result, tool_call.call_id, step_id=step_id)
                    self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
                    self._store_tool_result(tool_call.call_id, result, iteration)
                    continue

            start_time = time.time()
            output_chunks = []
            tool_failed = False
            error_message = None

            try:
                async for output_event in tool.run_stream(tool_call.arguments, iteration, effective_mode):
                    output_event.call_id = tool_call.call_id
                    output_event.step_id = step_id
                    yield output_event
                    output_chunks.append(output_event.output)

                execution_time = time.time() - start_time

                if len(output_chunks) == 0:
                    final_output = None
                elif len(output_chunks) == 1:
                    final_output = output_chunks[0]
                else:
                    final_output = output_chunks

                result = ToolResult(
                    name=internal_name,
                    output=final_output,
                    success=True,
                    error_message=None,
                    execution_time=execution_time,
                    iteration=iteration,
                    call_id=tool_call.call_id
                )
                yield ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id)
                self._agent.context.set(tool_state_key, b"finished", iteration=iteration)
                self._store_tool_result(tool_call.call_id, result, iteration)

            except Exception as e:
                execution_time = time.time() - start_time
                result = ToolResult(
                    name=tool_call.name,
                    output=None,
                    success=False,
                    error_message=f"Tool execution failed: {str(e)}",
                    execution_time=execution_time,
                    iteration=iteration,
                    call_id=tool_call.call_id
                )
                yield ErrorEvent("tool_execution_error", result.error_message, recoverable=True, step_id=step_id)
                yield ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id)
                self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
                self._store_tool_result(tool_call.call_id, result, iteration)

    async def _execute_single_tool_concurrent(
        self,
        tool_call: ToolCall,
        iteration: int,
        processing_mode: ProcessingMode | None,
        results_list: list[ToolResult],
        event_queue: asyncio.Queue,
        step_id: str = ""
    ) -> None:
        """
        Execute a single tool concurrently and append result to results_list.

        Used for concurrent tool execution during LLM streaming.
        Emits events to event_queue for consumption by main loop.
        """
        config = self._agent.get_config()
        effective_mode = processing_mode if processing_mode is not None else config.processing_mode

        tool_state_key = f"tool_state:{tool_call.call_id}"
        self._agent.context.set(tool_state_key, b"started", iteration=iteration)

        await event_queue.put(ToolStartEvent(tool_call.name, tool_call.arguments, iteration, tool_call.call_id, step_id=step_id))

        internal_name = self._resolve_tool_name(tool_call.name)

        if internal_name not in config.tools_allowed:
            result = self._create_tool_not_allowed_error(internal_name, iteration, tool_call.call_id)
            await event_queue.put(ErrorEvent("tool_not_allowed", result.error_message, recoverable=True, step_id=step_id))
            await event_queue.put(ToolEndEvent(internal_name, result, tool_call.call_id, step_id=step_id))
            results_list.append(result)

            self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
            self._store_tool_result(tool_call.call_id, result, iteration)
            return

        tool = self._agent.tools.get(internal_name)
        if tool is None:
            result = self._create_tool_not_found_error(internal_name, iteration, tool_call.call_id)
            await event_queue.put(ErrorEvent("tool_not_found", result.error_message, recoverable=True, step_id=step_id))
            await event_queue.put(ToolEndEvent(internal_name, result, tool_call.call_id, step_id=step_id))
            results_list.append(result)

            self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
            self._store_tool_result(tool_call.call_id, result, iteration)
            return

        if config.validate_tool_arguments:
            is_valid, validation_errors = tool.validate_arguments(tool_call.arguments)
            if not is_valid:
                result = self._create_tool_validation_error(internal_name, validation_errors, iteration, tool_call.call_id)
                await event_queue.put(ErrorEvent("tool_validation_error", result.error_message, recoverable=True, step_id=step_id))
                await event_queue.put(ToolValidationEvent(internal_name,
                    [{"field": e.field, "message": e.message, "value": e.value} for e in validation_errors],
                    step_id=step_id))
                await event_queue.put(ToolEndEvent(internal_name, result, tool_call.call_id, step_id=step_id))
                results_list.append(result)

                self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
                self._store_tool_result(tool_call.call_id, result, iteration)
                return

        start_time = time.time()
        output_chunks = []

        try:
            async for output_event in tool.run_stream(tool_call.arguments, iteration, effective_mode):
                output_event.call_id = tool_call.call_id
                output_event.step_id = step_id
                await event_queue.put(output_event)
                output_chunks.append(output_event.output)

            execution_time = time.time() - start_time

            if len(output_chunks) == 0:
                final_output = None
            elif len(output_chunks) == 1:
                final_output = output_chunks[0]
            else:
                final_output = output_chunks

            result = ToolResult(
                name=internal_name,
                output=final_output,
                success=True,
                error_message=None,
                execution_time=execution_time,
                iteration=iteration,
                call_id=tool_call.call_id
            )
            await event_queue.put(ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id))
            results_list.append(result)

            self._agent.context.set(tool_state_key, b"finished", iteration=iteration)
            self._store_tool_result(tool_call.call_id, result, iteration)

        except Exception as e:
            execution_time = time.time() - start_time
            result = ToolResult(
                name=tool_call.name,
                output=None,
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                execution_time=execution_time,
                iteration=iteration,
                call_id=tool_call.call_id
            )
            await event_queue.put(ErrorEvent("tool_execution_error", result.error_message, recoverable=True, step_id=step_id))
            await event_queue.put(ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id))
            results_list.append(result)

            self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
            self._store_tool_result(tool_call.call_id, result, iteration)

    def _execute_tools(self, tool_calls: list[ToolCall], iteration: int, processing_mode: ProcessingMode | None = None) -> list[ToolResult]:
        """Execute tool calls and store results in context."""
        config = self._agent.get_config()
        results = []
        effective_mode = processing_mode if processing_mode is not None else config.processing_mode

        for tool_index, tool_call in enumerate(tool_calls):
            tool_state_key = f"tool_state:{tool_call.call_id}"
            self._agent.context.set(tool_state_key, b"started", iteration=iteration)

            internal_name = self._resolve_tool_name(tool_call.name)

            if internal_name not in config.tools_allowed:
                result = self._create_tool_not_allowed_error(internal_name, iteration)
                results.append(result)
                self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
                self._store_tool_result(tool_call.call_id, result, iteration)
                continue

            tool = self._agent.tools.get(internal_name)
            if tool is None:
                result = self._create_tool_not_found_error(internal_name, iteration)
                results.append(result)
                self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
                self._store_tool_result(tool_call.call_id, result, iteration)
                continue

            if config.validate_tool_arguments:
                is_valid, validation_errors = tool.validate_arguments(tool_call.arguments)
                if not is_valid:
                    result = self._create_tool_validation_error(internal_name, validation_errors, iteration)
                    results.append(result)
                    self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
                    self._store_tool_result(tool_call.call_id, result, iteration)
                    continue

            result = tool.run(tool_call.arguments, iteration, processing_mode=effective_mode)
            results.append(result)
            if result.success:
                self._agent.context.set(tool_state_key, b"finished", iteration=iteration)
            else:
                self._agent.context.set(tool_state_key, b"failed", iteration=iteration)
            self._store_tool_result(tool_call.call_id, result, iteration)

        return results

    def _store_tool_result(self, call_id: str, result: ToolResult, iteration: int) -> None:
        """Store tool result in context."""
        result_key = f"tool:result:{iteration}:{call_id}"
        result_data = json.dumps({
            "tool_name": result.name,
            "success": result.success,
            "output": serialize_tool_output(result.output),
            "error_message": result.error_message,
            "execution_time": result.execution_time,
            "iteration": iteration,
            "call_id": call_id
        }).encode('utf-8')
        self._agent.context.set(result_key, result_data, iteration=iteration)

    def _update_context_from_output(
        self,
        raw_output: str,
        segments: ExtractedSegments,
        tool_results: list[ToolResult],
        iteration: int
    ) -> None:
        """Update context based on output_mapping rules."""
        config = self._agent.get_config()

        for context_key, operation in config.output_mapping:
            if operation == "set_latest":
                self._agent.context.set(context_key, raw_output, iteration=iteration)

            elif operation == "append_version":
                existing = self._agent.context.get(context_key)
                if existing:
                    combined = existing + "\n\n" + raw_output
                else:
                    combined = raw_output
                self._agent.context.set(context_key, combined, iteration=iteration)

            elif operation == "set_response":
                if segments.response:
                    self._agent.context.set(context_key, segments.response, iteration=iteration)

            elif operation == "set_reasoning":
                if segments.reasoning:
                    reasoning_text = "\n".join(segments.reasoning)
                    self._agent.context.set(context_key, reasoning_text, iteration=iteration)

            elif operation == "set_tools":
                if tool_results:
                    tools_data = json.dumps([
                        {
                            "name": tr.name,
                            "success": tr.success,
                            "output": serialize_tool_output(tr.output)
                        }
                        for tr in tool_results
                    ])
                    self._agent.context.set(context_key, tools_data, iteration=iteration)

    def _step_in_thread(self, user_input: str | None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._step_impl, user_input, processing_mode)
            return future.result()

    def _step_in_process(self, user_input: str | None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._step_impl, user_input, processing_mode)
            return future.result()

    def _step_async(self, user_input: str | None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call sync _step_async from within an async context. "
                "Use async/await pattern instead."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e) or "no current event loop" in str(e):
                return asyncio.run(self._async_wrapper(user_input, processing_mode))
            else:
                raise

    async def _async_wrapper(self, user_input: str | None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._step_impl, user_input, processing_mode)
