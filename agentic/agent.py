"""
Agent abstraction and execution runner.
"""
from typing import Protocol, AsyncIterator
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .core import AgentConfig, AgentStatus, AgentStepResult, ExtractedSegments, ToolResult, ToolCall, ProcessingMode, new_uuid
from .context import ContextManager
from .patterns import PatternRegistry, PatternExtractor, StreamingPatternExtractor
from .tools import ToolRegistry
from .events import (
    AgentEvent, LLMTokenEvent, LLMCompleteEvent, StatusEvent,
    ToolStartEvent, ToolEndEvent,
    ContextWriteEvent, ErrorEvent, StepCompleteEvent,
    PatternStartEvent, PatternContentEvent, PatternEndEvent
)


class LLMProvider(Protocol):
    """
    Protocol for LLM provider implementations.

    Providers can implement streaming or non-streaming generation.
    If stream() is not implemented, framework will simulate streaming
    by emitting the full generate() output as a single token.
    """

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate complete text from prompt (blocking)."""
        ...

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream tokens from prompt (optional).

        If not implemented, framework falls back to generate()
        and simulates streaming.
        """
        # Default implementation: simulate streaming with generate()
        text = self.generate(prompt, max_tokens, temperature, **kwargs)
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
        """Get agent ID."""
        return self._config.agent_id

    def get_config(self) -> AgentConfig:
        """Get agent configuration."""
        return self._config

    def set_config(self, config: AgentConfig) -> None:
        """Update agent configuration."""
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


class AgentRunner:
    """
    Executes agent steps: prompt building, LLM generation, tool execution, context updates.
    """

    def __init__(self, agent: Agent):
        self._agent = agent

    def step(self, user_input: str | None = None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        """
        Execute a single agent step (batch mode).

        This is a convenience wrapper around step_stream() that aggregates
        all events and returns the final result.
        """
        # Collect events from stream
        events = []
        final_result = None

        # Run stream in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def collect():
                nonlocal final_result
                async for event in self.step_stream(user_input, processing_mode):
                    events.append(event)
                    if isinstance(event, StepCompleteEvent):
                        final_result = event.result

            loop.run_until_complete(collect())
        finally:
            loop.close()

        return final_result

    async def step_stream(
        self,
        user_input: str | None = None,
        processing_mode: ProcessingMode | None = None
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute agent step with streaming events.

        Yields events as execution progresses:
        - LLMTokenEvent: As LLM generates tokens
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

        # Get/increment iteration at step START for consistent indexing
        if config.auto_increment_iteration:
            current_iteration = self._agent.context.next_iteration()
        else:
            current_iteration = self._agent.context.get_iteration()

        # Generate unique step ID for event correlation
        step_id = new_uuid()

        # Build prompt
        prompt = self._build_prompt(user_input)

        # Yield initial status
        yield StatusEvent(AgentStatus.OK, "Starting agent step", step_id=step_id)

        # Initialize streaming pattern extractor
        pattern_set_name = config.pattern_set or "default"
        pattern_set = self._agent.patterns.get_pattern_set(pattern_set_name)

        if pattern_set is None:
            # No pattern set - treat entire output as response
            pattern_extractor = None
        else:
            pattern_extractor = StreamingPatternExtractor(
                pattern_set=pattern_set,
                stream_content=config.stream_pattern_content
            )

        # Stream LLM generation with incremental pattern detection
        # Execute tools concurrently as they are detected
        raw_output_buffer = []
        detected_tools: list[ToolCall] = []
        tool_execution_tasks: list[asyncio.Task] = []
        tool_results: list[ToolResult] = []

        # Event queue for concurrent tool execution
        tool_event_queue: asyncio.Queue = asyncio.Queue()

        try:
            async for token in self._agent.provider.stream(
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature
            ):
                raw_output_buffer.append(token)
                yield LLMTokenEvent(token, step_id=step_id)

                # Drain concurrent tool event queue
                while not tool_event_queue.empty():
                    try:
                        event = tool_event_queue.get_nowait()
                        yield event
                    except asyncio.QueueEmpty:
                        break

                # Incremental DB write for LLM output - EVERY token (no versioning during stream)
                partial_output = "".join(raw_output_buffer)
                streaming_key = f"llm_streaming:{current_iteration}"
                self._agent.context.update(streaming_key, partial_output.encode('utf-8'))

                # Feed token to pattern extractor
                if pattern_extractor:
                    for event_data in pattern_extractor.feed_token(token):
                        event_type = event_data[0]

                        if event_type == "pattern_start":
                            _, pattern_name, pattern_type = event_data
                            yield PatternStartEvent(pattern_name, pattern_type, step_id=step_id)

                            # Emit WAITING_FOR_TOOL status when tool pattern starts
                            if pattern_type == "tool":
                                yield StatusEvent(AgentStatus.WAITING_FOR_TOOL, f"Tool pattern detected: {pattern_name}", step_id=step_id)

                        elif event_type == "pattern_content":
                            _, pattern_name, content = event_data
                            yield PatternContentEvent(pattern_name, content, is_partial=True, step_id=step_id)

                            # Incremental DB write for partial pattern content (no versioning during stream)
                            partial_key = f"pattern_partial:{pattern_name}:{current_iteration}"
                            existing = self._agent.context.get(partial_key)
                            if existing:
                                accumulated = existing.value.decode('utf-8') + content
                            else:
                                accumulated = content
                            self._agent.context.update(partial_key, accumulated.encode('utf-8'))

                        elif event_type == "pattern_end":
                            _, pattern_name, pattern_type, full_content, tool_call = event_data
                            yield PatternEndEvent(pattern_name, pattern_type, full_content, step_id=step_id)

                            # Incremental DB write for detected pattern (versioned - completion)
                            pattern_key = f"pattern:{pattern_type}:{current_iteration}:{len(detected_tools)}"
                            self._agent.context.set(pattern_key, full_content.encode('utf-8'))

                            # Clean up partial pattern key
                            partial_key = f"pattern_partial:{pattern_name}:{current_iteration}"
                            self._agent.context.delete(partial_key)

                            # Handle tool execution if concurrent mode enabled
                            if tool_call:
                                detected_tools.append(tool_call)

                                # If concurrent execution enabled, execute tool immediately
                                if config.concurrent_tool_execution:
                                    # Check if tool should be executed via callback
                                    should_execute = True

                                    if config.on_tool_detected:
                                        try:
                                            should_execute = config.on_tool_detected(tool_call)
                                        except Exception as e:
                                            yield ErrorEvent(
                                                error_type="tool_callback_error",
                                                error_message=f"on_tool_detected callback failed: {str(e)}",
                                                recoverable=True,
                                                step_id=step_id
                                            )
                                            should_execute = False

                                    if should_execute:
                                        # Spawn async task to execute tool concurrently with LLM streaming
                                        yield StatusEvent(AgentStatus.WAITING_FOR_TOOL, f"Starting concurrent execution of tool '{tool_call.name}'", step_id=step_id)
                                        task = asyncio.create_task(
                                            self._execute_single_tool_concurrent(
                                                tool_call, current_iteration, effective_mode,
                                                tool_results, tool_event_queue, step_id
                                            )
                                        )
                                        tool_execution_tasks.append(task)
                                    else:
                                        yield StatusEvent(
                                            AgentStatus.WAITING_FOR_TOOL,
                                            f"Tool '{tool_call.name}' execution rejected by callback",
                                            step_id=step_id
                                        )

            raw_output = "".join(raw_output_buffer)
            yield LLMCompleteEvent(raw_output, step_id=step_id)

            # Wait for all concurrent tool executions to complete
            if tool_execution_tasks:
                yield StatusEvent(AgentStatus.WAITING_FOR_TOOL, f"Waiting for {len(tool_execution_tasks)} concurrent tool(s) to complete", step_id=step_id)
                await asyncio.gather(*tool_execution_tasks, return_exceptions=True)

                # Drain remaining events from queue
                while not tool_event_queue.empty():
                    try:
                        event = tool_event_queue.get_nowait()
                        yield event
                    except asyncio.QueueEmpty:
                        break

        except Exception as e:
            # LLM error - cancel any running tool tasks
            for task in tool_execution_tasks:
                if not task.done():
                    task.cancel()

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

        # Finalize pattern extraction
        if pattern_extractor:
            segments, malformed_patterns = pattern_extractor.finalize(iteration=current_iteration)

            # Emit error events for malformed patterns and write to DB
            if malformed_patterns:
                for pattern_name, partial_content in malformed_patterns.items():
                    yield ErrorEvent(
                        error_type="malformed_pattern",
                        error_message=f"Pattern '{pattern_name}' missing end tag",
                        recoverable=True,
                        partial_data=partial_content,
                        step_id=step_id
                    )

                    # Write malformed pattern to DB (transient, for crash recovery - versioned)
                    malformed_key = f"pattern_malformed:{pattern_name}:{current_iteration}"
                    self._agent.context.set(malformed_key, partial_content.encode('utf-8'))
        else:
            # No pattern extraction - entire output is response
            segments = ExtractedSegments(response=raw_output)
            malformed_patterns = None

        # If not concurrent mode, execute tools sequentially now
        if not config.concurrent_tool_execution and detected_tools:
            # Filter tools based on callback
            tools_to_execute = []
            for tool_call in detected_tools:
                should_execute = True

                if config.on_tool_detected:
                    try:
                        should_execute = config.on_tool_detected(tool_call)
                    except Exception as e:
                        yield ErrorEvent(
                            error_type="tool_callback_error",
                            error_message=f"on_tool_detected callback failed: {str(e)}",
                            recoverable=True,
                            step_id=step_id
                        )
                        should_execute = False

                if should_execute:
                    tools_to_execute.append(tool_call)
                else:
                    yield StatusEvent(
                        AgentStatus.WAITING_FOR_TOOL,
                        f"Tool '{tool_call.name}' execution rejected by callback",
                        step_id=step_id
                    )

            # Execute approved tools using existing streaming path
            if tools_to_execute:
                yield StatusEvent(AgentStatus.WAITING_FOR_TOOL, f"Executing {len(tools_to_execute)} approved tool(s)", step_id=step_id)

                async for event in self._execute_tools_stream(tools_to_execute, current_iteration, effective_mode, step_id):
                    yield event
                    if isinstance(event, ToolEndEvent):
                        tool_results.append(event.result)

        # Check tool execution results
        tool_execution_failed = any(not tr.success for tr in tool_results)

        # Update context
        self._update_context_from_output(raw_output, segments, tool_results)

        # Emit context write event if enabled
        if config.incremental_context_writes:
            for context_key, _ in config.output_mapping:
                record = self._agent.context.get(context_key)
                if record:
                    preview = record.value.decode('utf-8')[:100] if len(record.value) < 100 else record.value.decode('utf-8')[:97] + "..."
                    yield ContextWriteEvent(
                        key=context_key,
                        value_preview=preview,
                        version=record.version,
                        iteration=record.iteration,
                        step_id=step_id
                    )

        # Determine final status and collect error details
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
            yield ErrorEvent(error_type, error_message, recoverable=False, step_id=step_id)
        elif detected_tools and tool_results:
            # Tools were executed successfully
            status = AgentStatus.TOOL_EXECUTED
        elif detected_tools and not tool_results:
            # Tools detected but none executed (all rejected by callback)
            status = AgentStatus.WAITING_FOR_TOOL
        elif not segments.response and not detected_tools:
            status = AgentStatus.DONE
        else:
            status = AgentStatus.OK

        # Emit final status
        yield StatusEvent(status, "Agent step complete", step_id=step_id)

        # Yield final result (iteration already set at step start)
        final_result = AgentStepResult(
            status=status,
            raw_output=raw_output,
            segments=segments,
            tool_results=tool_results,
            iteration=current_iteration,
            error_message=error_message,
            error_type=error_type,
            partial_malformed_patterns=malformed_patterns
        )
        yield StepCompleteEvent(final_result, step_id=step_id)

    def _step_impl(self, user_input: str | None = None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        """Internal synchronous implementation of agent step."""
        config = self._agent.get_config()
        # Fix: argument overrides config (consistent with step_stream)
        effective_mode = processing_mode if processing_mode is not None else config.processing_mode

        # Get/increment iteration at step START for consistent indexing
        if config.auto_increment_iteration:
            current_iteration = self._agent.context.next_iteration()
        else:
            current_iteration = self._agent.context.get_iteration()

        prompt = self._build_prompt(user_input)

        try:
            raw_output = self._agent.provider.generate(
                prompt=prompt,
                max_tokens=self._agent.get_config().max_tokens,
                temperature=self._agent.get_config().temperature
            )
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

        tool_results = []
        tool_execution_failed = False

        if segments.tools:
            tool_results = self._execute_tools(segments.tools, current_iteration, processing_mode=effective_mode)
            tool_execution_failed = any(not tr.success for tr in tool_results)

        self._update_context_from_output(raw_output, segments, tool_results)

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
            # Tools executed successfully - use TOOL_EXECUTED status
            status = AgentStatus.TOOL_EXECUTED
        elif not segments.response and not segments.tools:
            status = AgentStatus.DONE
        else:
            status = AgentStatus.OK

        # Return result (iteration already set at step start)
        return AgentStepResult(
            status=status,
            raw_output=raw_output,
            segments=segments,
            tool_results=tool_results,
            iteration=current_iteration,
            error_message=error_message,
            error_type=error_type
        )

    def _build_prompt(self, user_input: str | None) -> str:
        """Build prompt from context based on input_mapping rules."""
        config = self._agent.get_config()
        parts = []

        for context_key, position in config.input_mapping:
            if context_key.startswith("literal:"):
                literal_text = context_key[8:]
                if position == "prepend":
                    parts.insert(0, literal_text)
                else:
                    parts.append(literal_text)
            else:
                record = self._agent.context.get(context_key)
                if record is not None:
                    try:
                        text = record.value.decode('utf-8')
                        if position == "prepend":
                            parts.insert(0, text)
                        else:
                            parts.append(text)
                    except UnicodeDecodeError:
                        pass

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
            # Write tool state: started
            tool_state_key = f"tool_state:{tool_call.call_id}"
            self._agent.context.set(tool_state_key, b"started")

            # Emit start event
            yield ToolStartEvent(tool_call.name, tool_call.arguments, iteration, tool_call.call_id, step_id=step_id)

            # Check if tool is allowed
            if tool_call.name not in config.tools_allowed:
                result = ToolResult(
                    name=tool_call.name,
                    output={},
                    success=False,
                    error_message=f"Tool '{tool_call.name}' not in allowed list",
                    execution_time=0.0,
                    iteration=iteration
                )
                yield ErrorEvent("tool_not_allowed", result.error_message, recoverable=True, step_id=step_id)
                yield ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id)
                self._agent.context.set(tool_state_key, b"failed")
                self._store_tool_result(tool_call.call_id, result, iteration)
                continue

            # Get tool from registry
            tool = self._agent.tools.get(tool_call.name)
            if tool is None:
                result = ToolResult(
                    name=tool_call.name,
                    output={},
                    success=False,
                    error_message=f"Tool '{tool_call.name}' not found in registry",
                    execution_time=0.0,
                    iteration=iteration
                )
                yield ErrorEvent("tool_not_found", result.error_message, recoverable=True, step_id=step_id)
                yield ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id)
                self._agent.context.set(tool_state_key, b"failed")
                self._store_tool_result(tool_call.call_id, result, iteration)
                continue

            # Execute tool with streaming support
            start_time = time.time()
            output_chunks = []
            tool_failed = False
            error_message = None

            try:
                # Use tool.run_stream() which handles both streaming and non-streaming tools
                async for output_event in tool.run_stream(tool_call.arguments, iteration, effective_mode):
                    # Enrich output event with call_id and step_id
                    output_event.call_id = tool_call.call_id
                    output_event.step_id = step_id
                    # Forward the output event
                    yield output_event

                    # Collect output chunks
                    output_chunks.append(output_event.output)

                execution_time = time.time() - start_time

                # Build final result from collected chunks
                if len(output_chunks) == 1:
                    final_output = output_chunks[0]
                else:
                    # Multiple chunks - combine them
                    if all(isinstance(chunk, str) for chunk in output_chunks):
                        final_output = "".join(output_chunks)
                    elif all(isinstance(chunk, dict) for chunk in output_chunks):
                        final_output = {f"chunk_{i}": chunk for i, chunk in enumerate(output_chunks)}
                    else:
                        final_output = output_chunks

                result = ToolResult(
                    name=tool_call.name,
                    output=final_output,
                    success=True,
                    error_message=None,
                    execution_time=execution_time,
                    iteration=iteration
                )
                yield ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id)
                self._agent.context.set(tool_state_key, b"finished")
                self._store_tool_result(tool_call.call_id, result, iteration)

            except Exception as e:
                execution_time = time.time() - start_time
                result = ToolResult(
                    name=tool_call.name,
                    output={},
                    success=False,
                    error_message=f"Tool execution failed: {str(e)}",
                    execution_time=execution_time,
                    iteration=iteration
                )
                yield ErrorEvent("tool_execution_error", result.error_message, recoverable=True, step_id=step_id)
                yield ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id)
                self._agent.context.set(tool_state_key, b"failed")
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

        # Write tool state: started
        tool_state_key = f"tool_state:{tool_call.call_id}"
        self._agent.context.set(tool_state_key, b"started")

        # Emit start event
        await event_queue.put(ToolStartEvent(tool_call.name, tool_call.arguments, iteration, tool_call.call_id, step_id=step_id))

        # Check if tool allowed
        if tool_call.name not in config.tools_allowed:
            result = ToolResult(
                name=tool_call.name,
                output={},
                success=False,
                error_message=f"Tool '{tool_call.name}' not in allowed list",
                execution_time=0.0,
                iteration=iteration
            )
            await event_queue.put(ErrorEvent("tool_not_allowed", result.error_message, recoverable=True, step_id=step_id))
            await event_queue.put(ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id))
            results_list.append(result)

            # Write tool state: failed
            self._agent.context.set(tool_state_key, b"failed")

            # Incremental DB write with call_id
            self._store_tool_result(tool_call.call_id, result, iteration)
            return

        # Get tool from registry
        tool = self._agent.tools.get(tool_call.name)
        if tool is None:
            result = ToolResult(
                name=tool_call.name,
                output={},
                success=False,
                error_message=f"Tool '{tool_call.name}' not found in registry",
                execution_time=0.0,
                iteration=iteration
            )
            await event_queue.put(ErrorEvent("tool_not_found", result.error_message, recoverable=True, step_id=step_id))
            await event_queue.put(ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id))
            results_list.append(result)

            # Write tool state: failed
            self._agent.context.set(tool_state_key, b"failed")

            # Incremental DB write with call_id
            self._store_tool_result(tool_call.call_id, result, iteration)
            return

        # Execute tool with streaming support
        start_time = time.time()
        output_chunks = []

        try:
            async for output_event in tool.run_stream(tool_call.arguments, iteration, effective_mode):
                # Enrich output event with call_id and step_id before queuing
                output_event.call_id = tool_call.call_id
                output_event.step_id = step_id
                await event_queue.put(output_event)
                output_chunks.append(output_event.output)

            execution_time = time.time() - start_time

            # Combine output chunks
            if len(output_chunks) == 1:
                final_output = output_chunks[0]
            else:
                if all(isinstance(c, str) for c in output_chunks):
                    final_output = "".join(output_chunks)
                elif all(isinstance(c, dict) for c in output_chunks):
                    final_output = {f"chunk_{i}": c for i, c in enumerate(output_chunks)}
                else:
                    final_output = output_chunks

            result = ToolResult(
                name=tool_call.name,
                output=final_output,
                success=True,
                error_message=None,
                execution_time=execution_time,
                iteration=iteration
            )
            await event_queue.put(ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id))
            results_list.append(result)

            # Write tool state: finished
            self._agent.context.set(tool_state_key, b"finished")

            # Incremental DB write with call_id
            self._store_tool_result(tool_call.call_id, result, iteration)

        except Exception as e:
            execution_time = time.time() - start_time
            result = ToolResult(
                name=tool_call.name,
                output={},
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                execution_time=execution_time,
                iteration=iteration
            )
            await event_queue.put(ErrorEvent("tool_execution_error", result.error_message, recoverable=True, step_id=step_id))
            await event_queue.put(ToolEndEvent(tool_call.name, result, tool_call.call_id, step_id=step_id))
            results_list.append(result)

            # Write tool state: failed
            self._agent.context.set(tool_state_key, b"failed")

            # Incremental DB write with call_id
            self._store_tool_result(tool_call.call_id, result, iteration)

    def _execute_tools(self, tool_calls: list[ToolCall], iteration: int, processing_mode: ProcessingMode | None = None) -> list[ToolResult]:
        """Execute tool calls and store results in context."""
        config = self._agent.get_config()
        results = []
        effective_mode = processing_mode if processing_mode is not None else config.processing_mode

        for tool_index, tool_call in enumerate(tool_calls):
            if tool_call.name not in config.tools_allowed:
                result = ToolResult(
                    name=tool_call.name,
                    output={},
                    success=False,
                    error_message=f"Tool '{tool_call.name}' not in allowed list",
                    execution_time=0.0,
                    iteration=iteration
                )
                results.append(result)
                self._store_tool_result(tool_call.call_id, result, iteration)
                continue

            tool = self._agent.tools.get(tool_call.name)
            if tool is None:
                result = ToolResult(
                    name=tool_call.name,
                    output={},
                    success=False,
                    error_message=f"Tool '{tool_call.name}' not found in registry",
                    execution_time=0.0,
                    iteration=iteration
                )
                results.append(result)
                self._store_tool_result(tool_call.call_id, result, iteration)
                continue

            result = tool.run(tool_call.arguments, iteration, processing_mode=effective_mode)
            results.append(result)
            self._store_tool_result(tool_call.call_id, result, iteration)

        return results

    def _store_tool_result(self, call_id: str, result: ToolResult, iteration: int) -> None:
        """Store tool result in context with call_id-based key."""
        result_key = f"tool:result:{iteration}:{call_id}"
        result_data = json.dumps({
            "tool_name": result.name,
            "success": result.success,
            "output": str(result.output),
            "error_message": result.error_message,
            "execution_time": result.execution_time,
            "iteration": iteration,
            "call_id": call_id
        }).encode('utf-8')
        self._agent.context.set(result_key, result_data)

    def _update_context_from_output(
        self,
        raw_output: str,
        segments: ExtractedSegments,
        tool_results: list[ToolResult]
    ) -> None:
        """Update context based on output_mapping rules."""
        config = self._agent.get_config()

        for context_key, operation in config.output_mapping:
            if operation == "set_latest":
                self._agent.context.set(context_key, raw_output.encode('utf-8'))

            elif operation == "append_version":
                existing = self._agent.context.get(context_key)
                if existing:
                    combined = existing.value.decode('utf-8') + "\n\n" + raw_output
                else:
                    combined = raw_output
                self._agent.context.set(context_key, combined.encode('utf-8'))

            elif operation == "set_response":
                if segments.response:
                    self._agent.context.set(context_key, segments.response.encode('utf-8'))

            elif operation == "set_reasoning":
                if segments.reasoning:
                    reasoning_text = "\n".join(segments.reasoning)
                    self._agent.context.set(context_key, reasoning_text.encode('utf-8'))

            elif operation == "set_tools":
                if tool_results:
                    tools_data = json.dumps([
                        {
                            "name": tr.name,
                            "success": tr.success,
                            "output": str(tr.output)
                        }
                        for tr in tool_results
                    ])
                    self._agent.context.set(context_key, tools_data.encode('utf-8'))

    def _step_in_thread(self, user_input: str | None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        """Execute agent step in a separate thread."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._step_impl, user_input, processing_mode)
            return future.result()

    def _step_in_process(self, user_input: str | None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        """Execute agent step in a separate process."""
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._step_impl, user_input, processing_mode)
            return future.result()

    def _step_async(self, user_input: str | None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        """Execute agent step asynchronously."""
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call sync _step_async from within an async context. "
                "Use async/await pattern instead."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e) or "no current event loop" in str(e):
                # Not in async context, create new loop
                return asyncio.run(self._async_wrapper(user_input, processing_mode))
            else:
                # Re-raise if it's our error message
                raise

    async def _async_wrapper(self, user_input: str | None, processing_mode: ProcessingMode | None = None) -> AgentStepResult:
        """Wrapper to call step_impl in async context."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._step_impl, user_input, processing_mode)


class MockLLMProvider:
    """
    Mock LLM provider for testing that returns canned responses.

    Supports both batch (generate) and streaming (stream) modes.
    """

    def __init__(self, response: str = "This is a mock response.", simulate_streaming: bool = False):
        self._response = response
        self._simulate_streaming = simulate_streaming

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        return self._response

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream response tokens.

        If simulate_streaming=True, yields word-by-word.
        Otherwise yields entire response as one token.
        """
        if self._simulate_streaming:
            # Simulate streaming by yielding word by word
            words = self._response.split()
            for i, word in enumerate(words):
                if i < len(words) - 1:
                    yield word + " "
                else:
                    yield word
        else:
            # Non-streaming: yield full response
            yield self._response

    def set_response(self, response: str) -> None:
        self._response = response
