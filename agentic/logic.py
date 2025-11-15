"""
Logic flows for controlling agent execution across iterations.
"""
from dataclasses import dataclass, field
import re
import fnmatch
from typing import AsyncIterator
import asyncio

from .agent import AgentRunner
from .context import ContextManager
from .patterns import PatternRegistry
from .core import AgentStepResult, AgentStatus, ProcessingMode, output_to_string
from .events import (
    AgentEvent, StatusEvent, StepCompleteEvent,
    LLMCompleteEvent, LLMChunkEvent, PatternEndEvent, ToolEndEvent,
    PatternStartEvent, ToolStartEvent, ToolOutputEvent, ContextWriteEvent, ErrorEvent, PatternContentEvent,
    ContextHealthEvent
)


@dataclass
class LogicCondition:
    pattern_set: str
    pattern_name: str
    match_type: str  # "contains" | "equals" | "regex"
    target: str  # "response" | "reasoning" | "tool_output" | "context:{key}"
    evaluation_point: str = "auto"  # "auto" | "llm_chunk" | "llm_complete" | "tool_detected" | "tool_finished" | "step_complete" | "any_event"
    # "auto" uses smart defaults: pattern/regex → llm_complete, context → step_complete


@dataclass
class ContextHealthCheck:
    check_type: str  # "size" | "version_count" | "growth_rate"
    key_pattern: str  # Glob pattern: "llm_output:*", "tool_result:*", "*" for all
    threshold: float
    action: str = "warn"  # "warn" | "stop"
    evaluation_point: str = "step_complete"
    max_versions_limit: int = 10000  # Safety limit for version_count checks to prevent memory exhaustion


@dataclass
class LogicConfig:
    logic_id: str
    max_iterations: int | None = None
    stop_conditions: list[LogicCondition] = field(default_factory=list)
    loop_until_conditions: list[LogicCondition] = field(default_factory=list)
    break_on_error: bool = True
    processing_mode: ProcessingMode | None = ProcessingMode.THREAD  # Default to THREAD if not specified
    context_health_checks: list[ContextHealthCheck] = field(default_factory=list)


class LogicRunner:
    """
    Manages iterative execution of agent with conditional control flow.
    """

    def __init__(
        self,
        agent_runner: AgentRunner,
        context: ContextManager,
        patterns: PatternRegistry,
        config: LogicConfig
    ):
        self._agent_runner = agent_runner
        self._context = context
        self._patterns = patterns
        self._config = config

    def run(self, initial_input: str | None = None, processing_mode: ProcessingMode | None = None) -> list[AgentStepResult]:
        """
        Execute agent in loop with condition checking (batch mode).

        This aggregates all events from run_stream() and returns final results.
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "LogicRunner.run() cannot be called from an async context. "
                "Use 'await run_stream()' instead, or call from a synchronous context."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                raise

        return asyncio.run(self._collect_run_events(initial_input, processing_mode))

    async def _collect_run_events(
        self,
        initial_input: str | None,
        processing_mode: ProcessingMode | None
    ) -> list[AgentStepResult]:
        results = []
        async for event in self.run_stream(initial_input, processing_mode):
            if isinstance(event, StepCompleteEvent):
                results.append(event.result)
        return results

    async def run_stream(
        self,
        initial_input: str | None = None,
        processing_mode: ProcessingMode | None = None
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute agent loop with streaming events.

        Yields all events from underlying agent steps plus logic-level
        status events for loop control flow.

        Evaluates conditions at appropriate points based on evaluation_point:
        - "auto": infers from target (context → step_complete, patterns → llm_complete)
        - "llm_chunk": on every LLM chunk as it streams (LLMChunkEvent)
        - "llm_complete": after LLMCompleteEvent
        - "tool_detected": after PatternEndEvent with tool type
        - "tool_finished": after ToolEndEvent (tool execution completes)
        - "step_complete": after StepCompleteEvent
        - "any_event": on every event
        """
        if processing_mode is None:
            processing_mode = self._config.processing_mode

        yield StatusEvent(AgentStatus.OK, f"Starting logic loop: {self._config.logic_id}")

        needs_llm_chunk_eval = self._has_conditions_for_event("llm_chunk")
        needs_any_event_eval = self._has_conditions_for_event("any_event")
        should_accumulate_partial = needs_llm_chunk_eval or needs_any_event_eval

        max_buffer_size = self._agent_runner._agent.get_config().max_partial_buffer_size

        results: list[AgentStepResult] = []
        iteration_count = 0
        current_input = initial_input
        current_step_result = None
        partial_raw_output = "" if should_accumulate_partial else None

        while True:
            if self._config.max_iterations is not None:
                if iteration_count >= self._config.max_iterations:
                    yield StatusEvent(AgentStatus.DONE, f"Max iterations reached: {self._config.max_iterations}")
                    break

            if should_accumulate_partial:
                partial_raw_output = ""

            async for event in self._agent_runner.step_stream(current_input, processing_mode):
                yield event

                if should_accumulate_partial and isinstance(event, LLMChunkEvent):
                    partial_raw_output += event.chunk

                    if len(partial_raw_output) > max_buffer_size:
                        partial_raw_output = partial_raw_output[-max_buffer_size:]

                should_check = False
                event_type = None
                eval_context = None

                if isinstance(event, LLMChunkEvent):
                    event_type = "llm_chunk"
                    should_check = self._has_conditions_for_event("llm_chunk")
                    eval_context = {"raw_output": partial_raw_output or ""}

                elif isinstance(event, LLMCompleteEvent):
                    event_type = "llm_complete"
                    should_check = self._has_conditions_for_event("llm_complete")
                    eval_context = {"raw_output": event.full_text}

                elif isinstance(event, PatternEndEvent):
                    if event.pattern_type == "tool":
                        event_type = "tool_detected"
                        should_check = self._has_conditions_for_event("tool_detected")
                        eval_context = {
                            "tool_output": event.full_content,
                            "pattern_name": event.pattern_name,
                            "raw_output": partial_raw_output
                        }
                    else:
                        event_type = "pattern_end"
                        should_check = self._has_conditions_for_event("pattern_end")
                        eval_context = {
                            "pattern_name": event.pattern_name,
                            "pattern_type": event.pattern_type,
                            "full_content": event.full_content,
                            "raw_output": partial_raw_output
                        }

                elif isinstance(event, ToolEndEvent):
                    event_type = "tool_finished"
                    should_check = self._has_conditions_for_event("tool_finished")
                    eval_context = {
                        "tool_name": event.tool_name,
                        "tool_result": event.result,
                        "tool_output": output_to_string(event.result.output),
                        "tool_success": event.result.success,
                        "raw_output": partial_raw_output
                    }

                elif isinstance(event, PatternStartEvent):
                    event_type = "pattern_start"
                    should_check = self._has_conditions_for_event("pattern_start")
                    eval_context = {
                        "pattern_name": event.pattern_name,
                        "pattern_type": event.pattern_type,
                        "raw_output": partial_raw_output
                    }

                elif isinstance(event, PatternContentEvent):
                    event_type = "pattern_content"
                    should_check = self._has_conditions_for_event("pattern_content")
                    eval_context = {
                        "pattern_name": event.pattern_name,
                        "pattern_content": event.content,
                        "is_partial": event.is_partial,
                        "raw_output": partial_raw_output
                    }

                elif isinstance(event, ToolStartEvent):
                    event_type = "tool_start"
                    should_check = self._has_conditions_for_event("tool_start")
                    eval_context = {
                        "tool_name": event.tool_name,
                        "tool_arguments": event.arguments,
                        "iteration": event.iteration,
                        "raw_output": partial_raw_output
                    }

                elif isinstance(event, ToolOutputEvent):
                    event_type = "tool_output"
                    should_check = self._has_conditions_for_event("tool_output")
                    eval_context = {
                        "tool_name": event.tool_name,
                        "tool_output": output_to_string(event.output),
                        "is_partial": event.is_partial,
                        "raw_output": partial_raw_output
                    }

                elif isinstance(event, ContextWriteEvent):
                    event_type = "context_write"
                    should_check = self._has_conditions_for_event("context_write")
                    eval_context = {
                        "context_key": event.key,
                        "value_preview": event.value_preview,
                        "version": event.version,
                        "iteration": event.iteration,
                        "raw_output": partial_raw_output
                    }

                elif isinstance(event, ErrorEvent):
                    event_type = "error"
                    should_check = self._has_conditions_for_event("error")
                    eval_context = {
                        "error_type": event.error_type,
                        "error_message": event.error_message,
                        "recoverable": event.recoverable,
                        "raw_output": partial_raw_output
                    }

                elif isinstance(event, StatusEvent):
                    event_type = "status"
                    should_check = self._has_conditions_for_event("status")
                    eval_context = {
                        "status": event.status,
                        "status_message": event.message,
                        "raw_output": partial_raw_output
                    }

                elif isinstance(event, StepCompleteEvent):
                    event_type = "step_complete"
                    should_check = True  # Always check at step complete
                    current_step_result = event.result
                    results.append(current_step_result)
                    iteration_count += 1

                    if current_step_result.status == AgentStatus.ERROR and self._config.break_on_error:
                        yield StatusEvent(AgentStatus.ERROR, "Breaking on error")
                        return

                    # Check context health at step_complete
                    for health_check in self._config.context_health_checks:
                        if health_check.evaluation_point == event_type:
                            health_events = self._check_context_health(health_check)
                            for health_event in health_events:
                                yield health_event
                                if health_check.action == "stop":
                                    yield StatusEvent(AgentStatus.ERROR, f"Stopping due to health check: {health_check.check_type}")
                                    return

                if should_check:
                    if event_type == "step_complete" and current_step_result:
                        should_stop, loop_satisfied = self._check_conditions_for_event(
                            current_step_result,
                            event_type
                        )
                    elif eval_context:
                        should_stop, loop_satisfied = self._check_conditions_on_partial_context(
                            eval_context,
                            event_type
                        )
                    else:
                        should_stop, loop_satisfied = False, False

                    if should_stop:
                        yield StatusEvent(AgentStatus.DONE, f"Stop condition met at {event_type}")
                        return

                    if loop_satisfied:
                        yield StatusEvent(AgentStatus.DONE, f"Loop-until condition satisfied at {event_type}")
                        return

                if isinstance(event, StepCompleteEvent):
                    if current_step_result.segments.response:
                        current_input = current_step_result.segments.response
                    else:
                        current_input = None

                    if current_step_result.status == AgentStatus.DONE and not current_input:
                        yield StatusEvent(AgentStatus.DONE, "Agent completed with no further input")
                        return

                    current_step_result = None

    def _run_impl(self, initial_input: str | None = None) -> list[AgentStepResult]:
        results: list[AgentStepResult] = []
        iteration_count = 0
        current_input = initial_input

        while True:
            if self._config.max_iterations is not None:
                if iteration_count >= self._config.max_iterations:
                    break

            result = self._agent_runner.step(current_input, processing_mode=self._config.processing_mode)
            results.append(result)
            iteration_count += 1

            if result.status == AgentStatus.ERROR and self._config.break_on_error:
                break

            should_stop = False
            loop_satisfied = False

            if not should_stop and not loop_satisfied:
                stop, satisfied = self._check_conditions_for_event(result, "llm_complete")
                should_stop = should_stop or stop
                loop_satisfied = loop_satisfied or satisfied

            if not should_stop and not loop_satisfied and result.segments.tools:
                stop, satisfied = self._check_conditions_for_event(result, "tool_detected")
                should_stop = should_stop or stop
                loop_satisfied = loop_satisfied or satisfied

            if not should_stop and not loop_satisfied and result.tool_results:
                stop, satisfied = self._check_conditions_for_event(result, "tool_finished")
                should_stop = should_stop or stop
                loop_satisfied = loop_satisfied or satisfied

            if not should_stop and not loop_satisfied:
                stop, satisfied = self._check_conditions_for_event(result, "step_complete")
                should_stop = should_stop or stop
                loop_satisfied = loop_satisfied or satisfied

            if should_stop:
                break

            if loop_satisfied:
                break

            if result.segments.response:
                current_input = result.segments.response
            else:
                current_input = None

            if result.status == AgentStatus.DONE and not current_input:
                break

        return results

    def _evaluate_condition(self, condition: LogicCondition, result: AgentStepResult) -> bool:
        if condition.match_type == "contains":
            return self._check_pattern_in_result(condition, result)

        elif condition.match_type == "equals":
            target_text = self._get_target_text(condition.target, result)
            if target_text is None:
                return False
            return target_text == condition.pattern_name

        elif condition.match_type == "regex":
            target_text = self._get_target_text(condition.target, result)
            if target_text is None:
                return False
            try:
                pattern = re.compile(condition.pattern_name)
                return pattern.search(target_text) is not None
            except re.error:
                return False

        return False

    def _check_pattern_in_result(self, condition: LogicCondition, result: AgentStepResult) -> bool:
        """
        Check if specific named pattern exists in result.
        Uses raw_output to detect specific pattern instance, not just segment type.
        """
        pattern_set = self._patterns.get_pattern_set(condition.pattern_set)
        if pattern_set is None:
            return False

        pattern_obj = None
        for p in pattern_set.patterns:
            if p.name == condition.pattern_name:
                pattern_obj = p
                break

        if pattern_obj is None:
            return False

        start_tag = pattern_obj.start_tag
        end_tag = pattern_obj.end_tag

        start_escaped = re.escape(start_tag)
        end_escaped = re.escape(end_tag)
        quantifier = ".*" if pattern_obj.greedy else ".*?"
        regex = f"{start_escaped}({quantifier}){end_escaped}"

        target_text = self._get_target_text_for_pattern_check(condition.target, result)
        if target_text is None:
            return False

        matches = re.search(regex, target_text, re.DOTALL)
        return matches is not None

    def _get_target_text_for_pattern_check(self, target: str, result: AgentStepResult) -> str | None:
        """
        Get text for pattern matching.
        Returns raw_output for segment targets to preserve pattern tags.
        """
        if target == "response" or target == "reasoning" or target == "tool_output":
            return result.raw_output
        elif target.startswith("context:"):
            context_key = target[8:]
            return self._context.get(context_key)
        return None

    def _get_target_text(self, target: str, result: AgentStepResult) -> str | None:
        if target == "response":
            return result.segments.response

        elif target == "reasoning":
            if result.segments.reasoning:
                return "\n".join(result.segments.reasoning)
            return None

        elif target == "tool_output":
            if result.tool_results:
                outputs = [output_to_string(tr.output) for tr in result.tool_results]
                return "\n".join(outputs)
            return None

        elif target.startswith("context:"):
            context_key = target[8:]
            return self._context.get(context_key)

        return None

    def _has_conditions_for_event(self, event_type: str) -> bool:
        for condition in self._config.stop_conditions + self._config.loop_until_conditions:
            if self._should_evaluate_at_event(condition, event_type):
                return True
        return False

    def _should_evaluate_at_event(self, condition: LogicCondition, event_type: str) -> bool:
        eval_point = condition.evaluation_point

        if eval_point == "any_event":
            return True
        elif eval_point == event_type:
            return True
        elif eval_point == "auto":
            if condition.target.startswith("context:"):
                return event_type == "step_complete"
            elif condition.match_type == "contains":
                return event_type == "llm_complete"
            else:
                return event_type == "step_complete"

        return False

    def _check_conditions_for_event(
        self,
        result: AgentStepResult,
        event_type: str
    ) -> tuple[bool, bool]:
        """
        Check conditions that should be evaluated at this event type.

        Returns (should_stop, loop_satisfied).
        """
        should_stop = False
        loop_satisfied = False

        for condition in self._config.stop_conditions:
            if self._should_evaluate_at_event(condition, event_type):
                if self._evaluate_condition(condition, result):
                    should_stop = True
                    break

        for condition in self._config.loop_until_conditions:
            if self._should_evaluate_at_event(condition, event_type):
                if self._evaluate_condition(condition, result):
                    loop_satisfied = True
                    break

        return should_stop, loop_satisfied

    def _check_conditions_on_partial_context(
        self,
        context: dict,
        event_type: str
    ) -> tuple[bool, bool]:
        """
        Check conditions using partial evaluation context from events.

        Context dict contains keys like:
        - "raw_output": LLM output text
        - "tool_output": Tool pattern content or tool execution output
        - "pattern_name": Pattern name

        Returns (should_stop, loop_satisfied).
        """
        should_stop = False
        loop_satisfied = False

        for condition in self._config.stop_conditions:
            if self._should_evaluate_at_event(condition, event_type):
                if self._evaluate_condition_on_context(condition, context):
                    should_stop = True
                    break

        for condition in self._config.loop_until_conditions:
            if self._should_evaluate_at_event(condition, event_type):
                if self._evaluate_condition_on_context(condition, context):
                    loop_satisfied = True
                    break

        return should_stop, loop_satisfied

    def _evaluate_condition_on_context(self, condition: LogicCondition, context: dict) -> bool:
        if condition.match_type == "regex":
            target_text = None

            if condition.target == "response" or condition.target == "reasoning":
                target_text = context.get("raw_output")
            elif condition.target == "tool_output":
                target_text = context.get("tool_output")
            elif condition.target.startswith("context:"):
                context_key = condition.target[8:]
                target_text = self._context.get(context_key)

            if target_text is None:
                return False

            try:
                pattern = re.compile(condition.pattern_name)
                return pattern.search(target_text) is not None
            except re.error:
                return False

        elif condition.match_type == "contains":
            target_text = None

            if condition.target == "response" or condition.target == "reasoning":
                target_text = context.get("raw_output")
            elif condition.target == "tool_output":
                target_text = context.get("tool_output")
            elif condition.target.startswith("context:"):
                context_key = condition.target[8:]
                target_text = self._context.get(context_key)

            if target_text is None:
                return False

            pattern_set = self._patterns.get_pattern_set(condition.pattern_set)
            if pattern_set is None:
                return False

            pattern_obj = None
            for p in pattern_set.patterns:
                if p.name == condition.pattern_name:
                    pattern_obj = p
                    break

            if pattern_obj is None:
                return False

            start_escaped = re.escape(pattern_obj.start_tag)
            end_escaped = re.escape(pattern_obj.end_tag)
            quantifier = ".*" if pattern_obj.greedy else ".*?"
            regex = f"{start_escaped}({quantifier}){end_escaped}"

            matches = re.search(regex, target_text, re.DOTALL)
            return matches is not None

        elif condition.match_type == "equals":
            target_text = None

            if condition.target == "response":
                target_text = context.get("raw_output")
            elif condition.target == "tool_output":
                target_text = context.get("tool_output")
            elif condition.target.startswith("context:"):
                context_key = condition.target[8:]
                target_text = self._context.get(context_key)

            if target_text is None:
                return False

            return target_text == condition.pattern_name

        return False

    def _check_context_health(self, check: ContextHealthCheck) -> list[ContextHealthEvent]:
        """
        Check context health based on configuration.

        Args:
            check: Health check configuration

        Returns:
            List of ContextHealthEvent for any violations found
        """
        events = []

        if check.key_pattern == "*":
            matching_keys = self._context.list_keys()
        elif check.key_pattern.endswith("*") and "*" not in check.key_pattern[:-1]:
            prefix = check.key_pattern[:-1]
            matching_keys = self._context.list_keys(prefix=prefix if prefix else None)
        else:
            all_keys = self._context.list_keys()
            matching_keys = [k for k in all_keys if fnmatch.fnmatch(k, check.key_pattern)]

        for key in matching_keys:
            if check.check_type == "size":
                value = self._context.get_bytes(key)
                if value and len(value) > check.threshold:
                    events.append(ContextHealthEvent(
                        check_type="size",
                        key=key,
                        current_value=float(len(value)),
                        threshold=check.threshold,
                        recommended_action=check.action
                    ))

            elif check.check_type == "version_count":
                max_versions_to_fetch = min(
                    int(check.threshold) + 1,
                    check.max_versions_limit
                )
                history = self._context.get_history(key, max_versions=max_versions_to_fetch)
                if len(history) > check.threshold:
                    events.append(ContextHealthEvent(
                        check_type="version_count",
                        key=key,
                        current_value=float(len(history)),
                        threshold=check.threshold,
                        recommended_action=check.action
                    ))

        return events


def loop_n_times(agent_runner: AgentRunner, context: ContextManager, patterns: PatternRegistry, n: int) -> LogicRunner:
    config = LogicConfig(
        logic_id=f"loop_{n}",
        max_iterations=n
    )
    return LogicRunner(agent_runner, context, patterns, config)


def loop_until_pattern(
    agent_runner: AgentRunner,
    context: ContextManager,
    patterns: PatternRegistry,
    pattern_set: str,
    pattern_name: str,
    target: str = "response",
    max_iterations: int | None = None
) -> LogicRunner:
    config = LogicConfig(
        logic_id=f"loop_until_{pattern_name}",
        max_iterations=max_iterations,
        loop_until_conditions=[
            LogicCondition(
                pattern_set=pattern_set,
                pattern_name=pattern_name,
                match_type="contains",
                target=target
            )
        ]
    )
    return LogicRunner(agent_runner, context, patterns, config)


def loop_until_regex(
    agent_runner: AgentRunner,
    context: ContextManager,
    patterns: PatternRegistry,
    regex_pattern: str,
    target: str = "response",
    max_iterations: int | None = None
) -> LogicRunner:
    config = LogicConfig(
        logic_id=f"loop_until_regex",
        max_iterations=max_iterations,
        loop_until_conditions=[
            LogicCondition(
                pattern_set="default",  # Not used for regex
                pattern_name=regex_pattern,
                match_type="regex",
                target=target
            )
        ]
    )
    return LogicRunner(agent_runner, context, patterns, config)


def stop_on_error(
    agent_runner: AgentRunner,
    context: ContextManager,
    patterns: PatternRegistry,
    max_iterations: int | None = None
) -> LogicRunner:
    config = LogicConfig(
        logic_id="stop_on_error",
        max_iterations=max_iterations,
        break_on_error=True
    )
    return LogicRunner(agent_runner, context, patterns, config)
