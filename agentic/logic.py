"""
Logic flows for controlling agent execution across iterations.
"""
from dataclasses import dataclass, field
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import AsyncIterator

from .agent import AgentRunner
from .context import ContextManager
from .patterns import PatternRegistry
from .core import AgentStepResult, AgentStatus, ProcessingMode
from .events import (
    AgentEvent, StatusEvent, StepCompleteEvent,
    LLMCompleteEvent, LLMTokenEvent, PatternEndEvent, ToolEndEvent,
    PatternStartEvent, ToolStartEvent, ToolOutputEvent, ContextWriteEvent, ErrorEvent, PatternContentEvent
)


@dataclass
class LogicCondition:
    """Condition for controlling logic flow."""
    pattern_set: str
    pattern_name: str
    match_type: str  # "contains" | "equals" | "regex"
    target: str  # "response" | "reasoning" | "tool_output" | "context:{key}"
    evaluation_point: str = "auto"  # "auto" | "llm_token" | "llm_complete" | "tool_detected" | "tool_finished" | "step_complete" | "any_event"
    # "auto" uses smart defaults: pattern/regex → llm_complete, context → step_complete


@dataclass
class LogicConfig:
    """Configuration for logic execution."""
    logic_id: str
    max_iterations: int | None = None
    stop_conditions: list[LogicCondition] = field(default_factory=list)
    loop_until_conditions: list[LogicCondition] = field(default_factory=list)
    break_on_error: bool = True
    processing_mode: ProcessingMode | None = ProcessingMode.THREAD  # Default to THREAD if not specified


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
        results = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def collect():
                async for event in self.run_stream(initial_input, processing_mode):
                    if isinstance(event, StepCompleteEvent):
                        results.append(event.result)
            loop.run_until_complete(collect())
        finally:
            loop.close()
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
        - "llm_token": on every LLM chunk as it streams (LLMTokenEvent)
        - "llm_complete": after LLMCompleteEvent
        - "tool_detected": after PatternEndEvent with tool type
        - "tool_finished": after ToolEndEvent (tool execution completes)
        - "step_complete": after StepCompleteEvent
        - "any_event": on every event
        """
        if processing_mode is None:
            processing_mode = self._config.processing_mode

        yield StatusEvent(AgentStatus.OK, f"Starting logic loop: {self._config.logic_id}")

        results: list[AgentStepResult] = []
        iteration_count = 0
        current_input = initial_input
        current_step_result = None
        partial_raw_output = ""

        while True:
            if self._config.max_iterations is not None:
                if iteration_count >= self._config.max_iterations:
                    yield StatusEvent(AgentStatus.DONE, f"Max iterations reached: {self._config.max_iterations}")
                    break

            partial_raw_output = ""

            async for event in self._agent_runner.step_stream(current_input, processing_mode):
                yield event

                if isinstance(event, LLMTokenEvent):
                    partial_raw_output += event.token

                should_check = False
                event_type = None
                eval_context = None

                if isinstance(event, LLMTokenEvent):
                    event_type = "llm_token"
                    should_check = self._has_conditions_for_event("llm_token")
                    eval_context = {"raw_output": partial_raw_output}

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
                        "tool_output": str(event.result.output) if event.result.output else "",
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
                        "tool_output": str(event.output),
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
        """Internal synchronous implementation of logic loop."""
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
        """Evaluate condition against result."""
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
            record = self._context.get(context_key)
            if record:
                try:
                    return record.value.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    return None
        return None

    def _get_target_text(self, target: str, result: AgentStepResult) -> str | None:
        """Extract target text from result based on target specification."""
        if target == "response":
            return result.segments.response

        elif target == "reasoning":
            if result.segments.reasoning:
                return "\n".join(result.segments.reasoning)
            return None

        elif target == "tool_output":
            if result.tool_results:
                outputs = [str(tr.output) for tr in result.tool_results]
                return "\n".join(outputs)
            return None

        elif target.startswith("context:"):
            context_key = target[8:]
            record = self._context.get(context_key)
            if record:
                try:
                    return record.value.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    return None
            return None

        return None

    def _has_conditions_for_event(self, event_type: str) -> bool:
        """Check if any conditions should be evaluated for this event type."""
        for condition in self._config.stop_conditions + self._config.loop_until_conditions:
            if self._should_evaluate_at_event(condition, event_type):
                return True
        return False

    def _should_evaluate_at_event(self, condition: LogicCondition, event_type: str) -> bool:
        """Determine if condition should be evaluated at this event type."""
        eval_point = condition.evaluation_point

        if eval_point == "any_event":
            return True
        elif eval_point == event_type:
            return True
        elif eval_point == "auto":
            # Infer from target
            if condition.target.startswith("context:"):
                # Context conditions evaluate at step_complete
                return event_type == "step_complete"
            elif condition.match_type == "contains":
                # Pattern matching evaluates at llm_complete
                return event_type == "llm_complete"
            else:
                # Default to step_complete
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
        """Evaluate condition using partial context dict."""
        if condition.match_type == "regex":
            # For regex, get target text from context
            target_text = None

            if condition.target == "response" or condition.target == "reasoning":
                target_text = context.get("raw_output")
            elif condition.target == "tool_output":
                target_text = context.get("tool_output")
            elif condition.target.startswith("context:"):
                # Read from DB context
                context_key = condition.target[8:]
                record = self._context.get(context_key)
                if record:
                    try:
                        target_text = record.value.decode('utf-8')
                    except (UnicodeDecodeError, AttributeError):
                        return False

            if target_text is None:
                return False

            try:
                pattern = re.compile(condition.pattern_name)
                return pattern.search(target_text) is not None
            except re.error:
                return False

        elif condition.match_type == "contains":
            # For pattern contains, check if pattern exists in target text
            # Get target text based on condition.target
            target_text = None

            if condition.target == "response" or condition.target == "reasoning":
                target_text = context.get("raw_output")
            elif condition.target == "tool_output":
                target_text = context.get("tool_output")
            elif condition.target.startswith("context:"):
                context_key = condition.target[8:]
                record = self._context.get(context_key)
                if record:
                    try:
                        target_text = record.value.decode('utf-8')
                    except (UnicodeDecodeError, AttributeError):
                        return False

            if target_text is None:
                return False

            pattern_set = self._patterns.get_pattern_set(condition.pattern_set)
            if pattern_set is None:
                return False

            # Find the pattern object
            pattern_obj = None
            for p in pattern_set.patterns:
                if p.name == condition.pattern_name:
                    pattern_obj = p
                    break

            if pattern_obj is None:
                return False

            # Check if pattern tags exist in target text
            start_escaped = re.escape(pattern_obj.start_tag)
            end_escaped = re.escape(pattern_obj.end_tag)
            quantifier = ".*" if pattern_obj.greedy else ".*?"
            regex = f"{start_escaped}({quantifier}){end_escaped}"

            matches = re.search(regex, target_text, re.DOTALL)
            return matches is not None

        elif condition.match_type == "equals":
            # For equals, compare directly
            target_text = None

            if condition.target == "response":
                target_text = context.get("raw_output")
            elif condition.target == "tool_output":
                target_text = context.get("tool_output")
            elif condition.target.startswith("context:"):
                context_key = condition.target[8:]
                record = self._context.get(context_key)
                if record:
                    try:
                        target_text = record.value.decode('utf-8')
                    except (UnicodeDecodeError, AttributeError):
                        return False

            if target_text is None:
                return False

            return target_text == condition.pattern_name

        return False

    def _run_in_thread(self, initial_input: str | None) -> list[AgentStepResult]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_impl, initial_input)
            return future.result()

    def _run_in_process(self, initial_input: str | None) -> list[AgentStepResult]:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_impl, initial_input)
            return future.result()

    def _run_async(self, initial_input: str | None) -> list[AgentStepResult]:
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call sync _run_async from within an async context. "
                "Use async/await pattern instead."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e) or "no current event loop" in str(e):
                return asyncio.run(self._async_wrapper(initial_input))
            else:
                raise

    async def _async_wrapper(self, initial_input: str | None) -> list[AgentStepResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_impl, initial_input)


# Convenience functions for common logic patterns

def loop_n_times(agent_runner: AgentRunner, context: ContextManager, patterns: PatternRegistry, n: int) -> LogicRunner:
    """Create a LogicRunner that loops N times."""
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
    """Create a LogicRunner that loops until a pattern is found."""
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
    """Create a LogicRunner that loops until a regex matches."""
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
    """Create a LogicRunner that stops on first error."""
    config = LogicConfig(
        logic_id="stop_on_error",
        max_iterations=max_iterations,
        break_on_error=True
    )
    return LogicRunner(agent_runner, context, patterns, config)
