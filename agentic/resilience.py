"""
Resilience utilities: retry logic and rate limiting for operations.

Universal retry and rate limiting that works with any async iterator (LLM, tools, custom operations).
"""
from dataclasses import dataclass
from typing import AsyncIterator, Callable, TypeVar
import asyncio
import time
import random

from .events import RetryEvent, RateLimitEvent

T = TypeVar('T')


@dataclass
class RetryConfig:
    max_attempts: int = 3
    backoff: str = "exponential"  # "exponential" | "linear" | "constant"
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = (TimeoutError, ConnectionError, asyncio.TimeoutError)


@dataclass
class RateLimitConfig:
    requests_per_second: float | None = None
    requests_per_minute: float | None = None
    requests_per_hour: float | None = None
    burst_size: int = 10  # Token bucket burst capacity


def _calculate_backoff(attempt: int, config: RetryConfig) -> float:
    if config.backoff == "exponential":
        delay = min(config.base_delay * (2 ** attempt), config.max_delay)
    elif config.backoff == "linear":
        delay = min(config.base_delay * (attempt + 1), config.max_delay)
    else:  # constant
        delay = config.base_delay

    if config.jitter:
        # Add random jitter ±25%
        jitter = delay * 0.25 * (2 * random.random() - 1)
        delay = max(0, delay + jitter)

    return delay


async def retry_stream(
    stream_fn: Callable[[], AsyncIterator[T]],
    config: RetryConfig,
    operation_name: str,
    operation_type: str = "custom",
    step_id: str = ""
) -> AsyncIterator[T | RetryEvent]:
    """
    Universal retry wrapper for any async iterator.

    Wraps the actual operation (LLM provider call, tool execution, etc.) and retries on failure.
    Emits RetryEvent when retrying.

    Args:
        stream_fn: Function that returns async iterator (the operation to retry)
        config: Retry configuration
        operation_name: Name for logging/events
        operation_type: Type of operation ("llm", "tool", "custom")
        step_id: Optional step ID for event correlation

    Yields:
        Items from stream_fn() or RetryEvent instances
    """
    for attempt in range(config.max_attempts):
        try:
            async for item in stream_fn():
                yield item
            return
        except config.retry_on as e:
            if attempt < config.max_attempts - 1:
                delay = _calculate_backoff(attempt, config)

                yield RetryEvent(
                    operation_type=operation_type,
                    operation_name=operation_name,
                    attempt=attempt + 1,
                    max_attempts=config.max_attempts,
                    error=str(e),
                    next_delay_seconds=delay,
                    step_id=step_id
                )

                await asyncio.sleep(delay)
            else:
                raise


class RateLimiter:
    """
    Token bucket rate limiter - works for any operation.

    Limits requests per second, minute, and/or hour using token bucket algorithm.
    Thread-safe via asyncio Condition.
    """

    def __init__(self, config: RateLimitConfig):
        self._config = config
        self._tokens = config.burst_size
        self._last_update = time.time()
        self._condition = asyncio.Condition()

        self._refill_rate = self._calculate_refill_rate()

    def _calculate_refill_rate(self) -> float:
        rates = []
        if self._config.requests_per_second:
            rates.append(self._config.requests_per_second)
        if self._config.requests_per_minute:
            rates.append(self._config.requests_per_minute / 60.0)
        if self._config.requests_per_hour:
            rates.append(self._config.requests_per_hour / 3600.0)

        if not rates:
            raise ValueError("At least one rate limit must be configured")

        # Use most restrictive rate
        return min(rates)

    def _refill_tokens(self):
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._tokens + new_tokens, self._config.burst_size)

    async def acquire(self, tokens: int = 1, operation_name: str = "") -> None:
        """
        Acquire rate limit tokens - blocks if insufficient tokens available.

        Args:
            tokens: Number of tokens to acquire (default 1)
            operation_name: Name for logging
        """
        async with self._condition:
            await self._wait_for_tokens(tokens)
            self._tokens -= tokens

    async def _wait_for_tokens(self, tokens: int):
        while self._tokens < tokens:
            self._refill_tokens()
            if self._tokens < tokens:
                deficit = tokens - self._tokens
                wait_time = deficit / self._refill_rate
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=min(wait_time, 0.1))
                except asyncio.TimeoutError:
                    pass  
            else:
                self._condition.notify_all()

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if insufficient
        """
        async with self._condition:
            self._refill_tokens()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def tokens_available(self) -> float:
        """Get current token count (non-blocking)."""
        return self._tokens


async def rate_limited_stream(
    stream_fn: Callable[[], AsyncIterator[T]],
    limiter: RateLimiter,
    operation_name: str,
    step_id: str = ""
) -> AsyncIterator[T | RateLimitEvent]:
    """
    Universal rate limiting wrapper for any async iterator.

    Acquires rate limit before executing operation. Emits RateLimitEvent after acquisition.

    Args:
        stream_fn: Function that returns async iterator
        limiter: RateLimiter instance
        operation_name: Name for logging/events
        step_id: Optional step ID for event correlation

    Yields:
        RateLimitEvent (once) then items from stream_fn()
    """
    await limiter.acquire(1, operation_name)

    yield RateLimitEvent(
        operation_name=operation_name,
        acquired_at=time.time(),
        tokens_remaining=limiter.tokens_available(),
        step_id=step_id
    )

    async for item in stream_fn():
        yield item


async def resilient_stream(
    stream_fn: Callable[[], AsyncIterator[T]],
    retry_config: RetryConfig | None = None,
    rate_limiter: RateLimiter | None = None,
    operation_name: str = "",
    operation_type: str = "custom",
    step_id: str = ""
) -> AsyncIterator[T | RetryEvent | RateLimitEvent]:
    """
    Combined retry + rate limiting wrapper for any async iterator.

    Universal resilience layer that works for LLM calls, tool execution, or any async operation.
    Rate limiting is applied first (before request), then retry wraps the operation.

    Args:
        stream_fn: Function that returns async iterator (the operation)
        retry_config: Optional retry configuration
        rate_limiter: Optional rate limiter instance
        operation_name: Name for logging/events
        operation_type: Type of operation ("llm", "tool", "custom")
        step_id: Optional step ID for event correlation

    Yields:
        RateLimitEvent (if rate limited), RetryEvent (if retried), and items from stream_fn()

    Example:
        async def my_llm_call():
            async for chunk in provider.stream(prompt):
                yield chunk

        async for item in resilient_stream(
            my_llm_call,
            retry_config=RetryConfig(max_attempts=3),
            rate_limiter=my_rate_limiter,
            operation_name="gpt-4",
            operation_type="llm"
        ):
            if isinstance(item, RetryEvent):
                print(f"Retrying after {item.next_delay_seconds}s...")
            elif isinstance(item, RateLimitEvent):
                print(f"Rate limit acquired, {item.tokens_remaining} tokens remaining")
            else:
                # Actual LLM chunk
                print(item, end="")
    """
    if rate_limiter:
        await rate_limiter.acquire(1, operation_name)
        yield RateLimitEvent(
            operation_name=operation_name,
            acquired_at=time.time(),
            tokens_remaining=rate_limiter.tokens_available(),
            step_id=step_id
        )

    if retry_config:
        async for item in retry_stream(
            stream_fn,
            retry_config,
            operation_name,
            operation_type,
            step_id
        ):
            yield item
    else:
        async for item in stream_fn():
            yield item
