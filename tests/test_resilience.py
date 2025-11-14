"""
Tests for resilience utilities: retry logic and rate limiting.

Covers:
- RetryConfig with different backoff strategies
- Backoff calculation with jitter
- retry_stream() successful retry after failures
- retry_stream() exhausting max attempts
- RateLimiter token bucket mechanics
- RateLimiter acquire() blocking behavior
- RateLimiter try_acquire() non-blocking behavior
- rate_limited_stream() token acquisition
- resilient_stream() combining retry + rate limiting
- Event emission (RetryEvent, RateLimitEvent)
"""
import pytest
import asyncio
import time

from agentic.resilience import (
    RetryConfig,
    RateLimitConfig,
    RateLimiter,
    retry_stream,
    rate_limited_stream,
    resilient_stream,
    _calculate_backoff
)
from agentic.events import RetryEvent, RateLimitEvent


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_retry_config_defaults(self):
        """Test RetryConfig with default values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.backoff == "exponential"
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True
        assert TimeoutError in config.retry_on
        assert ConnectionError in config.retry_on

    def test_retry_config_custom(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_attempts=5,
            backoff="linear",
            base_delay=2.0,
            max_delay=30.0,
            jitter=False,
            retry_on=(ValueError,)
        )
        assert config.max_attempts == 5
        assert config.backoff == "linear"
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.jitter is False
        assert config.retry_on == (ValueError,)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_rate_limit_config_defaults(self):
        """Test RateLimitConfig with default values."""
        config = RateLimitConfig()
        assert config.requests_per_second is None
        assert config.requests_per_minute is None
        assert config.requests_per_hour is None
        assert config.burst_size == 10

    def test_rate_limit_config_custom(self):
        """Test RateLimitConfig with custom values."""
        config = RateLimitConfig(
            requests_per_second=5.0,
            requests_per_minute=300.0,
            burst_size=20
        )
        assert config.requests_per_second == 5.0
        assert config.requests_per_minute == 300.0
        assert config.burst_size == 20


class TestBackoffCalculation:
    """Tests for backoff delay calculation."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(backoff="exponential", base_delay=1.0, jitter=False)

        delay_0 = _calculate_backoff(0, config)
        delay_1 = _calculate_backoff(1, config)
        delay_2 = _calculate_backoff(2, config)

        assert delay_0 == 1.0  # 1.0 * 2^0
        assert delay_1 == 2.0  # 1.0 * 2^1
        assert delay_2 == 4.0  # 1.0 * 2^2

    def test_exponential_backoff_max_delay(self):
        """Test exponential backoff respects max_delay."""
        config = RetryConfig(backoff="exponential", base_delay=1.0, max_delay=5.0, jitter=False)

        delay_10 = _calculate_backoff(10, config)
        assert delay_10 == 5.0  # Capped at max_delay

    def test_linear_backoff(self):
        """Test linear backoff calculation."""
        config = RetryConfig(backoff="linear", base_delay=2.0, jitter=False)

        delay_0 = _calculate_backoff(0, config)
        delay_1 = _calculate_backoff(1, config)
        delay_2 = _calculate_backoff(2, config)

        assert delay_0 == 2.0  # 2.0 * (0 + 1)
        assert delay_1 == 4.0  # 2.0 * (1 + 1)
        assert delay_2 == 6.0  # 2.0 * (2 + 1)

    def test_linear_backoff_max_delay(self):
        """Test linear backoff respects max_delay."""
        config = RetryConfig(backoff="linear", base_delay=10.0, max_delay=15.0, jitter=False)

        delay_10 = _calculate_backoff(10, config)
        assert delay_10 == 15.0  # Capped at max_delay

    def test_constant_backoff(self):
        """Test constant backoff calculation."""
        config = RetryConfig(backoff="constant", base_delay=3.0, jitter=False)

        delay_0 = _calculate_backoff(0, config)
        delay_1 = _calculate_backoff(1, config)
        delay_2 = _calculate_backoff(2, config)

        assert delay_0 == 3.0
        assert delay_1 == 3.0
        assert delay_2 == 3.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds random variation to backoff."""
        config = RetryConfig(backoff="constant", base_delay=10.0, jitter=True)

        # Run multiple times to check variation
        delays = [_calculate_backoff(0, config) for _ in range(10)]

        # Should have variation (not all identical)
        assert len(set(delays)) > 1

        # All delays should be within ±25% of base_delay
        for delay in delays:
            assert 7.5 <= delay <= 12.5

    def test_jitter_non_negative(self):
        """Test that jitter never produces negative delays."""
        config = RetryConfig(backoff="constant", base_delay=0.5, jitter=True)

        for _ in range(20):
            delay = _calculate_backoff(0, config)
            assert delay >= 0


@pytest.mark.asyncio
class TestRetryStream:
    """Tests for retry_stream() function."""

    async def test_retry_stream_success_first_attempt(self):
        """Test retry_stream when operation succeeds on first attempt."""
        call_count = 0

        async def successful_stream():
            nonlocal call_count
            call_count += 1
            yield "result"

        config = RetryConfig(max_attempts=3)
        results = []

        async for item in retry_stream(successful_stream, config, "test_op"):
            results.append(item)

        assert call_count == 1
        assert results == ["result"]

    async def test_retry_stream_success_after_failure(self):
        """Test retry_stream succeeds after initial failures."""
        call_count = 0

        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Simulated timeout")
            yield "success"

        config = RetryConfig(max_attempts=5, base_delay=0.01)
        results = []
        retry_events = []

        async for item in retry_stream(failing_then_success, config, "test_op", "llm"):
            if isinstance(item, RetryEvent):
                retry_events.append(item)
            else:
                results.append(item)

        assert call_count == 3
        assert results == ["success"]
        assert len(retry_events) == 2  # Failed twice, then succeeded

    async def test_retry_stream_exhausts_attempts(self):
        """Test retry_stream raises after exhausting max attempts."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")
            yield "never_reached"

        config = RetryConfig(max_attempts=3, base_delay=0.01)

        with pytest.raises(ConnectionError):
            async for item in retry_stream(always_fails, config, "test_op"):
                pass

        assert call_count == 3

    async def test_retry_stream_emits_retry_events(self):
        """Test that retry_stream emits RetryEvent with correct fields."""
        call_count = 0

        async def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("First attempt fails")
            yield "success"

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry_events = []

        async for item in retry_stream(fail_once, config, "my_operation", "tool", "step_123"):
            if isinstance(item, RetryEvent):
                retry_events.append(item)

        assert len(retry_events) == 1
        event = retry_events[0]
        assert event.type == "retry"
        assert event.operation_type == "tool"
        assert event.operation_name == "my_operation"
        assert event.attempt == 1
        assert event.max_attempts == 3
        assert "First attempt fails" in event.error
        assert event.next_delay_seconds > 0
        assert event.step_id == "step_123"

    async def test_retry_stream_respects_retry_on(self):
        """Test that retry_stream only retries configured exceptions."""
        call_count = 0

        async def wrong_exception():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not in retry_on")
            yield "never"

        config = RetryConfig(max_attempts=3, retry_on=(TimeoutError,))

        # Should not retry ValueError
        with pytest.raises(ValueError):
            async for item in retry_stream(wrong_exception, config, "test"):
                pass

        assert call_count == 1  # Only attempted once

    async def test_retry_stream_multiple_yields(self):
        """Test retry_stream with operation that yields multiple items."""
        call_count = 0

        async def multi_yield():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Fail first")
            yield "item1"
            yield "item2"
            yield "item3"

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        results = []

        async for item in retry_stream(multi_yield, config, "test"):
            if not isinstance(item, RetryEvent):
                results.append(item)

        assert results == ["item1", "item2", "item3"]


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)

        assert limiter._tokens == 5  # Initial burst_size
        assert limiter._refill_rate == 10.0

    def test_rate_limiter_refill_rate_calculation(self):
        """Test refill rate calculation from config."""
        # Test with requests_per_second
        config1 = RateLimitConfig(requests_per_second=10.0)
        limiter1 = RateLimiter(config1)
        assert limiter1._refill_rate == 10.0

        # Test with requests_per_minute
        config2 = RateLimitConfig(requests_per_minute=120.0)
        limiter2 = RateLimiter(config2)
        assert limiter2._refill_rate == 2.0  # 120 / 60

        # Test with requests_per_hour
        config3 = RateLimitConfig(requests_per_hour=3600.0)
        limiter3 = RateLimiter(config3)
        assert limiter3._refill_rate == 1.0  # 3600 / 3600

    def test_rate_limiter_most_restrictive_rate(self):
        """Test that rate limiter uses most restrictive rate."""
        config = RateLimitConfig(
            requests_per_second=10.0,  # 10/s
            requests_per_minute=300.0,  # 5/s (more restrictive)
            requests_per_hour=36000.0   # 10/s
        )
        limiter = RateLimiter(config)

        # Should use most restrictive: 300/60 = 5/s
        assert limiter._refill_rate == 5.0

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_immediate(self):
        """Test acquire() when tokens are available."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=10)
        limiter = RateLimiter(config)

        start = time.time()
        await limiter.acquire(1)
        elapsed = time.time() - start

        # Should be immediate
        assert elapsed < 0.1
        assert limiter._tokens == 9  # 10 - 1

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=10)
        limiter = RateLimiter(config)

        await limiter.acquire(3)
        assert limiter._tokens == 7  # 10 - 3

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_blocks_when_insufficient(self):
        """Test that acquire() blocks when insufficient tokens."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)

        # Consume all tokens
        await limiter.acquire(5)
        assert limiter._tokens == 0

        # This should block and wait for refill
        start = time.time()
        await limiter.acquire(1)
        elapsed = time.time() - start

        # Should wait ~0.1 seconds for 1 token at 10/s rate
        assert elapsed >= 0.05  # At least some wait

    @pytest.mark.asyncio
    async def test_rate_limiter_try_acquire_success(self):
        """Test try_acquire() succeeds when tokens available."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)

        result = await limiter.try_acquire(2)
        assert result is True
        assert limiter._tokens == 3

    @pytest.mark.asyncio
    async def test_rate_limiter_try_acquire_failure(self):
        """Test try_acquire() fails when insufficient tokens."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)

        # Try to acquire more than available
        result = await limiter.try_acquire(10)
        assert result is False
        assert limiter._tokens == 5  # Unchanged

    @pytest.mark.asyncio
    async def test_rate_limiter_refill_over_time(self):
        """Test that tokens refill over time."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=10)
        limiter = RateLimiter(config)

        # Consume all tokens
        await limiter.acquire(10)
        assert limiter._tokens == 0

        # Wait for refill
        await asyncio.sleep(0.05)  # Should refill ~5 tokens at 100/s

        # Should have some tokens now
        limiter._refill_tokens()
        assert limiter._tokens > 0
        assert limiter._tokens <= 10  # Capped at burst_size

    def test_rate_limiter_tokens_available(self):
        """Test tokens_available() method."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=10)
        limiter = RateLimiter(config)

        assert limiter.tokens_available() == 10

    def test_rate_limiter_invalid_config_raises(self):
        """Test RateLimiter raises ValueError with no configured rates."""
        config = RateLimitConfig()  # All rates are None

        with pytest.raises(ValueError, match="At least one rate limit must be configured"):
            RateLimiter(config)

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_acquire(self):
        """Test RateLimiter handles concurrent acquire() calls correctly."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)

        results = []

        async def acquire_token(task_id: int):
            await limiter.acquire(1)
            results.append(task_id)

        # Launch 5 concurrent tasks (exactly burst_size)
        tasks = [asyncio.create_task(acquire_token(i)) for i in range(5)]
        await asyncio.gather(*tasks)

        # All tasks should complete
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}
        # All tokens should be consumed
        assert limiter._tokens == 0

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_blocking(self):
        """Test RateLimiter blocks concurrent requests when tokens exhausted."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=2)
        limiter = RateLimiter(config)

        acquired = []

        async def acquire_and_record():
            await limiter.acquire(1)
            acquired.append(time.time())

        # Launch 5 concurrent tasks with only 2 tokens available
        start = time.time()
        tasks = [asyncio.create_task(acquire_and_record()) for i in range(5)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        # Should take time as tokens refill for the 3rd, 4th, 5th requests
        assert len(acquired) == 5
        # With 100/s rate and 2 burst, need to wait for 3 more tokens
        # Minimum ~0.03s (3 tokens at 100/s)
        assert elapsed > 0.02  # At least some waiting occurred

        # Verify tokens were acquired sequentially with proper timing
        # First 2 should be immediate, then 3 more with refill delays
        assert acquired[1] - acquired[0] < 0.01  # First 2 are fast


@pytest.mark.asyncio
class TestRateLimitedStream:
    """Tests for rate_limited_stream() function."""

    async def test_rate_limited_stream_acquires_token(self):
        """Test that rate_limited_stream acquires token before execution."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)

        async def simple_stream():
            yield "data"

        initial_tokens = limiter.tokens_available()

        async for item in rate_limited_stream(simple_stream, limiter, "test_op"):
            pass

        # Should have consumed 1 token
        assert limiter.tokens_available() < initial_tokens

    async def test_rate_limited_stream_emits_event(self):
        """Test that rate_limited_stream emits RateLimitEvent."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)

        async def simple_stream():
            yield "data"

        events = []
        async for item in rate_limited_stream(simple_stream, limiter, "my_op", "step_456"):
            if isinstance(item, RateLimitEvent):
                events.append(item)

        assert len(events) == 1
        event = events[0]
        assert event.type == "rate_limit"
        assert event.operation_name == "my_op"
        assert event.step_id == "step_456"
        assert isinstance(event.acquired_at, float)
        assert event.tokens_remaining >= 0

    async def test_rate_limited_stream_passes_through_data(self):
        """Test that rate_limited_stream passes through all data."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=10)
        limiter = RateLimiter(config)

        async def multi_yield():
            yield "item1"
            yield "item2"
            yield "item3"

        results = []
        async for item in rate_limited_stream(multi_yield, limiter, "test"):
            if not isinstance(item, RateLimitEvent):
                results.append(item)

        assert results == ["item1", "item2", "item3"]


@pytest.mark.asyncio
class TestResilientStream:
    """Tests for resilient_stream() combining retry + rate limiting."""

    async def test_resilient_stream_no_resilience(self):
        """Test resilient_stream with no retry or rate limiting."""
        async def simple_stream():
            yield "data"

        results = []
        async for item in resilient_stream(simple_stream):
            results.append(item)

        assert results == ["data"]

    async def test_resilient_stream_retry_only(self):
        """Test resilient_stream with only retry."""
        call_count = 0

        async def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Fail")
            yield "success"

        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        results = []

        async for item in resilient_stream(
            fail_once,
            retry_config=retry_config,
            operation_name="test",
            operation_type="llm"
        ):
            if not isinstance(item, RetryEvent):
                results.append(item)

        assert results == ["success"]
        assert call_count == 2

    async def test_resilient_stream_rate_limit_only(self):
        """Test resilient_stream with only rate limiting."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=10)
        limiter = RateLimiter(config)

        async def simple_stream():
            yield "data"

        events = []
        async for item in resilient_stream(simple_stream, rate_limiter=limiter, operation_name="test"):
            if isinstance(item, RateLimitEvent):
                events.append(item)

        assert len(events) == 1

    async def test_resilient_stream_combined(self):
        """Test resilient_stream with both retry and rate limiting."""
        call_count = 0

        async def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Fail")
            yield "success"

        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        rate_config = RateLimitConfig(requests_per_second=100.0, burst_size=10)
        limiter = RateLimiter(rate_config)

        rate_events = []
        retry_events = []
        results = []

        async for item in resilient_stream(
            fail_once,
            retry_config=retry_config,
            rate_limiter=limiter,
            operation_name="test_op",
            operation_type="custom",
            step_id="step_789"
        ):
            if isinstance(item, RateLimitEvent):
                rate_events.append(item)
            elif isinstance(item, RetryEvent):
                retry_events.append(item)
            else:
                results.append(item)

        # Should have rate limit event first, then retry, then success
        assert len(rate_events) == 1
        assert len(retry_events) == 1
        assert results == ["success"]

    async def test_resilient_stream_event_order(self):
        """Test that events are emitted in correct order (rate limit before retry)."""
        call_count = 0

        async def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Fail")
            yield "data"

        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        rate_config = RateLimitConfig(requests_per_second=100.0, burst_size=10)
        limiter = RateLimiter(rate_config)

        event_sequence = []

        async for item in resilient_stream(
            fail_once,
            retry_config=retry_config,
            rate_limiter=limiter,
            operation_name="test"
        ):
            if isinstance(item, RateLimitEvent):
                event_sequence.append("rate_limit")
            elif isinstance(item, RetryEvent):
                event_sequence.append("retry")
            else:
                event_sequence.append("data")

        # Rate limit should come first
        assert event_sequence[0] == "rate_limit"


class TestResilientStreamEdgeCases:
    """Tests for edge cases in resilience utilities."""

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Test resilient_stream with empty stream."""
        async def empty_stream():
            return
            yield  # Never reached

        results = []
        async for item in resilient_stream(empty_stream):
            results.append(item)

        assert results == []

    @pytest.mark.asyncio
    async def test_exception_during_stream(self):
        """Test exception raised during stream (not during initial call)."""
        call_count = 0

        async def fail_mid_stream():
            nonlocal call_count
            call_count += 1
            yield "item1"
            if call_count < 2:
                raise RuntimeError("Mid-stream error")
            yield "item2"

        config = RetryConfig(max_attempts=3, retry_on=(RuntimeError,), base_delay=0.01)
        results = []

        async for item in resilient_stream(fail_mid_stream, retry_config=config, operation_name="test"):
            if not isinstance(item, RetryEvent):
                results.append(item)

        # Should have retried and succeeded
        assert "item1" in results
        assert "item2" in results
        assert call_count == 2
