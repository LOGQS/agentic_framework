"""
Mock LLM provider for testing.

Provides a simple implementation of the LLMProvider protocol for test purposes.
"""
from typing import AsyncIterator


class MockLLMProvider:
    """Mock LLM provider for testing that returns canned responses."""

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
        if self._simulate_streaming:
            words = self._response.split()
            for i, word in enumerate(words):
                if i < len(words) - 1:
                    yield word + " "
                else:
                    yield word
        else:
            yield self._response

    def set_response(self, response: str) -> None:
        self._response = response
