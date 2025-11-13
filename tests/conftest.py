"""
Shared fixtures for pytest test suite.

This module provides fixtures for:
- Temporary storage directories
- Storage instances
- Context managers
- Pattern registries
- Tool registries
- Mock agents
- Cleanup utilities
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator

from agentic.storage import RocksDBStorage, StorageConfig
from agentic.context import ContextManager, IterationManager
from agentic.patterns import PatternRegistry, PatternSet, Pattern, create_default_pattern_set
from agentic.tools import ToolRegistry, Tool, ToolDefinition, create_tool
from agentic.agent import Agent, AgentRunner, MockLLMProvider
from agentic.core import AgentConfig, ProcessingMode, SegmentType


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test storage.

    Yields:
        Path to temporary directory that is cleaned up after test.
    """
    tmp = tempfile.mkdtemp()
    tmp_path = Path(tmp)
    yield tmp_path
    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def storage_config(temp_dir: Path) -> StorageConfig:
    """
    Create storage configuration with temporary directory.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        StorageConfig instance configured for testing
    """
    return StorageConfig(
        base_dir=temp_dir,
        db_name_prefix="test_context",
        app_id="test_app"
    )


@pytest.fixture
def storage(storage_config: StorageConfig) -> Generator[RocksDBStorage, None, None]:
    """
    Create and initialize RocksDB storage instance.

    Args:
        storage_config: Storage configuration fixture

    Yields:
        Initialized RocksDBStorage instance that is closed after test.
    """
    storage_instance = RocksDBStorage(storage_config)
    storage_instance.initialize()
    yield storage_instance
    storage_instance.close()


@pytest.fixture
def iteration_manager(storage: RocksDBStorage) -> IterationManager:
    """
    Create iteration manager instance.

    Args:
        storage: Storage fixture

    Returns:
        IterationManager instance for tracking iterations
    """
    return IterationManager(storage)


@pytest.fixture
def context_manager(storage: RocksDBStorage, iteration_manager: IterationManager) -> ContextManager:
    """
    Create context manager instance.

    Args:
        storage: Storage fixture
        iteration_manager: Iteration manager fixture

    Returns:
        ContextManager instance for managing versioned context
    """
    return ContextManager(storage, iteration_manager)


@pytest.fixture
def pattern_registry(storage: RocksDBStorage) -> PatternRegistry:
    """
    Create pattern registry with default patterns registered.

    Args:
        storage: Storage fixture

    Returns:
        PatternRegistry instance with default pattern set
    """
    registry = PatternRegistry(storage)
    default_set = create_default_pattern_set()
    registry.register_pattern_set(default_set)
    return registry


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """
    Create tool registry with sample tools.

    Returns:
        ToolRegistry instance with basic test tools registered
    """
    registry = ToolRegistry()

    # Simple echo tool
    def echo_func(inputs: dict) -> dict:
        return {"result": inputs.get("message", "")}

    echo_tool = create_tool(
        name="echo",
        func=echo_func,
        input_schema={"message": "string"},
        output_schema={"result": "string"},
        timeout_seconds=5.0,
        description="Echo the input message"
    )
    registry.register(echo_tool)

    # Calculator tool
    def calc_func(inputs: dict) -> dict:
        a = inputs.get("a", 0)
        b = inputs.get("b", 0)
        op = inputs.get("operation", "add")

        if op == "add":
            result = a + b
        elif op == "subtract":
            result = a - b
        elif op == "multiply":
            result = a * b
        elif op == "divide":
            result = a / b if b != 0 else None
        else:
            result = None

        return {"result": result}

    calc_tool = create_tool(
        name="calculator",
        func=calc_func,
        input_schema={"a": "number", "b": "number", "operation": "string"},
        output_schema={"result": "number"},
        timeout_seconds=5.0,
        description="Perform arithmetic operations"
    )
    registry.register(calc_tool)

    return registry


@pytest.fixture
def mock_llm_provider() -> MockLLMProvider:
    """
    Create mock LLM provider with default response.

    Returns:
        MockLLMProvider instance for testing agent execution
    """
    return MockLLMProvider(
        response="This is a test response.",
        simulate_streaming=False
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    """
    Create basic agent configuration for testing.

    Returns:
        AgentConfig instance with default test settings
    """
    return AgentConfig(
        agent_id="test_agent",
        provider="mock",
        model="mock-model",
        max_tokens=1000,
        temperature=0.7,
        tools_allowed=["echo", "calculator"],
        input_mapping=[("system_prompt", "prepend")],
        output_mapping=[("last_output", "set_latest")],
        pattern_set="default",
        auto_increment_iteration=True,
        processing_mode=ProcessingMode.THREAD
    )


@pytest.fixture
def agent(
    agent_config: AgentConfig,
    context_manager: ContextManager,
    pattern_registry: PatternRegistry,
    tool_registry: ToolRegistry,
    mock_llm_provider: MockLLMProvider
) -> Agent:
    """
    Create fully configured agent instance.

    Args:
        agent_config: Agent configuration fixture
        context_manager: Context manager fixture
        pattern_registry: Pattern registry fixture
        tool_registry: Tool registry fixture
        mock_llm_provider: Mock LLM provider fixture

    Returns:
        Agent instance ready for testing
    """
    return Agent(
        config=agent_config,
        context=context_manager,
        patterns=pattern_registry,
        tools=tool_registry,
        provider_client=mock_llm_provider
    )


@pytest.fixture
def agent_runner(agent: Agent) -> AgentRunner:
    """
    Create agent runner instance.

    Args:
        agent: Agent fixture

    Returns:
        AgentRunner instance for executing agent steps
    """
    return AgentRunner(agent)


@pytest.fixture
def sample_pattern_set() -> PatternSet:
    """
    Create a sample pattern set for testing.

    Returns:
        PatternSet with custom patterns for testing
    """
    return PatternSet(
        name="test_patterns",
        patterns=[
            Pattern(
                name="custom_tool",
                start_tag="[TOOL:",
                end_tag=":END]",
                segment_type=SegmentType.TOOL,
                greedy=False
            ),
            Pattern(
                name="thought",
                start_tag="<thought>",
                end_tag="</thought>",
                segment_type=SegmentType.REASONING,
                greedy=False
            ),
            Pattern(
                name="answer",
                start_tag="<answer>",
                end_tag="</answer>",
                segment_type=SegmentType.RESPONSE,
                greedy=False
            )
        ],
        default_response_behavior="all_remaining"
    )


# Utility functions for tests

def create_test_storage(temp_dir: Path, db_name: str = "test_db") -> RocksDBStorage:
    """
    Utility function to create and initialize test storage.

    Args:
        temp_dir: Temporary directory path
        db_name: Database name prefix

    Returns:
        Initialized RocksDBStorage instance
    """
    config = StorageConfig(
        base_dir=temp_dir,
        db_name_prefix=db_name,
        app_id="test_util"
    )
    storage = RocksDBStorage(config)
    storage.initialize()
    return storage


def cleanup_storage(storage: RocksDBStorage) -> None:
    """
    Utility function to properly close storage.

    Args:
        storage: Storage instance to close
    """
    if storage:
        storage.close()
