"""
Tests for security fixes: buffer overflow protection and history limits.
"""
import pytest
from agentic.patterns import StreamingPatternExtractor, PatternSet, Pattern, SegmentType
from agentic.context import ContextManager, IterationManager
from agentic.storage import RocksDBStorage, StorageConfig
import tempfile
import shutil


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    temp_dir = tempfile.mkdtemp()
    config = StorageConfig(base_dir=temp_dir, db_name_prefix="test")
    storage = RocksDBStorage(config)
    storage.initialize()
    yield storage
    storage.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestBufferOverflowProtection:
    """Test buffer overflow protection in StreamingPatternExtractor."""

    def test_buffer_size_enforced(self):
        """Test that buffer size limit is enforced."""
        pattern_set = PatternSet(
            name="test",
            patterns=[
                Pattern("response", "<response>", "</response>", SegmentType.RESPONSE)
            ]
        )

        # Create extractor with small buffer limit
        extractor = StreamingPatternExtractor(pattern_set, max_buffer_size=100)

        # Feed chunks within limit - should work
        events = list(extractor.feed_chunk("Hello "))
        events = list(extractor.feed_chunk("world"))

        # Try to exceed buffer limit
        large_chunk = "x" * 200
        with pytest.raises(ValueError) as exc_info:
            list(extractor.feed_chunk(large_chunk))

        assert "exceeded maximum size" in str(exc_info.value)
        assert "100 bytes" in str(exc_info.value)

    def test_default_buffer_size(self):
        """Test that default buffer size is 10MB."""
        pattern_set = PatternSet(
            name="test",
            patterns=[
                Pattern("response", "<response>", "</response>", SegmentType.RESPONSE)
            ]
        )

        extractor = StreamingPatternExtractor(pattern_set)
        assert extractor._max_buffer_size == StreamingPatternExtractor.DEFAULT_MAX_BUFFER_SIZE
        assert extractor._max_buffer_size == 10_000_000

    def test_custom_buffer_size(self):
        """Test that custom buffer size can be set."""
        pattern_set = PatternSet(
            name="test",
            patterns=[
                Pattern("response", "<response>", "</response>", SegmentType.RESPONSE)
            ]
        )

        custom_size = 500_000
        extractor = StreamingPatternExtractor(pattern_set, max_buffer_size=custom_size)
        assert extractor._max_buffer_size == custom_size

    def test_buffer_overflow_with_patterns(self):
        """Test buffer overflow protection with actual pattern matching."""
        pattern_set = PatternSet(
            name="test",
            patterns=[
                Pattern("response", "<response>", "</response>", SegmentType.RESPONSE)
            ]
        )

        extractor = StreamingPatternExtractor(pattern_set, max_buffer_size=50)

        # Start feeding a pattern
        list(extractor.feed_chunk("<response>"))
        list(extractor.feed_chunk("Some text"))

        # Try to add more that would exceed limit
        with pytest.raises(ValueError):
            list(extractor.feed_chunk("x" * 100))


class TestHistoryDefaultLimit:
    """Test default history limit in ContextManager."""

    def test_history_default_limit(self, temp_storage):
        """Test that get_history defaults to 100 versions."""
        iteration_mgr = IterationManager(temp_storage)
        context = ContextManager(temp_storage, iteration_mgr)

        key = "test_key"

        # Create 150 versions
        for i in range(150):
            context.set(key, f"version_{i}".encode('utf-8'))

        # Get history without specifying limit - should default to 100
        history = context.get_history(key)

        assert len(history) == 100
        # Should be newest 100 (149 down to 50)
        assert history[0].value == b"version_149"
        assert history[99].value == b"version_50"

    def test_history_explicit_limit(self, temp_storage):
        """Test that explicit limit still works."""
        iteration_mgr = IterationManager(temp_storage)
        context = ContextManager(temp_storage, iteration_mgr)

        key = "test_key"

        # Create 50 versions
        for i in range(50):
            context.set(key, f"version_{i}".encode('utf-8'))

        # Get with explicit limit
        history = context.get_history(key, max_versions=10)

        assert len(history) == 10
        assert history[0].value == b"version_49"
        assert history[9].value == b"version_40"

    def test_history_all_versions_if_under_limit(self, temp_storage):
        """Test that all versions are returned if under default limit."""
        iteration_mgr = IterationManager(temp_storage)
        context = ContextManager(temp_storage, iteration_mgr)

        key = "test_key"

        # Create only 10 versions
        for i in range(10):
            context.set(key, f"version_{i}".encode('utf-8'))

        # Should get all 10
        history = context.get_history(key)

        assert len(history) == 10
        assert history[0].value == b"version_9"
        assert history[9].value == b"version_0"

    def test_history_large_explicit_limit(self, temp_storage):
        """Test that users can still request more than default if needed."""
        iteration_mgr = IterationManager(temp_storage)
        context = ContextManager(temp_storage, iteration_mgr)

        key = "test_key"

        # Create 200 versions
        for i in range(200):
            context.set(key, f"version_{i}".encode('utf-8'))

        # Request all 200 explicitly
        history = context.get_history(key, max_versions=200)

        assert len(history) == 200
        assert history[0].value == b"version_199"
        assert history[199].value == b"version_0"
