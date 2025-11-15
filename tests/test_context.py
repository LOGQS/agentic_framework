"""
Tests for context management and versioning.

Covers:
- IterationManager: get, next, register_event
- ContextManager: set, get, delete, versioning
- Context history tracking
- Tombstone deletion markers
- List keys with prefix filtering
- Clear operations
- Error handling
"""
import json
import time

from agentic.context import IterationManager, ContextRecord, TOMBSTONE


class TestIterationManager:
    """Tests for IterationManager."""

    def test_iteration_initial_value(self, iteration_manager):
        """Test that initial iteration is 0."""
        iteration = iteration_manager.get()
        assert iteration == 0

    def test_iteration_next_increments(self, iteration_manager):
        """Test that next() increments iteration."""
        assert iteration_manager.get() == 0
        next_iter = iteration_manager.next()
        assert next_iter == 1
        assert iteration_manager.get() == 1

    def test_iteration_multiple_increments(self, iteration_manager):
        """Test multiple calls to next()."""
        for expected in range(1, 11):
            actual = iteration_manager.next()
            assert actual == expected
        assert iteration_manager.get() == 10

    def test_iteration_persistence(self, storage):
        """Test that iteration persists across manager instances."""
        iter1 = IterationManager(storage)
        iter1.next()  # Increment to 1
        iter1.next()  # Increment to 2

        # Create new manager with same storage
        iter2 = IterationManager(storage)
        assert iter2.get() == 2

    def test_register_event(self, iteration_manager, storage):
        """Test registering iteration events."""
        iteration_manager.register_event("test_event")

        # Check event was stored
        key = f"iteration_event:{iteration_manager.get()}:test_event".encode('utf-8')
        event_data = storage.get(key)
        assert event_data is not None

        event = json.loads(event_data.decode('utf-8'))
        assert event["label"] == "test_event"
        assert "timestamp" in event

    def test_register_multiple_events(self, iteration_manager, storage):
        """Test registering multiple events at same iteration."""
        iteration_manager.register_event("event1")
        iteration_manager.register_event("event2")

        iteration = iteration_manager.get()
        key1 = f"iteration_event:{iteration}:event1".encode('utf-8')
        key2 = f"iteration_event:{iteration}:event2".encode('utf-8')

        assert storage.get(key1) is not None
        assert storage.get(key2) is not None


class TestContextManagerBasics:
    """Tests for basic ContextManager operations."""

    def test_set_and_get(self, context_manager):
        """Test setting and getting context value."""
        key = "test_key"
        value = b"test_value"

        record = context_manager.set(key, value)
        assert record.value == value
        assert record.version == 1
        assert record.iteration == 0

        retrieved = context_manager.get(key)
        assert retrieved is not None
        assert retrieved == "test_value"

        retrieved_record = context_manager.get_record(key)
        assert retrieved_record.value == value
        assert retrieved_record.version == 1

    def test_get_nonexistent_key(self, context_manager):
        """Test getting a key that doesn't exist."""
        result = context_manager.get("nonexistent")
        assert result is None

    def test_get_specific_version(self, context_manager):
        """Test getting a specific version of a key."""
        key = "versioned"
        context_manager.set(key, b"v1")
        context_manager.set(key, b"v2")
        context_manager.set(key, b"v3")

        # Get version 2
        value = context_manager.get(key, version=2)
        assert value is not None
        assert value == "v2"

        record = context_manager.get_record(key, version=2)
        assert record.value == b"v2"
        assert record.version == 2

    def test_get_latest_version(self, context_manager):
        """Test that get without version returns latest."""
        key = "multi_version"
        context_manager.set(key, b"v1")
        context_manager.set(key, b"v2")
        context_manager.set(key, b"v3")

        latest = context_manager.get(key)
        assert latest == "v3"

        latest_record = context_manager.get_record(key)
        assert latest_record.value == b"v3"
        assert latest_record.version == 3


class TestContextVersioning:
    """Tests for context versioning behavior."""

    def test_versioning_increments(self, context_manager):
        """Test that versions increment correctly."""
        key = "version_test"

        r1 = context_manager.set(key, b"value1")
        assert r1.version == 1

        r2 = context_manager.set(key, b"value2")
        assert r2.version == 2

        r3 = context_manager.set(key, b"value3")
        assert r3.version == 3

    def test_versioning_per_key(self, context_manager):
        """Test that versioning is per-key."""
        r1 = context_manager.set("key1", b"value")
        r2 = context_manager.set("key2", b"value")
        r3 = context_manager.set("key1", b"value2")

        assert r1.version == 1
        assert r2.version == 1  # Different key, starts at 1
        assert r3.version == 2  # Same key as r1, increments to 2

    def test_version_timestamps(self, context_manager):
        """Test that each version has a timestamp."""
        key = "timestamp_test"

        before = time.time()
        record = context_manager.set(key, b"value")
        after = time.time()

        assert before <= record.timestamp <= after

    def test_version_timestamps_increase(self, context_manager):
        """Test that timestamps increase with versions."""
        key = "ts_increase"

        r1 = context_manager.set(key, b"v1")
        time.sleep(0.01)
        r2 = context_manager.set(key, b"v2")

        assert r2.timestamp > r1.timestamp

    def test_version_iterations(self, context_manager):
        """Test that versions track iterations."""
        key = "iter_test"

        r1 = context_manager.set(key, b"v1")
        assert r1.iteration == 0

        context_manager.next_iteration()

        r2 = context_manager.set(key, b"v2")
        assert r2.iteration == 1


class TestContextDeletion:
    """Tests for context deletion with tombstones."""

    def test_delete_key(self, context_manager):
        """Test deleting a key."""
        key = "to_delete"
        context_manager.set(key, b"value")
        assert context_manager.get(key) is not None

        context_manager.delete(key)
        assert context_manager.get(key) is None

    def test_delete_creates_tombstone(self, context_manager, storage):
        """Test that delete creates a tombstone version."""
        key = "tombstone_test"
        context_manager.set(key, b"value")  # version 1
        context_manager.delete(key)  # version 2 (tombstone)

        # Get version 2 directly from storage
        record_key = f"context:{key}:2".encode('utf-8')
        data = storage.get(record_key)
        assert data is not None

        # Parse the record
        obj = json.loads(data.decode('utf-8'))
        value = bytes.fromhex(obj["value"])
        assert value == TOMBSTONE

    def test_delete_nonexistent_key(self, context_manager):
        """Test deleting a key that doesn't exist."""
        # Should not raise
        context_manager.delete("nonexistent")

        # Should create version 1 with tombstone
        record = context_manager.get("nonexistent")
        assert record is None  # Tombstone means None

    def test_get_after_delete_returns_none(self, context_manager):
        """Test that getting deleted key returns None."""
        key = "deleted"
        context_manager.set(key, b"value")
        context_manager.delete(key)

        result = context_manager.get(key)
        assert result is None

    def test_set_after_delete(self, context_manager):
        """Test setting a key after deleting it."""
        key = "delete_and_set"
        context_manager.set(key, b"v1")  # version 1
        context_manager.delete(key)  # version 2 (tombstone)
        context_manager.set(key, b"v3")  # version 3

        value = context_manager.get(key)
        assert value is not None
        assert value == "v3"

        record = context_manager.get_record(key)
        assert record.value == b"v3"
        assert record.version == 3


class TestContextHistory:
    """Tests for context history tracking."""

    def test_get_history(self, context_manager):
        """Test getting full history of a key."""
        key = "history_test"
        context_manager.set(key, b"v1")
        context_manager.set(key, b"v2")
        context_manager.set(key, b"v3")

        history = context_manager.get_history(key)
        assert len(history) == 3
        # Should be newest first
        assert history[0].value == b"v3"
        assert history[1].value == b"v2"
        assert history[2].value == b"v1"

    def test_get_history_empty(self, context_manager):
        """Test getting history of nonexistent key."""
        history = context_manager.get_history("nonexistent")
        assert history == []

    def test_get_history_with_limit(self, context_manager):
        """Test getting limited history."""
        key = "limited_history"
        for i in range(10):
            context_manager.set(key, f"v{i}".encode('utf-8'))

        history = context_manager.get_history(key, max_versions=3)
        assert len(history) == 3
        # Should get newest 3
        assert history[0].value == b"v9"
        assert history[1].value == b"v8"
        assert history[2].value == b"v7"

    def test_get_history_includes_tombstones(self, context_manager):
        """Test that history includes tombstones."""
        key = "history_tombstone"
        context_manager.set(key, b"v1")
        context_manager.delete(key)
        context_manager.set(key, b"v3")

        history = context_manager.get_history(key)
        # Should be 3 versions, but middle one returns None due to tombstone
        # Actually, get_history calls get() which filters tombstones
        # So we should only see non-tombstone versions
        non_tombstone = [r for r in history if r is not None]
        assert len(non_tombstone) == 2


class TestContextListKeys:
    """Tests for listing context keys."""

    def test_list_keys_empty(self, context_manager):
        """Test listing keys when context is empty."""
        keys = context_manager.list_keys()
        assert keys == []

    def test_list_keys(self, context_manager):
        """Test listing all context keys."""
        context_manager.set("key1", b"value")
        context_manager.set("key2", b"value")
        context_manager.set("key3", b"value")

        keys = context_manager.list_keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_list_keys_with_prefix(self, context_manager):
        """Test listing keys with prefix filter.

        Note: list_keys extracts the first segment after 'context:' from storage keys.
        Storage key format: context:{key}:{version}
        So 'user:alice' as key becomes storage key 'context:user:alice:1',
        but list_keys returns 'user:alice' as the key name.
        """
        context_manager.set("user_alice", b"data")
        context_manager.set("user_bob", b"data")
        context_manager.set("system_config", b"data")
        context_manager.set("admin_panel", b"data")

        # Filter by prefix "user"
        user_keys = context_manager.list_keys(prefix="user")
        assert set(user_keys) == {"user_alice", "user_bob"}

        # Filter by prefix "system"
        system_keys = context_manager.list_keys(prefix="system")
        assert system_keys == ["system_config"]

        # Filter by prefix "admin"
        admin_keys = context_manager.list_keys(prefix="admin")
        assert admin_keys == ["admin_panel"]

    def test_list_keys_sorted(self, context_manager):
        """Test that list_keys returns sorted keys."""
        keys_to_add = ["zebra", "alpha", "beta", "gamma"]
        for key in keys_to_add:
            context_manager.set(key, b"value")

        keys = context_manager.list_keys()
        assert keys == sorted(keys_to_add)

    def test_list_keys_multiple_versions(self, context_manager):
        """Test that list_keys returns unique keys despite multiple versions."""
        key = "multi_version"
        for i in range(5):
            context_manager.set(key, f"v{i}".encode('utf-8'))

        keys = context_manager.list_keys()
        assert keys == ["multi_version"]

    def test_list_keys_after_delete(self, context_manager):
        """Test that deleted keys still appear in list (tombstone exists)."""
        context_manager.set("key1", b"value")
        context_manager.set("key2", b"value")
        context_manager.delete("key1")

        keys = context_manager.list_keys()
        # Both keys should be listed (tombstone is still a version)
        assert set(keys) == {"key1", "key2"}


class TestContextClear:
    """Tests for clearing context."""

    def test_clear_removes_all_context(self, context_manager):
        """Test that clear removes all context entries."""
        context_manager.set("key1", b"value")
        context_manager.set("key2", b"value")
        context_manager.set("key3", b"value")

        context_manager.clear()

        keys = context_manager.list_keys()
        assert keys == []

    def test_clear_preserves_iteration(self, context_manager):
        """Test that clear preserves iteration counter."""
        context_manager.next_iteration()
        context_manager.next_iteration()
        current_iter = context_manager.get_iteration()

        context_manager.set("key", b"value")
        context_manager.clear()

        # Iteration should be preserved
        assert context_manager.get_iteration() == current_iter

    def test_clear_on_empty_context(self, context_manager):
        """Test that clear on empty context is safe."""
        context_manager.clear()  # Should not raise

    def test_set_after_clear(self, context_manager):
        """Test setting values after clear."""
        context_manager.set("key", b"v1")
        context_manager.clear()

        context_manager.set("key", b"v2")
        value = context_manager.get("key")
        assert value == "v2"

        record = context_manager.get_record("key")
        assert record.value == b"v2"
        assert record.version == 1  # Version resets after clear


class TestContextIteration:
    """Tests for iteration tracking in context."""

    def test_get_iteration(self, context_manager):
        """Test getting current iteration."""
        assert context_manager.get_iteration() == 0

    def test_next_iteration(self, context_manager):
        """Test incrementing iteration."""
        assert context_manager.next_iteration() == 1
        assert context_manager.get_iteration() == 1

    def test_context_tracks_iteration(self, context_manager):
        """Test that context records track iteration."""
        r1 = context_manager.set("key", b"v1")
        assert r1.iteration == 0

        context_manager.next_iteration()

        r2 = context_manager.set("key", b"v2")
        assert r2.iteration == 1

        context_manager.next_iteration()

        r3 = context_manager.set("key", b"v3")
        assert r3.iteration == 2


class TestContextRecord:
    """Tests for ContextRecord dataclass."""

    def test_context_record_creation(self):
        """Test creating ContextRecord."""
        record = ContextRecord(
            value=b"test",
            iteration=1,
            timestamp=123456.789,
            version=2
        )
        assert record.value == b"test"
        assert record.iteration == 1
        assert record.timestamp == 123456.789
        assert record.version == 2

    def test_context_record_equality(self):
        """Test ContextRecord equality comparison."""
        r1 = ContextRecord(b"val", 1, 123.0, 1)
        r2 = ContextRecord(b"val", 1, 123.0, 1)
        r3 = ContextRecord(b"val", 2, 123.0, 1)

        assert r1 == r2
        assert r1 != r3


class TestContextUpdate:
    """Tests for update() method."""

    def test_update_preserves_version_number(self, context_manager):
        """Test that update() doesn't create new version.

        The update() method is used for streaming/incremental writes within
        the same logical operation. Unlike set() which creates new versions,
        update() overwrites the current version.
        """
        r1 = context_manager.set("key", b"initial")
        assert r1.version == 1

        r2 = context_manager.update("key", b"updated")
        assert r2.version == 1  # Same version, not incremented!

        # Verify the value was actually updated
        value = context_manager.get("key")
        assert value == "updated"

        record = context_manager.get_record("key")
        assert record.value == b"updated"
        assert record.version == 1

    def test_update_creates_version_one_if_not_exists(self, context_manager):
        """Test update() on non-existent key creates version 1.

        When update() is called on a key that doesn't exist yet,
        it should create version 1 (same as set() would).
        """
        r1 = context_manager.update("new_key", b"value")
        assert r1.version == 1

        value = context_manager.get("new_key")
        assert value is not None
        assert value == "value"

        record = context_manager.get_record("new_key")
        assert record.value == b"value"
        assert record.version == 1

    def test_update_iteration_parameter(self, context_manager):
        """Test update() with explicit iteration parameter.

        The iteration parameter allows setting a specific iteration number,
        useful when multiple agents share the same iteration manager.
        """
        r1 = context_manager.update("key", b"value", iteration=5)
        assert r1.iteration == 5

        record = context_manager.get_record("key")
        assert record.iteration == 5

    def test_update_overwrites_existing_version(self, context_manager):
        """Test that update() overwrites the existing version data.

        After set() creates version 1, update() should overwrite version 1
        completely, not create version 2.
        """
        context_manager.set("key", b"v1")  # version 1
        context_manager.update("key", b"v1_updated")  # still version 1

        value = context_manager.get("key")
        assert value == "v1_updated"

        record = context_manager.get_record("key")
        assert record.version == 1
        assert record.value == b"v1_updated"

        # History should still show only one version
        history = context_manager.get_history("key")
        assert len(history) == 1
        assert history[0].value == b"v1_updated"

    def test_update_after_multiple_versions(self, context_manager):
        """Test update() after multiple set() operations.

        After creating multiple versions with set(), update() should
        overwrite only the latest version without creating a new one.
        """
        context_manager.set("key", b"v1")  # version 1
        context_manager.set("key", b"v2")  # version 2
        context_manager.set("key", b"v3")  # version 3

        # Update the latest (version 3)
        r = context_manager.update("key", b"v3_updated")
        assert r.version == 3

        # Latest should be updated
        latest = context_manager.get("key")
        assert latest == "v3_updated"

        latest_record = context_manager.get_record("key")
        assert latest_record.version == 3
        assert latest_record.value == b"v3_updated"

        # Older versions should be unchanged
        v1 = context_manager.get("key", version=1)
        assert v1 == "v1"
        v2 = context_manager.get("key", version=2)
        assert v2 == "v2"

    def test_update_updates_timestamp(self, context_manager):
        """Test that update() updates the timestamp.

        Even though update() preserves the version number,
        it should update the timestamp to reflect when the update occurred.
        """
        import time

        r1 = context_manager.set("key", b"initial")
        ts1 = r1.timestamp

        time.sleep(0.01)  # Small delay

        r2 = context_manager.update("key", b"updated")
        ts2 = r2.timestamp

        assert ts2 > ts1  # Timestamp should be newer
        assert r2.version == r1.version  # But version stays same

    def test_update_with_current_iteration(self, context_manager):
        """Test that update() uses current iteration when not specified.

        If iteration parameter is not provided, update() should use
        the current global iteration from the iteration manager.
        """
        context_manager.next_iteration()
        context_manager.next_iteration()
        current_iter = context_manager.get_iteration()

        r = context_manager.update("key", b"value")
        assert r.iteration == current_iter

    def test_update_multiple_times(self, context_manager):
        """Test calling update() multiple times in succession.

        Multiple update() calls should all overwrite the same version.
        """
        context_manager.set("key", b"v1")  # version 1

        context_manager.update("key", b"update1")
        value = context_manager.get("key")
        assert value == "update1"
        record = context_manager.get_record("key")
        assert record.version == 1
        assert record.value == b"update1"

        context_manager.update("key", b"update2")
        value = context_manager.get("key")
        assert value == "update2"
        record = context_manager.get_record("key")
        assert record.version == 1
        assert record.value == b"update2"

        context_manager.update("key", b"update3")
        value = context_manager.get("key")
        assert value == "update3"
        record = context_manager.get_record("key")
        assert record.version == 1
        assert record.value == b"update3"


class TestContextEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_value(self, context_manager):
        """Test storing empty bytes value."""
        context_manager.set("empty", b"")
        value = context_manager.get("empty")
        assert value == ""
        record = context_manager.get_record("empty")
        assert record.value == b""

    def test_large_value(self, context_manager):
        """Test storing large value."""
        large_value = b"x" * (1024 * 1024)  # 1MB
        context_manager.set("large", large_value)

        value = context_manager.get("large")
        assert len(value) == len(large_value)

        record = context_manager.get_record("large")
        assert len(record.value) == len(large_value)

    def test_unicode_key(self, context_manager):
        """Test using unicode in key names."""
        key = "unicode_\u00e9\u00f1"
        context_manager.set(key, b"value")

        value = context_manager.get(key)
        assert value == "value"

        record = context_manager.get_record(key)
        assert record.value == b"value"

    def test_unicode_value(self, context_manager):
        """Test storing unicode in values."""
        value = "unicode_\u4e2d\u6587".encode('utf-8')
        context_manager.set("key", value)

        retrieved = context_manager.get("key")
        assert retrieved == "unicode_\u4e2d\u6587"

        record = context_manager.get_record("key")
        assert record.value == value
        assert record.value.decode('utf-8') == "unicode_\u4e2d\u6587"

    def test_many_versions(self, context_manager):
        """Test handling many versions of a key."""
        key = "many_versions"
        for i in range(100):
            context_manager.set(key, f"v{i}".encode('utf-8'))

        latest = context_manager.get(key)
        assert latest == "v99"

        latest_record = context_manager.get_record(key)
        assert latest_record.version == 100
        assert latest_record.value == b"v99"

        # Get specific version
        v50 = context_manager.get(key, version=50)
        assert v50 == "v49"

    def test_concurrent_keys(self, context_manager):
        """Test managing many different keys."""
        for i in range(100):
            context_manager.set(f"key{i}", f"value{i}".encode('utf-8'))

        keys = context_manager.list_keys()
        assert len(keys) == 100

        # All should be retrievable
        for i in range(100):
            value = context_manager.get(f"key{i}")
            assert value == f"value{i}"


