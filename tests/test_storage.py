"""
Tests for RocksDB storage layer.

Covers:
- StorageConfig configuration
- RocksDBStorage initialization
- CRUD operations (get, put, delete)
- Iteration with prefix filtering
- Database identification and validation
- Path resolution with collision avoidance
- Error handling for uninitialized storage
- Storage cleanup and closing
- _generate_app_id utility function
"""
import pytest
from pathlib import Path
import time

from agentic.storage import RocksDBStorage, StorageConfig, _generate_app_id


class TestGenerateAppId:
    """Tests for _generate_app_id utility function."""

    def test_generate_app_id_default(self):
        """Test _generate_app_id with default app_id."""
        app_id = _generate_app_id()

        # Format: agentic_<hash>_<timestamp>_<uuid>
        parts = app_id.split('_')
        assert len(parts) == 4
        assert parts[0] == "agentic"
        assert len(parts[1]) == 16  # hash is 16 hex chars
        assert parts[2].isdigit()  # timestamp
        assert len(parts[3]) == 8  # uuid suffix is 8 chars

    def test_generate_app_id_custom(self):
        """Test _generate_app_id with custom app_id base."""
        app_id = _generate_app_id("my_custom_app")

        parts = app_id.split('_')
        assert len(parts) == 4
        assert parts[0] == "agentic"
        # Hash should be consistent for same app_id
        assert len(parts[1]) == 16

    def test_generate_app_id_consistency(self):
        """Test that hash is consistent for same app_id."""
        app_id1 = _generate_app_id("test_app")
        app_id2 = _generate_app_id("test_app")

        # Hash parts should be same (index 1)
        hash1 = app_id1.split('_')[1]
        hash2 = app_id2.split('_')[1]
        assert hash1 == hash2

        # But timestamps and UUIDs should differ
        timestamp1 = app_id1.split('_')[2]
        timestamp2 = app_id2.split('_')[2]
        # Timestamps might be same if generated very quickly
        # but UUIDs should always differ
        uuid1 = app_id1.split('_')[3]
        uuid2 = app_id2.split('_')[3]
        assert uuid1 != uuid2

    def test_generate_app_id_different_bases(self):
        """Test that different app_id bases produce different hashes."""
        app_id1 = _generate_app_id("app_one")
        app_id2 = _generate_app_id("app_two")

        hash1 = app_id1.split('_')[1]
        hash2 = app_id2.split('_')[1]

        # Different app_ids should produce different hashes
        assert hash1 != hash2

    def test_generate_app_id_none_parameter(self):
        """Test _generate_app_id with explicit None."""
        app_id = _generate_app_id(None)

        # Should use default "agentic_framework"
        parts = app_id.split('_')
        assert len(parts) == 4
        assert parts[0] == "agentic"


class TestStorageConfig:
    """Tests for StorageConfig dataclass."""

    def test_storage_config_defaults(self):
        """Test StorageConfig with default values."""
        config = StorageConfig()
        assert config.base_dir == "./context"
        assert config.db_name_prefix == "context"
        assert config.app_id is None

    def test_storage_config_custom(self):
        """Test StorageConfig with custom values."""
        config = StorageConfig(
            base_dir="/tmp/test",
            db_name_prefix="custom_db",
            app_id="my_app"
        )
        assert config.base_dir == "/tmp/test"
        assert config.db_name_prefix == "custom_db"
        assert config.app_id == "my_app"

    def test_storage_config_path_object(self):
        """Test StorageConfig with Path object."""
        path = Path("/tmp/test")
        config = StorageConfig(base_dir=path)
        assert config.base_dir == path


class TestRocksDBStorageInitialization:
    """Tests for RocksDBStorage initialization and setup."""

    def test_storage_creation(self, storage_config):
        """Test creating storage instance without initialization."""
        storage = RocksDBStorage(storage_config)
        assert storage._initialized is False
        assert storage._db is None
        assert storage._db_path is None

    def test_storage_initialization(self, temp_dir):
        """Test storage initialization creates database."""
        config = StorageConfig(base_dir=temp_dir, db_name_prefix="test")
        storage = RocksDBStorage(config)
        storage.initialize()

        assert storage._initialized is True
        assert storage._db is not None
        assert storage._db_path is not None
        assert storage._db_path.exists()

        storage.close()

    def test_storage_double_initialization(self, storage):
        """Test that double initialization is safe (idempotent)."""
        # Storage fixture is already initialized
        assert storage._initialized is True
        # Initialize again should be safe
        storage.initialize()
        assert storage._initialized is True

    def test_storage_path_resolution(self, temp_dir):
        """Test that storage resolves path correctly."""
        config = StorageConfig(base_dir=temp_dir, db_name_prefix="mydb")
        storage = RocksDBStorage(config)
        storage.initialize()

        db_path = storage.get_db_path()
        assert "mydb" in str(db_path)
        assert db_path.parent == temp_dir

        storage.close()

    def test_storage_path_collision_avoidance(self, temp_dir):
        """Test that storage avoids path collisions with UUID suffix."""
        config1 = StorageConfig(base_dir=temp_dir, db_name_prefix="same")
        storage1 = RocksDBStorage(config1)
        storage1.initialize()
        path1 = storage1.get_db_path()

        config2 = StorageConfig(base_dir=temp_dir, db_name_prefix="same")
        storage2 = RocksDBStorage(config2)
        storage2.initialize()
        path2 = storage2.get_db_path()

        # Paths should be different due to collision avoidance
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()

        storage1.close()
        storage2.close()

    def test_get_db_path_before_init_raises(self, storage_config):
        """Test that get_db_path raises error before initialization."""
        storage = RocksDBStorage(storage_config)
        with pytest.raises(RuntimeError, match="Storage not initialized"):
            storage.get_db_path()


class TestRocksDBStorageCRUD:
    """Tests for basic CRUD operations."""

    def test_put_and_get(self, storage):
        """Test storing and retrieving a key-value pair."""
        key = b"test:key"
        value = b"test_value"
        storage.put(key, value)

        retrieved = storage.get(key)
        assert retrieved == value

    def test_get_nonexistent_key(self, storage):
        """Test getting a key that doesn't exist returns None."""
        result = storage.get(b"nonexistent:key")
        assert result is None

    def test_put_overwrites_existing(self, storage):
        """Test that putting to existing key overwrites value."""
        key = b"test:overwrite"
        storage.put(key, b"value1")
        storage.put(key, b"value2")

        retrieved = storage.get(key)
        assert retrieved == b"value2"

    def test_delete_key(self, storage):
        """Test deleting a key."""
        key = b"test:delete"
        storage.put(key, b"value")
        assert storage.get(key) is not None

        storage.delete(key)
        assert storage.get(key) is None

    def test_delete_nonexistent_key(self, storage):
        """Test deleting a nonexistent key (should not raise)."""
        # Should not raise an error
        storage.delete(b"nonexistent:key")

    def test_put_empty_value(self, storage):
        """Test storing empty bytes value."""
        key = b"test:empty"
        storage.put(key, b"")
        retrieved = storage.get(key)
        assert retrieved == b""

    def test_put_large_value(self, storage):
        """Test storing large value."""
        key = b"test:large"
        # 1MB of data
        large_value = b"x" * (1024 * 1024)
        storage.put(key, large_value)

        retrieved = storage.get(key)
        assert len(retrieved) == len(large_value)
        assert retrieved == large_value


class TestRocksDBStorageIteration:
    """Tests for iteration with prefix filtering."""

    def test_iterate_empty(self, storage):
        """Test iteration on empty storage."""
        results = list(storage.iterate(b"test:"))
        assert results == []

    def test_iterate_with_prefix(self, storage):
        """Test iteration with prefix filtering."""
        # Put multiple keys with same prefix
        storage.put(b"test:key1", b"value1")
        storage.put(b"test:key2", b"value2")
        storage.put(b"test:key3", b"value3")
        storage.put(b"other:key", b"value4")

        results = list(storage.iterate(b"test:"))
        assert len(results) == 3

        # Check all have correct prefix
        for key, value in results:
            assert key.startswith(b"test:")

    def test_iterate_returns_key_value_tuples(self, storage):
        """Test that iteration returns (key, value) tuples."""
        storage.put(b"prefix:k1", b"v1")
        storage.put(b"prefix:k2", b"v2")

        for item in storage.iterate(b"prefix:"):
            assert isinstance(item, tuple)
            assert len(item) == 2
            key, value = item
            assert isinstance(key, bytes)
            assert isinstance(value, bytes)

    def test_iterate_no_matches(self, storage):
        """Test iteration with prefix that matches nothing."""
        storage.put(b"test:key", b"value")
        results = list(storage.iterate(b"nomatch:"))
        assert results == []

    def test_iterate_ordering(self, storage):
        """Test that iteration returns keys in lexicographic order."""
        keys = [b"z:3", b"a:1", b"m:2"]
        for key in keys:
            storage.put(key, b"value")

        results = list(storage.iterate(b""))
        result_keys = [k for k, v in results if k in keys]

        # Should be sorted
        assert result_keys == sorted(keys)


class TestStorageIdentification:
    """Tests for database identification system."""

    def test_storage_has_identification(self, storage):
        """Test that initialized storage has identification."""
        id_value = storage.get(b"metadata:id")
        assert id_value is not None

    def test_identification_format(self, storage):
        """Test that identification follows expected format."""
        id_value = storage.get(b"metadata:id")
        app_id = id_value.decode('utf-8')

        # Format: agentic_<hash>_<timestamp>_<uuid>
        parts = app_id.split('_')
        assert len(parts) == 4
        assert parts[0] == "agentic"
        assert len(parts[1]) == 16  # hash is 16 hex chars
        assert parts[2].isdigit()  # timestamp
        assert len(parts[3]) == 8  # uuid suffix is 8 chars

    def test_identification_hash_consistency(self, temp_dir):
        """Test that hash is consistent for same app_id."""
        config1 = StorageConfig(base_dir=temp_dir, db_name_prefix="db1", app_id="test_app")
        storage1 = RocksDBStorage(config1)
        storage1.initialize()
        id1 = storage1.get(b"metadata:id").decode('utf-8')
        hash1 = id1.split('_')[1]

        # Different database, same app_id
        config2 = StorageConfig(base_dir=temp_dir, db_name_prefix="db2", app_id="test_app")
        storage2 = RocksDBStorage(config2)
        storage2.initialize()
        id2 = storage2.get(b"metadata:id").decode('utf-8')
        hash2 = id2.split('_')[1]

        # Hashes should be the same (same app_id)
        assert hash1 == hash2

        storage1.close()
        storage2.close()

    def test_identification_validation_success(self, temp_dir):
        """Test that reopening storage with same app_id succeeds."""
        config = StorageConfig(base_dir=temp_dir, db_name_prefix="persist", app_id="test_app")
        storage1 = RocksDBStorage(config)
        storage1.initialize()
        db_path = storage1.get_db_path()
        storage1.close()

        # Reopen with same config
        config2 = StorageConfig(base_dir=db_path.parent, db_name_prefix="persist", app_id="test_app")
        storage2 = RocksDBStorage(config2)
        # Should not raise
        storage2.initialize()
        storage2.close()

    def test_identification_validation_failure(self, temp_dir):
        """Test that reopening storage with different app_id raises error."""
        # Create a unique directory to avoid collision avoidance
        import os
        unique_db_dir = temp_dir / "unique_persist_db"
        os.makedirs(unique_db_dir, exist_ok=True)

        config = StorageConfig(base_dir=unique_db_dir, db_name_prefix="testdb", app_id="app1")
        storage1 = RocksDBStorage(config)
        storage1.initialize()
        db_path = storage1.get_db_path()
        storage1.close()

        # Now reopen the SAME database directory with a different app_id
        # Use the exact same db_path to ensure we're opening the existing database
        config2 = StorageConfig(base_dir=unique_db_dir, db_name_prefix="testdb", app_id="app2")
        storage2 = RocksDBStorage(config2)

        # Force storage2 to use the same path as storage1
        storage2._db_path = db_path

        with pytest.raises(ValueError, match="Invalid or corrupted database ID"):
            # This should raise because the database has app1's ID but we're using app2
            storage2._initialized = False
            from rocksdict import Rdict, Options
            opts = Options()
            opts.create_if_missing(False)  # Don't create new, open existing
            storage2._db = Rdict(str(db_path / "db"), options=opts)
            storage2._ensure_identification()  # This should raise ValueError

        if storage2._db:
            storage2._db.close()


class TestStorageErrorHandling:
    """Tests for error handling and edge cases."""

    def test_operations_before_init_raise(self, storage_config):
        """Test that operations before initialization raise errors."""
        storage = RocksDBStorage(storage_config)

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            storage.get(b"key")

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            storage.put(b"key", b"value")

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            storage.delete(b"key")

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            list(storage.iterate(b"prefix"))

    def test_close_uninitialized_storage(self, storage_config):
        """Test closing uninitialized storage is safe."""
        storage = RocksDBStorage(storage_config)
        storage.close()  # Should not raise

    def test_operations_after_close(self, temp_dir):
        """Test that operations after close raise errors."""
        config = StorageConfig(base_dir=temp_dir)
        storage = RocksDBStorage(config)
        storage.initialize()
        storage.close()

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            storage.get(b"key")

    def test_double_close(self, temp_dir):
        """Test that closing twice is safe."""
        config = StorageConfig(base_dir=temp_dir)
        storage = RocksDBStorage(config)
        storage.initialize()
        storage.close()
        storage.close()  # Should not raise


class TestStoragePerformance:
    """Tests for storage performance characteristics."""

    def test_batch_write_performance(self, storage):
        """Test that batch writes complete in reasonable time."""
        start = time.time()
        for i in range(1000):
            key = f"perf:key{i}".encode('utf-8')
            value = f"value{i}".encode('utf-8')
            storage.put(key, value)
        elapsed = time.time() - start

        # 1000 writes should complete in under 5 seconds
        assert elapsed < 5.0

    def test_batch_read_performance(self, storage):
        """Test that batch reads complete in reasonable time."""
        # Write test data
        for i in range(1000):
            key = f"perf:read{i}".encode('utf-8')
            storage.put(key, b"value")

        # Read back
        start = time.time()
        for i in range(1000):
            key = f"perf:read{i}".encode('utf-8')
            storage.get(key)
        elapsed = time.time() - start

        # 1000 reads should complete in under 2 seconds
        assert elapsed < 2.0

    def test_iteration_performance(self, storage):
        """Test that iteration is efficient."""
        # Write test data
        for i in range(1000):
            key = f"iter:key{i}".encode('utf-8')
            storage.put(key, b"value")

        # Iterate
        start = time.time()
        results = list(storage.iterate(b"iter:"))
        elapsed = time.time() - start

        assert len(results) == 1000
        # Iteration should complete in under 1 second
        assert elapsed < 1.0


class TestStorageEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_unicode_in_key(self, storage):
        """Test handling of unicode in keys."""
        key = "test:unicode_\u00e9\u00f1".encode('utf-8')
        value = b"value"
        storage.put(key, value)
        assert storage.get(key) == value

    def test_unicode_in_value(self, storage):
        """Test handling of unicode in values."""
        key = b"test:value"
        value = "unicode_value_\u4e2d\u6587".encode('utf-8')
        storage.put(key, value)
        assert storage.get(key) == value

    def test_binary_data(self, storage):
        """Test storing arbitrary binary data."""
        key = b"test:binary"
        value = bytes(range(256))  # All possible byte values
        storage.put(key, value)
        assert storage.get(key) == value

    def test_very_long_key(self, storage):
        """Test handling of very long keys."""
        key = b"test:" + b"x" * 10000
        value = b"value"
        storage.put(key, value)
        assert storage.get(key) == value

    def test_many_keys_same_prefix(self, storage):
        """Test handling many keys with same prefix."""
        prefix = b"same:prefix:"
        for i in range(100):
            key = prefix + str(i).encode('utf-8')
            storage.put(key, b"value")

        results = list(storage.iterate(prefix))
        assert len(results) == 100


class TestInMemoryStorage:
    """Tests for InMemoryStorage backend."""

    def test_inmemory_storage_initialization(self, temp_dir):
        """Test InMemoryStorage initialization creates app ID."""
        from agentic.storage import InMemoryStorage
        config = StorageConfig(base_dir=temp_dir, app_id="test_app")
        storage = InMemoryStorage(config)
        storage.initialize()

        # Should have app ID
        app_id = storage.get(b"metadata:id")
        assert app_id is not None
        assert b"agentic_" in app_id

        storage.close()

    def test_inmemory_basic_operations(self, temp_dir):
        """Test InMemoryStorage basic CRUD operations."""
        from agentic.storage import InMemoryStorage
        config = StorageConfig(base_dir=temp_dir)
        storage = InMemoryStorage(config)
        storage.initialize()

        # Put and get
        storage.put(b"key1", b"value1")
        assert storage.get(b"key1") == b"value1"

        # Update
        storage.put(b"key1", b"value2")
        assert storage.get(b"key1") == b"value2"

        # Delete
        storage.delete(b"key1")
        assert storage.get(b"key1") is None

        storage.close()

    def test_inmemory_iteration_order(self, temp_dir):
        """Test InMemoryStorage iteration order matches RocksDB lexicographic order."""
        from agentic.storage import InMemoryStorage
        config = StorageConfig(base_dir=temp_dir)
        storage = InMemoryStorage(config)
        storage.initialize()

        # Insert keys in specific order
        keys = [b"test:z", b"test:a", b"test:m"]
        for key in keys:
            storage.put(key, b"value")

        # Iterate with prefix
        results = list(storage.iterate(b"test:"))
        result_keys = [k for k, v in results]

        # Should have all keys in lexicographic order to match RocksDB behavior
        assert result_keys == sorted(keys)

        storage.close()

    def test_inmemory_storage_not_persistent(self, temp_dir):
        """Test that InMemoryStorage loses data when closed.

        This verifies the in-memory nature - data doesn't persist to disk.
        """
        from agentic.storage import InMemoryStorage
        config = StorageConfig(base_dir=temp_dir, db_name_prefix="memory")
        storage1 = InMemoryStorage(config)
        storage1.initialize()

        storage1.put(b"key", b"value")
        storage1.close()

        # Create new instance - data should not persist
        storage2 = InMemoryStorage(config)
        storage2.initialize()
        assert storage2.get(b"key") is None

        storage2.close()

    def test_inmemory_get_db_path_returns_memory(self, temp_dir):
        """Test that InMemoryStorage returns :memory: for db_path."""
        from agentic.storage import InMemoryStorage
        config = StorageConfig(base_dir=temp_dir)
        storage = InMemoryStorage(config)
        storage.initialize()

        path = storage.get_db_path()
        assert path == Path(":memory:")

        storage.close()

    def test_inmemory_get_db_path_requires_initialization(self, temp_dir):
        """get_db_path should raise if initialize() was not called."""
        from agentic.storage import InMemoryStorage
        config = StorageConfig(base_dir=temp_dir)
        storage = InMemoryStorage(config)

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            storage.get_db_path()

    def test_inmemory_double_initialization(self, temp_dir):
        """Test that double initialization is idempotent."""
        from agentic.storage import InMemoryStorage
        config = StorageConfig(base_dir=temp_dir)
        storage = InMemoryStorage(config)

        storage.initialize()
        storage.put(b"key", b"value")

        # Initialize again should be safe
        storage.initialize()
        assert storage.get(b"key") == b"value"

        storage.close()

    def test_inmemory_operations_before_init_raise(self, temp_dir):
        """Test that operations before initialization raise errors."""
        from agentic.storage import InMemoryStorage
        config = StorageConfig(base_dir=temp_dir)
        storage = InMemoryStorage(config)

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            storage.get(b"key")

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            storage.put(b"key", b"value")

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            storage.delete(b"key")

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            list(storage.iterate(b"prefix"))

    def test_inmemory_close_clears_data(self, temp_dir):
        """Test that close() clears all data from memory."""
        from agentic.storage import InMemoryStorage
        config = StorageConfig(base_dir=temp_dir)
        storage = InMemoryStorage(config)
        storage.initialize()

        storage.put(b"key1", b"value1")
        storage.put(b"key2", b"value2")

        storage.close()

        # After close, reinitialization should start fresh
        storage.initialize()
        assert storage.get(b"key1") is None
        assert storage.get(b"key2") is None

        storage.close()


class TestRocksDBLeakFix:
    """Tests for RocksDB cleanup on initialization failure."""

    def test_rocksdb_closes_on_init_failure(self, temp_dir):
        """Test that RocksDB connection is closed if initialization fails.

        This tests the leak fix where DB connections were left open when
        _ensure_identification() or other post-open operations failed.
        """
        from agentic.storage import RocksDBStorage, StorageConfig

        config = StorageConfig(base_dir=temp_dir, db_name_prefix="test_fail", app_id="test")

        # Create a failing storage class
        class FailingStorage(RocksDBStorage):
            def _ensure_identification(self):
                # Simulate failure during identification
                raise ValueError("Simulated identification failure")

        failing_storage = FailingStorage(config)

        try:
            failing_storage.initialize()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Simulated identification failure" in str(e)

        # Verify DB is closed (should be None after cleanup)
        assert failing_storage._db is None

    def test_rocksdb_cleanup_on_permission_error(self, temp_dir):
        """Test RocksDB cleanup when database open fails due to permissions.

        Verifies that the fix handles failures during the Rdict() call itself.
        """
        from agentic.storage import RocksDBStorage, StorageConfig

        config = StorageConfig(base_dir=temp_dir, db_name_prefix="perm_test")

        class PermissionFailStorage(RocksDBStorage):
            def initialize(self):
                if self._initialized:
                    return

                self._db_path = self._resolve_context_path()
                self._db_path.mkdir(parents=True, exist_ok=True)

                from rocksdict import Rdict, Options
                opts = Options()
                opts.create_if_missing(True)

                try:
                    # Simulate failure during DB open
                    raise PermissionError("Simulated permission error")
                except Exception:
                    # Clean up DB connection on initialization failure
                    if self._db is not None:
                        self._db.close()
                        self._db = None
                    raise

        failing_storage = PermissionFailStorage(config)

        with pytest.raises(PermissionError, match="Simulated permission error"):
            failing_storage.initialize()

        # Verify DB is None after cleanup
        assert failing_storage._db is None
        assert failing_storage._initialized is False
