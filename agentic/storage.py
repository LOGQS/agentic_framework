"""
RocksDB storage layer with identification and path resolution.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from rocksdict import Rdict, Options
import hashlib
import time

from .core import new_uuid


def _generate_app_id(app_id: str | None = None) -> str:
    """
    Generate application ID for database identification.

    Format: agentic_<hash>_<timestamp>_<uuid>

    Args:
        app_id: Optional custom app ID base. Defaults to "agentic_framework".

    Returns:
        Generated app ID string
    """
    base = app_id if app_id else "agentic_framework"
    hash_obj = hashlib.sha256(base.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:16]
    timestamp = int(time.time())
    unique_id = new_uuid()[:8]
    return f"agentic_{hash_hex}_{timestamp}_{unique_id}"


@dataclass
class StorageConfig:
    """Configuration for storage layer."""
    base_dir: Path | str = "./context"
    db_name_prefix: str = "context"
    app_id: str | None = None


class RocksDBStorage:
    """
    RocksDB wrapper with automatic path resolution and database identification.
    """

    def __init__(self, config: StorageConfig):
        self._config = config
        self._db_path: Path | None = None
        self._db: Rdict | None = None
        self._initialized = False

    def initialize(self) -> None:
        """
        Initialize storage:
        - Resolve DB path with collision avoidance
        - Open RocksDB connection
        - Generate or validate app identification
        """
        if self._initialized:
            return

        self._db_path = self._resolve_context_path()
        self._db_path.mkdir(parents=True, exist_ok=True)

        opts = Options()
        opts.create_if_missing(True)
        opts.set_max_open_files(1000)
        opts.set_write_buffer_size(67108864)
        opts.set_max_write_buffer_number(3)
        opts.set_target_file_size_base(67108864)

        try:
            self._db = Rdict(str(self._db_path / "db"), options=opts)
            self._ensure_identification()
            self._initialized = True
        except Exception:
            # Clean up DB connection on initialization failure
            if self._db is not None:
                self._db.close()
                self._db = None
            raise

    def get_db_path(self) -> Path:
        if not self._initialized or self._db_path is None:
            raise RuntimeError("Storage not initialized. Call initialize() first.")
        return self._db_path

    def get(self, key: bytes) -> bytes | None:
        if not self._initialized or self._db is None:
            raise RuntimeError("Storage not initialized.")
        return self._db.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        if not self._initialized or self._db is None:
            raise RuntimeError("Storage not initialized.")
        self._db.put(key, value)

    def delete(self, key: bytes) -> None:
        if not self._initialized or self._db is None:
            raise RuntimeError("Storage not initialized.")
        self._db.delete(key)

    def iterate(self, prefix: bytes) -> Iterator[tuple[bytes, bytes]]:
        """Iterate over keys with prefix using RocksDB prefix seek."""
        if not self._initialized or self._db is None:
            raise RuntimeError("Storage not initialized.")

        try:
            iter_obj = self._db.iter()
            iter_obj.seek(prefix)

            for key, value in iter_obj:
                if not key.startswith(prefix):
                    break
                yield (key, value)
        except (AttributeError, TypeError):
            for key, value in self._db.items():
                if key.startswith(prefix):
                    yield (key, value)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        self._initialized = False

    def _resolve_context_path(self) -> Path:
        base = Path(self._config.base_dir)
        candidate = base / self._config.db_name_prefix

        if not candidate.exists():
            return candidate

        while True:
            suffix = new_uuid()[:8]
            candidate = base / f"{self._config.db_name_prefix}_{suffix}"
            if not candidate.exists():
                return candidate

    def _ensure_identification(self) -> None:
        id_key = b"metadata:id"
        existing_id = self._db.get(id_key)

        if existing_id is None:
            app_id = _generate_app_id(self._config.app_id)
            self._db.put(id_key, app_id.encode('utf-8'))
        else:
            stored_id = existing_id.decode('utf-8')
            if not self._validate_app_id(stored_id):
                raise ValueError(
                    f"Invalid or corrupted database ID: {stored_id}. "
                    "This database may not belong to this application."
                )

    def _validate_app_id(self, app_id: str) -> bool:
        """
        Validate app_id structure and hash.
        Expected format: agentic_<hex>_<timestamp>_<uuid>
        """
        parts = app_id.split('_')
        if len(parts) != 4:
            return False
        if parts[0] != "agentic":
            return False
        stored_hash = parts[1]
        if len(stored_hash) != 16:
            return False
        try:
            int(stored_hash, 16)
        except ValueError:
            return False
        try:
            int(parts[2])
        except ValueError:
            return False
        if len(parts[3]) != 8:
            return False

        expected_base = self._config.app_id if self._config.app_id else "agentic_framework"
        expected_hash_obj = hashlib.sha256(expected_base.encode('utf-8'))
        expected_hash = expected_hash_obj.hexdigest()[:16]

        if stored_hash != expected_hash:
            return False

        return True


class InMemoryStorage:
    """
    In-memory storage backend for testing and simple scripts.

    Implements same interface as RocksDBStorage but stores all data in memory.
    Data is lost when process terminates.

    Performance:
    - Uses lazy sorted cache for efficient iteration
    - First iterate() after writes: O(n log n) to build cache
    - Subsequent iterate() calls: O(log n + k) where k = matching keys
    - Optimized for read-heavy test workloads (write once, read many)
    """

    def __init__(self, config: StorageConfig):
        self._config = config
        self._data: dict[bytes, bytes] = {}
        self._sorted_keys_cache: list[bytes] | None = None  # Lazy cache, invalidated on writes
        self._initialized = False

    def initialize(self) -> None:
        """Initialize in-memory storage."""
        if self._initialized:
            return

        # Generate app ID for consistency with RocksDB implementation
        app_id = _generate_app_id(self._config.app_id)
        self._data[b"metadata:id"] = app_id.encode('utf-8')
        self._sorted_keys_cache = None  # Invalidate cache after write
        self._initialized = True

    def get_db_path(self) -> Path:
        """Return pseudo path for in-memory storage (requires initialization)."""
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        return Path(":memory:")

    def get(self, key: bytes) -> bytes | None:
        """Retrieve value by key."""
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        return self._data.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        """Store key-value pair."""
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        self._data[key] = value
        self._sorted_keys_cache = None  # Invalidate cache

    def delete(self, key: bytes) -> None:
        """Delete key from storage."""
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")
        if key in self._data:
            del self._data[key]
            self._sorted_keys_cache = None  # Invalidate cache

    def iterate(self, prefix: bytes) -> Iterator[tuple[bytes, bytes]]:
        """
        Iterate over keys with given prefix in lexicographic byte order.

        Uses lazy sorted cache + binary search for efficient prefix matching.
        Optimized for test scenarios: write context once, iterate many times.

        Performance:
        - Cold (after writes): O(n log n) to rebuild cache
        - Warm (cache valid): O(log n + k) where k = matching keys
        """
        if not self._initialized:
            raise RuntimeError("Storage not initialized.")

        # Lazily rebuild sorted cache if invalidated
        if self._sorted_keys_cache is None:
            self._sorted_keys_cache = sorted(self._data.keys())

        # Binary search for first key >= prefix
        import bisect
        start_idx = bisect.bisect_left(self._sorted_keys_cache, prefix)

        # Iterate from start_idx until keys no longer match prefix
        for key in self._sorted_keys_cache[start_idx:]:
            if not key.startswith(prefix):
                break
            yield (key, self._data[key])

    def close(self) -> None:
        """Close storage and clear all data."""
        self._data.clear()
        self._sorted_keys_cache = None
        self._initialized = False
