"""
Context management with versioning and global iteration tracking.
"""
from dataclasses import dataclass
import json

from .storage import RocksDBStorage
from .core import now_timestamp

# Tombstone marker for deleted keys
TOMBSTONE = b"__TOMBSTONE__"


@dataclass
class ContextRecord:
    """Internal representation of a versioned context entry."""
    value: bytes
    iteration: int
    timestamp: float
    version: int


class IterationManager:
    """
    Manages global iteration counter stored in RocksDB.
    """

    def __init__(self, storage: RocksDBStorage):
        self._storage = storage
        self._key_global = b"iteration:global"

    def get(self) -> int:
        """Get current iteration."""
        value = self._storage.get(self._key_global)
        if value is None:
            self._storage.put(self._key_global, b"0")
            return 0
        return int(value.decode('utf-8'))

    def next(self) -> int:
        """Increment and return next iteration."""
        current = self.get()
        new_value = current + 1
        self._storage.put(self._key_global, str(new_value).encode('utf-8'))
        return new_value

    def register_event(self, label: str) -> None:
        """
        Register an event marker at current iteration.
        Key: iteration_event:{iteration}:{label}
        """
        iteration = self.get()
        timestamp = now_timestamp()
        key = f"iteration_event:{iteration}:{label}".encode('utf-8')
        value = json.dumps({"timestamp": timestamp, "label": label}).encode('utf-8')
        self._storage.put(key, value)


class ContextManager:
    """
    Manages versioned context with automatic iteration tracking.

    All context keys are stored with versions:
    - context:{key}:{version} -> ContextRecord
    - context:{key}:latest -> version number
    """

    def __init__(self, storage: RocksDBStorage, iteration: IterationManager):
        self._storage = storage
        self._iteration = iteration

    def set(self, key: str, value: bytes, processing_mode=None) -> ContextRecord:
        """
        Set context value with automatic versioning.

        Note: Currently synchronous regardless of processing_mode.
        Mode parameter reserved for future async I/O optimization.
        """
        latest_key = f"context:{key}:latest".encode('utf-8')
        latest_value = self._storage.get(latest_key)

        if latest_value is None:
            version = 1
        else:
            version = int(latest_value.decode('utf-8')) + 1

        record = ContextRecord(
            value=value,
            iteration=self._iteration.get(),
            timestamp=now_timestamp(),
            version=version
        )

        record_key = f"context:{key}:{version}".encode('utf-8')
        record_data = self._serialize_record(record)
        self._storage.put(record_key, record_data)
        self._storage.put(latest_key, str(version).encode('utf-8'))

        return record

    def update(self, key: str, value: bytes, processing_mode=None) -> ContextRecord:
        """
        Update context value without creating a new version.
        Overwrites the current version. If key doesn't exist, creates version 1.

        Use for streaming/incremental writes within same logical operation.
        Use set() for new generations/completions that should create new versions.

        Note: Currently synchronous regardless of processing_mode.
        Mode parameter reserved for future async I/O optimization.
        """
        latest_key = f"context:{key}:latest".encode('utf-8')
        latest_value = self._storage.get(latest_key)

        if latest_value is None:
            # First write: create version 1
            version = 1
        else:
            # Overwrite current version (don't increment)
            version = int(latest_value.decode('utf-8'))

        record = ContextRecord(
            value=value,
            iteration=self._iteration.get(),
            timestamp=now_timestamp(),
            version=version
        )

        record_key = f"context:{key}:{version}".encode('utf-8')
        record_data = self._serialize_record(record)
        self._storage.put(record_key, record_data)
        self._storage.put(latest_key, str(version).encode('utf-8'))

        return record

    def get(self, key: str, version: int | None = None, processing_mode=None) -> ContextRecord | None:
        """
        Get context value (latest version if version not specified).
        Returns None if key doesn't exist or is deleted.
        """
        if version is None:
            latest_key = f"context:{key}:latest".encode('utf-8')
            latest_value = self._storage.get(latest_key)
            if latest_value is None:
                return None
            version = int(latest_value.decode('utf-8'))

        record_key = f"context:{key}:{version}".encode('utf-8')
        record_data = self._storage.get(record_key)

        if record_data is None:
            return None

        record = self._deserialize_record(record_data)

        if record.value == TOMBSTONE:
            return None

        return record

    def delete(self, key: str, processing_mode=None) -> None:
        """
        Mark key as deleted with tombstone version.
        History is preserved but get() will return None.
        """
        self.set(key, TOMBSTONE, processing_mode=processing_mode)

    def list_keys(self, prefix: str | None = None) -> list[str]:
        """List all unique context keys, optionally filtered by prefix."""
        search_prefix = b"context:"
        keys_set = set()

        for key, _ in self._storage.iterate(search_prefix):
            key_str = key.decode('utf-8')
            parts = key_str.split(':')

            if len(parts) >= 3:
                context_key = parts[1]
                if prefix is None or context_key.startswith(prefix):
                    keys_set.add(context_key)

        return sorted(list(keys_set))

    def clear(self) -> None:
        """Clear all context entries (metadata and iteration preserved)."""
        keys_to_delete = []

        for key, _ in self._storage.iterate(b"context:"):
            keys_to_delete.append(key)

        for key in keys_to_delete:
            self._storage.delete(key)

    def get_iteration(self) -> int:
        """Get current iteration."""
        return self._iteration.get()

    def next_iteration(self) -> int:
        """Increment and return next iteration."""
        return self._iteration.next()

    def get_history(self, key: str, max_versions: int | None = None) -> list[ContextRecord]:
        """Get version history for key, newest first, optionally limited."""
        latest_key = f"context:{key}:latest".encode('utf-8')
        latest_value = self._storage.get(latest_key)

        if latest_value is None:
            return []

        latest_version = int(latest_value.decode('utf-8'))
        history = []

        for v in range(latest_version, 0, -1):
            record = self.get(key, version=v)
            if record is not None:
                history.append(record)

            if max_versions is not None and len(history) >= max_versions:
                break

        return history

    def _serialize_record(self, record: ContextRecord) -> bytes:
        """Serialize ContextRecord to bytes."""
        data = {
            "value": record.value.hex(),
            "iteration": record.iteration,
            "timestamp": record.timestamp,
            "version": record.version
        }
        return json.dumps(data).encode('utf-8')

    def _deserialize_record(self, data: bytes) -> ContextRecord:
        """Deserialize bytes to ContextRecord."""
        obj = json.loads(data.decode('utf-8'))
        return ContextRecord(
            value=bytes.fromhex(obj["value"]),
            iteration=obj["iteration"],
            timestamp=obj["timestamp"],
            version=obj["version"]
        )
