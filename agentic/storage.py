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

        self._db = Rdict(str(self._db_path / "db"), options=opts)

        # Ensure identification
        self._ensure_identification()
        self._initialized = True

    def get_db_path(self) -> Path:
        """Return database path."""
        if not self._initialized or self._db_path is None:
            raise RuntimeError("Storage not initialized. Call initialize() first.")
        return self._db_path

    def get(self, key: bytes) -> bytes | None:
        """Get value for key."""
        if not self._initialized or self._db is None:
            raise RuntimeError("Storage not initialized.")
        return self._db.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        """Store key-value pair."""
        if not self._initialized or self._db is None:
            raise RuntimeError("Storage not initialized.")
        self._db.put(key, value)

    def delete(self, key: bytes) -> None:
        """Delete key."""
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
        """Close database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None
        self._initialized = False

    def _resolve_context_path(self) -> Path:
        """Resolve context directory with collision avoidance via UUID suffix."""
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
        """Ensure database has valid identification."""
        id_key = b"metadata:id"
        existing_id = self._db.get(id_key)

        if existing_id is None:
            app_id = self._generate_app_id()
            self._db.put(id_key, app_id.encode('utf-8'))
        else:
            stored_id = existing_id.decode('utf-8')
            if not self._validate_app_id(stored_id):
                raise ValueError(
                    f"Invalid or corrupted database ID: {stored_id}. "
                    "This database may not belong to this application."
                )

    def _generate_app_id(self) -> str:
        """Generate app ID. Format: agentic_<hash>_<timestamp>_<uuid>"""
        if self._config.app_id:
            base = self._config.app_id
        else:
            base = "agentic_framework"

        hash_obj = hashlib.sha256(base.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:16]
        timestamp = int(time.time())
        unique_id = new_uuid()[:8]

        return f"agentic_{hash_hex}_{timestamp}_{unique_id}"

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
