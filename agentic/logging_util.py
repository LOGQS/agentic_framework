"""
Structured logging utilities for the aegeantic framework.

Provides consistent logging across all modules with structured context.
Uses Python's built-in logging module for zero additional dependencies.
"""
import logging
import json
import threading
from typing import Any


# Thread-safe logger initialization
_logger_lock = threading.RLock()
_initialized_loggers: set[str] = set()


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs structured log records in JSON format.

    Each log record includes:
    - timestamp
    - level
    - logger name
    - message
    - structured context (passed as extra={...})
    """

    # Dynamically detect standard LogRecord attributes for forward compatibility
    # This prevents new Python LogRecord fields from leaking into extra context
    _STANDARD_ATTRS = frozenset(
        vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
    )

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add structured context from extra fields
        # Only include non-standard LogRecord attributes (custom fields passed via extra={})
        for key, value in record.__dict__.items():
            if key not in self._STANDARD_ATTRS and key != "message":
                # Convert non-serializable values to strings
                try:
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        try:
            return json.dumps(log_data)
        except (TypeError, ValueError) as e:
            # Fallback: convert all values to strings if serialization fails
            safe_data = {k: str(v) for k, v in log_data.items()}
            safe_data["_serialization_error"] = str(e)
            return json.dumps(safe_data)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Thread-safe initialization with proper prefix handling.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Normalize name: strip 'agentic.' prefix if present (handle edge cases like 'agentic.agentic.foo')
    while name.startswith("agentic."):
        name = name[8:]  # Remove "agentic." prefix

    # Construct logger name with consistent prefix
    logger_name = f"agentic.{name}"

    # Thread-safe initialization
    with _logger_lock:
        logger = logging.getLogger(logger_name)

        # Only initialize if not already initialized (idempotent)
        if logger_name not in _initialized_loggers:
            logger.setLevel(logging.INFO)

            # Default to console output with structured format
            handler = logging.StreamHandler()
            handler.setFormatter(StructuredFormatter())
            logger.addHandler(handler)

            # Don't propagate to root logger to avoid duplicate logs
            logger.propagate = False

            # Mark as initialized
            _initialized_loggers.add(logger_name)

    return logger
