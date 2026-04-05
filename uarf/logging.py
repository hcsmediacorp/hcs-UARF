"""
Compatibility logging module.

Historically tests and examples imported `uarf.logging`.
The active implementation lives in `uarf.uarf_logging`.
"""

from uarf.uarf_logging import UARFLogger, get_logger, debug, info, warning, error


def setup_logger(*args, **kwargs):
    """Compatibility wrapper: return logger created by get_logger."""
    return get_logger(*args, **kwargs)


def success(msg: str, **kwargs):
    """Compatibility helper used by legacy callers."""
    get_logger().info(f"✅ {msg}", **kwargs)

__all__ = [
    "UARFLogger",
    "get_logger",
    "setup_logger",
    "debug",
    "info",
    "warning",
    "error",
    "success",
]
