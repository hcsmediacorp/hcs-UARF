"""
UARF Logging - Alias for uarf_logging module
Backward compatibility and cleaner import path
"""

from uarf.uarf_logging import (
    get_logger,
    UARFLogger,
    ColoredFormatter,
    JSONFormatter,
    debug,
    info,
    warning,
    error,
    critical,
    exception,
)

__all__ = [
    "get_logger",
    "UARFLogger",
    "ColoredFormatter",
    "JSONFormatter",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
]
