"""
UARF Logging - Alias for uarf_logging module
Backward compatibility and cleaner import path
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from uarf_logging import (
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
