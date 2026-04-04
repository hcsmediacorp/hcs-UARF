"""
UARF - Universal AutoResearch Framework (Lite)
Lightweight orchestration for low-RAM environments.

Made with ❤️ by hcsmedia
Refactored for cloud-friendly deployment.
"""

__version__ = "2.0.0-lite"
__author__ = "hcsmedia"

# Lite components (new, memory-efficient)
from .core.config_lite import LiteConfig, load_config, quick_config
from .core.device_manager import DeviceManager, select_device, get_device
from .models.registry import ModelRegistry, ModelEntry, get_registry, suggest_model, list_tiny_models
from .logging.debug_logger import DebugLogger, setup_logger, get_logger, debug, info, warning, error, success
from .controller import UARFController, quick_start, detect, suggest, TaskResult

__all__ = [
    # Config
    "LiteConfig",
    "load_config",
    "quick_config",
    
    # Device
    "DeviceManager",
    "select_device",
    "get_device",
    
    # Models
    "ModelRegistry",
    "ModelEntry",
    "get_registry",
    "suggest_model",
    "list_tiny_models",
    
    # Logging
    "DebugLogger",
    "setup_logger",
    "get_logger",
    "debug",
    "info",
    "warning",
    "error",
    "success",
    
    # Controller
    "UARFController",
    "quick_start",
    "detect",
    "suggest",
    "TaskResult",
]
