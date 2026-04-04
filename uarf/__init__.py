"""
UARF - Universal AutoResearch Framework
LLM Training auf jedem Gerät (Linux, Mac, Windows, Termux)

Made with ❤️ by hcsmedia
"""

__version__ = "1.0.0"
__author__ = "hcsmedia"

from .core.hardware_detector import HardwareDetector
from .core.config import UARFConfig
from .core.trainer import UniversalTrainer
from .core.model_selector import ModelSelector
from .core.swap_manager import SwapManager, SwapConfig
from .auto_mode import AutoMode, auto_train

__all__ = [
    "HardwareDetector",
    "UARFConfig",
    "UniversalTrainer",
    "ModelSelector",
    "SwapManager",
    "SwapConfig",
    "AutoMode",
    "auto_train",
]
