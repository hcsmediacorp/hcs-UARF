"""
UARF - Universal AutoResearch Framework
LLM Training auf jedem Gerät (Linux, Mac, Windows, Termux)
"""

__version__ = "0.3.0"
__author__ = "UARF Team"

from .core.hardware_detector import HardwareDetector
from .core.config import UARFConfig
from .core.trainer import UniversalTrainer
from .core.model_selector import ModelSelector

__all__ = [
    "HardwareDetector",
    "UARFConfig", 
    "UniversalTrainer",
    "ModelSelector",
]
