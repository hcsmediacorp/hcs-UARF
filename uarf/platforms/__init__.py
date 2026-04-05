"""
UARF Platform Adapters - Base Interface

Provides abstract base classes for platform-specific implementations.
All platform adapters must inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class PlatformType(Enum):
    """Supported platform types"""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    ANDROID = "android"
    COLAB = "colab"
    CLUSTER = "cluster"
    UNKNOWN = "unknown"


@dataclass
class PlatformInfo:
    """Platform information structure"""
    platform_type: PlatformType
    os_name: str
    os_version: str
    python_version: str
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    gpu_count: int
    gpu_name: Optional[str]
    cuda_version: Optional[str]
    special_features: List[str]


class PlatformAdapter(ABC):
    """
    Abstract base class for platform-specific adapters

    Each platform (Android, Colab, Windows, Cluster) must implement
    this interface to provide platform-specific functionality.
    """

    def __init__(self):
        self.platform_info: Optional[PlatformInfo] = None

    @abstractmethod
    def detect(self) -> PlatformInfo:
        """
        Detect platform capabilities and configuration

        Returns:
            PlatformInfo with detected capabilities
        """
        pass

    @abstractmethod
    def setup_environment(self) -> Dict[str, str]:
        """
        Set up platform-specific environment variables

        Returns:
            Dictionary of environment variables to set
        """
        pass

    @abstractmethod
    def get_optimal_config(self) -> Dict[str, Any]:
        """
        Get optimal training configuration for this platform

        Returns:
            Configuration dictionary optimized for this platform
        """
        pass

    @abstractmethod
    def check_prerequisites(self) -> tuple[bool, List[str]]:
        """
        Check if all prerequisites are met

        Returns:
            Tuple of (success, list of missing requirements)
        """
        pass

    @abstractmethod
    def install_dependencies(self, requirements: List[str]) -> bool:
        """
        Install required dependencies for this platform

        Args:
            requirements: List of package requirements

        Returns:
            True if installation successful
        """
        pass

    def get_platform_name(self) -> str:
        """Get human-readable platform name"""
        return self.__class__.__name__.replace("Adapter", "")


class TrainingBackend(ABC):
    """
    Abstract base class for training backends

    Different platforms may use different training backends
    (native PyTorch, DeepSpeed, FSDP, etc.)
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize training backend"""
        pass

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute single training step"""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str, state: Dict[str, Any]) -> Path:
        """Save training checkpoint"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        pass


def get_platform_adapter(platform_type: PlatformType) -> PlatformAdapter:
    """
    Factory function to get appropriate platform adapter

    Args:
        platform_type: Type of platform

    Returns:
        PlatformAdapter instance
    """
    # Import adapters lazily to avoid circular imports
    if platform_type == PlatformType.ANDROID:
        from .android.adapter import AndroidAdapter
        return AndroidAdapter()
    if platform_type == PlatformType.COLAB:
        from .colab.adapter import ColabAdapter
        return ColabAdapter()

    # Optional adapters may not be implemented yet.
    if platform_type == PlatformType.WINDOWS:
        try:
            from .windows.adapter import WindowsAdapter
            return WindowsAdapter()
        except Exception as e:
            raise NotImplementedError("Windows adapter is not implemented yet") from e

    if platform_type == PlatformType.CLUSTER:
        try:
            from .cluster.adapter import ClusterAdapter
            return ClusterAdapter()
        except Exception as e:
            raise NotImplementedError("Cluster adapter is not implemented yet") from e

    raise NotImplementedError(
        f"No adapter implemented for platform type '{platform_type.value}'. "
        "Available adapters: android, colab."
    )


__all__ = [
    "PlatformType",
    "PlatformInfo",
    "PlatformAdapter",
    "TrainingBackend",
    "get_platform_adapter",
]
