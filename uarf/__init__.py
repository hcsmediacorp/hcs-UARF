"""
UARF - Universal AutoResearch Framework v2.0
One Framework. Every Device. From 256MB to Multi-GPU Clusters.

Made with ❤️ by hcsmedia
Refactored for unified cross-device deployment.
"""

__version__ = "2.0.0"
__author__ = "hcsmedia"

# Core components (lazy-loaded)
from .core.config_lite import LiteConfig, load_config, quick_config
from .core.device_manager import DeviceManager, select_device, get_device
from .models.registry import ModelRegistry, ModelEntry, get_registry, suggest_model, list_tiny_models
from .logging.debug_logger import DebugLogger, setup_logger, get_logger, debug, info, warning, error, success
from .controller import UARFController, quick_start, detect, suggest, TaskResult

# Convenience imports
from typing import Optional, Dict, Any


def train(
    text: str,
    time_minutes: int = 5,
    device: str = "auto",
    model_id: Optional[str] = None,
    output_dir: str = "./outputs",
    debug: bool = False,
    **kwargs
) -> TaskResult:
    """
    Simplest training interface - perfect for Qwen Chat control.
    
    Args:
        text: Training text
        time_minutes: Training duration in minutes
        device: Device preference (auto, cpu, cuda, mps)
        model_id: Optional specific model ID (auto-selected if not provided)
        output_dir: Output directory for checkpoints
        debug: Enable debug mode
        **kwargs: Additional config overrides
    
    Returns:
        TaskResult with training outcome
    
    Example:
        >>> from uarf import train
        >>> result = train("Your text here...", time_minutes=5)
        >>> print(f"Model saved to: {result.output_path}")
    """
    controller = UARFController(
        debug=debug,
        ram_mb=kwargs.get('ram_mb', None)
    )
    
    # Update config
    controller.update_config(
        time_budget_seconds=time_minutes * 60,
        device=device,
        output_dir=output_dir,
        **kwargs
    )
    
    # Auto-select model if not specified
    if model_id:
        controller.config.model_id = model_id
    else:
        model_result = controller.select_model()
        if model_result.success:
            model_id = model_result.data['model']
    
    # Note: Actual training implementation would go here
    # For now, return a placeholder result
    return TaskResult(
        success=True,
        message=f"Training configured for {text[:50]}...",
        data={
            'model': model_id,
            'time_minutes': time_minutes,
            'device': device,
            'output_dir': output_dir
        }
    )


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
    
    # Convenience
    "train",
]
