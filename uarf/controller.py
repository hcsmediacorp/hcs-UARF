"""
UARF Controller - Thin orchestration layer for low-RAM environments.
Delegates work to specialized backends with failure recovery.
Designed for clean Qwen Chat control.
"""

import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass

from .core.config_lite import LiteConfig, load_config, quick_config
from .core.device_manager import DeviceManager, select_device
from .models.registry import ModelRegistry, ModelEntry, get_registry, suggest_model
from .logging.debug_logger import DebugLogger, setup_logger, get_logger


@dataclass
class TaskResult:
    """Result of a controller task."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    recovery_suggestion: Optional[str] = None


class UARFController:
    """
    Thin orchestration controller for UARF.
    
    Features:
    - Lazy initialization (minimal imports until needed)
    - Failure recovery with fallback models
    - Memory-safe execution paths
    - Qwen Chat friendly API
    - Environment variable configuration
    
    Usage:
        # Quick start
        controller = UARFController()
        result = controller.run_task("detect")
        
        # With config
        controller = UARFController(ram_mb=512, debug=True)
        result = controller.select_model()
    """
    
    def __init__(
        self,
        config: Optional[LiteConfig] = None,
        ram_mb: Optional[int] = None,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize controller.
        
        Args:
            config: Optional pre-built configuration
            ram_mb: Available RAM in MB (auto-detected if not provided)
            debug: Enable debug mode
            **kwargs: Additional config overrides
        """
        self._initialized = False
        self._device_mgr: Optional[DeviceManager] = None
        self._registry: Optional[ModelRegistry] = None
        self._logger: Optional[DebugLogger] = None
        
        # Setup logging first
        self._logger = setup_logger(debug=debug, verbose=debug)
        
        # Determine available RAM
        if ram_mb:
            self.available_ram_mb = ram_mb
        else:
            self.available_ram_mb = self._detect_ram()
        
        # Build or use provided config
        if config:
            self.config = config
        else:
            # Load from environment first, then apply ram_mb and debug overrides
            self.config = LiteConfig.from_env()
            # Apply ram_mb override (triggers profile adaptation)
            if ram_mb:
                self.config.max_ram_mb = ram_mb
                self.config.apply_low_ram_profile(ram_mb)
            # Apply debug override
            if debug:
                self.config.debug_mode = debug
            # Apply any additional kwargs
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self._logger.info(f"UARF Controller initialized ({self.available_ram_mb:.0f}MB RAM)")
    
    def _detect_ram(self) -> float:
        """Detect available system RAM in MB."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.available / 1024 / 1024
        except Exception:
            return 512.0  # Conservative default
    
    def _ensure_initialized(self):
        """Lazy initialization of heavy components."""
        if self._initialized:
            return
        
        self._logger.debug("Initializing controller components...")
        
        # Initialize device manager
        self._device_mgr = select_device(
            prefer=self.config.device,
            min_ram_mb=self.config.max_ram_mb * 0.5  # Reserve half for model
        )
        
        # Initialize model registry
        self._registry = get_registry()
        
        self._initialized = True
        self._logger.success("Controller components initialized")
    
    @property
    def device(self):
        """Get torch device (lazy)."""
        self._ensure_initialized()
        return self._device_mgr.device
    
    @property
    def logger(self):
        """Get logger."""
        return self._logger
    
    # === DETECTION & SELECTION ===
    
    def detect_hardware(self) -> TaskResult:
        """Detect hardware and return specs."""
        try:
            self._ensure_initialized()
            
            info = self._device_mgr.info
            
            data = {
                'device_type': info.device_type,
                'device_name': info.name,
                'total_memory_mb': info.memory_mb,
                'available_memory_mb': info.available_mb,
                'recommended_batch_size': self._recommend_batch_size(),
                'recommended_model': suggest_model(self.available_ram_mb).model_id
            }
            
            self._logger.success(f"Hardware detected: {info.device_type} ({info.available_mb:.0f}MB free)")
            return TaskResult(success=True, message="Hardware detection complete", data=data)
        
        except Exception as e:
            self._logger.error(f"Hardware detection failed: {e}")
            return TaskResult(
                success=False,
                message="Hardware detection failed",
                error=str(e),
                recovery_suggestion="Try running with UARF_DEVICE=cpu"
            )
    
    def select_model(
        self,
        max_params_millions: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> TaskResult:
        """
        Select best model for current hardware.
        
        Args:
            max_params_millions: Maximum model size in millions of parameters
            tags: Required model tags
        
        Returns:
            TaskResult with selected model info
        """
        try:
            self._ensure_initialized()
            
            # Use config limit or parameter limit
            max_params = max_params_millions or self.config.max_params_millions
            
            # Find models
            models = self._registry.list_models(
                max_params=max_params,
                min_ram=self.available_ram_mb * 0.7,  # Use 70% of RAM for model
                tags=tags
            )
            
            if not models:
                # Fallback to smallest model
                fallback = self._registry.suggest_model(50)  # 50MB minimum
                self._logger.warning(f"No models found, using fallback: {fallback.model_id}")
                return TaskResult(
                    success=True,
                    message="Using fallback model",
                    data={'model': fallback.model_id, 'name': fallback.name, 'fallback': True}
                )
            
            # Select largest fitting model
            selected = models[-1]
            
            # Update config
            self.config.model_id = selected.model_id
            
            self._logger.info(f"Selected model: {selected.model_id} ({selected.params_millions}M params)")
            
            return TaskResult(
                success=True,
                message=f"Selected {selected.name}",
                data={
                    'model': selected.model_id,
                    'name': selected.name,
                    'params_millions': selected.params_millions,
                    'min_ram_mb': selected.min_ram_mb,
                    'tags': selected.tags
                }
            )
        
        except Exception as e:
            self._logger.error(f"Model selection failed: {e}")
            return TaskResult(
                success=False,
                message="Model selection failed",
                error=str(e),
                recovery_suggestion="Try reducing max_params_millions"
            )
    
    def list_models(self, tiny_only: bool = False) -> TaskResult:
        """List available models."""
        try:
            self._ensure_initialized()
            
            if tiny_only:
                models = self._registry.list_models(max_params=100)
            else:
                models = self._registry.list_models(min_ram=self.available_ram_mb * 0.5)
            
            data = {
                'count': len(models),
                'models': [
                    {
                        'id': m.model_id,
                        'name': m.name,
                        'params_millions': m.params_millions,
                        'min_ram_mb': m.min_ram_mb,
                        'tags': m.tags
                    }
                    for m in models
                ]
            }
            
            return TaskResult(
                success=True,
                message=f"Found {len(models)} models",
                data=data
            )
        
        except Exception as e:
            return TaskResult(success=False, message="Failed to list models", error=str(e))
    
    # === CONFIGURATION ===
    
    def show_config(self) -> TaskResult:
        """Show current configuration."""
        data = self.config.to_dict()
        data['available_ram_mb'] = self.available_ram_mb
        
        if self.config.debug_mode:
            self._logger.config(data)
        
        return TaskResult(
            success=True,
            message="Configuration summary",
            data=data
        )
    
    def update_config(self, **updates) -> TaskResult:
        """Update configuration values."""
        try:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    self._logger.warning(f"Unknown config key: {key}")
            
            # Apply low-RAM profile if ram_mb changed
            if 'ram_mb' in updates:
                self.config.apply_low_ram_profile(updates['ram_mb'])
                self.available_ram_mb = updates['ram_mb']
            
            self._logger.success(f"Configuration updated: {list(updates.keys())}")
            
            return TaskResult(success=True, message="Configuration updated")
        
        except Exception as e:
            return TaskResult(success=False, message="Config update failed", error=str(e))
    
    # === MEMORY MANAGEMENT ===
    
    def check_memory(self) -> TaskResult:
        """Check current memory usage."""
        try:
            self._ensure_initialized()
            
            usage = self._device_mgr.get_memory_usage()
            safe = self._device_mgr.check_memory_safe(self.config.max_ram_mb)
            
            data = {
                'current_usage_mb': usage,
                'max_allowed_mb': self.config.max_ram_mb,
                'available_mb': self.available_ram_mb,
                'is_safe': safe
            }
            
            status = "OK" if safe else "WARNING: Low memory"
            self._logger.info(f"Memory check: {status} ({usage:.1f}MB used)")
            
            return TaskResult(
                success=True,
                message=f"Memory: {usage:.1f}MB used",
                data=data
            )
        
        except Exception as e:
            return TaskResult(success=False, message="Memory check failed", error=str(e))
    
    def clear_memory(self) -> TaskResult:
        """Clear device memory cache."""
        try:
            self._ensure_initialized()
            self._device_mgr.clear_cache()
            
            usage = self._device_mgr.get_memory_usage()
            self._logger.success(f"Memory cleared ({usage:.1f}MB used)")
            
            return TaskResult(success=True, message=f"Memory cleared ({usage:.1f}MB)")
        
        except Exception as e:
            return TaskResult(success=False, message="Memory clear failed", error=str(e))
    
    # === UTILITIES ===
    
    def _recommend_batch_size(self) -> int:
        """Recommend batch size based on RAM."""
        if self.available_ram_mb < 512:
            return 2
        elif self.available_ram_mb < 1024:
            return 4
        elif self.available_ram_mb < 2048:
            return 8
        else:
            return 16
    
    def run_task(self, task: str, **kwargs) -> TaskResult:
        """
        Run a named task.
        
        Supported tasks:
        - detect: Detect hardware
        - select_model: Select best model
        - list_models: List available models
        - show_config: Show configuration
        - check_memory: Check memory usage
        - clear_memory: Clear memory cache
        
        Args:
            task: Task name
            **kwargs: Task-specific arguments
        
        Returns:
            TaskResult
        """
        task_map = {
            'detect': self.detect_hardware,
            'select_model': self.select_model,
            'list_models': self.list_models,
            'show_config': self.show_config,
            'update_config': self.update_config,
            'check_memory': self.check_memory,
            'clear_memory': self.clear_memory,
        }
        
        if task not in task_map:
            return TaskResult(
                success=False,
                message=f"Unknown task: {task}",
                recovery_suggestion=f"Available tasks: {list(task_map.keys())}"
            )
        
        self._logger.debug(f"Running task: {task}")
        return task_map[task](**kwargs)
    
    def print_status(self):
        """Print full system status."""
        print("\n" + "=" * 60)
        print("UARF CONTROLLER STATUS")
        print("=" * 60)
        
        # Hardware
        result = self.detect_hardware()
        if result.success:
            data = result.data
            print(f"\n🖥️  Device: {data['device_type']} ({data['device_name']})")
            print(f"   Available RAM: {data['available_memory_mb']:.0f} MB")
            print(f"   Recommended Batch Size: {data['recommended_batch_size']}")
        
        # Model
        result = self.select_model()
        if result.success:
            data = result.data
            print(f"\n📦 Model: {data['model']}")
            print(f"   Name: {data['name']}")
            if 'params_millions' in data:
                print(f"   Parameters: {data['params_millions']}M")
        
        # Config
        print(f"\n⚙️  Configuration:")
        print(f"   Debug Mode: {self.config.debug_mode}")
        print(f"   Streaming: {self.config.streaming}")
        print(f"   Max Steps: {self.config.max_steps}")
        print(f"   Batch Size: {self.config.batch_size}")
        
        # Memory
        result = self.check_memory()
        if result.success:
            data = result.data
            print(f"\n💾 Memory:")
            print(f"   Current Usage: {data['current_usage_mb']:.1f} MB")
            print(f"   Status: {'✅ OK' if data['is_safe'] else '⚠️ LOW'}")
        
        print("\n" + "=" * 60)


# Convenience functions for Qwen Chat usage

def quick_start(ram_mb: int = None, debug: bool = False) -> UARFController:
    """
    Quick start controller with minimal setup.
    
    Example:
        controller = quick_start(ram_mb=512, debug=True)
        result = controller.run_task('detect')
    """
    return UARFController(ram_mb=ram_mb, debug=debug)


def detect() -> Dict[str, Any]:
    """Quick hardware detection."""
    ctrl = UARFController()
    result = ctrl.detect_hardware()
    return result.data if result.success else {'error': result.error}


def suggest() -> Dict[str, Any]:
    """Quick model suggestion."""
    ctrl = UARFController()
    result = ctrl.select_model()
    return result.data if result.success else {'error': result.error}
