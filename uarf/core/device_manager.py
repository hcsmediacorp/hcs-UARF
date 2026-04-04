"""
UARF Device Manager - Lightweight device setup and memory management.
Extracted from UniversalTrainer for modularity and testability.
Supports lazy initialization and memory-safe device selection.
"""

import os
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Device information."""
    device_type: str  # cpu, cuda, mps
    device_id: int = 0
    name: str = ""
    memory_mb: float = 0.0
    available_mb: float = 0.0


class DeviceManager:
    """
    Manages device selection and initialization.
    
    Features:
    - Lazy device initialization (no torch import until needed)
    - Memory-safe device selection
    - Automatic fallback to CPU
    - Minimal overhead
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize device manager.
        
        Args:
            device: Device specification (auto, cpu, cuda, mps, cuda:N)
        """
        self.device_spec = device
        self._device = None
        self._torch = None
        self._info: Optional[DeviceInfo] = None
    
    def _import_torch(self):
        """Lazy torch import."""
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch
    
    @property
    def device(self):
        """Get torch.device (lazy initialization)."""
        if self._device is None:
            self._setup_device()
        return self._device
    
    @property
    def info(self) -> DeviceInfo:
        """Get device info (lazy initialization)."""
        if self._info is None:
            self._setup_device()
        return self._info
    
    def _setup_device(self):
        """Setup device based on specification."""
        torch = self._import_torch()
        
        if self.device_spec == "auto":
            # Auto-detect best available device
            if torch.cuda.is_available():
                self._device = torch.device("cuda:0")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        elif self.device_spec.startswith("cuda:"):
            # Specific CUDA device
            self._device = torch.device(self.device_spec)
        elif self.device_spec == "cuda":
            if torch.cuda.is_available():
                self._device = torch.device("cuda:0")
            else:
                raise RuntimeError("CUDA not available")
        elif self.device_spec == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                raise RuntimeError("MPS not available")
        elif self.device_spec == "cpu":
            self._device = torch.device("cpu")
        else:
            raise ValueError(f"Unknown device spec: {self.device_spec}")
        
        # Gather device info
        self._info = self._gather_info(torch)
    
    def _gather_info(self, torch) -> DeviceInfo:
        """Gather device information."""
        if self._device.type == "cuda":
            try:
                gpu_id = self._device.index or 0
                name = torch.cuda.get_device_name(gpu_id)
                total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 / 1024
                alloc_mem = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
                free_mem = total_mem - alloc_mem
                return DeviceInfo(
                    device_type="cuda",
                    device_id=gpu_id,
                    name=name,
                    memory_mb=total_mem,
                    available_mb=free_mem
                )
            except Exception:
                return DeviceInfo(device_type="cuda")
        
        elif self._device.type == "mps":
            # MPS doesn't have good memory introspection yet
            return DeviceInfo(device_type="mps", name="Apple Silicon")
        
        else:
            # CPU
            import psutil
            try:
                mem = psutil.virtual_memory()
                available_mb = mem.available / 1024 / 1024
            except Exception:
                available_mb = 0
            
            return DeviceInfo(
                device_type="cpu",
                name=psutil.cpu_brand() if hasattr(psutil, 'cpu_brand') else "CPU",
                available_mb=available_mb
            )
    
    def get_dtype(self, precision: str = "auto") -> 'torch.dtype':
        """
        Get appropriate dtype for precision.
        
        Args:
            precision: auto, fp32, fp16, bf16, int8
        
        Returns:
            torch.dtype
        """
        torch = self._import_torch()
        
        if precision == "auto":
            if self._device.type == "cuda":
                cap = torch.cuda.get_device_capability(0)
                if cap[0] >= 8:  # Ampere or newer
                    return torch.bfloat16
                else:
                    return torch.float16
            elif self._device.type == "mps":
                return torch.float16
            else:
                return torch.float32
        
        precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int8": torch.int8,
        }
        
        return precision_map.get(precision, torch.float32)
    
    def clear_cache(self):
        """Clear device memory cache."""
        if self._device is None:
            return
        
        torch = self._import_torch()
        
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self._device is None:
            return 0.0
        
        torch = self._import_torch()
        
        if self._device.type == "cuda":
            return torch.cuda.memory_allocated() / 1024 / 1024
        elif self._device.type == "mps":
            # MPS memory tracking is limited
            return 0.0
        else:
            # CPU memory via psutil
            try:
                import psutil
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024
            except Exception:
                return 0.0
    
    def check_memory_safe(self, required_mb: float) -> bool:
        """
        Check if allocating required_mb is safe.
        
        Args:
            required_mb: Required memory in MB
        
        Returns:
            True if allocation is likely safe
        """
        if self._info is None:
            self._setup_device()
        
        if self._device.type == "cpu":
            # Conservative: use 80% of available RAM
            return self._info.available_mb * 0.8 > required_mb
        
        elif self._device.type == "cuda":
            # Conservative: use 90% of free VRAM
            return self._info.available_mb * 0.9 > required_mb
        
        else:
            # Unknown/conservative
            return True
    
    def print_status(self):
        """Print device status."""
        if self._info is None:
            self._setup_device()
        
        print(f"\n🖥️  Device: {self._device}")
        print(f"   Type: {self._info.device_type}")
        if self._info.name:
            print(f"   Name: {self._info.name}")
        if self._info.memory_mb > 0:
            print(f"   Total Memory: {self._info.memory_mb:.1f} MB")
        if self._info.available_mb > 0:
            print(f"   Available: {self._info.available_mb:.1f} MB")
        print()


def select_device(
    prefer: str = "auto",
    min_ram_mb: float = 0,
    allow_cpu: bool = True
) -> DeviceManager:
    """
    Select best available device.
    
    Args:
        prefer: Preferred device type
        min_ram_mb: Minimum required RAM
        allow_cpu: Allow CPU fallback
    
    Returns:
        Configured DeviceManager
    """
    # Try preferred device first
    if prefer != "auto":
        try:
            mgr = DeviceManager(prefer)
            _ = mgr.device  # Trigger setup
            return mgr
        except Exception:
            pass
    
    # Auto-detect
    try:
        mgr = DeviceManager("auto")
        _ = mgr.device
        
        # Check if it meets requirements
        if mgr.info.available_mb >= min_ram_mb:
            return mgr
        
        # Doesn't meet requirements, try CPU
        if allow_cpu:
            return DeviceManager("cpu")
        
        raise RuntimeError(f"No device with {min_ram_mb}MB RAM available")
    
    except Exception as e:
        if allow_cpu:
            return DeviceManager("cpu")
        raise e


# Convenience function
def get_device(device: str = "auto") -> 'torch.device':
    """Get torch.device quickly."""
    mgr = DeviceManager(device)
    return mgr.device
