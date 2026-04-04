"""
Edge Device Support Module for UARF
Provides optimization and deployment support for edge devices:
- NVIDIA Jetson (Nano, Xavier, Orin)
- Google Coral (Edge TPU)
- Raspberry Pi
- Mobile devices (Android/iOS)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EdgeDeviceType(Enum):
    """Supported edge device types"""
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    JETSON_ORIN = "jetson_orin"
    CORAL_USB = "coral_usb"
    CORAL_EDGE_TPU = "coral_edge_tpu"
    RASPBERRY_PI_4 = "raspberry_pi_4"
    RASPBERRY_PI_5 = "raspberry_pi_5"
    ANDROID = "android"
    IOS = "ios"
    UNKNOWN = "unknown"


@dataclass
class EdgeDeviceSpec:
    """Edge device specifications"""
    name: str
    device_type: EdgeDeviceType
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    tpu_available: bool
    max_model_size_mb: float
    recommended_quantization: str
    platform_specific: Dict[str, Any]


class EdgeDeviceDetector:
    """Detects and profiles edge devices"""
    
    DEVICE_SPECS = {
        EdgeDeviceType.JETSON_NANO: EdgeDeviceSpec(
            name="NVIDIA Jetson Nano",
            device_type=EdgeDeviceType.JETSON_NANO,
            cpu_cores=4,
            ram_gb=4.0,
            gpu_available=True,
            tpu_available=False,
            max_model_size_mb=512,
            recommended_quantization="int8",
            platform_specific={"cuda_compute_capability": 5.3, "jetpack_version": "4.x"}
        ),
        EdgeDeviceType.JETSON_XAVIER: EdgeDeviceSpec(
            name="NVIDIA Jetson Xavier NX",
            device_type=EdgeDeviceType.JETSON_XAVIER,
            cpu_cores=6,
            ram_gb=8.0,
            gpu_available=True,
            tpu_available=False,
            max_model_size_mb=2048,
            recommended_quantization="fp16",
            platform_specific={"cuda_compute_capability": 7.2, "jetpack_version": "5.x"}
        ),
        EdgeDeviceType.JETSON_ORIN: EdgeDeviceSpec(
            name="NVIDIA Jetson Orin",
            device_type=EdgeDeviceType.JETSON_ORIN,
            cpu_cores=12,
            ram_gb=16.0,
            gpu_available=True,
            tpu_available=False,
            max_model_size_mb=8192,
            recommended_quantization="fp16",
            platform_specific={"cuda_compute_capability": 8.7, "jetpack_version": "6.x"}
        ),
        EdgeDeviceType.CORAL_USB: EdgeDeviceSpec(
            name="Google Coral USB Accelerator",
            device_type=EdgeDeviceType.CORAL_USB,
            cpu_cores=4,
            ram_gb=4.0,
            gpu_available=False,
            tpu_available=True,
            max_model_size_mb=256,
            recommended_quantization="int8",
            platform_specific={"edgetpu_version": "1.0"}
        ),
        EdgeDeviceType.RASPBERRY_PI_4: EdgeDeviceSpec(
            name="Raspberry Pi 4",
            device_type=EdgeDeviceType.RASPBERRY_PI_4,
            cpu_cores=4,
            ram_gb=8.0,
            gpu_available=False,
            tpu_available=False,
            max_model_size_mb=512,
            recommended_quantization="int8",
            platform_specific={"neon_support": True}
        ),
        EdgeDeviceType.RASPBERRY_PI_5: EdgeDeviceSpec(
            name="Raspberry Pi 5",
            device_type=EdgeDeviceType.RASPBERRY_PI_5,
            cpu_cores=4,
            ram_gb=8.0,
            gpu_available=False,
            tpu_available=False,
            max_model_size_mb=1024,
            recommended_quantization="int8",
            platform_specific={"neon_support": True, "pcie_support": True}
        ),
    }
    
    def detect(self) -> EdgeDeviceSpec:
        """Detect current edge device"""
        import platform
        import os
        
        system = platform.system()
        machine = platform.machine()
        
        # Check for Jetson
        if os.path.exists('/etc/nv_tegra_release'):
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                if 'ORIN' in content:
                    return self.DEVICE_SPECS[EdgeDeviceType.JETSON_ORIN]
                elif 'XAVIER' in content:
                    return self.DEVICE_SPECS[EdgeDeviceType.JETSON_XAVIER]
                else:
                    return self.DEVICE_SPECS[EdgeDeviceType.JETSON_NANO]
        
        # Check for Coral
        try:
            import edgetpu
            return self.DEVICE_SPECS[EdgeDeviceType.CORAL_USB]
        except ImportError:
            pass
        
        # Check for Raspberry Pi
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi 5' in model:
                    return self.DEVICE_SPECS[EdgeDeviceType.RASPBERRY_PI_5]
                elif 'Raspberry Pi 4' in model:
                    return self.DEVICE_SPECS[EdgeDeviceType.RASPBERRY_PI_4]
        
        # Default based on system
        if system == 'Android':
            return EdgeDeviceSpec(
                name="Android Device",
                device_type=EdgeDeviceType.ANDROID,
                cpu_cores=8,
                ram_gb=6.0,
                gpu_available=True,
                tpu_available=False,
                max_model_size_mb=1024,
                recommended_quantization="fp16",
                platform_specific={}
            )
        
        return EdgeDeviceSpec(
            name="Unknown Device",
            device_type=EdgeDeviceType.UNKNOWN,
            cpu_cores=4,
            ram_gb=4.0,
            gpu_available=False,
            tpu_available=False,
            max_model_size_mb=256,
            recommended_quantization="int8",
            platform_specific={}
        )


class EdgeOptimizer:
    """Optimizes models for edge deployment"""
    
    def __init__(self, device_spec: EdgeDeviceSpec):
        """
        Initialize edge optimizer
        
        Args:
            device_spec: Target device specifications
        """
        self.device_spec = device_spec
        logger.info(f"Initialized optimizer for {device_spec.name}")
    
    def optimize(self, model_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize model for target edge device
        
        Args:
            model_state: Model state dict
            config: Model configuration
            
        Returns:
            Optimized model state
        """
        logger.info(f"Optimizing model for {self.device_spec.name}")
        
        # Apply quantization
        if self.device_spec.recommended_quantization == "int8":
            model_state = self._quantize_int8(model_state)
        elif self.device_spec.recommended_quantization == "fp16":
            model_state = self._quantize_fp16(model_state)
        
        # Prune if needed for small devices
        if self.device_spec.max_model_size_mb < 512:
            model_state = self._prune_weights(model_state, pruning_ratio=0.3)
        
        return model_state
    
    def _quantize_int8(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize model to INT8"""
        import numpy as np
        
        quantized = {}
        for name, tensor in model_state.items():
            if hasattr(tensor, 'cpu'):
                tensor = tensor.cpu()
            if hasattr(tensor, 'numpy'):
                tensor = tensor.numpy()
            
            # Simple symmetric quantization
            min_val = np.min(tensor)
            max_val = np.max(tensor)
            scale = max(abs(min_val), abs(max_val)) / 127.0
            
            if scale > 0:
                quantized_tensor = np.round(tensor / scale).astype(np.int8)
                # Store scale as metadata
                quantized[f"{name}_scale"] = np.array([scale], dtype=np.float32)
                quantized[name] = quantized_tensor.astype(np.float32)  # Keep as float for compatibility
            else:
                quantized[name] = tensor
        
        return quantized
    
    def _quantize_fp16(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize model to FP16"""
        import numpy as np
        
        quantized = {}
        for name, tensor in model_state.items():
            if hasattr(tensor, 'cpu'):
                tensor = tensor.cpu()
            if hasattr(tensor, 'numpy'):
                tensor = tensor.numpy()
            
            quantized[name] = tensor.astype(np.float16)
        
        return quantized
    
    def _prune_weights(self, model_state: Dict[str, Any], 
                       pruning_ratio: float) -> Dict[str, Any]:
        """Prune model weights"""
        import numpy as np
        
        pruned = {}
        for name, tensor in model_state.items():
            if hasattr(tensor, 'cpu'):
                tensor = tensor.cpu()
            if hasattr(tensor, 'numpy'):
                tensor = tensor.numpy()
            
            threshold = np.percentile(np.abs(tensor), pruning_ratio * 100)
            mask = np.abs(tensor) > threshold
            pruned[name] = tensor * mask
        
        return pruned
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration for target device"""
        return {
            "device_name": self.device_spec.name,
            "device_type": self.device_spec.device_type.value,
            "max_model_size_mb": self.device_spec.max_model_size_mb,
            "quantization": self.device_spec.recommended_quantization,
            "gpu_acceleration": self.device_spec.gpu_available,
            "tpu_acceleration": self.device_spec.tpu_available,
            "recommended_batch_size": 1 if self.device_spec.ram_gb < 4 else 4,
            "platform_specific": self.device_spec.platform_specific
        }


def detect_edge_device() -> EdgeDeviceSpec:
    """Convenience function to detect edge device"""
    detector = EdgeDeviceDetector()
    return detector.detect()


def optimize_for_edge(model_state: Dict[str, Any], config: Dict[str, Any],
                      device_type: Optional[EdgeDeviceType] = None) -> tuple:
    """
    Optimize model for edge deployment
    
    Args:
        model_state: Model state dict
        config: Model configuration
        device_type: Target device type (auto-detect if None)
        
    Returns:
        Tuple of (optimized_model, deployment_config)
    """
    detector = EdgeDeviceDetector()
    
    if device_type:
        device_spec = detector.DEVICE_SPECS.get(device_type, detector.detect())
    else:
        device_spec = detector.detect()
    
    optimizer = EdgeOptimizer(device_spec)
    optimized_model = optimizer.optimize(model_state, config)
    deployment_config = optimizer.get_deployment_config()
    
    return optimized_model, deployment_config
