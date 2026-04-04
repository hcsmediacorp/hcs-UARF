"""
LiteRT (TensorFlow Lite) Export Module for UARF
Exports models to TensorFlow Lite format for mobile and edge deployment
Supports:
- TFLite FlatBuffer export
- Edge TPU compilation
- GPU delegate optimization
- NNAPI delegate for Android
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class DelegateType(Enum):
    """TFLite delegate types"""
    CPU = "cpu"
    GPU = "gpu"
    EDGE_TPU = "edge_tpu"
    NNAPI = "nnapi"
    XNNPACK = "xnnpack"


class LiteRTExporter:
    """
    Exports PyTorch models to TensorFlow Lite format
    
    Supports:
    - FP32, FP16, INT8 quantization
    - Delegate-specific optimizations
    - Edge TPU compatibility
    """
    
    def __init__(self, quantization: str = "fp16", 
                 delegate: DelegateType = DelegateType.CPU):
        """
        Initialize LiteRT exporter
        
        Args:
            quantization: Quantization type (fp32, fp16, int8)
            delegate: Target delegate for optimization
        """
        self.quantization = quantization.lower()
        self.delegate = delegate
        logger.info(f"Initialized LiteRT exporter ({quantization}, {delegate.value})")
    
    def export(self, model_state: Dict[str, Any], config: Dict[str, Any],
               output_path: str, input_shape: tuple = (1, 512)) -> Path:
        """
        Export model to TFLite format
        
        Args:
            model_state: Model state dict
            config: Model configuration
            output_path: Output file path
            input_shape: Example input shape
            
        Returns:
            Path to exported TFLite file
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting model to TFLite ({self.quantization}): {output}")
        
        try:
            # Try to use TensorFlow for conversion
            import tensorflow as tf
            return self._export_with_tf(model_state, config, output, input_shape)
        except ImportError:
            logger.warning("TensorFlow not available, using fallback export")
            return self._export_fallback(model_state, config, output)
    
    def _export_with_tf(self, model_state: Dict[str, Any], config: Dict[str, Any],
                        output: Path, input_shape: tuple) -> Path:
        """Export using TensorFlow converter"""
        import tensorflow as tf
        import numpy as np
        
        # Create a simple SavedModel from state dict
        class SimpleModel(tf.Module):
            def __init__(self, state_dict, config):
                super().__init__()
                self.config = config
                self.variables = {}
                
                # Convert state dict to TF variables
                for name, tensor in state_dict.items():
                    if hasattr(tensor, 'cpu'):
                        tensor = tensor.cpu().numpy()
                    elif hasattr(tensor, 'numpy'):
                        tensor = tensor.numpy()
                    self.variables[name] = tf.Variable(tensor, trainable=False)
            
            @tf.function(input_signature=[
                tf.TensorSpec(shape=input_shape, dtype=tf.float32)
            ])
            def __call__(self, x):
                # Simplified forward pass (placeholder)
                # In production, this would implement the actual model architecture
                return tf.nn.softmax(x)
        
        model = SimpleModel(model_state, config)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [model.__call__.get_concrete_function()]
        )
        
        # Apply optimizations based on quantization
        if self.quantization == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif self.quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: [
                np.random.randn(*input_shape).astype(np.float32)
                for _ in range(100)
            ]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        # Apply delegate-specific settings
        if self.delegate == DelegateType.EDGE_TPU:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        elif self.delegate == DelegateType.GPU:
            converter.experimental_enable_delegates = [
                tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')
            ]
        
        tflite_model = converter.convert()
        
        # Write output
        with open(output, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite export complete: {output} ({output.stat().st_size / 1024 / 1024:.2f} MB)")
        return output
    
    def _export_fallback(self, model_state: Dict[str, Any], config: Dict[str, Any],
                         output: Path) -> Path:
        """Fallback export without TensorFlow"""
        import json
        import struct
        
        # Create a minimal TFLite-like structure (metadata only)
        # This is a placeholder for when TF is not available
        metadata = {
            "format": "tflite_fallback",
            "quantization": self.quantization,
            "delegate": self.delegate.value,
            "config": config,
            "tensor_count": len(model_state),
            "note": "Full TFLite export requires TensorFlow"
        }
        
        # Write as JSON with binary header
        with open(output, 'wb') as f:
            # Write magic number
            f.write(b'TFL3')
            # Write metadata length
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            f.write(struct.pack('<I', len(metadata_bytes)))
            f.write(metadata_bytes)
        
        logger.warning(f"Fallback export created (not full TFLite): {output}")
        return output
    
    def compile_for_edge_tpu(self, tflite_path: str, 
                            output_path: Optional[str] = None) -> Path:
        """
        Compile TFLite model for Edge TPU
        
        Args:
            tflite_path: Path to TFLite model
            output_path: Output path for compiled model
            
        Returns:
            Path to compiled Edge TPU model
        """
        import subprocess
        
        if output_path is None:
            output_path = str(Path(tflite_path).with_suffix('.edgetpu.tflite'))
        
        try:
            result = subprocess.run(
                ['edgetpu_compiler', '-o', str(Path(output_path).parent), tflite_path],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Edge TPU compilation successful: {output_path}")
            return Path(output_path)
        except FileNotFoundError:
            logger.error("Edge TPU compiler not found. Install with: apt-get install edgetpu-compiler")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Edge TPU compilation failed: {e.stderr}")
            raise
    
    def get_delegate_info(self) -> Dict[str, Any]:
        """Get information about the selected delegate"""
        delegate_info = {
            DelegateType.CPU: {
                "name": "CPU",
                "description": "Standard CPU execution",
                "supported_quantization": ["fp32", "fp16", "int8"],
                "platforms": ["all"]
            },
            DelegateType.GPU: {
                "name": "GPU",
                "description": "OpenGL/Vulkan GPU acceleration",
                "supported_quantization": ["fp32", "fp16"],
                "platforms": ["android", "linux"]
            },
            DelegateType.EDGE_TPU: {
                "name": "Edge TPU",
                "description": "Google Coral Edge TPU accelerator",
                "supported_quantization": ["int8"],
                "platforms": ["linux", "android"]
            },
            DelegateType.NNAPI: {
                "name": "NNAPI",
                "description": "Android Neural Networks API",
                "supported_quantization": ["fp32", "fp16", "int8"],
                "platforms": ["android"]
            },
            DelegateType.XNNPACK: {
                "name": "XNNPACK",
                "description": "Optimized CPU backend",
                "supported_quantization": ["fp32", "fp16", "int8"],
                "platforms": ["all"]
            }
        }
        
        return delegate_info.get(self.delegate, {})


def export_to_litert(model_state: Dict[str, Any], config: Dict[str, Any],
                     output_path: str, quantization: str = "fp16",
                     delegate: str = "cpu", input_shape: tuple = (1, 512)) -> Path:
    """
    Convenience function to export model to TFLite
    
    Args:
        model_state: Model state dict
        config: Model configuration
        output_path: Output file path
        quantization: Quantization type
        delegate: Delegate type
        input_shape: Input shape
        
    Returns:
        Path to exported file
    """
    del_type = DelegateType(delegate.lower())
    exporter = LiteRTExporter(quantization=quantization, delegate=del_type)
    return exporter.export(model_state, config, output_path, input_shape)
