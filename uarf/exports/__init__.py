"""
UARF Export Modules
Unified interface for all export formats
"""

from .gguf import GGUFExporter, export_to_gguf
from .edge import (
    EdgeDeviceType, EdgeDeviceDetector, EdgeOptimizer,
    detect_edge_device, optimize_for_edge
)
from .litert import (
    DelegateType, LiteRTExporter, export_to_litert
)
from .bitnet import (
    BitNetQuantizer, BitNetInference,
    quantize_to_1bit, create_inference_engine
)
from .webgpu import (
    WebGPUConfig, WebGPUExporter, export_for_webgpu
)
from .turboquant import (
    QuantizationLevel, TurboQuantConfig, TurboQuantEngine,
    TurboQuantInference, create_turboquant_inference
)


class UniversalExporter:
    """
    Unified exporter supporting all formats
    
    Usage:
        exporter = UniversalExporter()
        exporter.export(model, config, "model.gguf", format="gguf")
        exporter.export(model, config, "output/", format="webgpu")
    """
    
    def __init__(self):
        self.exporters = {
            "gguf": self._export_gguf,
            "tflite": self._export_tflite,
            "litert": self._export_tflite,
            "edge": self._export_edge,
            "bitnet": self._export_bitnet,
            "webgpu": self._export_webgpu,
            "turboquant": self._export_turboquant,
        }
    
    def export(self, model_state: dict, config: dict, output_path: str,
               format: str = "gguf", **kwargs) -> any:
        """
        Export model to specified format
        
        Args:
            model_state: Model state dict
            config: Model configuration
            output_path: Output path
            format: Export format (gguf, tflite, edge, bitnet, webgpu, turboquant)
            **kwargs: Format-specific arguments
            
        Returns:
            Export result (path or engine)
        """
        if format not in self.exporters:
            raise ValueError(f"Unsupported format: {format}")
        
        return self.exporters[format](model_state, config, output_path, **kwargs)
    
    def _export_gguf(self, model_state, config, output_path, **kwargs):
        quantization = kwargs.get("quantization", "f16")
        return export_to_gguf(model_state, config, output_path, quantization)
    
    def _export_tflite(self, model_state, config, output_path, **kwargs):
        quantization = kwargs.get("quantization", "fp16")
        delegate = kwargs.get("delegate", "cpu")
        input_shape = kwargs.get("input_shape", (1, 512))
        return export_to_litert(model_state, config, output_path, quantization, delegate, input_shape)
    
    def _export_edge(self, model_state, config, output_path, **kwargs):
        device_type = kwargs.get("device_type", None)
        optimized_model, deploy_config = optimize_for_edge(model_state, config, device_type)
        return optimized_model, deploy_config
    
    def _export_bitnet(self, model_state, config, output_path, **kwargs):
        quant_type = kwargs.get("quantization_type", "bitnet_b158")
        pack = kwargs.get("pack_weights", True)
        return quantize_to_1bit(model_state, config, quant_type, pack)
    
    def _export_webgpu(self, model_state, config, output_path, **kwargs):
        precision = kwargs.get("precision", "fp16")
        return export_for_webgpu(model_state, config, output_path, precision)
    
    def _export_turboquant(self, model_state, config, output_path, **kwargs):
        quantization = kwargs.get("quantization", "int8")
        return create_turboquant_inference(model_state, config, quantization)
    
    def list_formats(self) -> list:
        """List all supported export formats"""
        return list(self.exporters.keys())


__all__ = [
    # Main exporter
    "UniversalExporter",
    
    # GGUF
    "GGUFExporter", "export_to_gguf",
    
    # Edge
    "EdgeDeviceType", "EdgeDeviceDetector", "EdgeOptimizer",
    "detect_edge_device", "optimize_for_edge",
    
    # LiteRT
    "DelegateType", "LiteRTExporter", "export_to_litert",
    
    # BitNet
    "BitNetQuantizer", "BitNetInference",
    "quantize_to_1bit", "create_inference_engine",
    
    # WebGPU
    "WebGPUConfig", "WebGPUExporter", "export_for_webgpu",
    
    # TurboQuant
    "QuantizationLevel", "TurboQuantConfig", "TurboQuantEngine",
    "TurboQuantInference", "create_turboquant_inference",
]
