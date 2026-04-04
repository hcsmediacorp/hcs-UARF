"""
GGUF Export Module for UARF
Exports trained models to GGUF format for llama.cpp compatibility
"""

import struct
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from enum import IntEnum

logger = logging.getLogger(__name__)


class GGMLType(IntEnum):
    """GGML tensor types"""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 5
    Q5_1 = 6
    Q8_0 = 7
    Q8_1 = 8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 30


class GGUFVersion(IntEnum):
    """GGUF specification versions"""
    V1 = 1
    V2 = 2
    V3 = 3


class GGUFExporter:
    """
    Exports PyTorch models to GGUF format
    
    Supports:
    - F32, F16, Q8_0 quantization
    - Metadata embedding
    - llama.cpp compatible output
    """
    
    def __init__(self, quantization: str = "f16"):
        """
        Initialize GGUF exporter
        
        Args:
            quantization: Quantization type (f32, f16, q8_0)
        """
        self.quantization = quantization.lower()
        self.gguf_version = GGUFVersion.V3
        self.alignment = 32
        
        self.type_map = {
            "f32": GGMLType.F32,
            "f16": GGMLType.F16,
            "q8_0": GGMLType.Q8_0,
        }
        
        if self.quantization not in self.type_map:
            raise ValueError(f"Unsupported quantization: {quantization}")
    
    def export(self, model_state: Dict[str, Any], config: Dict[str, Any], 
               output_path: str) -> Path:
        """
        Export model to GGUF format
        
        Args:
            model_state: PyTorch state dict
            config: Model configuration
            output_path: Output file path
            
        Returns:
            Path to exported GGUF file
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting model to GGUF ({self.quantization}): {output}")
        
        with open(output, 'wb') as f:
            # Write header
            self._write_header(f)
            
            # Write metadata
            self._write_metadata(f, config)
            
            # Write tensors
            self._write_tensors(f, model_state, config)
        
        logger.info(f"GGUF export complete: {output} ({output.stat().st_size / 1024 / 1024:.2f} MB)")
        return output
    
    def _write_header(self, f):
        """Write GGUF magic and version"""
        f.write(b'GGUF')
        f.write(struct.pack('<I', int(self.gguf_version)))
        # Tensor count and KV count written later
        f.write(struct.pack('<Q', 0))  # Placeholder for tensor count
        f.write(struct.pack('<Q', 0))  # Placeholder for KV count
    
    def _write_metadata(self, f, config: Dict[str, Any]):
        """Write key-value metadata"""
        kv_data = []
        
        # General metadata
        kv_data.append(("general.architecture", "llama"))
        kv_data.append(("general.name", config.get("model_name", "uarf_model")))
        kv_data.append(("general.file_type", self.type_map[self.quantization].value))
        
        # Model architecture
        if "hidden_size" in config:
            kv_data.append(("llama.embedding_length", config["hidden_size"]))
        if "num_heads" in config:
            kv_data.append(("llama.attention.head_count", config["num_heads"]))
        if "num_layers" in config:
            kv_data.append(("llama.block_count", config["num_layers"]))
        if "vocab_size" in config:
            kv_data.append(("llama.vocab_size", config["vocab_size"]))
        if "max_seq_len" in config:
            kv_data.append(("llama.context_length", config["max_seq_len"]))
        
        # Write KV count (update header later)
        self.kv_count = len(kv_data)
        
        for key, value in kv_data:
            self._write_kv_pair(f, key, value)
    
    def _write_kv_pair(self, f, key: str, value):
        """Write a single key-value pair"""
        # Write key
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        
        # Write value type and data
        if isinstance(value, str):
            f.write(struct.pack('<I', 1))  # STRING type
            val_bytes = value.encode('utf-8')
            f.write(struct.pack('<Q', len(val_bytes)))
            f.write(val_bytes)
        elif isinstance(value, int):
            f.write(struct.pack('<I', 0))  # UINT32 type
            f.write(struct.pack('<I', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 2))  # FLOAT32 type
            f.write(struct.pack('<f', value))
    
    def _write_tensors(self, f, state_dict: Dict, config: Dict[str, Any]):
        """Write model tensors"""
        tensor_count = len(state_dict)
        
        # Go back and update tensor count in header
        f.seek(8)
        f.write(struct.pack('<Q', tensor_count))
        f.write(struct.pack('<Q', self.kv_count))
        f.seek(0, 2)  # Seek to end
        
        for name, tensor in state_dict.items():
            self._write_tensor(f, name, tensor)
    
    def _write_tensor(self, f, name: str, tensor):
        """Write a single tensor"""
        import numpy as np
        
        # Convert to numpy if needed
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        if hasattr(tensor, 'numpy'):
            tensor = tensor.numpy()
        elif not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
        
        # Quantize if needed
        if self.quantization == "q8_0":
            tensor = self._quantize_q8_0(tensor)
            ggml_type = GGMLType.Q8_0
        elif self.quantization == "f16":
            tensor = tensor.astype(np.float16)
            ggml_type = GGMLType.F16
        else:
            tensor = tensor.astype(np.float32)
            ggml_type = GGMLType.F32
        
        # Write tensor name
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('<Q', len(name_bytes)))
        f.write(name_bytes)
        
        # Write dimensions
        n_dims = len(tensor.shape)
        f.write(struct.pack('<I', n_dims))
        for dim in reversed(tensor.shape):  # GGUF uses reverse order
            f.write(struct.pack('<Q', dim))
        
        # Write type
        f.write(struct.pack('<I', int(ggml_type)))
        
        # Write offset (calculated later)
        offset_pos = f.tell()
        f.write(struct.pack('<Q', 0))  # Placeholder
        
        # Align and write data
        current_pos = f.tell()
        aligned_pos = ((current_pos + self.alignment - 1) // self.alignment) * self.alignment
        padding = aligned_pos - current_pos
        if padding > 0:
            f.write(b'\x00' * padding)
        
        # Update offset
        data_offset = f.tell()
        f.seek(offset_pos)
        f.write(struct.pack('<Q', data_offset))
        f.seek(data_offset)
        
        # Write tensor data
        f.write(tensor.tobytes())
    
    def _quantize_q8_0(self, tensor):
        """Simple Q8_0 quantization"""
        import numpy as np
        
        # Reshape for block processing (block size = 32)
        original_shape = tensor.shape
        flat = tensor.flatten().astype(np.float32)
        
        # Pad to multiple of 32
        block_size = 32
        pad_size = (block_size - len(flat) % block_size) % block_size
        if pad_size > 0:
            flat = np.pad(flat, (0, pad_size))
        
        # Quantize each block
        num_blocks = len(flat) // block_size
        quantized = np.zeros(len(flat), dtype=np.int8)
        
        for i in range(num_blocks):
            block = flat[i * block_size:(i + 1) * block_size]
            max_abs = np.max(np.abs(block))
            if max_abs > 0:
                scale = max_abs / 127.0
                quantized_block = np.round(block / scale).astype(np.int8)
                quantized[i * block_size:(i + 1) * block_size] = quantized_block
        
        return quantized.reshape(original_shape).astype(np.float32)


def export_to_gguf(model_state: Dict[str, Any], config: Dict[str, Any], 
                   output_path: str, quantization: str = "f16") -> Path:
    """
    Convenience function to export model to GGUF
    
    Args:
        model_state: Model state dict
        config: Model configuration
        output_path: Output file path
        quantization: Quantization type (f32, f16, q8_0)
        
    Returns:
        Path to exported file
    """
    exporter = GGUFExporter(quantization=quantization)
    return exporter.export(model_state, config, output_path)
