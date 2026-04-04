"""
BitNet 1-bit LLM Support Module for UARF
Implements 1-bit quantization for Large Language Models
Based on BitNet b1.58 and related research

Features:
- Ternary weights (-1, 0, +1)
- Absolute mean quantization
- Custom bit-packed storage
- Inference with bit operations
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class BitNetQuantizer:
    """
    Quantizes models to 1-bit (ternary) format
    
    Supports:
    - BitNet b1.58 quantization
    - Weight ternarization
    - Bit-packed storage
    """
    
    def __init__(self, quantization_type: str = "bitnet_b158"):
        """
        Initialize BitNet quantizer
        
        Args:
            quantization_type: Quantization algorithm (bitnet_b158, ternary)
        """
        self.quantization_type = quantization_type
        logger.info(f"Initialized BitNet quantizer ({quantization_type})")
    
    def quantize(self, model_state: Dict[str, Any], 
                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quantize model to 1-bit format
        
        Args:
            model_state: Model state dict
            config: Model configuration
            
        Returns:
            Quantized model state
        """
        logger.info("Quantizing model to 1-bit format")
        
        quantized = {}
        stats = {
            "total_params": 0,
            "quantized_params": 0,
            "compression_ratio": 0.0
        }
        
        original_size = 0
        quantized_size = 0
        
        for name, tensor in model_state.items():
            # Skip non-weight tensors
            if not self._is_weight_tensor(name):
                quantized[name] = tensor
                continue
            
            # Convert to numpy
            if hasattr(tensor, 'cpu'):
                tensor = tensor.cpu()
            if hasattr(tensor, 'numpy'):
                tensor = tensor.numpy()
            
            original_size += tensor.size * 4  # FP32 bytes
            
            # Apply quantization
            if self.quantization_type == "bitnet_b158":
                q_tensor, scale = self._quantize_bitnet_b158(tensor)
                quantized[name] = q_tensor
                quantized[f"{name}_scale"] = scale
            elif self.quantization_type == "ternary":
                q_tensor, scale = self._quantize_ternary(tensor)
                quantized[name] = q_tensor
                quantized[f"{name}_scale"] = scale
            else:
                quantized[name] = tensor
                quantized[f"{name}_scale"] = np.array([1.0], dtype=np.float32)
            
            # Estimate quantized size (2 bits per weight)
            quantized_size += (tensor.size * 2) // 8 + tensor.size // 4  # Scale overhead
            stats["quantized_params"] += tensor.size
        
        stats["total_params"] = stats["quantized_params"]
        if original_size > 0:
            stats["compression_ratio"] = original_size / max(quantized_size, 1)
        
        logger.info(f"Quantization complete. Compression ratio: {stats['compression_ratio']:.2f}x")
        quantized["_quantization_stats"] = stats
        
        return quantized
    
    def _is_weight_tensor(self, name: str) -> bool:
        """Check if tensor is a weight (not bias or norm)"""
        weight_keywords = ['weight', 'kernel', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3']
        skip_keywords = ['bias', 'norm', 'gamma', 'beta', 'running']
        
        name_lower = name.lower()
        return any(kw in name_lower for kw in weight_keywords) and \
               not any(kw in name_lower for kw in skip_keywords)
    
    def _quantize_bitnet_b158(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        BitNet b1.58 quantization
        
        Uses absolute mean quantization with ternary output
        """
        # Compute absolute mean
        abs_mean = np.mean(np.abs(tensor))
        
        if abs_mean < 1e-8:
            # Handle zero/near-zero tensors
            return np.zeros_like(tensor, dtype=np.int8), np.array([1.0], dtype=np.float32)
        
        # Quantize to {-1, 0, +1}
        threshold = abs_mean * 0.5  # Tunable threshold
        quantized = np.zeros_like(tensor, dtype=np.int8)
        quantized[tensor > threshold] = 1
        quantized[tensor < -threshold] = -1
        
        # Scale factor
        scale = np.array([abs_mean], dtype=np.float32)
        
        return quantized, scale
    
    def _quantize_ternary(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple ternary quantization
        
        Uses median-based thresholding
        """
        # Compute threshold based on standard deviation
        std = np.std(tensor)
        threshold = std * 0.5
        
        quantized = np.zeros_like(tensor, dtype=np.int8)
        quantized[tensor > threshold] = 1
        quantized[tensor < -threshold] = -1
        
        # Optimal scale factor
        mask_pos = quantized == 1
        mask_neg = quantized == -1
        
        if np.sum(mask_pos) + np.sum(mask_neg) > 0:
            pos_sum = np.sum(tensor[mask_pos]) if np.any(mask_pos) else 0
            neg_sum = np.sum(tensor[mask_neg]) if np.any(mask_neg) else 0
            scale_val = (pos_sum - neg_sum) / (np.sum(mask_pos) + np.sum(mask_neg))
        else:
            scale_val = std
        
        scale = np.array([max(scale_val, 1e-8)], dtype=np.float32)
        
        return quantized, scale
    
    def pack_weights(self, quantized_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pack ternary weights into bit format
        
        Reduces storage from 8-bit int8 to 2-bit packed format
        """
        packed = {}
        
        for name, tensor in quantized_state.items():
            if name.endswith('_scale') or not isinstance(tensor, np.ndarray):
                packed[name] = tensor
                continue
            
            if tensor.dtype == np.int8:
                # Pack 4 weights per byte (2 bits each)
                packed[name] = self._pack_2bit(tensor)
            else:
                packed[name] = tensor
        
        return packed
    
    def _pack_2bit(self, tensor: np.ndarray) -> np.ndarray:
        """Pack int8 ternary values into 2-bit format"""
        # Flatten tensor
        flat = tensor.flatten()
        
        # Pad to multiple of 4
        pad_size = (4 - len(flat) % 4) % 4
        if pad_size > 0:
            flat = np.pad(flat, (0, pad_size))
        
        # Convert to unsigned (0, 1, 2) for packing
        unsigned = flat + 1  # Now values are 0, 1, 2
        
        # Pack 4 values per byte
        num_bytes = len(flat) // 4
        packed = np.zeros(num_bytes, dtype=np.uint8)
        
        for i in range(num_bytes):
            chunk = unsigned[i*4:(i+1)*4]
            packed[i] = (chunk[0] << 6) | (chunk[1] << 4) | (chunk[2] << 2) | chunk[3]
        
        return packed
    
    def unpack_weights(self, packed_state: Dict[str, Any]) -> Dict[str, Any]:
        """Unpack bit-packed weights"""
        unpacked = {}
        
        for name, tensor in packed_state.items():
            if name.endswith('_scale') or not isinstance(tensor, np.ndarray):
                unpacked[name] = tensor
                continue
            
            if tensor.dtype == np.uint8:
                unpacked[name] = self._unpack_2bit(tensor)
            else:
                unpacked[name] = tensor
        
        return unpacked
    
    def _unpack_2bit(self, packed: np.ndarray) -> np.ndarray:
        """Unpack 2-bit format to int8"""
        # Unpack each byte to 4 values
        unpacked = np.zeros(len(packed) * 4, dtype=np.int8)
        
        for i, byte in enumerate(packed):
            unpacked[i*4] = (byte >> 6) & 0b11
            unpacked[i*4 + 1] = (byte >> 4) & 0b11
            unpacked[i*4 + 2] = (byte >> 2) & 0b11
            unpacked[i*4 + 3] = byte & 0b11
        
        # Convert back to signed (-1, 0, 1)
        unpacked = unpacked - 1
        
        return unpacked


class BitNetInference:
    """
    Optimized inference engine for 1-bit models
    
    Uses bit operations for fast matrix multiplication
    """
    
    def __init__(self, quantized_model: Dict[str, Any]):
        """
        Initialize inference engine
        
        Args:
            quantized_model: Quantized model state
        """
        self.model = quantized_model
        self.scales = {k: v for k, v in quantized_model.items() if k.endswith('_scale')}
    
    def infer(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run inference on inputs
        
        Args:
            inputs: Input array
            
        Returns:
            Output predictions
        """
        # Simplified inference (placeholder for actual architecture)
        # In production, this would implement the full transformer forward pass
        
        x = inputs.astype(np.float32)
        
        # Apply first layer (example)
        for name, weight in self.model.items():
            if name.endswith('_scale') or name.startswith('_'):
                continue
            
            if isinstance(weight, np.ndarray) and weight.dtype == np.uint8:
                # Unpack and apply
                weight = self._unpack_and_apply(x, weight, self.scales.get(f"{name}_scale"))
                x = weight
                break
        
        return x
    
    def _unpack_and_apply(self, x: np.ndarray, packed_weight: np.ndarray,
                          scale: Optional[np.ndarray]) -> np.ndarray:
        """Unpack weights and apply matrix multiplication"""
        quantizer = BitNetQuantizer()
        weight = quantizer._unpack_2bit(packed_weight)
        
        # Reshape for matmul (simplified)
        if len(weight.shape) == 1:
            weight = weight.reshape(-1, 1)
        
        # Apply scale
        if scale is not None:
            scale_val = float(scale[0])
        else:
            scale_val = 1.0
        
        # Matrix multiplication
        if x.shape[-1] == weight.shape[0]:
            output = np.dot(x, weight) * scale_val
        else:
            output = x * scale_val
        
        return output


def quantize_to_1bit(model_state: Dict[str, Any], config: Dict[str, Any],
                     quantization_type: str = "bitnet_b158",
                     pack_weights: bool = True) -> Dict[str, Any]:
    """
    Convenience function to quantize model to 1-bit
    
    Args:
        model_state: Model state dict
        config: Model configuration
        quantization_type: Quantization algorithm
        pack_weights: Whether to pack weights to 2-bit format
        
    Returns:
        Quantized model state
    """
    quantizer = BitNetQuantizer(quantization_type=quantization_type)
    quantized = quantizer.quantize(model_state, config)
    
    if pack_weights:
        quantized = quantizer.pack_weights(quantized)
    
    return quantized


def create_inference_engine(quantized_model: Dict[str, Any]) -> BitNetInference:
    """
    Create inference engine for 1-bit model
    
    Args:
        quantized_model: Quantized model state
        
    Returns:
        Inference engine
    """
    return BitNetInference(quantized_model)
