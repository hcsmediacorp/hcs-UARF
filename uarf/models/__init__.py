"""
UARF Models - Pre-trained Model Definitions and Loading Utilities

This module provides model architectures, loading utilities, and 
pre-configured model definitions for various tasks.

NOTE: Heavy imports (torch, transformers) are deferred to lazy loading.
Import registry functions directly from uarf.models.registry for low-RAM operation.
"""

from typing import Dict, List, Optional, Any, Type, Tuple, Union

# Test/mocking compatibility symbols
AutoTokenizer = None


def _lazy_import_torch():
    """Lazy import torch only when needed."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for model loading. "
            "Install with: pip install torch"
        )


def _lazy_import_transformers():
    """Lazy import transformers only when needed."""
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoModelForQuestionAnswering,
            AutoTokenizer as _AutoTokenizer,
            PreTrainedModel,
            PreTrainedTokenizer,
        )
        global AutoTokenizer
        AutoTokenizer = _AutoTokenizer
        return {
            'AutoModelForCausalLM': AutoModelForCausalLM,
            'AutoModelForSequenceClassification': AutoModelForSequenceClassification,
            'AutoModelForTokenClassification': AutoModelForTokenClassification,
            'AutoModelForQuestionAnswering': AutoModelForQuestionAnswering,
            'AutoTokenizer': _AutoTokenizer,
            'PreTrainedModel': PreTrainedModel,
            'PreTrainedTokenizer': PreTrainedTokenizer,
        }
    except ImportError:
        raise ImportError(
            "Transformers is required for model loading. "
            "Install with: pip install transformers"
        )


# Type aliases for torch-free usage
DeviceType = Union[str, Any]  # Will be torch.device when torch is available
DtypeType = Any  # Will be torch.dtype when torch is available
ModelType = Any  # Will be PreTrainedModel when transformers is available


from uarf.core.config import UARFConfig


class ModelRegistry:
    """
    Registry for all supported models in UARF.
    
    Provides centralized model management with automatic
    compatibility checking and optimized loading.
    """
    
    # Supported model families
    SUPPORTED_FAMILIES = {
        "qwen": {
            "prefix": "Qwen/",
            "description": "Alibaba's Qwen series - efficient multilingual models",
            "sizes": ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"],
            "license": "Apache-2.0 / Qwen License"
        },
        "phi": {
            "prefix": "microsoft/phi-",
            "description": "Microsoft's Phi series - compact high-quality models",
            "sizes": ["1", "2", "3"],
            "license": "MIT"
        },
        "llama": {
            "prefix": "meta-llama/",
            "description": "Meta's Llama series - state-of-the-art open models",
            "sizes": ["1B", "3B", "8B", "70B"],
            "license": "Llama Community License"
        },
        "gemma": {
            "prefix": "google/gemma-",
            "description": "Google's Gemma series - lightweight open models",
            "sizes": ["2B", "7B", "27B"],
            "license": "Gemma Terms of Use"
        },
        "tinyllama": {
            "prefix": "TinyLlama/",
            "description": "TinyLlama - ultra-efficient small models",
            "sizes": ["1.1B"],
            "license": "Apache-2.0"
        },
        "bert": {
            "prefix": "bert-base",
            "description": "BERT - classic encoder models",
            "sizes": ["uncased", "cased", "multilingual"],
            "license": "Apache-2.0"
        },
        "distilbert": {
            "prefix": "distilbert",
            "description": "DistilBERT - distilled BERT variants",
            "sizes": ["base", "multilingual"],
            "license": "Apache-2.0"
        }
    }
    
    # Task-specific default models
    TASK_DEFAULTS = {
        "text-generation": "Qwen/Qwen2.5-1.5B",
        "text-classification": "distilbert-base-uncased",
        "token-classification": "bert-base-cased",
        "question-answering": "bert-base-uncased",
        "fill-mask": "bert-base-uncased",
        "summarization": "google-t5/t5-small",
        "translation": "google-t5/t5-small"
    }
    
    @classmethod
    def get_model_info(cls, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        info = {
            "model_id": model_id,
            "family": None,
            "size": None,
            "description": None,
            "license": None,
            "supported_tasks": []
        }
        
        # Detect family (case-insensitive matching)
        model_id_lower = model_id.lower()
        for family, details in cls.SUPPORTED_FAMILIES.items():
            prefix_lower = details["prefix"].lower()
            if prefix_lower in model_id_lower:
                info["family"] = family
                info["description"] = details["description"]
                info["license"] = details["license"]
                
                # Detect size
                for size in details["sizes"]:
                    if size.lower() in model_id_lower:
                        info["size"] = size
                        break
                break
        
        return info
    
    @classmethod
    def list_models(cls, family: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all supported models, optionally filtered by family."""
        models = []
        
        if family:
            if family not in cls.SUPPORTED_FAMILIES:
                raise ValueError(f"Unknown family: {family}. Available: {list(cls.SUPPORTED_FAMILIES.keys())}")
            
            fam_info = cls.SUPPORTED_FAMILIES[family]
            for size in fam_info["sizes"]:
                model_id = f"{fam_info['prefix']}{size}"
                models.append({
                    "model_id": model_id,
                    "family": family,
                    "size": size,
                    "description": fam_info["description"],
                    "license": fam_info["license"]
                })
        else:
            for fam, details in cls.SUPPORTED_FAMILIES.items():
                for size in details["sizes"]:
                    model_id = f"{details['prefix']}{size}"
                    models.append({
                        "model_id": model_id,
                        "family": fam,
                        "size": size,
                        "description": details["description"],
                        "license": details["license"]
                    })
        
        return models
    
    @classmethod
    def get_default_for_task(cls, task: str) -> str:
        """Get the default model ID for a given task."""
        return cls.TASK_DEFAULTS.get(task, cls.TASK_DEFAULTS["text-generation"])


class ModelLoader:
    """
    Universal model loader with optimized loading strategies.
    
    Features:
    - Automatic device placement
    - Memory-efficient loading (meta device, quantization)
    - Mixed precision support
    - Gradient checkpointing
    - Torch compile integration
    """
    
    def __init__(self, config: UARFConfig):
        self.config = config
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
    
    def load_tokenizer(self) -> Any:
        """Load tokenizer with optimal settings."""
        global AutoTokenizer
        if AutoTokenizer is None:
            transformers = _lazy_import_transformers()
            tokenizer_cls = transformers['AutoTokenizer']
        else:
            tokenizer_cls = AutoTokenizer
        
        self.tokenizer = tokenizer_cls.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
            use_fast=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.tokenizer
    
    def load_model(self, device: Any, dtype: Any) -> Any:
        """
        Load model with memory-efficient strategy.
        
        Args:
            device: Target device for model
            dtype: Target data type
            
        Returns:
            Loaded PyTorch model
        """
        torch, _ = _lazy_import_torch()
        transformers = _lazy_import_transformers()
        AutoConfig = transformers['AutoTokenizer'].__class__.__bases__[0].__class__  # Hack to get module
        
        from transformers import AutoConfig
        
        print(f"\n📦 Loading model: {self.config.model_id}")
        
        # Determine model class based on task
        model_class = self._get_model_class()
        
        # Load configuration
        model_config = AutoConfig.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Memory-efficient loading using meta device
        with torch.device('meta'):
            self.model = model_class.from_config(model_config)
        
        # Move to actual device and initialize weights
        self.model = self.model.to_empty(device=device)
        self.model.init_weights()
        
        # Apply optimizations
        self._apply_optimizations(device, dtype)
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded: {param_count:,} parameters")
        
        return self.model
    
    def _get_model_class(self) -> Type[Any]:
        """Determine the appropriate model class for the task."""
        transformers = _lazy_import_transformers()
        
        task_mapping = {
            "text-generation": transformers['AutoModelForCausalLM'],
            "causal-lm": transformers['AutoModelForCausalLM'],
            "text-classification": transformers['AutoModelForSequenceClassification'],
            "sequence-classification": transformers['AutoModelForSequenceClassification'],
            "token-classification": transformers['AutoModelForTokenClassification'],
            "question-answering": transformers['AutoModelForQuestionAnswering'],
            "qa": transformers['AutoModelForQuestionAnswering'],
        }
        
        task = self.config.task_type or "text-generation"
        return task_mapping.get(task, transformers['AutoModelForCausalLM'])
    
    def _apply_optimizations(self, device: Any, dtype: Any):
        """Apply performance optimizations to the model."""
        
        # Gradient Checkpointing
        if self.config.use_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✅ Gradient Checkpointing enabled")
        
        # Torch Compile (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            import torch
            print("🔧 Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        # Attention implementation optimization
        if hasattr(self.model.config, '_attn_implementation'):
            if device.type == "cuda":
                # Use Flash Attention 2 if available
                try:
                    from flash_attn import flash_attn_func
                    self.model.config._attn_implementation = "flash_attention_2"
                    print("✅ Flash Attention 2 enabled")
                except ImportError:
                    # Fall back to sdpa
                    self.model.config._attn_implementation = "sdpa"
                    print("✅ SDPA attention enabled")
            elif device.type == "mps":
                self.model.config._attn_implementation = "sdpa"
                print("✅ SDPA attention enabled (MPS)")
    
    def load_from_checkpoint(self, checkpoint_path: str, device: DeviceType) -> ModelType:
        """Load model from a local checkpoint."""
        print(f"\n📥 Loading from checkpoint: {checkpoint_path}")
        
        torch, _ = _lazy_import_torch()
        transformers = _lazy_import_transformers()
        AutoTokenizer = transformers['AutoTokenizer']
        model_class = self._get_model_class()
        
        self.model = model_class.from_pretrained(
            checkpoint_path,
            trust_remote_code=self.config.trust_remote_code
        ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=self.config.trust_remote_code
        )
        
        print(f"✅ Checkpoint loaded successfully")
        return self.model


class QuantizedModelLoader(ModelLoader):
    """
    Loader for quantized models (INT8, INT4, NF4).
    
    Supports:
    - bitsandbytes quantization
    - GPTQ quantization
    - AWQ quantization
    """
    
    def __init__(self, config: UARFConfig, quantization_config: Optional[Dict] = None):
        super().__init__(config)
        self.quantization_config = quantization_config or {}
    
    def load_quantized_model(self, device: DeviceType) -> ModelType:
        """Load model with quantization applied."""
        torch, _ = _lazy_import_torch()
        transformers = _lazy_import_transformers()
        AutoTokenizer = transformers['AutoTokenizer']
        
        from transformers import BitsAndBytesConfig
        
        # Default 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.quantization_config.get("load_in_4bit", True),
            load_in_8bit=self.quantization_config.get("load_in_8bit", False),
            bnb_4bit_quant_type=self.quantization_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=self.quantization_config.get("use_double_quant", True),
        )
        
        print(f"\n📦 Loading quantized model: {self.config.model_id}")
        
        model_class = self._get_model_class()
        
        self.model = model_class.from_pretrained(
            self.config.model_id,
            quantization_config=bnb_config,
            trust_remote_code=self.config.trust_remote_code,
            device_map=device.type if device.type != "mps" else "cpu",
            torch_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code
        )
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Quantized model loaded: {param_count:,} parameters")
        
        return self.model


def create_model(config: UARFConfig, device: DeviceType, dtype: DtypeType) -> Tuple[Any, Any]:
    """
    Factory function to create a model and tokenizer.
    
    Args:
        config: UARF configuration
        device: Target device
        dtype: Target data type
        
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = ModelLoader(config)
    loader.load_tokenizer()
    model = loader.load_model(device, dtype)
    
    return model, loader.tokenizer


def create_quantized_model(config: UARFConfig, device: DeviceType, 
                           quant_config: Optional[Dict] = None) -> Tuple[Any, Any]:
    """
    Factory function to create a quantized model and tokenizer.
    
    Args:
        config: UARF configuration
        device: Target device
        quant_config: Quantization configuration dict
        
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = QuantizedModelLoader(config, quant_config)
    model = loader.load_quantized_model(device)
    
    return model, loader.tokenizer
