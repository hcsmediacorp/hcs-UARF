"""
UARF Model Registry - Lightweight model metadata and selection.
Supports tiny models (<100M params), fallback models, and remote providers.
Lazy loading to minimize memory footprint.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ModelEntry:
    """Single model entry with minimal metadata."""
    model_id: str
    name: str
    params_millions: int
    size_mb: float
    min_ram_mb: float
    tags: List[str] = field(default_factory=list)
    provider: str = "huggingface"  # huggingface, local, ollama, remote
    description: str = ""
    
    def fits_in_ram(self, available_ram_mb: float) -> bool:
        """Check if model fits in available RAM."""
        return self.min_ram_mb <= available_ram_mb
    
    def is_tiny(self) -> bool:
        """Check if model is <100M parameters."""
        return self.params_millions < 100


class ModelRegistry:
    """
    Lightweight model registry with lazy loading.
    
    Features:
    - Tiny models (<100M) for low-RAM environments
    - Fallback models for OOM scenarios
    - Remote provider stubs
    - No hardcoded large metadata blobs
    """
    
    # Minimal built-in registry of tiny models
    TINY_MODELS = [
        ModelEntry(
            model_id="hf-internal-testing/tiny-random-gpt2",
            name="Tiny GPT-2 (Test)",
            params_millions=1,  # ~1M params
            size_mb=5,
            min_ram_mb=50,
            tags=["test", "tiny", "gpt2", "debug"],
            description="Minimal GPT-2 for testing and debugging"
        ),
        ModelEntry(
            model_id="hf-internal-testing/tiny-random-llama",
            name="Tiny Llama (Test)",
            params_millions=1,
            size_mb=8,
            min_ram_mb=64,
            tags=["test", "tiny", "llama", "debug"],
            description="Minimal Llama for testing"
        ),
        ModelEntry(
            model_id="HuggingFaceTB/SmolLM-360M-Instruct",
            name="SmolLM 360M",
            params_millions=360,
            size_mb=720,
            min_ram_mb=512,
            tags=["smollm", "instruct", "small"],
            description="Smallest useful instruct model"
        ),
        ModelEntry(
            model_id="HuggingFaceTB/SmolLM-135M-Instruct",
            name="SmolLM 135M",
            params_millions=135,
            size_mb=270,
            min_ram_mb=256,
            tags=["smollm", "instruct", "tiny"],
            description="Very small instruct model"
        ),
        ModelEntry(
            model_id="onnx/tinybert",
            name="TinyBERT (ONNX)",
            params_millions=14,
            size_mb=60,
            min_ram_mb=100,
            tags=["bert", "onnx", "tiny", "classification"],
            provider="onnx",
            description="Tiny BERT for classification tasks"
        ),
    ]
    
    # Fallback models when OOM occurs (progressively smaller)
    FALLBACK_CHAIN = [
        "hf-internal-testing/tiny-random-gpt2",  # 1M - ultimate fallback
        "hf-internal-testing/tiny-random-llama",  # 1M
        "HuggingFaceTB/SmolLM-135M-Instruct",     # 135M
        "HuggingFaceTB/SmolLM-360M-Instruct",     # 360M
    ]
    
    # Remote provider templates (lazy expansion)
    REMOTE_PROVIDERS = {
        "huggingface": {
            "template": "https://huggingface.co/{model_id}",
            "load_fn": "transformers.AutoModelForCausalLM.from_pretrained"
        },
        "ollama": {
            "template": "http://localhost:11434/api/generate",
            "load_fn": "ollama.generate"
        },
        "local": {
            "template": "{model_id}",  # Direct path
            "load_fn": "transformers.AutoModelForCausalLM.from_pretrained"
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize registry.
        
        Args:
            cache_dir: Optional directory to cache model metadata
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/uarf/models")
        self._models: Dict[str, ModelEntry] = {}
        self._initialized = False
    
    def _lazy_init(self):
        """Lazy initialization to avoid loading on import."""
        if self._initialized:
            return
        
        # Load built-in tiny models
        for model in self.TINY_MODELS:
            self._models[model.model_id] = model
        
        # Try to load additional models from cache
        self._load_from_cache()
        
        self._initialized = True
    
    def _load_from_cache(self):
        """Load additional model metadata from cache file."""
        cache_file = Path(self.cache_dir) / "registry.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                for model_id, info in data.items():
                    if model_id not in self._models:
                        self._models[model_id] = ModelEntry(**info)
            except Exception:
                pass  # Ignore cache errors
    
    def save_to_cache(self):
        """Save current registry to cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = Path(self.cache_dir) / "registry.json"
        data = {k: self._entry_to_dict(v) for k, v in self._models.items()}
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _entry_to_dict(self, entry: ModelEntry) -> dict:
        """Convert ModelEntry to dictionary."""
        return {
            'model_id': entry.model_id,
            'name': entry.name,
            'params_millions': entry.params_millions,
            'size_mb': entry.size_mb,
            'min_ram_mb': entry.min_ram_mb,
            'tags': entry.tags,
            'provider': entry.provider,
            'description': entry.description
        }
    
    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model by ID."""
        self._lazy_init()
        return self._models.get(model_id)
    
    def list_models(
        self,
        max_params: Optional[int] = None,
        min_ram: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelEntry]:
        """
        List models matching criteria.
        
        Args:
            max_params: Maximum parameters in millions
            min_ram: Available RAM in MB (filters to fitting models)
            tags: Required tags (all must match)
        
        Returns:
            List of matching ModelEntry objects
        """
        self._lazy_init()
        
        results = []
        for model in self._models.values():
            # Filter by max params
            if max_params and model.params_millions > max_params:
                continue
            
            # Filter by RAM
            if min_ram and not model.fits_in_ram(min_ram):
                continue
            
            # Filter by tags
            if tags:
                if not all(tag in model.tags for tag in tags):
                    continue
            
            results.append(model)
        
        # Sort by size (smallest first)
        results.sort(key=lambda m: m.params_millions)
        return results
    
    def suggest_model(self, available_ram_mb: float, task: str = "text-generation") -> ModelEntry:
        """
        Suggest best model for available RAM.
        
        Args:
            available_ram_mb: Available RAM in MB
            task: Task type (future use)
        
        Returns:
            Best fitting ModelEntry
        """
        self._lazy_init()
        
        # Find largest model that fits
        fitting = self.list_models(min_ram=available_ram_mb)
        
        if fitting:
            # Return largest fitting model
            return fitting[-1]
        
        # Ultimate fallback
        return self._models.get(
            self.FALLBACK_CHAIN[0],
            self.TINY_MODELS[0]
        )
    
    def get_fallback_chain(self, current_model_id: str) -> List[str]:
        """
        Get fallback chain starting from current model.
        
        Returns progressively smaller models for OOM recovery.
        """
        self._lazy_init()
        
        try:
            idx = self.FALLBACK_CHAIN.index(current_model_id)
            return self.FALLBACK_CHAIN[idx:]
        except ValueError:
            # Current model not in fallback chain, return full chain
            return self.FALLBACK_CHAIN.copy()
    
    def add_model(self, entry: ModelEntry):
        """Add custom model to registry."""
        self._lazy_init()
        self._models[entry.model_id] = entry
    
    def remove_model(self, model_id: str):
        """Remove model from registry."""
        self._lazy_init()
        if model_id in self._models:
            del self._models[model_id]
    
    def get_provider_info(self, provider: str) -> Optional[dict]:
        """Get provider configuration."""
        return self.REMOTE_PROVIDERS.get(provider)
    
    def print_catalog(self, available_ram_mb: Optional[float] = None):
        """Print model catalog."""
        self._lazy_init()
        
        print("\n" + "=" * 70)
        print("UARF MODEL CATALOG")
        print("=" * 70)
        
        if available_ram_mb:
            print(f"Filtering for available RAM: {available_ram_mb:.0f} MB\n")
            models = self.list_models(min_ram=available_ram_mb)
        else:
            print("Showing all models:\n")
            models = list(self._models.values())
        
        print(f"{'Model ID':<45} {'Params':<8} {'RAM':<8} {'Tags'}")
        print("-" * 70)
        
        for model in models:
            tags_str = ", ".join(model.tags[:3])  # Show first 3 tags
            print(f"{model.model_id:<45} {model.params_millions:<8} {model.min_ram_mb:<8.0f}MB {tags_str}")
        
        print("=" * 70)
        print(f"Total: {len(models)} models")
        
        if available_ram_mb:
            fallback = self.suggest_model(available_ram_mb)
            print(f"\n💡 Recommended: {fallback.model_id} ({fallback.name})")
        
        print()


# Global registry instance (lazy initialized)
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get global registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def suggest_model(ram_mb: float) -> ModelEntry:
    """Convenience function to suggest model for RAM."""
    return get_registry().suggest_model(ram_mb)


def list_tiny_models() -> List[ModelEntry]:
    """List all tiny models (<100M params)."""
    return get_registry().list_models(max_params=100)


def create_tiny_model(vocab_size: int = 8192, d_model: int = 256, 
                      n_layers: int = 4, n_heads: int = 8, 
                      max_seq_len: int = 512):
    """
    Create a tiny transformer model for demo/testing purposes.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length
    
    Returns:
        nn.Module: Tiny transformer model
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class TinyTransformer(nn.Module):
        """Minimal transformer for testing and demo."""
        
        def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=n_heads, 
                    dim_feedforward=d_model*4, 
                    activation='gelu', 
                    batch_first=True, 
                    norm_first=True
                ) for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        def forward(self, input_ids, attention_mask=None, targets=None):
            B, T = input_ids.shape
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
            h = self.token_emb(input_ids) + self.pos_emb(positions)
            for layer in self.layers:
                h = layer(h)
            h = self.norm(h)
            logits = self.lm_head(h)
            
            if targets is not None:
                return F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1)
                )
            return logits
    
    return TinyTransformer(vocab_size, d_model, n_layers, n_heads, max_seq_len)
