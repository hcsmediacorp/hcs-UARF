"""
UARF LiteConfig - Lightweight configuration with env var support
Optimized for low-RAM environments (<2GB) and cloud deployment.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class LiteConfig:
    """
    Lightweight configuration for UARF orchestration.
    
    Supports:
    - Environment variables (UARF_*)
    - JSON/YAML config files
    - Programmatic overrides
    - Sensible low-RAM defaults
    """
    
    # === MODEL SELECTION ===
    model_id: str = field(default_factory=lambda: os.getenv('UARF_MODEL', 'hf-internal-testing/tiny-random-gpt2'))
    model_source: str = field(default_factory=lambda: os.getenv('UARF_MODEL_SOURCE', 'huggingface'))  # huggingface, local, remote
    max_params_millions: int = field(default_factory=lambda: int(os.getenv('UARF_MAX_PARAMS_M', '100')))
    
    # === HARDWARE & MEMORY ===
    device: str = field(default_factory=lambda: os.getenv('UARF_DEVICE', 'auto'))  # auto, cpu, cuda, mps
    max_ram_mb: int = field(default_factory=lambda: int(os.getenv('UARF_MAX_RAM_MB', '512')))
    batch_size: int = field(default_factory=lambda: int(os.getenv('UARF_BATCH_SIZE', '4')))
    max_seq_len: int = field(default_factory=lambda: int(os.getenv('UARF_MAX_SEQ_LEN', '256')))
    gradient_accumulation_steps: int = field(default_factory=lambda: int(os.getenv('UARF_GRAD_ACCUM', '4')))
    
    # === TRAINING ===
    learning_rate: float = field(default_factory=lambda: float(os.getenv('UARF_LR', '1e-4')))
    max_steps: int = field(default_factory=lambda: int(os.getenv('UARF_MAX_STEPS', '100')))
    time_budget_seconds: int = field(default_factory=lambda: int(os.getenv('UARF_TIME_BUDGET', '120')))
    warmup_ratio: float = field(default_factory=lambda: float(os.getenv('UARF_WARMUP', '0.1')))
    
    # === DATA ===
    dataset_path: str = field(default_factory=lambda: os.getenv('UARF_DATASET', ''))
    streaming: bool = field(default_factory=lambda: os.getenv('UARF_STREAMING', 'true').lower() == 'true')
    val_split: float = field(default_factory=lambda: float(os.getenv('UARF_VAL_SPLIT', '0.1')))
    
    # === OUTPUT ===
    output_dir: str = field(default_factory=lambda: os.getenv('UARF_OUTPUT', './outputs'))
    save_every_steps: int = field(default_factory=lambda: int(os.getenv('UARF_SAVE_EVERY', '50')))
    
    # === LOGGING & DEBUG ===
    log_level: str = field(default_factory=lambda: os.getenv('UARF_LOG_LEVEL', 'INFO'))
    debug_mode: bool = field(default_factory=lambda: os.getenv('UARF_DEBUG', 'false').lower() == 'true')
    verbose_errors: bool = field(default_factory=lambda: os.getenv('UARF_VERBOSE_ERRORS', 'false').lower() == 'true')
    
    # === ADVANCED ===
    seed: int = field(default_factory=lambda: int(os.getenv('UARF_SEED', '42')))
    compile_model: bool = field(default_factory=lambda: os.getenv('UARF_COMPILE', 'false').lower() == 'true')  # Disabled by default for low RAM
    use_gradient_checkpointing: bool = field(default_factory=lambda: os.getenv('UARF_GRAD_CKPT', 'true').lower() == 'true')
    
    @classmethod
    def from_env(cls, prefix: str = 'UARF_') -> 'LiteConfig':
        """
        Create config from environment variables.
        All UARF_* variables are automatically picked up.
        """
        # Start with defaults
        config = cls()
        
        # Override with any UARF_* env vars
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}
        
        for key, value in env_vars.items():
            attr_name = key[len(prefix):].lower()
            if hasattr(config, attr_name):
                current_val = getattr(config, attr_name)
                
                # Type conversion based on current default
                if isinstance(current_val, bool):
                    converted = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_val, int):
                    try:
                        converted = int(value)
                    except ValueError:
                        continue
                elif isinstance(current_val, float):
                    try:
                        converted = float(value)
                    except ValueError:
                        continue
                else:
                    converted = value
                
                setattr(config, attr_name, converted)
        
        return config
    
    @classmethod
    def from_json(cls, path: str) -> 'LiteConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LiteConfig':
        """Create config from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def apply_low_ram_profile(self, ram_mb: float):
        """
        Automatically adjust settings for low-RAM environments.
        Called by hardware detector.
        """
        if ram_mb < 512:
            # Extremely constrained (<512MB)
            self.batch_size = 2
            self.max_seq_len = 128
            self.gradient_accumulation_steps = 8
            self.max_params_millions = 50
            self.streaming = True
        elif ram_mb < 1024:
            # Very constrained (<1GB)
            self.batch_size = 4
            self.max_seq_len = 256
            self.gradient_accumulation_steps = 4
            self.max_params_millions = 100
            self.streaming = True
        elif ram_mb < 2048:
            # Constrained (<2GB)
            self.batch_size = 8
            self.max_seq_len = 512
            self.gradient_accumulation_steps = 2
            self.max_params_millions = 250
            self.streaming = False
        
        # Always disable torch.compile for low RAM
        if ram_mb < 4096:
            self.compile_model = False
        
        # Enable gradient checkpointing for memory savings
        self.use_gradient_checkpointing = True
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        
        if self.max_seq_len < 32:
            errors.append("max_seq_len must be >= 32")
        
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        
        if self.time_budget_seconds < 10:
            errors.append("time_budget_seconds must be >= 10")
        
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            errors.append("warmup_ratio must be between 0 and 1")
        
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if self.device not in valid_devices:
            errors.append(f"device must be one of {valid_devices}")
        
        return errors
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("UARF LITE CONFIGURATION")
        print("=" * 60)
        print(f"Model:           {self.model_id} (max {self.max_params_millions}M params)")
        print(f"Device:          {self.device}")
        print(f"Max RAM:         {self.max_ram_mb} MB")
        print(f"Batch Size:      {self.batch_size} (accum={self.gradient_accumulation_steps})")
        print(f"Seq Length:      {self.max_seq_len}")
        print(f"Learning Rate:   {self.learning_rate}")
        print(f"Max Steps:       {self.max_steps}")
        print(f"Time Budget:     {self.time_budget_seconds}s")
        print(f"Streaming:       {self.streaming}")
        print(f"Debug Mode:      {self.debug_mode}")
        print(f"Compile Model:   {self.compile_model}")
        print(f"Grad Checkpoint: {self.use_gradient_checkpointing}")
        print("=" * 60)


def load_config(
    config_file: Optional[str] = None,
    env_prefix: str = 'UARF_',
    **overrides
) -> LiteConfig:
    """
    Load configuration with priority: overrides > env vars > config file > defaults.
    
    Args:
        config_file: Optional path to JSON config file
        env_prefix: Environment variable prefix (default: UARF_)
        **overrides: Direct parameter overrides
    
    Returns:
        Merged LiteConfig instance
    """
    # Start with file config if provided
    if config_file and Path(config_file).exists():
        config = LiteConfig.from_json(config_file)
    else:
        config = LiteConfig()
    
    # Apply environment variables
    env_config = LiteConfig.from_env(env_prefix)
    for key, value in env_config.to_dict().items():
        setattr(config, key, value)
    
    # Apply direct overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# Convenience function for Qwen Chat usage
def quick_config(
    model: str = None,
    ram_mb: int = None,
    debug: bool = None,
    **kwargs
) -> LiteConfig:
    """
    Quick configuration helper for interactive use.
    
    Example:
        config = quick_config(model='tiny-model', ram_mb=512, debug=True)
    """
    config = LiteConfig.from_env()
    
    if model:
        config.model_id = model
    if ram_mb:
        config.max_ram_mb = ram_mb
        config.apply_low_ram_profile(ram_mb)
    if debug is not None:
        config.debug_mode = debug
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
