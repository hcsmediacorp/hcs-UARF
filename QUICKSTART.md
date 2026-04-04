# UARF Lite - Quick Start Guide

## Overview

UARF Lite is a refactored, lightweight version of UARF designed for:
- **Low-RAM environments** (<2GB, works down to 512MB)
- **Cloud deployment** with minimal overhead
- **Qwen Chat control** with clean APIs
- **Fast startup** (<100MB before model load)

## Installation

```bash
cd /workspace
pip install -e .
```

## Quick Start (Qwen Chat Friendly)

### Method 1: One-liner detection
```python
from uarf import detect, suggest

# Detect hardware
hw = detect()
print(hw)  # {'device_type': 'cpu', 'available_memory_mb': 512, ...}

# Suggest model
model = suggest()
print(model)  # {'model': 'HuggingFaceTB/SmolLM-135M-Instruct', ...}
```

### Method 2: Controller API
```python
from uarf import quick_start

# Initialize with RAM limit
ctrl = quick_start(ram_mb=512, debug=True)

# Run tasks
result = ctrl.run_task('detect')
print(result.data)

result = ctrl.run_task('select_model')
print(result.data)

result = ctrl.run_task('check_memory')
print(result.data)
```

### Method 3: Environment Variables
```bash
export UARF_MODEL=HuggingFaceTB/SmolLM-135M-Instruct
export UARF_BATCH_SIZE=4
export UARF_MAX_STEPS=100
export UARF_DEBUG=true
export UARF_RAM_MB=512

python3 -c "from uarf import UARFController; c = UARFController(); c.print_status()"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UARF_MODEL` | `hf-internal-testing/tiny-random-gpt2` | Model ID |
| `UARF_MODEL_SOURCE` | `huggingface` | Source (huggingface, local, ollama) |
| `UARF_MAX_PARAMS_M` | `100` | Max model size in millions |
| `UARF_DEVICE` | `auto` | Device (auto, cpu, cuda, mps) |
| `UARF_MAX_RAM_MB` | `512` | Max RAM to use |
| `UARF_BATCH_SIZE` | `4` | Batch size |
| `UARF_MAX_SEQ_LEN` | `256` | Sequence length |
| `UARF_GRAD_ACCUM` | `4` | Gradient accumulation steps |
| `UARF_LR` | `1e-4` | Learning rate |
| `UARF_MAX_STEPS` | `100` | Max training steps |
| `UARF_TIME_BUDGET` | `120` | Time budget in seconds |
| `UARF_STREAMING` | `true` | Enable streaming datasets |
| `UARF_DEBUG` | `false` | Debug mode |
| `UARF_LOG_LEVEL` | `INFO` | Log level |
| `UARF_VERBOSE_ERRORS` | `false` | Show full tracebacks |

### Programmatic Configuration

```python
from uarf import LiteConfig, quick_config

# Quick config
config = quick_config(
    model='HuggingFaceTB/SmolLM-135M-Instruct',
    ram_mb=512,
    debug=True,
    batch_size=8
)

# Or build from env + overrides
from uarf import load_config
config = load_config(
    config_file='config.json',  # Optional
    batch_size=4,
    max_steps=200
)
```

## Model Registry

### List Available Models
```python
from uarf import get_registry, list_tiny_models

registry = get_registry()

# All models fitting in 512MB
models = registry.list_models(min_ram=512)

# Only tiny models (<100M params)
tiny = list_tiny_models()

# Print catalog
registry.print_catalog(available_ram_mb=512)
```

### Fallback Chain
```python
# Get fallback chain for OOM recovery
chain = registry.get_fallback_chain('HuggingFaceTB/SmolLM-360M-Instruct')
# ['HuggingFaceTB/SmolLM-360M-Instruct', 'HuggingFaceTB/SmolLM-135M-Instruct', ...]
```

## Logging

```python
from uarf import setup_logger, debug, info, error, success

# Setup with options
logger = setup_logger(
    level='DEBUG',
    log_file='./outputs/uarf.log',
    debug=True,
    verbose=True
)

# Use convenience functions
debug('Debug message')
info('Info message')
success('Operation completed')
error('Error occurred', exc_info=True)
```

## Memory Management

```python
from uarf import UARFController

ctrl = UARFController(ram_mb=512)

# Check memory
result = ctrl.check_memory()
print(f"Using {result.data['current_usage_mb']:.1f}MB")

# Clear cache
ctrl.clear_memory()

# Device manager for fine control
from uarf import select_device

device_mgr = select_device()
print(device_mgr.get_memory_usage())
device_mgr.clear_cache()
```

## Task Reference

| Task | Description | Returns |
|------|-------------|---------|
| `detect` | Hardware detection | device_type, available_memory_mb, recommended_batch_size |
| `select_model` | Auto-select best model | model_id, name, params_millions |
| `list_models` | List available models | count, models[] |
| `show_config` | Show configuration | config dict |
| `update_config` | Update settings | success status |
| `check_memory` | Memory usage check | current_usage_mb, is_safe |
| `clear_memory` | Clear memory cache | cleared amount |

## Qwen Chat Workflow

### Typical Session
```python
# Step 1: Initialize
from uarf import quick_start
ctrl = quick_start(ram_mb=512, debug=True)

# Step 2: Detect hardware
hw = ctrl.run_task('detect')
print("Hardware:", hw.data['device_type'])

# Step 3: Select model
model = ctrl.run_task('select_model')
print("Model:", model.data['model'])

# Step 4: Configure
ctrl.update_config(max_steps=200, learning_rate=2e-4)

# Step 5: Verify
ctrl.check_memory()
ctrl.show_config()

# Step 6: Ready for training (Phase 2)
# ... training implementation coming next
```

### Recovery Pattern
```python
result = ctrl.run_task('select_model', max_params_millions=50)

if not result.success:
    print("Error:", result.error)
    print("Suggestion:", result.recovery_suggestion)
    
    # Try fallback
    result = ctrl.run_task('select_model', max_params_millions=10)
```

## Next Steps

1. **Phase 2**: Streaming data loader
2. **Phase 3**: Lightweight trainer backend
3. **Phase 4**: Full training pipeline integration

## Troubleshooting

### Out of Memory
```bash
# Reduce RAM target
export UARF_MAX_RAM_MB=256
export UARF_BATCH_SIZE=2
export UARF_MAX_SEQ_LEN=128

# Use smaller model
export UARF_MAX_PARAMS_M=50
```

### No CUDA Detected
```bash
# Force CPU mode
export UARF_DEVICE=cpu
```

### Debug Mode
```bash
export UARF_DEBUG=true
export UARF_LOG_LEVEL=DEBUG
export UARF_VERBOSE_ERRORS=true
```
