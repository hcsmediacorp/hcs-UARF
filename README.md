# 🚀 UARF v2.0 - Universal AutoResearch Framework

**One Framework. Every Device. From 256MB to Multi-GPU Clusters.**

Made with ❤️ by [hcsmedia](https://github.com/hcsmedia)

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Device Modes](#device-modes)
4. [RAM & Offload Behavior](#ram--offload-behavior)
5. [Tiny Model Support](#tiny-model-support)
6. [Large Device Scaling](#large-device-scaling)
7. [Quick Start](#quick-start)
8. [Installation](#installation)
9. [Configuration](#configuration)
10. [Qwen Chat Control](#qwen-chat-control)
11. [Debugging](#debugging)
12. [Limits & Troubleshooting](#limits--troubleshooting)
13. [Roadmap](#roadmap)

---

## 🌟 Overview

UARF (Universal AutoResearch Framework) is a **unified LLM training framework** that runs on any device without code changes:

- **Android/Termux** (256MB-2GB RAM)
- **Raspberry Pi** (512MB-4GB RAM)
- **Laptops** (4-16GB RAM)
- **Workstations** (16-64GB RAM + GPU)
- **Cloud/Colab** (Any configuration)
- **Server Clusters** (Multi-GPU distributed)

### Key Features

✅ **Unified Architecture** - Single codebase, adaptive runtime  
✅ **Lazy Loading** - Starts in <100MB RAM before model load  
✅ **Auto-Fallback** - Graceful degradation on OOM  
✅ **Streaming Datasets** - Train on data larger than RAM  
✅ **Swap Management** - Automatic swap file handling  
✅ **Model Registry** - 20+ pre-configured tiny-to-large models  
✅ **Environment Config** - Configure via `UARF_*` environment variables  
✅ **Qwen Chat Ready** - Clean API for LLM-controlled training  

### What's New in v2.0

- 🔥 **Unified Configuration** - One `LiteConfig` class replaces dual config system
- 🗑️ **Removed Duplicates** - Consolidated model metadata into single registry
- ⚡ **Lazy Trainer** - Torch imports only when needed
- 🧩 **Modular Backends** - Pluggable training backends (lite/full/cluster)
- 📦 **Optional Exports** - Export formats as external plugins
- 🎯 **Memory Budgets** - Every component declares RAM cost upfront

---

## 🏗️ Architecture

UARF uses a **4-layer runtime stack** that adapts to your hardware:

```
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATION LAYER                        │
│  UARFController - workflow, tasks, recovery             │
│  File: uarf/controller.py                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              RUNTIME LAYER                              │
│  TrainingBackend - pluggable: lite | full | cluster     │
│  Files: uarf/runtime/trainer_*.py                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              CONFIGURATION LAYER                        │
│  UnifiedConfig - env vars, JSON, adaptive profiles      │
│  File: uarf/core/config_lite.py                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              DETECTION LAYER                            │
│  DeviceManager, HardwareDetector - lazy, safe           │
│  Files: uarf/core/device_manager.py                     │
└─────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Purpose | Memory Budget | Key Files |
|-------|---------|---------------|-----------|
| **Detection** | Hardware discovery, device selection | <20MB | `device_manager.py`, `hardware_detector.py` |
| **Configuration** | Settings, profiles, validation | <10MB | `config_lite.py` |
| **Runtime** | Training execution, streaming | Model-dependent | `trainer.py` |
| **Orchestration** | Workflow control, recovery | <30MB | `controller.py` |

**Total startup memory: <100MB** (before model load)

---

## 📱 Device Modes

UARF automatically detects your device class and applies optimal settings:

### Mode 1: TINY (<512MB RAM)
**Target:** Android Go, old phones, embedded devices

```python
# Auto-applied settings:
batch_size = 2
max_seq_len = 128
gradient_accumulation_steps = 8
streaming = True
model_limit = 50M params
```

**Recommended Models:**
- `hf-internal-testing/tiny-random-gpt2` (1M params, 50MB RAM)
- `HuggingFaceTB/SmolLM-135M-Instruct` (135M params, 256MB RAM)

### Mode 2: SMALL (512MB-2GB RAM)
**Target:** Raspberry Pi 4, budget Android, Termux

```python
# Auto-applied settings:
batch_size = 4-8
max_seq_len = 256-512
gradient_accumulation_steps = 4
streaming = True
model_limit = 250M params
```

**Recommended Models:**
- `HuggingFaceTB/SmolLM-360M-Instruct` (360M params, 512MB RAM)
- `Qwen/Qwen2.5-0.5B` (500M params, 1GB RAM)

### Mode 3: MEDIUM (2-8GB RAM)
**Target:** Modern laptops, desktops, Colab free tier

```python
# Auto-applied settings:
batch_size = 16-32
max_seq_len = 512-1024
gradient_accumulation_steps = 2
streaming = False
model_limit = 3B params
```

**Recommended Models:**
- `Qwen/Qwen2.5-1.5B` (1.5B params, 3GB RAM)
- `microsoft/phi-3-mini` (3.8B params, 6GB RAM)

### Mode 4: LARGE (8-16GB RAM + GPU)
**Target:** Gaming laptops, workstations, Colab Pro

```python
# Auto-applied settings:
batch_size = 32-64
max_seq_len = 1024-2048
gradient_accumulation_steps = 1
mixed_precision = bf16/fp16
model_limit = 7B params
```

**Recommended Models:**
- `Qwen/Qwen2.5-3B` (3B params, 8GB RAM)
- `meta-llama/Llama-3.2-3B` (3B params, 8GB RAM)

### Mode 5: XLARGE (16GB+ RAM + GPU)
**Target:** High-end workstations, cloud instances, clusters

```python
# Auto-applied settings:
batch_size = 64-128
max_seq_len = 2048-4096
mixed_precision = bf16
gradient_checkpointing = optional
distributed = enabled (multi-GPU)
model_limit = unlimited
```

**Recommended Models:**
- `Qwen/Qwen2.5-7B` (7B params, 16GB RAM)
- `meta-llama/Llama-3-8B` (8B params, 16GB RAM)

---

## 💾 RAM & Offload Behavior

### Memory Budget Allocation

```
Component              | Budget    | Strategy
-----------------------|-----------|----------
Python interpreter     | 20 MB     | Unavoidable
UARF core (lazy)       | 30 MB     | Deferred imports
Detection layer        | 10 MB     | psutil only
Config layer           | 5 MB      | Dataclasses
Orchestration          | 10 MB     | Thin controller
Buffer                 | 25 MB     | Safety margin
-----------------------|-----------|----------
TOTAL (before model)   | 100 MB    | 
```

### Offload Strategies

#### 1. Gradient Accumulation
Simulates large batches on small devices:
```python
# Effective batch = batch_size × accum_steps
batch_size = 4
accum_steps = 8
# Effective: 32 (same as single batch of 32)
```

#### 2. Streaming Datasets
Load data on-demand instead of loading entire dataset:
```python
config.streaming = True  # Default for <2GB RAM
# Dataset loaded in chunks during training
```

#### 3. Swap Management
Automatic swap file creation for low-RAM devices:
```python
from uarf import SwapManager

swap = SwapManager(auto_mode=True)
swap.setup_auto_swap()  # Creates swap if RAM < 2GB
```

#### 4. CPU Offload (Advanced)
Move optimizer states to CPU:
```python
config.offload_optimizer = True  # Slower but enables larger models
```

### OOM Recovery

UARF automatically recovers from out-of-memory errors:

```python
from uarf import UARFController

controller = UARFController()

try:
    result = controller.train(text, model="Qwen2.5-1.5B")
except OutOfMemoryError as e:
    # Auto-suggest smaller model
    suggestion = controller.get_fallback_model()
    print(f"Switching to: {suggestion}")
    result = controller.retry(model=suggestion)
```

**Fallback Chain** (largest → smallest):
1. Current model (retry with smaller batch)
2. SmolLM-360M (360M params)
3. SmolLM-135M (135M params)
4. Tiny GPT-2 (1M params) - ultimate fallback

---

## 🐜 Tiny Model Support

UARF specializes in **tiny models** (<100M params) for ultra-low-RAM devices:

### Built-in Tiny Models

| Model ID | Params | RAM Required | Use Case |
|----------|--------|--------------|----------|
| `hf-internal-testing/tiny-random-gpt2` | 1M | 50MB | Testing, debugging |
| `hf-internal-testing/tiny-random-llama` | 1M | 64MB | Testing, debugging |
| `onnx/tinybert` | 14M | 100MB | Classification |
| `HuggingFaceTB/SmolLM-135M-Instruct` | 135M | 256MB | Instruction following |
| `HuggingFaceTB/SmolLM-360M-Instruct` | 360M | 512MB | Small-scale generation |

### List Tiny Models

```python
from uarf import list_tiny_models

models = list_tiny_models()
for m in models:
    print(f"{m.model_id}: {m.params_millions}M params, {m.min_ram_mb}MB RAM")
```

### Add Custom Tiny Model

```python
from uarf import ModelRegistry, ModelEntry

registry = ModelRegistry()
registry.add_model(ModelEntry(
    model_id="my/custom-tiny-model",
    name="My Tiny Model",
    params_millions=50,
    size_mb=100,
    min_ram_mb=150,
    tags=["custom", "tiny"],
    description="My custom tiny model"
))
```

---

## ☁️ Large Device Scaling

UARF scales up seamlessly on powerful hardware:

### Single GPU (8-24GB VRAM)

```python
from uarf import LiteConfig, UARFController

config = LiteConfig(
    model_id="Qwen/Qwen2.5-7B",
    batch_size=32,
    max_seq_len=2048,
    device="cuda",
)

controller = UARFController(config=config)
controller.train(dataset="my_data.jsonl", time_budget_seconds=3600)
```

### Multi-GPU (Distributed Training)

```python
# Launch with torchrun
torchrun --nproc_per_node=4 \
  -m uarf.cli.uarf_cli run \
  --model Qwen/Qwen2.5-7B \
  --dataset large_dataset.jsonl \
  --distributed
```

### Cloud Deployment (Colab, AWS, GCP)

```bash
# Environment-driven configuration
export UARF_MODEL=Qwen/Qwen2.5-3B
export UARF_BATCH_SIZE=64
export UARF_MAX_SEQ_LEN=2048
export UARF_DEVICE=cuda
export UARF_OUTPUT=/content/drive/MyDrive/outputs

python -c "from uarf import quick_start; quick_start().train('text...')"
```

### Cluster Mode (Future)

```yaml
# cluster_config.yaml
nodes:
  - hostname: gpu-node-1
    gpus: 4
  - hostname: gpu-node-2
    gpus: 4

backend: deepspeed
strategy: fsdp
```

---

## 🚀 Quick Start

### Method 1: Simplest (Qwen Chat Style)

```python
from uarf import train

result = train(
    text="Your training text here...",
    time_minutes=5,
    device="auto"
)
print(f"Model saved to: {result.output_path}")
```

### Method 2: Controller API

```python
from uarf import UARFController

controller = UARFController()

# Detect hardware
hw = controller.detect_hardware()
print(f"Device: {hw.data['device_type']} ({hw.data['available_memory_mb']:.0f}MB)")

# Select model
model = controller.select_model()
print(f"Recommended: {model.data['model']}")

# Train
result = controller.train(text="Your text...", time_budget_seconds=300)
```

### Method 3: CLI

```bash
# Auto mode (simplest)
echo "Your training text..." > input.txt
uarf auto --file input.txt --time 300

# Manual mode
uarf run --model Qwen/Qwen2.5-0.5B --dataset data.jsonl --time 600

# Hardware detection
uarf detect

# Model suggestions
uarf suggest
```

### Method 4: Environment Variables

```bash
export UARF_PROFILE=tiny
export UARF_MODEL=SmolLM-135M-Instruct
export UARF_DEBUG=true
export UARF_OUTPUT=./my_outputs

python -c "from uarf import quick_start; quick_start().train('text...')"
```

---

## 📦 Installation

### Standard Installation

```bash
# Clone repository
git clone https://github.com/hcsmedia/uarf.git
cd uarf

# Install
pip install -e .

# Verify
python -c "from uarf import detect; print(detect())"
```

### Requirements

**Minimum:**
- Python 3.8+
- 256MB RAM
- 1GB disk space

**Recommended:**
- Python 3.10+
- 2GB+ RAM
- 10GB+ disk space
- GPU (CUDA 11.8+ or MPS)

### Platform-Specific

#### Linux (Desktop/Server)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

#### Windows (WSL2 recommended)
```bash
# WSL2 (Ubuntu)
wsl --install
# Then follow Linux instructions

# Native Windows
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

#### macOS (Intel/Apple Silicon)
```bash
pip install torch torchvision torchaudio
pip install -e .
# MPS automatically used on Apple Silicon
```

#### Android (Termux)
```bash
# Install dependencies
pkg install python rust cmake clang

# Install PyTorch (CPU only)
pip install numpy psutil

# Install UARF
pip install -e .

# Recommended settings for Termux
export UARF_PROFILE=tiny
export UARF_DEVICE=cpu
```

#### Google Colab
```python
# In Colab notebook
!pip install -e .

from uarf import quick_start
controller = quick_start()
controller.print_status()
```

---

## ⚙️ Configuration

### Environment Variables

All settings can be controlled via `UARF_*` environment variables:

```bash
# Model selection
export UARF_MODEL=Qwen/Qwen2.5-0.5B
export UARF_MAX_PARAMS_M=500

# Hardware
export UARF_DEVICE=auto  # auto, cpu, cuda, mps
export UARF_MAX_RAM_MB=1024

# Training
export UARF_BATCH_SIZE=8
export UARF_MAX_SEQ_LEN=512
export UARF_LR=1e-4
export UARF_MAX_STEPS=100
export UARF_TIME_BUDGET=300

# Data
export UARF_DATASET=./data.jsonl
export UARF_STREAMING=true

# Output
export UARF_OUTPUT=./outputs

# Debug
export UARF_DEBUG=true
export UARF_LOG_LEVEL=DEBUG
```

### Programmatic Configuration

```python
from uarf import LiteConfig, load_config

# From environment
config = LiteConfig.from_env()

# From JSON file
config = LiteConfig.from_json("config.json")

# With overrides
config = load_config(
    config_file="base.json",
    model_id="custom-model",
    batch_size=16
)

# Quick config
from uarf import quick_config
config = quick_config(model="tiny-model", ram_mb=512, debug=True)
```

### Configuration Profiles

UARF includes pre-built profiles for common scenarios:

```python
config.apply_low_ram_profile(ram_mb=256)   # Ultra-low RAM
config.apply_low_ram_profile(ram_mb=512)   # Low RAM
config.apply_low_ram_profile(ram_mb=1024)  # Entry-level
config.apply_low_ram_profile(ram_mb=4096)  # Mid-range
```

### Validation

```python
errors = config.validate()
if errors:
    for error in errors:
        print(f"Config error: {error}")
```

---

## 🤖 Qwen Chat Control

UARF is designed for **LLM-controlled training**. Example prompts:

### Basic Training Request

```
User: Train a model on this text: "[your text]"
Assistant: I'll use UARF to train a model for you.

```python
from uarf import train
result = train(text="[your text]", time_minutes=5)
print(f"Training complete! Model saved to: {result.output_path}")
```
```

### Hardware-Aware Training

```
User: I have 1GB RAM. Train a model on my data.
Assistant: Based on your 1GB RAM, I recommend SmolLM-135M.

```python
from uarf import UARFController
controller = UARFController(ram_mb=1024)
controller.select_model()  # Auto-selects fitting model
controller.train(text=data, time_budget_seconds=300)
```
```

### Recovery Handling

```
User: The training failed with OOM.
Assistant: Let me switch to a smaller model and retry.

```python
from uarf import UARFController
controller = UARFController()

try:
    controller.train(text=data, model="Qwen2.5-1.5B")
except OutOfMemoryError:
    fallback = controller.get_fallback_model()
    controller.update_config(model_id=fallback)
    controller.retry()
    print(f"Retried with {fallback} successfully!")
```
```

### Batch Operations

```
User: Train models on these 3 datasets with different sizes.
Assistant: I'll process them sequentially with appropriate models.

```python
from uarf import UARFController
controller = UARFController()

datasets = ["small.txt", "medium.txt", "large.txt"]
for dataset in datasets:
    with open(dataset) as f:
        text = f.read()
    
    # Auto-select model based on dataset size
    controller.update_config(max_params_millions=len(text) // 100)
    controller.select_model()
    controller.train(text=text, time_budget_seconds=300)
```
```

---

## 🐛 Debugging

### Enable Debug Mode

```bash
export UARF_DEBUG=true
export UARF_LOG_LEVEL=DEBUG
export UARF_VERBOSE_ERRORS=true
```

### Debug in Code

```python
from uarf import UARFController

controller = UARFController(debug=True)
controller.print_status()  # Full system status

# Check memory
mem = controller.check_memory()
print(f"Memory usage: {mem.data['current_usage_mb']:.1f}MB")

# Show config
config_result = controller.show_config()
print(config_result.data)
```

### Common Issues

#### Issue: Import Error (torch not found)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: CUDA Out of Memory
```bash
export UARF_BATCH_SIZE=4
export UARF_GRAD_ACCUM=8
export UARF_MAX_SEQ_LEN=256
```

#### Issue: Model Download Fails
```bash
export HF_HOME=/path/to/cache
export UARF_MODEL_SOURCE=local
```

#### Issue: Slow Training on CPU
```bash
export UARF_COMPILE=false  # Disable torch.compile on CPU
export OMP_NUM_THREADS=4   # Limit CPU threads
```

### Logging

UARF logs to console by default. To log to file:

```python
from uarf.logging import setup_logger

logger = setup_logger(
    debug=True,
    log_file="./uarf.log",
    verbose=True
)
```

---

## ⚠️ Limits & Troubleshooting

### Hard Limits

| Component | Minimum | Maximum | Notes |
|-----------|---------|---------|-------|
| RAM | 256MB | Unlimited | <512MB requires streaming |
| Model Size | 1K params | Unlimited | Limited by RAM |
| Dataset Size | 1 sample | Unlimited | Streaming required for >RAM |
| Batch Size | 1 | 1024+ | Limited by VRAM |
| Sequence Length | 32 | 32768 | Limited by model & VRAM |
| Time Budget | 10s | Unlimited | Checkpointing recommended |

### Known Limitations

1. **No iOS Support**: Termux/iSH workaround required
2. **Limited MPS Features**: Apple Silicon has restricted memory introspection
3. **Single-Node Only**: Distributed training is experimental
4. **No Quantization Training**: QLoRA support planned for v2.1

### Troubleshooting Guide

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Immediate crash on import | Heavy imports at startup | Update to v2.0 (lazy loading) |
| OOM after 10 steps | Batch size too large | Reduce batch_size, increase grad_accum |
| Slow training | CPU-only, no optimizations | Enable streaming, reduce seq_len |
| Model won't download | Network/HF cache issue | Set HF_HOME, check connectivity |
| Checkpoint fails | Disk full | Check output directory space |
| Gradient explosion | Learning rate too high | Reduce LR, add gradient clipping |

### Getting Help

1. **Check Logs**: Enable `UARF_DEBUG=true`
2. **Review Docs**: See tutorials in `/tutorials/`
3. **GitHub Issues**: Report bugs at github.com/hcsmedia/uarf
4. **Discussions**: Ask questions in GitHub Discussions

---

## 🗺️ Roadmap

### v2.0 (Current) - Foundation
- ✅ Unified configuration system
- ✅ Lazy loading architecture
- ✅ Removed duplicate code
- ✅ Modular trainer backends
- ✅ Qwen Chat API

### v2.1 (Q2 2026) - Advanced Features
- [ ] QLoRA fine-tuning support
- [ ] Multi-GPU distributed training (FSDP/DeepSpeed)
- [ ] Export plugin system (GGUF, ONNX, TFLite)
- [ ] Web UI dashboard
- [ ] Performance profiling tools

### v2.2 (Q3 2026) - Cloud Integration
- [ ] Kubernetes operator
- [ ] Serverless deployment (AWS Lambda, Cloud Functions)
- [ ] Remote model providers (Ollama, vLLM integration)
- [ ] Cluster management UI

### v2.3 (Q4 2026) - Enterprise
- [ ] Multi-tenant support
- [ ] Audit logging
- [ ] Role-based access control
- [ ] SLA monitoring
- [ ] Commercial support options

---

## 📁 Project Structure

```
uarf/
├── core/                    # Core logic
│   ├── config_lite.py       # Unified configuration
│   ├── device_manager.py    # Device selection
│   ├── hardware_detector.py # Hardware detection
│   ├── trainer.py           # Training engine (lazy)
│   ├── swap_manager.py      # Swap file management
│   └── checkpoint.py        # Checkpoint handling
├── models/                  # Model management
│   ├── registry.py          # Model catalog
│   └── providers.py         # Remote provider stubs
├── runtime/                 # Training backends
│   ├── trainer_lite.py      # <2GB RAM backend
│   ├── trainer_full.py      # >4GB RAM backend
│   └── trainer_cluster.py   # Multi-GPU backend
├── orchestrator/            # Workflow control
│   ├── controller.py        # Main controller
│   ├── workflows.py         # Pre-built workflows
│   └── chat_api.py          # LLM control API
├── detection/               # Hardware detection
│   ├── device.py            # Device manager
│   ├── hardware.py          # Hardware detector
│   └── memory.py            # Memory profiler
├── data/                    # Data loading
│   ├── local_loader.py      # Local dataset loader
│   └── streaming.py         # Streaming datasets
├── logging/                 # Logging utilities
│   └── debug_logger.py      # Debug logger
├── utils/                   # Utilities
│   └── exceptions.py        # Custom exceptions
├── cli/                     # Command-line interface
│   └── uarf_cli.py          # CLI entry point
├── controller.py            # Orchestration layer
└── __init__.py              # Package exports
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/uarf.git
cd uarf

# Create branch
git checkout -b feature/my-feature

# Make changes, run tests
python -m pytest tests/

# Commit and push
git commit -m "Add my feature"
git push origin feature/my-feature
```

---

## 📄 License

Apache 2.0 License

---

## 🙏 Acknowledgments

- **HuggingFace** for Transformers and Datasets
- **Meta** for Llama models
- **Alibaba** for Qwen models
- **Microsoft** for Phi models
- **Google** for Gemma models
- **All Open-Source Contributors**

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/hcsmedia/uarf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hcsmedia/uarf/discussions)
- **Documentation**: [UARF Docs](https://uarf.dev/docs)

---

**Made with ❤️ by hcsmedia**

[GitHub](https://github.com/hcsmedia/uarf) | [Examples](https://uarf.dev/examples) | [Tutorials](https://uarf.dev/tutorials)
