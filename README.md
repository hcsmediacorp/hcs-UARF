# UARF - Universal AutoResearch Framework

**v0.2.0** - Now with GGUF, Edge Devices, LiteRT, 1-bit LLMs, WebGPU & TurboQuant!

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

## 🚀 What's New in v0.2.0

The **Universal AutoResearch Framework (UARF)** extends the original autoresearch concept with enterprise-grade deployment capabilities:

### New Export Formats (v0.2.0)
- **GGUF Export** - llama.cpp compatibility for CPU inference
- **Edge Device Support** - Jetson Nano/Xavier/Orin, Coral USB, Raspberry Pi 4/5
- **LiteRT/TFLite** - TensorFlow Lite with GPU, Edge TPU, NNAPI delegates
- **1-bit LLM (BitNet)** - Ternary weights (-1, 0, +1) with 4x compression
- **WebGPU** - Browser-based inference with WGSL shaders
- **TurboQuant** - Mixed precision inference (FP8, INT4, INT8, FP16)

### Platform Support
- **Desktop**: Windows, macOS, Linux
- **Mobile**: Android, iOS (via CoreML)
- **Cloud**: Google Colab, AWS, Azure, GCP Clusters
- **Edge**: NVIDIA Jetson, Google Coral, Raspberry Pi
- **Browser**: WebGPU-enabled browsers (Chrome, Edge, Firefox)

---

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick Start

### Basic Setup (Original Autoresearch)

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

### UARF Framework Usage

```python
from uarf import UniversalTrainer, HardwareDetector, ModelSelector, UARFConfig
from uarf.exports import UniversalExporter

# Auto-detect hardware
detector = HardwareDetector()
hardware_info = detector.detect()
print(f"Detected: {hardware_info['platform']} - {hardware_info['device_name']}")

# Select optimal model
selector = ModelSelector()
model_config = selector.select_optimal(hardware_info)

# Configure training
config = UARFConfig.from_dict({
    "depth": model_config["depth"],
    "width": model_config["width"],
    "max_seq_len": model_config["max_seq_len"],
    "device_batch_size": model_config["batch_size"],
})

# Train model
trainer = UniversalTrainer(config)
model_state = trainer.train()

# Export to various formats
exporter = UniversalExporter()

# GGUF for llama.cpp
exporter.export(model_state, config.to_dict(), "model.gguf", format="gguf")

# TFLite for mobile
exporter.export(model_state, config.to_dict(), "model.tflite", format="tflite")

# Edge device optimization
optimized_model, deploy_config = exporter.export(
    model_state, config.to_dict(), "edge_output/", format="edge"
)

# 1-bit LLM with BitNet
bitnet_model = exporter.export(
    model_state, config.to_dict(), "bitnet_model/", format="bitnet"
)

# WebGPU for browser
webgpu_files = exporter.export(
    model_state, config.to_dict(), "webgpu_export/", format="webgpu"
)

# TurboQuant for high-performance inference
inference_engine = exporter.export(
    model_state, config.to_dict(), "turboquant/", format="turboquant"
)
```

### CLI Interface

```bash
# Show UARF version
uarf --version

# Detect hardware
uarf detect

# Get recommended model config
uarf recommend

# Train with auto-config
uarf train --auto

# Export model
uarf export model.gguf --format gguf
uarf export model.tflite --format tflite --delegate gpu
uarf export output/ --format webgpu
```

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project Structure

### Core Files (Original Autoresearch)

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

### UARF Framework Structure

```
uarf/
├── __init__.py           # Main package exports
├── core/
│   ├── hardware_detector.py   # Auto-detect GPU, CPU, TPU, NPU
│   ├── model_selector.py      # Recommend optimal model config
│   ├── config.py              # UARF configuration system
│   └── trainer.py             # Universal training engine
├── exports/
│   ├── __init__.py            # UniversalExporter interface
│   ├── gguf/                  # llama.cpp GGUF export
│   ├── edge/                  # Edge device optimization
│   ├── litert/                # TensorFlow Lite / LiteRT
│   ├── bitnet/                # 1-bit LLM (BitNet)
│   ├── webgpu/                # WebGPU browser inference
│   └── turboquant/            # Mixed precision inference
├── adapters/
│   ├── android/               # Android deployment
│   ├── windows/               # Windows-specific optimizations
│   └── ...                    # Platform-specific adapters
├── platforms/
│   ├── android/               # Android platform support
│   ├── colab/                 # Google Colab integration
│   ├── cluster/               # Multi-GPU cluster support
│   └── windows/               # Windows platform support
├── models/                    # Pre-trained model zoo
└── cli/
    └── uarf_cli.py            # Command-line interface
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform Support

### UARF Native Support (v0.2.0)

UARF now provides **native cross-platform support** without requiring forks:

| Platform | Status | Features |
|----------|--------|----------|
| **NVIDIA GPU** | ✅ Full | CUDA, Flash Attention 3, Multi-GPU |
| **AMD GPU** | ✅ Full | ROCm, HIP support |
| **Apple Silicon** | ✅ Full | MPS, Metal acceleration |
| **CPU (x86/ARM)** | ✅ Full | AVX2, NEON optimizations |
| **Google TPU** | ✅ Beta | JAX backend via Colab |
| **Android** | ✅ Full | NNAPI, Vulkan GPU |
| **iOS** | ✅ Beta | CoreML export |
| **Web Browser** | ✅ New | WebGPU (Chrome/Edge/Firefox) |
| **Edge Devices** | ✅ New | Jetson, Coral, Raspberry Pi |

### Export Formats

| Format | Use Case | Quantization | Size Reduction |
|--------|----------|--------------|----------------|
| **GGUF** | llama.cpp CPU inference | F32, F16, Q8_0, Q4_K_M | 2-8x |
| **TFLite/LiteRT** | Mobile/Edge deployment | FP16, INT8, UINT8 | 2-4x |
| **BitNet** | Ultra-low power 1-bit LLMs | 1.58-bit ternary | 4x |
| **WebGPU** | Browser-based inference | FP16, FP32 | 2x |
| **TurboQuant** | High-performance serving | FP8, INT4, INT8 | 2-8x |
| **Edge Optimized** | Jetson/Coral/Pi | INT8, Pruned | 4-10x |

### Original Autoresearch Tuning Guide

If you're running the original autoresearch on smaller compute platforms than an H100, here are recommendations for tuning the defaults:

1. **Dataset**: Use a dataset with less entropy, e.g. [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). GPT-4 generated short stories work well with smaller models.
2. **Vocab Size**: Decrease `vocab_size` from 8192 down to 4096, 2048, 1024, or even byte-level (256).
3. **Sequence Length**: Lower `MAX_SEQ_LEN` in `prepare.py` to 256-512 depending on your hardware. Compensate by increasing `DEVICE_BATCH_SIZE`.
4. **Evaluation**: Decrease `EVAL_TOKENS` in `prepare.py` for faster validation.
5. **Model Depth**: Reduce `DEPTH` in `train.py` from 8 to 4 or lower.
6. **Attention Pattern**: Use `WINDOW_PATTERN = "L"` instead of "SSSL" for efficiency.
7. **Batch Size**: Lower `TOTAL_BATCH_SIZE` to `2**14` (~16K) or lower, keeping powers of 2.

**Pro Tip**: Use `uarf detect` and `uarf recommend` to automatically get optimal configurations for your hardware!

## Notable Forks (Original Autoresearch)

These forks extend the original autoresearch for specific platforms:

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS MLX)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows RTX)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD ROCm)

**Note**: With UARF v0.2.0, most platform-specific forks are no longer needed! Use the main repository for cross-platform support.

---

## UARF Documentation

- **[Release Notes v0.2.0](UARF_RELEASE_v0.2.0.md)** - Detailed changelog and new features
- **[Master Plan](UARF_MASTERPLAN.md)** - Long-term roadmap and vision
- **[Status Report](UARF_STATUS_REPORT.md)** - Current implementation status

---

## License

MIT

---

**Built with ❤️ by the UARF Team** | Original concept by [@karpathy](https://github.com/karpathy)
