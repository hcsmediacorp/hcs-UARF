# 🚀 UARF - Universal AutoResearch Framework v0.3.0

**LLM Training auf JEDEM Gerät** - Linux, Mac, Windows, Termux (Android)

## ✨ Neue Features in v0.3.0

- **📁 Lokale Datensätze**: Training mit eigenen JSON/JSONL-Daten ohne Internet
- **🤖 Synthetische Daten**: Automatische Generierung bei fehlenden Datensätzen
- **⚡ Gradient Checkpointing**: Speichereffizientes Training auch auf wenig RAM
- **🔧 Flexible Konfiguration**: Minimalanforderungen reduziert (32 seq_len, 30s budget)
- **✅ Production Ready**: Alle kritischen Fehler behoben

## 📦 Installation

```bash
# Dependencies
pip install torch transformers datasets tqdm psutil huggingface_hub

# UARF installieren
cd /workspace && pip install -e .
```

## 🎯 Schnellstart

### Mit lokalem Dataset
```python
from uarf import UARFConfig, UniversalTrainer

config = UARFConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    dataset_name="./test_data/train.json",
    batch_size=8,
    max_seq_len=128,
    max_steps=100,
    time_budget_seconds=300
)

trainer = UniversalTrainer(config)
trainer.train()
```

### Test-Datensätze erstellen
```bash
python -m uarf.data.test_dataset ./test_data
```

### CLI verwenden
```bash
uarf detect                    # Hardware erkennen
uarf run --time 600            # Training starten
uarf run --dataset ./data.json # Mit eigenem Dataset
```

## 🔧 Minimale Konfiguration (für schnelle Tests)

```python
config = UARFConfig(
    batch_size=2,
    max_seq_len=32,
    time_budget_seconds=30,
    max_steps=10,
    use_gradient_checkpointing=True
)
```

## 📊 Hardware-Empfehlungen

| RAM | Batch Size | Max Seq Len | Gradient Checkpointing |
|-----|------------|-------------|------------------------|
| <2GB | 2-4 | 32-64 | ✓ |
| 2-4GB | 4-8 | 64-128 | ✓ |
| 4-8GB | 8-16 | 128-256 | ✗ |
| 8-16GB | 16-32 | 256-512 | ✗ |
| >16GB | 32+ | 512+ | ✗ |

## ✅ Getestete Funktionalität

- ✓ Hardware Detection (CPU, RAM, GPU)
- ✓ Model Loading (Qwen/Qwen2.5-0.5B)
- ✓ Local Dataset Loading (JSON/JSONL)
- ✓ Synthetic Dataset Generation
- ✓ Tokenization & Data Preparation
- ✓ Optimizer & Scheduler Setup
- ✓ Training Loop with Gradient Accumulation
- ✓ Checkpoint Saving/Loading
- ✓ Evaluation & Metrics Tracking

## 🧪 Tests

```bash
python -m pytest tests/ -v
python test_uarf_complete.py
```

## 📄 Lizenz

MIT License - Frei für Forschung und Bildung

---

**Version**: 0.3.0 (Stable Release) | **Status**: Production Ready ✅
