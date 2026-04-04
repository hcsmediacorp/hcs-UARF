# 🚀 UARF v0.3.0 - STABLE RELEASE

**Universal AutoResearch Framework** - LLM Training auf JEDEM Gerät

## ✅ Release Highlights v0.3.0

### Neue Features
- **Lokale Datensatz-Unterstützung**: Training mit eigenen JSON/JSONL-Daten ohne Internet
- **Synthetische Datensätze**: Automatische Generierung bei fehlenden Daten
- **Verbesserte Fehlerbehandlung**: Graceful Fallbacks für alle Komponenten
- **Gradient Checkpointing**: Speichereffizientes Training auch auf wenig RAM
- **Flexible Konfiguration**: Minimalanforderungen reduziert für schnelle Tests

### Behobene Fehler
- ✅ Dataset-Loading mit lokalen Pfaden
- ✅ Import-Pfade für alle Module korrigiert
- ✅ Validation-Grenzwerte angepasst (min. 32 seq_len, 30s time_budget)
- ✅ Tokenization mit dynamischen Text-Spalten
- ✅ Checkpoint-Manager Integration

### Getestete Funktionalität
```
✓ Hardware Detection (CPU, RAM, GPU)
✓ Model Loading (Qwen/Qwen2.5-0.5B)
✓ Local Dataset Loading (JSON/JSONL)
✓ Synthetic Dataset Generation
✓ Tokenization & Data Preparation
✓ Optimizer & Scheduler Setup
✓ Training Loop with Gradient Accumulation
✓ Checkpoint Saving/Loading
✓ Evaluation & Metrics Tracking
✓ Logging System
```

## 📦 Installation

```bash
# Dependencies installieren
pip install torch transformers datasets tqdm psutil huggingface_hub

# Oder mit uv (empfohlen)
uv pip install torch transformers datasets tqdm psutil huggingface_hub

# UARF installieren
cd /workspace
pip install -e .
```

## 🎯 Schnellstart

### Mit lokalem Dataset
```python
from uarf import UARFConfig, UniversalTrainer

config = UARFConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    dataset_name="./test_data/train.json",  # Lokaler Pfad
    batch_size=8,
    max_seq_len=128,
    max_steps=100,
    time_budget_seconds=300,
    output_dir="./outputs"
)

trainer = UniversalTrainer(config)
trainer.train()
```

### Mit synthetischem Dataset (kein Internet benötigt)
```python
config = UARFConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    dataset_name="nonexistent_dataset",  # Wird automatisch synthetisch ersetzt
    batch_size=8,
    max_seq_len=128,
    max_steps=100,
)

trainer = UniversalTrainer(config)
trainer.train()  # Verwendet automatisch synthetische Daten
```

### Test-Datensätze erstellen
```bash
# Erstellt verschiedene Test-Datensätze
python -m uarf.data.test_dataset ./my_test_data

# Verfügbare Datasets:
# - mini_dataset.json (100 Samples)
# - small_dataset.json (500 Samples)
# - medium_dataset.json (2000 Samples)
# - train.json / val.json (800/200 Split)
```

### CLI verwenden
```bash
# Hardware erkennen
uarf detect

# Training starten
uarf run --model Qwen/Qwen2.5-0.5B --dataset ./test_data/train.json --time 600

# Mit eigenem Dataset
uarf run --dataset ./my_data.json --batch-size 4 --max-seq-len 64
```

## 📊 Test-Datensätze

UARF enthält einen integrierten Test-Dataset-Generator:

```python
from uarf.data.test_dataset import create_test_datasets

# Erstellt alle Test-Datensätze
datasets = create_test_datasets("./test_data")

# Ergebnis:
# ✓ Mini-Dataset: 100 Samples
# ✓ Small-Dataset: 500 Samples
# ✓ Medium-Dataset: 2000 Samples
# ✓ JSONL-Dataset: 1000 Samples
# ✓ Train/Val Split: 800/200 Samples
```

## 🔧 Konfigurationsoptionen

### Minimale Anforderungen (für schnelle Tests)
```python
config = UARFConfig(
    batch_size=2,           # Minimum: 1
    max_seq_len=32,         # Minimum: 32
    time_budget_seconds=30, # Minimum: 30
    max_steps=10,           # Für schnellen Test
    use_gradient_checkpointing=True,  # Spart RAM
    compile_model=False,    # Schnellerer Start
)
```

### Empfohlene Einstellungen nach Hardware

| RAM | Batch Size | Max Seq Len | Gradient Checkpointing |
|-----|------------|-------------|------------------------|
| <2GB | 2-4 | 32-64 | ✓ |
| 2-4GB | 4-8 | 64-128 | ✓ |
| 4-8GB | 8-16 | 128-256 | ✗ |
| 8-16GB | 16-32 | 256-512 | ✗ |
| >16GB | 32+ | 512+ | ✗ |

## 🧪 Tests ausführen

```bash
# Alle Tests
python -m pytest tests/ -v

# Spezifische Tests
python -m pytest tests/test_core.py -v
python -m pytest tests/test_trainer.py -v

# Quick Test Script
python test_uarf_complete.py
```

## 📁 Projektstruktur

```
/workspace
├── uarf/
│   ├── core/
│   │   ├── config.py           # Konfigurationsverwaltung
│   │   ├── trainer.py          # Universal Trainer
│   │   ├── hardware_detector.py # Hardware-Erkennung
│   │   ├── checkpoint.py       # Checkpoint Manager
│   │   └── model_selector.py   # Modell-Empfehlungen
│   ├── data/
│   │   ├── test_dataset.py     # Test-Dataset Generator
│   │   └── local_loader.py     # Lokaler Dataset Loader
│   ├── cli/
│   │   └── uarf_cli.py         # Command Line Interface
│   ├── exports/                # Export-Module (GGUF, ONNX, etc.)
│   ├── platforms/              # Platform-Adapter
│   └── utils/
│       └── exceptions.py       # Custom Exceptions
├── tests/                      # Unit Tests
├── test_data/                  # Generierte Test-Datensätze
└── test_uarf_complete.py       # Kompletter Systemtest
```

## 🐛 Troubleshooting

### "Out of Memory" Fehler
```python
# Reduziere Batch Size und Sequenzlänge
config.batch_size = 2
config.max_seq_len = 32
config.use_gradient_checkpointing = True
```

### "Dataset not found"
```python
# Verwende lokalen Pfad oder synthetische Daten
config.dataset_name = "./my_data.json"  # Existierender Pfad
# ODER
config.dataset_name = "any_name"  # Erstellt synthetische Daten
```

### Langsames Training
```python
# Deaktiviere Compile für schnelleren Start
config.compile_model = False

# Verwende weniger Worker
config.num_workers = 0
```

## 📈 Metriken & Monitoring

Das Training protokolliert automatisch:
- Loss pro Step
- Validierungs-Loss
- Tokens pro Sekunde
- Speichernutzung
- Checkpoints

## 🤝 Contributing

Pull Requests willkommen für:
- Weitere Export-Formate
- Multi-GPU Support
- Experiment Tracking
- Mehr Datensatz-Loader
- Platform-spezifische Optimierungen

## 📄 Lizenz

MIT License - Frei für Forschung und Bildung

## 🙏 Credits

UARF wurde entwickelt um LLM-Training auf jeder Hardware zugänglich zu machen.

---

**Version**: 0.3.0 (Stable Release)  
**Release Date**: 2026  
**Status**: Production Ready ✅
