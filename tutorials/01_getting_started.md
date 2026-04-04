# Tutorial 01: Erste Schritte mit UARF

**Level:** 🟢 Beginner | **Dauer:** 10 Minuten

In diesem Tutorial lernst du die Grundlagen von UARF und führst dein erstes Training durch.

## 📋 Voraussetzungen

- Python 3.9 oder höher
- Mindestens 4GB RAM (8GB empfohlen)
- Internetverbindung für Modell-Download

## 📦 Schritt 1: Installation

### Option A: Installation via pip (Empfohlen)

```bash
pip install uarf
```

### Option B: Installation from Source

```bash
# Repository klonen
git clone https://github.com/hcsmediacorp/hcs-UARF.git
cd hcs-UARF

# Installieren
pip install -e .
```

### Option C: Mit CUDA-Support (für NVIDIA GPUs)

```bash
# PyTorch mit CUDA installieren
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Dann UARF installieren
pip install -e .
```

## ✅ Schritt 2: Installation überprüfen

Teste ob UARF korrekt installiert wurde:

```bash
python -c "from uarf import HardwareDetector; print('UARF ready!')"
```

Wenn keine Fehlermeldung erscheint, ist alles korrekt installiert.

## 🔍 Schritt 3: Hardware erkennen lassen

Bevor du trainierst, lass UARF deine Hardware analysieren:

```python
from uarf import HardwareDetector

detector = HardwareDetector()
detector.print_summary()
```

Das zeigt dir:
- Verfügbaren RAM
- GPU-Informationen (falls vorhanden)
- Empfohlene Einstellungen
- Optimale Modellgröße

**Beispiel-Ausgabe:**
```
🔍 HARDWARE DETECTION
============================================================
Platform:        Linux
CPU Cores:       8
Total RAM:       16.0 GB
Available RAM:   12.5 GB
GPU Available:   Yes
GPU Name:        NVIDIA GeForce RTX 3060
GPU VRAM:        12.0 GB
============================================================
✅ Empfohlenes Modell: Qwen/Qwen2.5-3B
✅ Batch Size: 32
✅ Max Seq Len: 2048
```

## 🎯 Schritt 4: Erstes Training mit Auto Mode

Der einfachste Weg zu deinem ersten Modell:

```python
from uarf import auto_train

# Dein Trainingstext
text = """
Künstliche Intelligenz verändert unsere Welt täglich.
Maschinelles Lernen ermöglicht es Computern, aus Daten zu lernen.
Neuronale Netze sind inspiriert von der Struktur des menschlichen Gehirns.
Deep Learning hat viele Durchbrüche in der Bilderkennung ermöglicht.
"""

# Training starten (2 Minuten)
auto_train(text, time_seconds=120)
```

Das war's! UARF erledigt automatisch:
- Text-Vorverarbeitung
- Hardware-Optimierung
- Modellauswahl
- Training
- Checkpoint-Speicherung

## 🎮 Schritt 5: Manuelles Training

Für mehr Kontrolle über das Training:

```python
from uarf import UARFConfig, UniversalTrainer, HardwareDetector

# Hardware erkennen
detector = HardwareDetector()
hardware_config = detector.get_optimal_config()

# Konfiguration erstellen
config = UARFConfig(
    model_id="Qwen/Qwen2.5-0.5B",  # Kleines Modell für den Start
    dataset_name="karpathy/tinyshakespeare",  # Beispiel-Dataset
    time_budget_seconds=300,  # 5 Minuten Training
    batch_size=hardware_config.get('batch_size', 16),
    max_seq_len=hardware_config.get('max_seq_len', 512),
    learning_rate=2e-4,
    output_dir="./my_first_model",
)

# Training starten
trainer = UniversalTrainer(config)
trainer.train()
```

## 📊 Schritt 6: Trainings-Ergebnisse ansehen

Nach dem Training findest du die Ergebnisse im Output-Verzeichnis:

```
my_first_model/
├── checkpoint-100/
│   ├── model.pt
│   ├── optimizer.pt
│   └── training_state.pt
├── final/
│   ├── model.pt
│   └── training_state.pt
└── config.json
```

## 🧪 Schritt 7: Modell testen

So testest du dein trainiertes Modell:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Modell laden
model_path = "./my_first_model/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Text generieren
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## 🐛 Häufige Probleme & Lösungen

### Problem: "CUDA out of memory"

**Lösung:** Reduziere Batch Size und Sequenzlänge:
```python
config.batch_size = 4
config.max_seq_len = 256
config.use_gradient_checkpointing = True
```

### Problem: "Dataset not found"

**Lösung:** Verwende ein lokales Dataset oder ein verfügbares HuggingFace Dataset:
```python
# Lokales Dataset
config.dataset_name = "./mein_dataset.jsonl"

# Oder HuggingFace Dataset
config.dataset_name = "karpathy/tinyshakespeare"
```

### Problem: "ImportError"

**Lösung:** Alle Dependencies neu installieren:
```bash
pip install --upgrade torch transformers datasets accelerate tqdm psutil
pip install -e . --force-reinstall
```

## 📝 Zusammenfassung

Du hast gelernt:
1. ✅ UARF zu installieren
2. ✅ Deine Hardware zu analysieren
3. ✅ Auto Mode für einfaches Training zu nutzen
4. ✅ Manuelles Training zu konfigurieren
5. ✅ Trainings-Ergebnisse zu finden und zu testen

## ➡️ Nächste Schritte

- **Tutorial 02:** [Auto Mode vertiefen](02_auto_mode.md)
- **Tutorial 03:** [Hardware-Erkennung verstehen](03_hardware_detection.md)
- **Tutorial 04:** [Eigene Datasets erstellen](04_custom_datasets.md)

## 📚 Weitere Ressourcen

- [Offizielle Dokumentation](https://github.com/hcsmediacorp/hcs-UARF)
- [Modell-Auswahl Guide](../docs/model_selection.md)
- [FAQ](../docs/faq.md)

---

**Made with ❤️ by hcsmedia**
