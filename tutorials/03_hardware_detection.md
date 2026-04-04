# Tutorial 03: Hardware-Erkennung

**Level:** 🟢 Beginner | **Dauer:** 5 Minuten

In diesem Tutorial lernst du, wie UARF deine Hardware erkennt und automatisch optimiert.

## 🔍 Warum Hardware-Erkennung?

Nicht jedes Gerät ist gleich. UARF passt sich automatisch an:
- Verfügbaren RAM
- GPU (Typ und VRAM)
- CPU-Kerne
- Plattform (Desktop, Mobile, Cloud)

## 🚀 Grundlagen

### Hardware erkennen

```python
from uarf import HardwareDetector

detector = HardwareDetector()
detector.print_summary()
```

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
Is Mobile:       No
Is Colab:        No
Is Cluster:      No
============================================================
✅ Empfohlenes Modell: Qwen/Qwen2.5-3B
✅ Batch Size: 32
✅ Max Seq Len: 2048
```

### Optimale Konfiguration abrufen

```python
from uarf import HardwareDetector

detector = HardwareDetector()
config = detector.get_optimal_config()

print(config)
# {'batch_size': 32, 'max_seq_len': 2048, 'gradient_accumulation': 4, ...}
```

## 📊 Hardware-Spezifikationen im Detail

### Alle Specs anzeigen

```python
from uarf import HardwareDetector

detector = HardwareDetector()
specs = detector.specs

print(f"Platform: {specs.platform}")
print(f"CPU Cores: {specs.cpu_count}")
print(f"RAM Total: {specs.ram_total} GB")
print(f"RAM Available: {specs.ram_available} GB")
print(f"GPU Available: {specs.gpu_available}")
print(f"GPU Name: {specs.gpu_name}")
print(f"GPU VRAM: {specs.gpu_vram} GB")
print(f"Is Mobile: {specs.is_mobile}")
print(f"Is Colab: {specs.is_colab}")
print(f"Is Cluster: {specs.is_cluster}")
```

### Als JSON exportieren

```python
from uarf import HardwareDetector
import json

detector = HardwareDetector()
specs_dict = {
    'platform': detector.specs.platform,
    'cpu_count': detector.specs.cpu_count,
    'ram_total_gb': detector.specs.ram_total,
    'ram_available_gb': detector.specs.ram_available,
    'gpu_available': detector.specs.gpu_available,
    'gpu_name': detector.specs.gpu_name,
    'gpu_vram_gb': detector.specs.gpu_vram,
}

print(json.dumps(specs_dict, indent=2))
```

## 🎯 Modell-Empfehlungen basierend auf Hardware

### Automatische Empfehlung

```python
from uarf import HardwareDetector, ModelSelector

detector = HardwareDetector()
selector = ModelSelector(detector.specs)

# Empfehlungen anzeigen
selector.print_suggestions(task="text-generation")
```

### Manuelle Auswahl basierend auf Hardware

```python
from uarf import HardwareDetector

detector = HardwareDetector()
specs = detector.specs

if specs.gpu_vram and specs.gpu_vram >= 16:
    model_id = "Qwen/Qwen2.5-7B"
elif specs.gpu_vram and specs.gpu_vram >= 8:
    model_id = "Qwen/Qwen2.5-3B"
elif specs.ram_available >= 8:
    model_id = "Qwen/Qwen2.5-1.5B"
else:
    model_id = "Qwen/Qwen2.5-0.5B"

print(f"Empfohlenes Modell: {model_id}")
```

## 📱 Platform-spezifische Erkennung

### Android / Termux

```python
from uarf import HardwareDetector

detector = HardwareDetector()

if detector.specs.is_mobile:
    print("📱 Mobile device detected!")
    print(f"  RAM: {detector.specs.ram_available} GB")
    print("  → Verwende kleine Modelle (<1B)")
    print("  → Aktiviere Swap Management")
```

### Google Colab

```python
from uarf import HardwareDetector

detector = HardwareDetector()

if detector.specs.is_colab:
    print("☁️ Google Colab detected!")
    print(f"  GPU: {detector.specs.gpu_name}")
    print(f"  VRAM: {detector.specs.gpu_vram} GB")
    print("  → Nutze die kostenlose GPU!")
```

### Server-Cluster

```python
from uarf import HardwareDetector

detector = HardwareDetector()

if detector.specs.is_cluster:
    print("🖥️ Cluster detected!")
    print("  → Multi-GPU Training möglich")
```

## ⚙️ Optimale Einstellungen berechnen

### Batch Size

```python
from uarf import HardwareDetector

detector = HardwareDetector()
config = detector.get_optimal_config()

batch_size = config['batch_size']
print(f"Optimale Batch Size: {batch_size}")
```

### Sequenzlänge

```python
from uarf import HardwareDetector

detector = HardwareDetector()
config = detector.get_optimal_config()

max_seq_len = config['max_seq_len']
print(f"Optimale Sequenzlänge: {max_seq_len}")
```

### Gradient Accumulation

```python
from uarf import HardwareDetector

detector = HardwareDetector()
config = detector.get_optimal_config()

grad_accum = config['gradient_accumulation']
print(f"Gradient Accumulation: {grad_accum}")
```

## 🔧 Manuelles Override

Manchmal willst du die automatischen Einstellungen überschreiben:

```python
from uarf import HardwareDetector, UARFConfig

detector = HardwareDetector()
hardware_config = detector.get_optimal_config()

# Config mit Hardware-Defaults
config = UARFConfig(
    model_id="Qwen/Qwen2.5-1.5B",
    dataset_name="karpathy/tinyshakespeare",
    time_budget_seconds=300,
)

# Hardware-Defaults übernehmen
config.batch_size = hardware_config.get('batch_size', 16)
config.max_seq_len = hardware_config.get('max_seq_len', 512)

# Aber manuell überschreiben
config.batch_size = 8  # Kleiner als empfohlen für mehr Stabilität
config.learning_rate = 1e-4  # Custom Learning Rate
```

## 🐛 Troubleshooting

### Falsche Hardware-Erkennung

Wenn Hardware nicht korrekt erkannt wird:

```python
from uarf import HardwareDetector

detector = HardwareDetector()

# Force CPU-only mode
detector.force_cpu_mode = True

# Oder spezifisches Device setzen
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
```

### GPU wird nicht erkannt

```bash
# NVIDIA: Treiber überprüfen
nvidia-smi

# PyTorch CUDA Check
python -c "import torch; print(torch.cuda.is_available())"
```

### Wenig verfügbarer RAM

```python
from uarf import HardwareDetector, SwapManager, SwapConfig

detector = HardwareDetector()

if detector.specs.ram_available < 4:
    print("⚠️ Wenig RAM verfügbar!")
    
    # Swap aktivieren
    swap = SwapManager(SwapConfig(auto_mode=True, enabled=True))
    swap.setup_auto_swap()
```

## 📈 Tipps

1. **Immer zuerst Hardware checken:** Bevor du trainierst
2. **Automatische Empfehlungen nutzen:** Meistens optimal
3. **Bei Problemen reduzieren:** Batch Size und Sequenzlänge verringern
4. **Swap bei wenig RAM:** Ab <8GB RAM empfehlenswert

## ➡️ Nächste Schritte

- **Tutorial 04:** [Custom Datasets](04_custom_datasets.md)
- **Tutorial 05:** [Manual Training](05_manual_training.md)
- **Tutorial 09:** [Swap Management](09_swap_management.md)

---

**Made with ❤️ by hcsmedia**
