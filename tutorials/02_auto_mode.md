# Tutorial 02: Auto Mode verwenden

**Level:** 🟢 Beginner | **Dauer:** 5 Minuten

Der Auto Mode ist der einfachste Weg, um mit UARF ein Modell zu trainieren. Du musst nur Text eingeben - alles andere erledigt UARF automatisch.

## 🎯 Was ist Auto Mode?

Auto Mode ist eine vollautomatische Pipeline, die:
- Deinen Text in Segmente aufteilt
- Die optimale Hardware-Konfiguration erkennt
- Das passende Modell auswählt
- Swap-Management bei Bedarf aktiviert
- Das Training durchführt
- Checkpoints speichert

## 🚀 Schnellstart

```python
from uarf import auto_train

text = """
Dein Text hier. Das kann alles sein:
- Eigene Geschichten
- Fachartikel
- Code-Dokumentation
- Chat-Verläufe
"""

# Training für 2 Minuten
auto_train(text, time_seconds=120)
```

Das war's! Mehr musst du nicht tun.

## 📝 Detaillierte Verwendung

### Grundlegende Optionen

```python
from uarf import auto_train

auto_train(
    text="Dein Trainingstext...",
    time_seconds=300,        # Trainingszeit in Sekunden
    output_dir="./outputs",  # Output-Verzeichnis
)
```

### Fortgeschrittene Optionen

```python
from uarf import auto_train

auto_train(
    text="Dein Trainingstext...",
    
    # Zeitbudget
    time_seconds=600,  # 10 Minuten
    
    # Modell-Auswahl (optional, sonst automatisch)
    model_id="Qwen/Qwen2.5-1.5B",
    
    # Output-Verzeichnis
    output_dir="./my_custom_outputs",
    
    # Swap Management
    swap_enabled=True,      # Swap aktivieren
    swap_size_gb=8.0,       # 8GB Swap (optional)
)
```

## 💡 Anwendungsfälle

### 1. Eigene Geschichten trainieren

```python
story = """
Es war einmal ein kleines KI-Modell, das träumte davon,
große Dinge zu lernen. Jeden Tag las es neue Texte und
verbesserte sich Schritt für Schritt. Eines Tages konnte
es plötzlich ganze Romane verstehen und selbst schreiben.
"""

auto_train(story, time_seconds=180, output_dir="./story_model")
```

### 2. Fachwissen trainieren

```python
medical_text = """
Die kardiovaskuläre Erkrankung betrifft das Herz und die Blutgefäße.
Risikofaktoren sind Bluthochdruck, hohe Cholesterinwerte und Rauchen.
Prävention umfasst gesunde Ernährung, regelmäßige Bewegung und 
Stressmanagement. Die Behandlung kann Medikamente und Lebensstiländerungen umfassen.
"""

auto_train(medical_text, time_seconds=300, output_dir="./medical_model")
```

### 3. Code-Dokumentation

```python
code_docs = """
def calculate_sum(numbers):
    '''Berechnet die Summe einer Liste von Zahlen.'''
    return sum(numbers)

def multiply(a, b):
    '''Multipliziert zwei Zahlen miteinander.'''
    return a * b

def divide(a, b):
    '''Teilt a durch b mit Fehlerbehandlung.'''
    if b == 0:
        raise ValueError("Division durch Null nicht erlaubt")
    return a / b
"""

auto_train(code_docs, time_seconds=240, output_dir="./code_model")
```

### 4. Mehrsprachiger Text

```python
multilingual = """
Hello world! This is English text for training.
Hallo Welt! Dies ist deutscher Text zum Trainieren.
Bonjour le monde! Ceci est du texte français.
¡Hola mundo! Este es texto en español.
"""

auto_train(multilingual, time_seconds=300, output_dir="./multilingual_model")
```

## 🔧 Auto Mode Klasse verwenden

Für mehr Kontrolle kannst du die `AutoMode` Klasse direkt verwenden:

```python
from uarf.auto_mode import AutoMode

# AutoMode Instanz erstellen
auto = AutoMode()

# Dataset aus Text erstellen
dataset_path = auto.create_dataset_from_text(
    text="Dein langer Text hier...",
    name="mein_dataset"
)

# Manuelles Setup mit Swap
auto.setup_swap(auto=True)  # Automatische Swap-Größe

# Training starten
trainer = auto.train(
    text="Dein Text...",
    time_seconds=300,
    output_dir="./outputs"
)
```

## 📊 Was passiert im Hintergrund?

1. **Text-Verarbeitung:**
   - Text wird in Segmente aufgeteilt (512 Wörter pro Segment)
   - 50% Überlappung zwischen Segmenten für besseren Kontext
   - JSONL-Format wird erstellt

2. **Hardware-Erkennung:**
   - RAM und VRAM werden analysiert
   - Optimale Batch Size wird berechnet
   - Passendes Modell wird ausgewählt

3. **Swap Management:**
   - Bei wenig RAM wird Swap automatisch aktiviert
   - Swap-Größe wird basierend auf verfügbarer Festplatte berechnet

4. **Training:**
   - Modell wird geladen
   - Training läuft für das angegebene Zeitbudget
   - Checkpoints werden regelmäßig gespeichert

## 🐛 Troubleshooting

### "Nicht genug Speicher"

Aktiviere Swap-Management explizit:

```python
auto_train(
    text="Dein Text...",
    swap_enabled=True,
    swap_size_gb=8.0  # 8GB Swap
)
```

### "Training zu langsam"

Reduziere die Modellgröße:

```python
auto_train(
    text="Dein Text...",
    model_id="Qwen/Qwen2.5-0.5B",  # Kleinstes Modell
    time_seconds=120
)
```

### "Zu wenig Text"

Auto Mode funktioniert auch mit wenig Text, aber für bessere Ergebnisse:
- Mindestens 500 Wörter empfohlen
- Optimal: 2000+ Wörter

## 📈 Tipps für beste Ergebnisse

1. **Qualität vor Quantität:** Lieber weniger, aber hochwertigen Text verwenden
2. **Konsistenter Stil:** Einheitlicher Schreibstil verbessert Ergebnisse
3. **Ausreichend Zeit:** Mindestens 2-3 Minuten für sinnvolles Training
4. **Swap bei wenig RAM:** Bei <8GB RAM immer Swap aktivieren

## ➡️ Nächste Schritte

- **Tutorial 03:** [Hardware-Erkennung](03_hardware_detection.md)
- **Tutorial 04:** [Custom Datasets](04_custom_datasets.md)
- **Tutorial 09:** [Swap Management](09_swap_management.md)

---

**Made with ❤️ by hcsmedia**
