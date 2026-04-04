# 📘 UARF Tutorials

Willkommen zu den UARF Tutorials! Hier lernst du Schritt für Schritt, wie du das Universal AutoResearch Framework optimal nutzt.

## 📑 Inhaltsverzeichnis

### Einsteiger-Tutorials
1. [Erste Schritte mit UARF](01_getting_started.md) - Installation und erstes Training
2. [Auto Mode verwenden](02_auto_mode.md) - Einfach Text pasten und trainieren
3. [Hardware-Erkennung](03_hardware_detection.md) - Optimale Einstellungen finden

### Fortgeschrittene Tutorials
4. [Custom Datasets erstellen](04_custom_datasets.md) - Eigene Daten vorbereiten
5. [Manuelles Training konfigurieren](05_manual_training.md) - Volle Kontrolle über Parameter
6. [Checkpointing und Resume](06_checkpointing.md) - Training fortsetzen

### Experten-Tutorials
7. [Modelle exportieren](07_export_models.md) - GGUF, Edge-Devices und mehr
8. [Platform-spezifische Optimierung](08_platform_optimization.md) - Android, Colab, Cluster
9. [Swap Management](09_swap_management.md) - Training mit wenig RAM

### Projekt-Tutorials
10. [Eigene Geschichten trainieren](10_train_stories.md) - Kreative Schreibassistenz
11. [Code-Dokumentation fine-tunen](11_code_docs.md) - Code-verstehende Modelle
12. [Mehrsprachige Modelle](12_multilingual.md) - Training auf mehreren Sprachen

---

## 🎯 Tutorial-Pfade

### Für absolute Anfänger
```
01 → 02 → 03 → 10
```

### Für Entwickler
```
01 → 04 → 05 → 07 → 11
```

### Für Data Scientists
```
03 → 05 → 06 → 07 → 09
```

### Für Mobile-Entwickler
```
01 → 02 → 07 → 08
```

---

## 📖 Alle Tutorials im Überblick

| Nr. | Titel | Level | Dauer |
|-----|-------|-------|-------|
| 01 | Erste Schritte | 🟢 Beginner | 10 min |
| 02 | Auto Mode | 🟢 Beginner | 5 min |
| 03 | Hardware Detection | 🟢 Beginner | 5 min |
| 04 | Custom Datasets | 🟡 Intermediate | 15 min |
| 05 | Manual Training | 🟡 Intermediate | 20 min |
| 06 | Checkpointing | 🟡 Intermediate | 10 min |
| 07 | Export Models | 🔴 Advanced | 25 min |
| 08 | Platform Optimization | 🔴 Advanced | 30 min |
| 09 | Swap Management | 🟡 Intermediate | 10 min |
| 10 | Train Stories | 🟢 Beginner | 15 min |
| 11 | Code Documentation | 🟡 Intermediate | 20 min |
| 12 | Multilingual | 🔴 Advanced | 30 min |

---

## 🚀 Schnellstart

Du willst sofort loslegen? Starte hier:

```bash
# Installation
pip install uarf

# Erstes Training (Auto Mode)
python -c "
from uarf import auto_train
text = 'Dein Text hier zum Trainieren...'
auto_train(text, time_seconds=120)
"
```

Oder folge dem [ersten Tutorial](01_getting_started.md).

---

## 💡 Tipps

- **Wenig RAM?** Lies zuerst Tutorial 09 (Swap Management)
- **Android/Termux?** Tutorial 08 hat platform-spezifische Tipps
- **Production Use?** Tutorial 06 (Checkpointing) ist essentiell
- **Modell teilen?** Tutorial 07 zeigt alle Export-Optionen

---

## 🤝 Feedback

Hast du Verbesserungsvorschläge oder vermisst du ein Tutorial? 
Öffne gerne ein Issue auf GitHub oder schreibe uns in den Discussions.

**Made with ❤️ by hcsmedia**
