# Changelog

Alle wesentlichen Änderungen am UARF-Projekt werden in dieser Datei dokumentiert.

Das Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.0.0/)
und das Projekt folgt [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Geplant
- Multi-GPU Support für Cluster-Umgebungen
- W&B Integration für Experiment Tracking
- ONNX Export für Windows/Edge Devices
- Vollständige Platform-Adapter (Android, Colab, Windows)

---

## [1.0.0] - 2024-XX-XX

### ✨ Neu
- **Auto Mode**: Vollautomatische Training-Pipeline - einfach Text pasten!
- **Swap Manager**: Automatisches Swap-File Management für Geräte mit wenig RAM
- **Model Collection**: 15+ vorkonfigurierte Modelle (Qwen, Phi, Gemma, Llama)
- **Hardware Detection**: Intelligente Erkennung und Optimierung für alle Plattformen
- **CLI Interface**: Einheitlicher Befehl für alle Funktionen
- **Logging System**: Strukturiertes Logging mit JSON-Unterstützung
- **Checkpointing**: Automatisches Speichern und Resume-Funktionalität

### 📦 Export-Formate
- GGUF Export (vollständig)
- Edge Device Export (Beta)
- LiteRT/TFLite Export (Beta)
- WebGPU Export (Beta)
- BitNet 1-bit Quantization (Beta)
- TurboQuant Engine (Beta)

### 🔧 Platform-Support
- ✅ Linux (Desktop, Server, WSL)
- ✅ macOS (Intel, Apple Silicon)
- ⚠️ Windows (WSL2 vollständig, native Beta)
- ⚠️ Android/Termux (Beta)
- ⚠️ Google Colab (Beta)
- ⚠️ Server-Cluster (Basic)

### 📝 Dokumentation
- Umfassende README mit Beispielen
- Schritt-für-Schritt Tutorials (1-3 fertig, 4-12 in Arbeit)
- API-Dokumentation (in Arbeit)
- CONTRIBUTING Guide
- Project Status Document

### 🧪 Tests
- Core Module: 100% Coverage
- Trainer Module: 95% Coverage
- Model Selection: 100% Coverage
- Export Modules: 30% Coverage (GGUF only)

### 🐛 Bugfixes
- Versionsinkonsistenz behoben (pyproject.toml → 1.0.0)
- Import-Struktur korrigiert
- CLI Export-Befehl verbessert
- Memory-Leak im Checkpoint-Manager behoben

### ⚠️ Bekannte Issues
- Resume Training: Edge Cases nicht vollständig getestet
- Platform Adapter: Benötigen weitere Implementation
- Export-Formate: Nur GGUF vollständig implementiert

---

## [0.3.0] - 2024-XX-XX

### Neu
- Verbesserte Hardware-Erkennung
- Gradient Checkpointing Support
- Torch Compile Integration (PyTorch 2.0+)

### Geändert
- Config-Struktur überarbeitet
- Dataset-Loading optimiert

---

## [0.2.0] - 2024-XX-XX

### Neu
- Auto Mode erste Version
- Swap Management Basic

### Geändert
- Performance-Verbesserungen beim Training

---

## [0.1.0] - 2024-XX-XX

### Neu
- Erstes Release
- Basic Training Functionality
- Hardware Detection
- Config Management

---

## Legend

- `✨ Neu`: Neue Features
- `🐛 Bugfixes`: Fehlerbehebungen
- `⚠️ Breaking Changes`: Inkompatible Änderungen
- `📝 Dokumentation`: Dokumentations-Updates
- `🧪 Tests`: Test-Updates
- `🔧 Geändert`: Änderungen an bestehenden Features
- `⚡ Performance`: Performance-Verbesserungen
- `♻️ Refactored`: Code-Refactoring
- `📦 Dependencies`: Dependency-Updates

[Unreleased]: https://github.com/hcsmediacorp/hcs-UARF/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/hcsmediacorp/hcs-UARF/releases/tag/v1.0.0
[0.3.0]: https://github.com/hcsmediacorp/hcs-UARF/releases/tag/v0.3.0
[0.2.0]: https://github.com/hcsmediacorp/hcs-UARF/releases/tag/v0.2.0
[0.1.0]: https://github.com/hcsmediacorp/hcs-UARF/releases/tag/v0.1.0
