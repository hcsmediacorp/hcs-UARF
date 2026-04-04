# CONTRIBUTING.md - Beiträge zu UARF

**Willkommen!** Wir freuen uns über jeden Beitrag zum UARF-Projekt.

## 📑 Inhaltsverzeichnis

1. [Code of Conduct](#code-of-conduct)
2. [Wie kann ich beitragen?](#wie-kann-ich-beitragen)
3. [Development Setup](#development-setup)
4. [Pull Request Prozess](#pull-request-prozess)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Dokumentation](#dokumentation)

---

## 🤝 Code of Conduct

### Unsere Verpflichtung

Wir als Contributing verpflichten uns, eine offene und einladende Community zu schaffen. Jeder ist willkommen, unabhängig von:
- Alter, Körpergröße, Behinderung
- Ethnischer Zugehörigkeit, Geschlecht, Identität
- Erfahrung, Nationalität, Aussehen
- Rasse, Religion, sexuelle Orientierung

### Unsere Standards

Beispiele für Verhalten, das zur Schaffung einer positiven Umgebung beiträgt:
- Verwendung von einladender und inklusiver Sprache
- Respektvoller Umgang mit unterschiedlichen Meinungen
- Konstruktive Kritik annehmen und geben
- Fokus auf das, was am besten für die Community ist

Beispiele für inakzeptables Verhalten:
- Verwendung sexualisierter Sprache oder Bilder
- Trolling, beleidigende Kommentare, persönliche Angriffe
- Öffentliche oder private Belästigung
- Veröffentlichung privater Informationen ohne Zustimmung

---

## 💡 Wie kann ich beitragen?

### 1. Bug Reports melden

Fehlermeldungen sind wertvoll! Bevor du einen Bug reportest:
- ✅ Überprüfe, ob der Bug bereits gemeldet wurde
- ✅ Aktualisiere auf die neueste Version
- ✅ Sammle Informationen (Logs, Steps to Reproduce)

**Bug Report Template:**
```markdown
**Beschreibung:**
Kurze Beschreibung des Problems

**Steps to Reproduce:**
1. ...
2. ...
3. ...

**Erwartetes Verhalten:**
Was sollte passieren?

**Tatsächliches Verhalten:**
Was ist passiert?

**Umgebung:**
- OS: [z.B. Ubuntu 22.04]
- Python: [z.B. 3.11]
- UARF Version: [z.B. 1.0.0]

**Logs:**
```
[Log-Ausgabe hier]
```
```

### 2. Feature Requests vorschlagen

Neue Ideen sind willkommen! Erstelle ein Issue mit dem Label "enhancement".

**Feature Request Template:**
```markdown
**Problem Statement:**
Welches Problem soll gelöst werden?

**Proposed Solution:**
Wie könnte die Lösung aussehen?

**Alternatives Considered:**
Gibt es alternative Lösungen?

**Additional Context:**
Weitere Informationen, Screenshots, etc.
```

### 3. Code beitragen

#### Einfache Fixes (Good First Issues)
- Dokumentation verbessern
- Typo korrigieren
- Kleine Bugfixes
- Tests hinzufügen

Look for issues labeled:
- `good first issue` - Perfekt für Einsteiger
- `help wanted` - Hilfe erwünscht
- `documentation` - Docs verbessern

#### Größere Features
Für größere Änderungen:
1. Erstelle zuerst ein Issue zur Diskussion
2. Warte auf Feedback vom Maintainer
3. Erstelle dann einen PR

---

## 🛠️ Development Setup

### 1. Repository forken

```bash
# Fork auf GitHub erstellen
# Dann klonen:
git clone https://github.com/DEIN_USERNAME/hcs-UARF.git
cd hcs-UARF
```

### 2. Entwicklungsumgebung einrichten

```bash
# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -e ".[dev]"

# Pre-commit Hooks installieren (optional)
pip install pre-commit
pre-commit install
```

### 3. Branch erstellen

```bash
# Immer von develop branch aus arbeiten
git checkout develop
git pull origin develop

# Feature Branch erstellen
git checkout -b feature/mein-feature-name
# oder für Bugfixes
git checkout -b fix/bugfix-beschreibung
```

---

## 🔄 Pull Request Prozess

### 1. Vor dem Commit

```bash
# Code formatieren
black uarf/ tests/
isort uarf/ tests/

# Linting
flake8 uarf/ tests/

# Tests laufen lassen
pytest tests/ -v
```

### 2. Commit Guidelines

**Commit Message Format:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: Neues Feature
- `fix`: Bugfix
- `docs`: Dokumentation
- `style`: Formatting (keine Logik-Änderung)
- `refactor`: Code-Refactoring
- `test`: Tests hinzufügen/ändern
- `chore`: Build/Tooling Changes

**Beispiele:**
```bash
# Good
git commit -m "feat(auto_mode): Add text segmentation for better training"
git commit -m "fix(trainer): Resolve memory leak in checkpoint saving"
git commit -m "docs(readme): Update installation instructions"

# Bad
git commit -m "fixed stuff"
git commit -m "WIP"
```

### 3. Pull Request erstellen

1. **Branch pushen:**
   ```bash
   git push origin feature/mein-feature-name
   ```

2. **PR auf GitHub erstellen:**
   - Base: `hcsmediacorp/hcs-UARF:develop`
   - Head: `DEIN_USERNAME/hcs-UARF:feature/mein-feature-name`

3. **PR Description ausfüllen:**
   ```markdown
   ## Beschreibung
   Kurze Beschreibung der Änderungen
   
   ## Related Issue
   Closes #123
   
   ## Checklist
   - [ ] Tests hinzugefügt
   - [ ] Dokumentation aktualisiert
   - [ ] Changelog aktualisiert
   - [ ] Code formatted (black, isort)
   - [ ] Alle Tests passing
   ```

### 4. Review Prozess

- Ein Maintainer reviewt deinen PR
- Feedback wird als Comments gegeben
- Nach Approval wird der PR gemerged
- Bei Änderungen requested: Address feedback und push again

---

## 📝 Coding Standards

### Python Style Guide

Wir folgen [PEP 8](https://pep8.org/) mit diesen Ergänzungen:

**Imports:**
```python
# Standard Library
import os
import sys
from pathlib import Path

# Third Party
import torch
import numpy as np
from transformers import AutoModel

# Local Imports
from .core.config import UARFConfig
from ..utils.helpers import format_output
```

**Type Hints:**
```python
def calculate_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> float:
    """Calculate loss between predictions and targets."""
    pass
```

**Docstrings:**
```python
class UniversalTrainer:
    """
    Universeller Trainer für alle Plattformen
    
    Unterstützt Training auf:
    - Mobile Geräten (Android/Termux)
    - Desktop (Windows/Linux/Mac)
    - Cloud (Google Colab)
    - Cluster (Multi-GPU)
    
    Example:
        >>> config = UARFConfig(model_id="Qwen/Qwen2.5-0.5B")
        >>> trainer = UniversalTrainer(config)
        >>> trainer.train()
    """
    pass
```

### File Structure

```
uarf/
├── core/           # Kernmodule
│   ├── __init__.py
│   ├── config.py   # Mit Docstrings und Type Hints
│   └── ...
├── tests/          # Tests parallel zur Struktur
│   ├── test_core.py
│   └── ...
└── ...
```

---

## 🧪 Testing

### Tests schreiben

```python
# tests/test_mein_module.py
import pytest
from uarf.core.config import UARFConfig

class TestUARFConfig:
    """Tests für UARFConfig"""
    
    def test_default_values(self):
        """Testet Standardwerte"""
        config = UARFConfig()
        assert config.batch_size == 16
        assert config.learning_rate == 2e-4
    
    def test_custom_values(self):
        """Testet benutzerdefinierte Werte"""
        config = UARFConfig(batch_size=32, learning_rate=1e-4)
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
    
    @pytest.mark.parametrize("batch_size", [4, 8, 16, 32])
    def test_batch_sizes(self, batch_size):
        """Testet verschiedene Batch Sizes"""
        config = UARFConfig(batch_size=batch_size)
        assert config.batch_size == batch_size
```

### Tests laufen lassen

```bash
# Alle Tests
pytest tests/ -v

# Mit Coverage
pytest tests/ -v --cov=uarf --cov-report=html

# Einzelner Test
pytest tests/test_core.py::TestUARFConfig::test_default_values -v

# Schnelle Tests (ohne slow marker)
pytest tests/ -v -m "not slow"
```

---

## 📖 Dokumentation

### README Updates

Bei neuen Features:
- README.md aktualisieren
- Screenshot/GIF hinzufügen (wenn relevant)
- Beispiel-Code bereitstellen

### Tutorials

Neue Tutorials erstellen im Format:
```markdown
# Tutorial XX: Titel

**Level:** 🟢 Beginner | **Dauer:** X Minuten

## Beschreibung
...

## Code Beispiele
```python
...
```

## Zusammenfassung
...
```

### API Dokumentation

Docstrings für alle öffentlichen Funktionen:
```python
def public_function(param: str) -> dict:
    """
    Kurzbeschreibung
    
    Ausführliche Beschreibung mit Details.
    
    Args:
        param: Beschreibung des Parameters
        
    Returns:
        Beschreibung des Rückgabewerts
        
    Raises:
        ValueError: Wenn Parameter ungültig ist
        
    Example:
        >>> result = public_function("test")
        >>> print(result)
        {'key': 'value'}
    """
    pass
```

---

## 📜 License

Mit deinem Beitrag stimmst du zu, dass dein Code unter der Apache 2.0 License veröffentlicht wird.

---

## 🙏 Danke!

Jeder Beitrag hilft, UARF besser zu machen. Egal ob Bug Report, Feature Request, Documentation oder Code - wir schätzen jede Hilfe!

**Fragen?** Öffne ein Issue oder schreibe in den Discussions.

---

**Made with ❤️ by the UARF Team**
