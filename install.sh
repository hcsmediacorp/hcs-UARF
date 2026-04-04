#!/bin/bash
# UARF Installation Script - Funktioniert auf Linux, Mac, Windows (WSL), Termux

set -e

echo "🚀 UARF Installation"
echo "===================="

# Platform erkennen
PLATFORM=$(uname -s 2>/dev/null || echo "Windows")
IS_TERMUX=false

if [ -n "$TERMUX_VERSION" ] || [ -d "/data/data/com.termux" ]; then
    IS_TERMUX=true
    echo "📱 Termux/Android erkannt!"
fi

# Python prüfen
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python nicht gefunden!"
    if [ "$IS_TERMUX" = true ]; then
        echo "   Installiere mit: pkg install python"
    else
        echo "   Bitte Python 3.8+ installieren"
    fi
    exit 1
fi

echo "✅ Python gefunden: $($PYTHON --version)"

# Virtuelle Umgebung erstellen
if [ ! -d ".venv" ]; then
    echo "📦 Erstelle virtuelle Umgebung..."
    $PYTHON -m venv .venv
fi

# Aktivieren
if [ "$IS_TERMUX" = true ]; then
    source .venv/bin/activate
else
    case "$PLATFORM" in
        MINGW*|CYGWIN*|MSYS*)
            source .venv/Scripts/activate
            ;;
        *)
            source .venv/bin/activate
            ;;
    esac
fi

echo "✅ Virtuelle Umgebung aktiviert"

# Dependencies installieren
echo "📦 Installiere Dependencies..."
pip install --upgrade pip

# PyTorch Installation basierend auf Platform
if [ "$IS_TERMUX" = true ]; then
    echo "   📱 Termux: Installiere CPU-only PyTorch..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    case "$PLATFORM" in
        Darwin)
            echo "   🍎 macOS: Installiere PyTorch..."
            pip install torch torchvision torchaudio
            ;;
        Linux)
            if lspci | grep -i nvidia &> /dev/null; then
                echo "   🐧 Linux + NVIDIA: Installiere PyTorch mit CUDA..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            else
                echo "   🐧 Linux (CPU): Installiere PyTorch..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            fi
            ;;
        MINGW*|CYGWIN*|MSYS*)
            echo "   🪟 Windows: Installiere PyTorch..."
            pip install torch torchvision torchaudio
            ;;
        *)
            echo "   🔧 Unbekannte Platform: Installiere Standard PyTorch..."
            pip install torch torchvision torchaudio
            ;;
    esac
fi

# Andere Dependencies
echo "   📦 Installiere weitere Pakete..."
pip install transformers datasets tqdm psutil huggingface_hub

echo ""
echo "✅ Installation abgeschlossen!"
echo ""
echo "🎯 Nächste Schritte:"
echo "   source .venv/bin/activate  # Umgebung aktivieren"
echo "   python uarf_run.py         # Training starten"
echo ""
echo "💡 Tipps:"
echo "   python uarf_run.py --demo          # Demo-Modus"
echo "   python uarf_run.py --detect-only   # Hardware erkennen"
echo "   python uarf_run.py --suggest       # Modell-Empfehlung"
