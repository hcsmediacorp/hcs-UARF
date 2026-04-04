#!/usr/bin/env python3
"""
UARF Quickstart - Teste ob alles funktioniert
"""

import sys

def test_imports():
    """Teste alle wichtigen Imports"""
    tests = []
    
    # PyTorch
    try:
        import torch
        tests.append(("✅", "PyTorch", f"v{torch.__version__}"))
        cuda = torch.cuda.is_available()
        mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        device = "CUDA" if cuda else "MPS" if mps else "CPU"
        tests.append(("📦", "Device", device))
    except ImportError as e:
        tests.append(("❌", "PyTorch", str(e)))
        return False, tests
    
    # Transformers
    try:
        import transformers
        tests.append(("✅", "Transformers", f"v{transformers.__version__}"))
    except ImportError as e:
        tests.append(("❌", "Transformers", str(e)))
    
    # Datasets
    try:
        import datasets
        tests.append(("✅", "Datasets", f"v{datasets.__version__}"))
    except ImportError as e:
        tests.append(("❌", "Datasets", str(e)))
    
    # TQDM
    try:
        import tqdm
        tests.append(("✅", "TQDM", f"v{tqdm.__version__}"))
    except ImportError as e:
        tests.append(("❌", "TQDM", str(e)))
    
    # PSUtil
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        tests.append(("✅", "PSUtil", f"{ram_gb:.1f} GB RAM erkannt"))
    except ImportError as e:
        tests.append(("❌", "PSUtil", str(e)))
    
    return True, tests


def test_hardware():
    """Teste Hardware-Erkennung"""
    print("\n🔍 Hardware-Check:")
    
    try:
        import torch
        import psutil
        
        ram = psutil.virtual_memory()
        print(f"   RAM: {ram.total / (1024**3):.1f} GB gesamt, {ram.available / (1024**3):.1f} GB frei")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   GPU: Apple Silicon")
        else:
            print(f"   GPU: Nicht verfügbar (CPU-Modus)")
        
        return True
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return False


def test_model_load():
    """Teste Modell-Laden mit minimalem Modell"""
    print("\n🧪 Teste Modell-Laden...")
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
        params = sum(getattr(config, attr, 0) for attr in ['hidden_size', 'num_hidden_layers', 'vocab_size'] if hasattr(config, attr))
        print(f"   ✅ Konfiguration geladen: ~{params:,} Parameter-Infos")
        return True
    except Exception as e:
        print(f"   ⚠️  Modell-Test fehlgeschlagen: {e}")
        print(f"      (Das ist OK für den ersten Start)")
        return True  # Nicht kritisch


def main():
    print("=" * 60)
    print("🚀 UARF Quickstart Check")
    print("=" * 60)
    
    # Imports testen
    print("\n📦 Import-Check:")
    success, tests = test_imports()
    
    for status, name, info in tests:
        print(f"   {status} {name}: {info}")
    
    if not success:
        print("\n❌ Kritische Abhängigkeiten fehlen!")
        print("\n💡 Installation:")
        print("   pip install torch transformers datasets tqdm psutil huggingface_hub")
        sys.exit(1)
    
    # Hardware testen
    hw_ok = test_hardware()
    
    # Modell-Test
    model_ok = test_model_load()
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    if success and hw_ok:
        print("✅ Alles bereit für UARF!")
        print("\n🎯 Nächste Schritte:")
        print("   python uarf_run.py --demo          # Demo Training starten")
        print("   python uarf_run.py --detect-only   # Hardware anzeigen")
        print("   python uarf_run.py --suggest       # Modell-Empfehlung")
    else:
        print("⚠️  Einige Tests fehlgeschlagen")
        print("   Siehe Fehlermeldungen oben")
    print("=" * 60)


if __name__ == '__main__':
    main()
