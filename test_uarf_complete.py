#!/usr/bin/env python3
"""
UARF Quick Test Script
Testet das komplette UARF System mit lokalem Dataset
"""

import sys
import time
from pathlib import Path

# Projekt-Root zum Path hinzufügen
sys.path.insert(0, str(Path(__file__).parent))

from uarf.core.config import UARFConfig
from uarf.core.hardware_detector import HardwareDetector
from uarf.core.trainer import UniversalTrainer


def run_quick_test():
    """Führt einen schnellen Test des gesamten Systems durch"""
    
    print("=" * 70)
    print("🚀 UARF v0.3.0 - STABLE RELEASE TEST")
    print("=" * 70)
    
    # 1. Hardware erkennen
    print("\n🔍 SCHRITT 1: Hardware-Erkennung")
    detector = HardwareDetector()
    detector.print_summary()
    
    hardware_config = detector.get_optimal_config()
    
    # 2. Konfiguration erstellen
    print("\n⚙️  SCHRITT 2: Konfiguration")
    
    # Sehr kleine Konfiguration für Systeme mit wenig RAM
    config = UARFConfig(
        model_id="Qwen/Qwen2.5-0.5B",
        dataset_name="./test_data/mini_dataset.json",  # Kleinstes Dataset
        time_budget_seconds=60,  # 1 Minute Test
        batch_size=4,  # Sehr klein für wenig RAM
        max_seq_len=64,  # Minimal für wenig RAM
        max_steps=20,  # Nur 20 Steps für schnellen Test
        learning_rate=2e-4,
        output_dir="./test_output",
        log_every_n_steps=5,
        eval_every_n_steps=10,
        save_every_n_steps=20,
        gradient_accumulation_steps=2,  # Accumulate um effektive batch size zu erhöhen
        use_gradient_checkpointing=True,  # Spart Speicher
        compile_model=False,  # Deaktiviert für schnelleren Start und weniger RAM
        val_split_ratio=0.1,
        precision="fp32",  # Kein Mixed Precision Overhead
        num_workers=0,  # Keine zusätzlichen Prozesse
    )
    
    config.print_summary()
    
    # 3. Trainer erstellen
    print("\n🏗️  SCHRITT 3: Trainer initialisieren")
    trainer = UniversalTrainer(config)
    print("✅ Trainer erstellt")
    
    # 4. Training starten
    print("\n🎯 SCHRITT 4: Training starten")
    start_time = time.time()
    
    try:
        trainer.train()
        
        elapsed = time.time() - start_time
        print(f"\n✅ TRAINING ERFOLGREICH in {elapsed:.1f}s")
        
        # Metriken ausgeben
        print("\n" + "=" * 70)
        print("📊 ERGEBNISSE")
        print("=" * 70)
        print(f"Steps:           {trainer.metrics.steps_completed}")
        print(f"Total Tokens:    {trainer.metrics.total_tokens:,}")
        print(f"Best Val Loss:   {trainer.metrics.best_val_loss:.4f}")
        print(f"Training Time:   {trainer.metrics.training_time_seconds:.1f}s")
        print(f"Peak Memory:     {trainer.metrics.peak_memory_mb:.1f} MB")
        
        # Check if output was created
        output_dir = Path(config.output_dir)
        if output_dir.exists():
            checkpoints = list(output_dir.glob("checkpoint-*"))
            final_model = output_dir / "final"
            print(f"\n💾 OUTPUT:")
            print(f"   Output Dir: {output_dir.absolute()}")
            print(f"   Checkpoints: {len(checkpoints)}")
            print(f"   Final Model: {'✓' if final_model.exists() else '✗'}")
        
        print("\n" + "=" * 70)
        print("✅ ALLE TESTS ERFOLGREICH!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ FEHLER IM TRAINING: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
