#!/usr/bin/env python3
"""
UARF Auto Mode - Einfache Text-zu-Modell Pipeline

Nur Text pasten, der Rest läuft automatisch!
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any

from core.config import UARFConfig
from core.trainer import UniversalTrainer
from core.hardware_detector import HardwareDetector
from core.swap_manager import SwapManager, SwapConfig


class AutoMode:
    """
    Vollautomatischer Trainingsmodus
    
    Verwendung:
        1. Text eingeben/pasten
        2. AutoMode starten
        3. Fertig!
    """
    
    def __init__(self):
        self.hardware = HardwareDetector()
        self.swap_manager = None
        self.config = None
        self.trainer = None
        
    def setup_swap(self, auto: bool = True, size_gb: Optional[float] = None):
        """Swap einrichten für speichereffizientes Training"""
        if auto:
            self.swap_manager = SwapManager(SwapConfig(auto_mode=True, enabled=True))
            self.swap_manager.setup_auto_swap()
        elif size_gb:
            self.swap_manager = SwapManager(SwapConfig(auto_mode=False, enabled=True))
            self.swap_manager.setup_manual_swap(size_gb)
        
        return self
    
    def create_dataset_from_text(self, text: str, name: str = "custom_text"):
        """
        Erstellt Dataset aus Text
        
        Args:
            text: Der Text zum Trainieren
            name: Name des Datasets
            
        Returns:
            Pfad zum erstellten Dataset
        """
        # Temporäres Verzeichnis erstellen
        temp_dir = Path(tempfile.mkdtemp(prefix="uarf_auto_"))
        
        # Text in Segmente aufteilen (für besseres Training)
        segments = self._split_text_into_segments(text, segment_size=512)
        
        # JSONL Datei erstellen
        jsonl_path = temp_dir / "dataset.jsonl"
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments):
                entry = {
                    "id": i,
                    "text": segment.strip(),
                    "source": name
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✅ Dataset erstellt: {jsonl_path} ({len(segments)} Segmente)")
        
        return str(jsonl_path)
    
    def _split_text_into_segments(self, text: str, segment_size: int = 512) -> list:
        """Teilt Text in überlappende Segmente für besseres Training"""
        words = text.split()
        segments = []
        
        for i in range(0, len(words), segment_size // 2):  # 50% Überlappung
            segment = ' '.join(words[i:i + segment_size])
            if segment.strip():
                segments.append(segment)
        
        return segments
    
    def train(self, text: str, model_id: Optional[str] = None, 
              time_seconds: int = 300, output_dir: str = "./auto_outputs",
              swap_enabled: bool = True, swap_size_gb: Optional[float] = None):
        """
        Vollautomatisches Training
        
        Args:
            text: Text zum Trainieren
            model_id: Modell-ID (optional, wird automatisch gewählt wenn None)
            time_seconds: Trainingszeit in Sekunden
            output_dir: Ausgabeverzeichnis
            swap_enabled: Swap aktivieren
            swap_size_gb: Swap-Größe in GB (optional)
        """
        print("\n" + "="*70)
        print("🚀 UARF AUTO MODE")
        print("="*70)
        print("📝 Text wird verarbeitet...")
        
        # Hardware erkennen
        print("\n🔍 Hardware-Erkennung...")
        self.hardware.print_summary()
        
        # Swap einrichten
        if swap_enabled:
            print("\n💾 Swap-Management...")
            self.setup_swap(auto=(swap_size_gb is None), size_gb=swap_size_gb)
        
        # Dataset erstellen
        print("\n📊 Dataset wird erstellt...")
        dataset_path = self.create_dataset_from_text(text)
        
        # Modell auswählen falls nicht angegeben
        if model_id is None:
            model_id = self._select_optimal_model()
            print(f"🎯 Automatisches Modell ausgewählt: {model_id}")
        
        # Konfiguration erstellen
        hardware_config = self.hardware.get_optimal_config()
        
        self.config = UARFConfig(
            model_id=model_id,
            dataset_name=dataset_path,
            time_budget_seconds=time_seconds,
            output_dir=output_dir,
            batch_size=hardware_config.get('batch_size', 16),
            max_seq_len=hardware_config.get('max_seq_len', 512),
            learning_rate=2e-4,
            gradient_accumulation_steps=hardware_config.get('gradient_accumulation', 4),
            use_gradient_checkpointing=True,
            compile_model=False,
        )
        
        print("\n📋 Konfiguration:")
        self.config.print_summary()
        
        # Training starten
        print("\n🔥 Training startet...")
        self.trainer = UniversalTrainer(self.config)
        self.trainer.train()
        
        print("\n" + "="*70)
        print("✅ TRAINING ABGESCHLOSSEN!")
        print("="*70)
        print(f"📁 Ergebnisse: {output_dir}")
        
        return self.trainer
    
    def _select_optimal_model(self) -> str:
        """Wählt optimales Modell basierend auf Hardware"""
        specs = self.hardware.specs
        
        if specs.gpu_vram and specs.gpu_vram >= 16:
            return "Qwen/Qwen2.5-3B"
        elif specs.gpu_vram and specs.gpu_vram >= 8:
            return "Qwen/Qwen2.5-1.5B"
        elif specs.ram_available >= 4:
            return "Qwen/Qwen2.5-0.5B"
        else:
            return "Qwen/Qwen2.5-0.5B"
    
    def quick_train(self, text: str):
        """Schnellstes Training mit Minimal-Konfiguration"""
        return self.train(
            text=text,
            time_seconds=120,
            swap_enabled=True,
        )


def auto_train(text: str, **kwargs):
    """
    Convenience-Funktion für Auto Mode
    
    Verwendung:
        from uarf import auto_train
        auto_train("Dein Text hier...")
    """
    auto = AutoMode()
    return auto.train(text, **kwargs)


if __name__ == '__main__':
    print("UARF Auto Mode - Test")
    print("="*50)
    
    demo_text = """
    Künstliche Intelligenz und maschinelles Lernen verändern unsere Welt.
    Neuronale Netze lernen aus Daten und erkennen Muster.
    Deep Learning ermöglicht komplexe Aufgaben wie Bilderkennung und Sprachverarbeitung.
    Transformer-Modelle haben die NLP-Landschaft revolutioniert.
    """
    
    auto = AutoMode()
    dataset_path = auto.create_dataset_from_text(demo_text, "demo")
    print(f"Demo-Dataset erstellt: {dataset_path}")
