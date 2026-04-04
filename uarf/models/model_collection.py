#!/usr/bin/env python3
"""
UARF Model Collection - Erweiterte Modellsammlung

Unterstützte Modelle von HuggingFace, GitHub und anderen Quellen.
Optimiert für verschiedene Hardware-Klassen.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Modell-Informationen"""
    model_id: str
    name: str
    size_params: float  # in Milliarden Parametern
    min_ram_gb: float
    min_vram_gb: Optional[float]
    license: str
    source: str  # huggingface, github, etc.
    tasks: List[str]
    description: str
    recommended_for: str  # z.B. "mobile", "desktop", "server"


class ModelCollection:
    """
    Sammlung aller unterstützten Modelle
    
    Enthält Modelle von:
    - HuggingFace (Qwen, Phi, Gemma, Llama, TinyLlama)
    - GitHub (Open Source Projekte)
    - Anderen Quellen
    """
    
    MODELS = [
        # Qwen Serie (Alibaba) - Sehr empfehlenswert!
        ModelInfo(
            model_id="Qwen/Qwen2.5-0.5B",
            name="Qwen 2.5 0.5B",
            size_params=0.5,
            min_ram_gb=2.0,
            min_vram_gb=None,
            license="Apache-2.0",
            source="huggingface",
            tasks=["text-generation", "chat", "code"],
            description="Kleinstes Qwen Modell, perfekt für Mobile und Low-RAM",
            recommended_for="mobile"
        ),
        ModelInfo(
            model_id="Qwen/Qwen2.5-1.5B",
            name="Qwen 2.5 1.5B",
            size_params=1.5,
            min_ram_gb=4.0,
            min_vram_gb=2.0,
            license="Apache-2.0",
            source="huggingface",
            tasks=["text-generation", "chat", "code", "multilingual"],
            description="Ausgezeichnetes Preis-Leistungs-Verhältnis",
            recommended_for="desktop"
        ),
        ModelInfo(
            model_id="Qwen/Qwen2.5-3B",
            name="Qwen 2.5 3B",
            size_params=3.0,
            min_ram_gb=6.0,
            min_vram_gb=4.0,
            license="Apache-2.0",
            source="huggingface",
            tasks=["text-generation", "chat", "code", "reasoning"],
            description="Sehr gutes Modell für Desktop-PCs",
            recommended_for="desktop"
        ),
        ModelInfo(
            model_id="Qwen/Qwen2.5-7B",
            name="Qwen 2.5 7B",
            size_params=7.0,
            min_ram_gb=12.0,
            min_vram_gb=8.0,
            license="Apache-2.0",
            source="huggingface",
            tasks=["text-generation", "chat", "code", "reasoning", "math"],
            description="Leistungsstarkes Modell für Workstations",
            recommended_for="server"
        ),
        
        # Microsoft Phi Serie
        ModelInfo(
            model_id="microsoft/phi-2",
            name="Phi-2",
            size_params=2.7,
            min_ram_gb=6.0,
            min_vram_gb=3.0,
            license="MIT",
            source="huggingface",
            tasks=["text-generation", "reasoning", "code"],
            description="Kompaktes Modell mit überraschenden Fähigkeiten",
            recommended_for="desktop"
        ),
        ModelInfo(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            name="Phi-3 Mini",
            size_params=3.8,
            min_ram_gb=8.0,
            min_vram_gb=4.0,
            license="MIT",
            source="huggingface",
            tasks=["text-generation", "chat", "reasoning"],
            description="Neueste Phi Generation mit verbessertem Reasoning",
            recommended_for="desktop"
        ),
        
        # Google Gemma Serie
        ModelInfo(
            model_id="google/gemma-2b",
            name="Gemma 2B",
            size_params=2.0,
            min_ram_gb=4.0,
            min_vram_gb=2.0,
            license="Gemma Terms",
            source="huggingface",
            tasks=["text-generation", "chat"],
            description="Leichtgewichtiges Google Modell",
            recommended_for="desktop"
        ),
        ModelInfo(
            model_id="google/gemma-7b",
            name="Gemma 7B",
            size_params=7.0,
            min_ram_gb=12.0,
            min_vram_gb=8.0,
            license="Gemma Terms",
            source="huggingface",
            tasks=["text-generation", "chat", "reasoning"],
            description="Leistungsstarkes Google Open Model",
            recommended_for="server"
        ),
        
        # TinyLlama
        ModelInfo(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            name="TinyLlama 1.1B Chat",
            size_params=1.1,
            min_ram_gb=3.0,
            min_vram_gb=None,
            license="Apache-2.0",
            source="huggingface",
            tasks=["text-generation", "chat"],
            description="Ultra-effizientes Chat-Modell",
            recommended_for="mobile"
        ),
        
        # Llama Serie (Meta)
        ModelInfo(
            model_id="meta-llama/Llama-3.2-1B",
            name="Llama 3.2 1B",
            size_params=1.0,
            min_ram_gb=3.0,
            min_vram_gb=None,
            license="Llama Community",
            source="huggingface",
            tasks=["text-generation", "chat"],
            description="Neuestes kleines Llama Modell",
            recommended_for="mobile"
        ),
        ModelInfo(
            model_id="meta-llama/Llama-3.2-3B",
            name="Llama 3.2 3B",
            size_params=3.0,
            min_ram_gb=6.0,
            min_vram_gb=4.0,
            license="Llama Community",
            source="huggingface",
            tasks=["text-generation", "chat", "reasoning"],
            description="Ausgewogenes Llama Modell",
            recommended_for="desktop"
        ),
        ModelInfo(
            model_id="meta-llama/Meta-Llama-3-8B",
            name="Llama 3 8B",
            size_params=8.0,
            min_ram_gb=14.0,
            min_vram_gb=10.0,
            license="Llama Community",
            source="huggingface",
            tasks=["text-generation", "chat", "reasoning", "code"],
            description="State-of-the-art Open Model",
            recommended_for="server"
        ),
        
        # SmolLM (sehr klein, für Mobile)
        ModelInfo(
            model_id="HuggingFaceTB/SmolLM-360M",
            name="SmolLM 360M",
            size_params=0.36,
            min_ram_gb=1.5,
            min_vram_gb=None,
            license="Apache-2.0",
            source="huggingface",
            tasks=["text-generation"],
            description="Minimalistisch für extrem ressourcenbeschränkte Geräte",
            recommended_for="mobile"
        ),
    ]
    
    @classmethod
    def get_all_models(cls) -> List[ModelInfo]:
        """Alle verfügbaren Modelle"""
        return cls.MODELS
    
    @classmethod
    def get_model_by_id(cls, model_id: str) -> Optional[ModelInfo]:
        """Sucht Modell nach ID"""
        for model in cls.MODELS:
            if model.model_id == model_id:
                return model
        return None
    
    @classmethod
    def filter_by_hardware(cls, ram_gb: float, vram_gb: Optional[float] = None) -> List[ModelInfo]:
        """Filtert Modelle nach verfügbarer Hardware"""
        compatible = []
        for model in cls.MODELS:
            if ram_gb >= model.min_ram_gb:
                if vram_gb is None or model.min_vram_gb is None:
                    compatible.append(model)
                elif vram_gb >= model.min_vram_gb:
                    compatible.append(model)
        return sorted(compatible, key=lambda x: x.size_params)
    
    @classmethod
    def filter_by_use_case(cls, use_case: str) -> List[ModelInfo]:
        """Filtert Modelle nach Einsatzzweck"""
        return [m for m in cls.MODELS if m.recommended_for == use_case]
    
    @classmethod
    def print_collection(cls):
        """Druckt gesamte Modellsammlung"""
        print("\n" + "="*80)
        print("📚 UARF MODEL COLLECTION")
        print("="*80)
        
        # Gruppieren nach Quelle
        by_source = {}
        for model in cls.MODELS:
            if model.source not in by_source:
                by_source[model.source] = []
            by_source[model.source].append(model)
        
        for source, models in by_source.items():
            print(f"\n📍 {source.upper()}")
            print("-"*60)
            for model in models:
                vram_req = f"{model.min_vram_gb}GB VRAM" if model.min_vram_gb else "CPU möglich"
                print(f"  • {model.name} ({model.size_params}B)")
                print(f"    RAM: {model.min_ram_gb}GB | {vram_req}")
                print(f"    License: {model.license}")
                print(f"    {model.description}")
                print()
        
        print("="*80)


if __name__ == '__main__':
    ModelCollection.print_collection()
