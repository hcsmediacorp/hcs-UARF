"""
Model Selector - Intelligente Modellauswahl basierend auf Hardware und Aufgabe
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .hardware_detector import HardwareSpecs


@dataclass
class ModelInfo:
    """Informationen über ein Modell"""
    model_id: str
    name: str
    size_mb: float
    params_millions: int
    min_ram_gb: float
    min_vram_gb: float
    recommended_for: List[str]
    license: str
    description: str


class ModelSelector:
    """Wählt passende Modelle basierend auf Hardware-Spezifikationen aus"""
    
    # Vordefinierte Modelle für verschiedene Use-Cases - Organisiert nach Größe
    AVAILABLE_MODELS = {
        "text-generation": [
            # === SEHR KLEINE MODELLE (< 1B) - Mobile/Edge ===
            ModelInfo(
                model_id="Qwen/Qwen2.5-0.5B",
                name="Qwen 2.5 0.5B",
                size_mb=950,
                params_millions=494,
                min_ram_gb=1.5,
                min_vram_gb=0.8,
                recommended_for=["mobile", "edge", "low-resource", "android", "iot"],
                license="Apache-2.0",
                description="Ultra-kompaktes Modell für mobile Geräte und IoT"
            ),
            ModelInfo(
                model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                name="TinyLlama 1.1B Chat",
                size_mb=2100,
                params_millions=1100,
                min_ram_gb=2.5,
                min_vram_gb=1.2,
                recommended_for=["mobile", "edge", "quick-experiments", "raspberry-pi"],
                license="Apache-2.0",
                description="Schnellstes Chat-Modell für Edge-Geräte"
            ),
            ModelInfo(
                model_id="google/gemma-2b-it",
                name="Gemma 2B Instruct",
                size_mb=3800,
                params_millions=2500,
                min_ram_gb=4.0,
                min_vram_gb=2.0,
                recommended_for=["mobile", "desktop", "colab-free"],
                license="Gemma-2.0",
                description="Googles leichtes Instruct-Modell"
            ),
            
            # === KLEINE MODELLE (1B - 3B) - Desktop/Entry GPU ===
            ModelInfo(
                model_id="Qwen/Qwen2.5-1.5B",
                name="Qwen 2.5 1.5B",
                size_mb=2800,
                params_millions=1540,
                min_ram_gb=3.5,
                min_vram_gb=1.8,
                recommended_for=["desktop", "colab-free", "laptop"],
                license="Apache-2.0",
                description="Ausgewogenes Modell für alltägliche Nutzung"
            ),
            ModelInfo(
                model_id="microsoft/phi-2",
                name="Phi-2",
                size_mb=5200,
                params_millions=2700,
                min_ram_gb=5.0,
                min_vram_gb=2.5,
                recommended_for=["desktop", "research", "reasoning"],
                license="MIT",
                description="Microsofts kompaktes Reasoning-Modell"
            ),
            ModelInfo(
                model_id="microsoft/phi-3-mini-4k-instruct",
                name="Phi-3 Mini",
                size_mb=4500,
                params_millions=3800,
                min_ram_gb=5.0,
                min_vram_gb=2.5,
                recommended_for=["desktop", "mobile", "reasoning"],
                license="MIT",
                description="Neuestes Phi-3 Mini mit 4K Kontext"
            ),
            ModelInfo(
                model_id="stabilityai/stablelm-2-zephyr-1_6b",
                name="StableLM 2 Zephyr 1.6B",
                size_mb=3200,
                params_millions=1600,
                min_ram_gb=3.5,
                min_vram_gb=1.8,
                recommended_for=["desktop", "chat", "roleplay"],
                license="CC-BY-SA-4.0",
                description="Feinabgestimmtes Chat-Modell von Stability AI"
            ),
            
            # === MITTLERE MODELLE (3B - 8B) - Standard GPU ===
            ModelInfo(
                model_id="Qwen/Qwen2.5-3B",
                name="Qwen 2.5 3B",
                size_mb=5600,
                params_millions=3200,
                min_ram_gb=7.0,
                min_vram_gb=3.5,
                recommended_for=["desktop-gpu", "colab-pro", "development"],
                license="Apache-2.0",
                description="Solides Mittelklasse-Modell für Entwicklung"
            ),
            ModelInfo(
                model_id="Qwen/Qwen2.5-7B-Instruct",
                name="Qwen 2.5 7B Instruct",
                size_mb=13500,
                params_millions=7200,
                min_ram_gb=14.0,
                min_vram_gb=6.0,
                recommended_for=["desktop-gpu", "colab-pro", "production"],
                license="Apache-2.0",
                description="Leistungsstarkes Instruct-Modell für Produktion"
            ),
            ModelInfo(
                model_id="mistralai/Mistral-7B-Instruct-v0.3",
                name="Mistral 7B Instruct v0.3",
                size_mb=14000,
                params_millions=7300,
                min_ram_gb=14.0,
                min_vram_gb=6.0,
                recommended_for=["desktop-gpu", "colab-pro", "enterprise"],
                license="Apache-2.0",
                description="Industrie-Standard für 7B Klasse"
            ),
            ModelInfo(
                model_id="google/gemma-7b-it",
                name="Gemma 7B Instruct",
                size_mb=15000,
                params_millions=8500,
                min_ram_gb=15.0,
                min_vram_gb=7.0,
                recommended_for=["desktop-gpu", "research", "multilingual"],
                license="Gemma-2.0",
                description="Googles starkes multilinguales Modell"
            ),
            ModelInfo(
                model_id="meta-llama/Llama-3.2-3B-Instruct",
                name="Llama 3.2 3B Instruct",
                size_mb=5800,
                params_millions=3200,
                min_ram_gb=7.0,
                min_vram_gb=3.5,
                recommended_for=["desktop", "mobile", "edge"],
                license="Llama-3.2",
                description="Metas neuestes effizientes 3B Modell"
            ),
            
            # === GROSSE MODELLE (8B - 14B) - High-End GPU ===
            ModelInfo(
                model_id="meta-llama/Llama-3.2-8B-Instruct",
                name="Llama 3.2 8B Instruct",
                size_mb=16000,
                params_millions=8000,
                min_ram_gb=16.0,
                min_vram_gb=8.0,
                recommended_for=["desktop-gpu", "colab-pro", "production"],
                license="Llama-3.2",
                description="Metas aktuelles 8B Produktionsmodell"
            ),
            ModelInfo(
                model_id="microsoft/Phi-3.5-mini-instruct",
                name="Phi-3.5 Mini",
                size_mb=7500,
                params_millions=3800,
                min_ram_gb=8.0,
                min_vram_gb=4.0,
                recommended_for=["desktop-gpu", "reasoning", "coding"],
                license="MIT",
                description="Verbessertes Phi-3 mit besserem Reasoning"
            ),
            ModelInfo(
                model_id="Qwen/Qwen2.5-14B-Instruct",
                name="Qwen 2.5 14B Instruct",
                size_mb=27000,
                params_millions=14000,
                min_ram_gb=28.0,
                min_vram_gb=12.0,
                recommended_for=["cluster", "multi-gpu", "enterprise"],
                license="Apache-2.0",
                description="Hochleistungsmodell für komplexe Aufgaben"
            ),
            
            # === SEHR GROSSE MODELLE (> 20B) - Cluster/Multi-GPU ===
            ModelInfo(
                model_id="Qwen/Qwen2.5-32B-Instruct",
                name="Qwen 2.5 32B Instruct",
                size_mb=62000,
                params_millions=32000,
                min_ram_gb=64.0,
                min_vram_gb=24.0,
                recommended_for=["cluster", "multi-gpu", "research"],
                license="Apache-2.0",
                description="Massives Modell für Forschungsaufgaben"
            ),
            ModelInfo(
                model_id="Qwen/Qwen2.5-72B-Instruct",
                name="Qwen 2.5 72B Instruct",
                size_mb=140000,
                params_millions=72000,
                min_ram_gb=128.0,
                min_vram_gb=48.0,
                recommended_for=["cluster", "multi-gpu", "state-of-art"],
                license="Apache-2.0",
                description="State-of-the-Art Open-Source Modell"
            ),
            ModelInfo(
                model_id="meta-llama/Llama-3.1-70B-Instruct",
                name="Llama 3.1 70B Instruct",
                size_mb=135000,
                params_millions=70000,
                min_ram_gb=128.0,
                min_vram_gb=48.0,
                recommended_for=["cluster", "multi-gpu", "enterprise"],
                license="Llama-3.1",
                description="Metas größtes verfügbares Modell"
            ),
            ModelInfo(
                model_id="tiiuae/falcon-180B",
                name="Falcon 180B",
                size_mb=350000,
                params_millions=180000,
                min_ram_gb=256.0,
                min_vram_gb=96.0,
                recommended_for=["cluster", "distributed", "supercomputer"],
                license="Apache-2.0",
                description="Enormes Modell für verteilte Systeme"
            ),
        ],
        
        # Text Classification Models
        "text-classification": [
            ModelInfo(
                model_id="distilbert-base-uncased",
                name="DistilBERT Base",
                size_mb=250,
                params_millions=66,
                min_ram_gb=1.0,
                min_vram_gb=0.5,
                recommended_for=["all", "mobile", "edge"],
                license="Apache-2.0",
                description="Leichte Klassifikations-Baseline"
            ),
            ModelInfo(
                model_id="bert-base-uncased",
                name="BERT Base",
                size_mb=420,
                params_millions=110,
                min_ram_gb=2.0,
                min_vram_gb=1.0,
                recommended_for=["desktop", "mobile"],
                license="Apache-2.0",
                description="Standard Klassifikationsmodell"
            ),
            ModelInfo(
                model_id="roberta-base",
                name="RoBERTa Base",
                size_mb=480,
                params_millions=125,
                min_ram_gb=2.0,
                min_vram_gb=1.0,
                recommended_for=["desktop", "research"],
                license="MIT",
                description="Verbessertes BERT Training"
            ),
            ModelInfo(
                model_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
                name="RoBERTa Sentiment",
                size_mb=480,
                params_millions=125,
                min_ram_gb=2.0,
                min_vram_gb=1.0,
                recommended_for=["social-media", "sentiment-analysis"],
                license="MIT",
                description="Vorgefertigtes Sentiment-Modell"
            ),
        ],
        
        # Fill-Mask Models
        "fill-mask": [
            ModelInfo(
                model_id="bert-base-uncased",
                name="BERT Base",
                size_mb=420,
                params_millions=110,
                min_ram_gb=2.0,
                min_vram_gb=1.0,
                recommended_for=["all"],
                license="Apache-2.0",
                description="Standard Mask-Filling Modell"
            ),
            ModelInfo(
                model_id="albert-base-v2",
                name="ALBERT Base v2",
                size_mb=45,
                params_millions=12,
                min_ram_gb=1.0,
                min_vram_gb=0.5,
                recommended_for=["mobile", "edge", "low-resource"],
                license="Apache-2.0",
                description="Ultraleichtes Mask-Filling Modell"
            ),
        ],
        
        # Question Answering Models
        "question-answering": [
            ModelInfo(
                model_id="distilbert-base-cased-distilled-squad",
                name="DistilBERT SQuAD",
                size_mb=250,
                params_millions=66,
                min_ram_gb=1.5,
                min_vram_gb=0.8,
                recommended_for=["all", "mobile"],
                license="Apache-2.0",
                description="Leichtes QA Modell"
            ),
            ModelInfo(
                model_id="deepset/roberta-base-squad2",
                name="RoBERTa SQuAD2",
                size_mb=480,
                params_millions=125,
                min_ram_gb=2.5,
                min_vram_gb=1.2,
                recommended_for=["desktop", "production"],
                license="Apache-2.0",
                description="Robustes QA Modell für Production"
            ),
        ],
        
        # Summarization Models
        "summarization": [
            ModelInfo(
                model_id="facebook/bart-large-cnn",
                name="BART Large CNN",
                size_mb=1600,
                params_millions=406,
                min_ram_gb=4.0,
                min_vram_gb=2.5,
                recommended_for=["desktop", "news"],
                license="Apache-2.0",
                description="Spezialisiert auf News-Zusammenfassung"
            ),
            ModelInfo(
                model_id="t5-small",
                name="T5 Small",
                size_mb=240,
                params_millions=60,
                min_ram_gb=1.5,
                min_vram_gb=0.8,
                recommended_for=["mobile", "edge", "general"],
                license="Apache-2.0",
                description="Vielseitiges kleines Summarization-Modell"
            ),
        ],
        
        # Translation Models
        "translation": [
            ModelInfo(
                model_id="Helsinki-NLP/opus-mt-en-de",
                name="OPUS-MT EN-DE",
                size_mb=300,
                params_millions=210,
                min_ram_gb=2.0,
                min_vram_gb=1.0,
                recommended_for=["desktop", "mobile"],
                license="Apache-2.0",
                description="Englisch-Deutsch Übersetzung"
            ),
            ModelInfo(
                model_id="facebook/marianMTModel",
                name="Marian MT",
                size_mb=1200,
                params_millions=310,
                min_ram_gb=3.5,
                min_vram_gb=2.0,
                recommended_for=["desktop", "multilingual"],
                license="Apache-2.0",
                description="Multilinguales Übersetzungsmodell"
            ),
        ],
        
        # Code Generation Models
        "code-generation": [
            ModelInfo(
                model_id="Salesforce/codegen-350M-mono",
                name="CodeGen 350M Mono",
                size_mb=700,
                params_millions=350,
                min_ram_gb=2.0,
                min_vram_gb=1.2,
                recommended_for=["mobile", "edge", "prototyping"],
                license="MIT",
                description="Leichtes Code-Modell für Python"
            ),
            ModelInfo(
                model_id="microsoft/phi-1_5",
                name="Phi-1.5",
                size_mb=2800,
                params_millions=1300,
                min_ram_gb=4.0,
                min_vram_gb=2.0,
                recommended_for=["desktop", "coding", "education"],
                license="MIT",
                description="Spezialisiert auf Code-Verständnis"
            ),
            ModelInfo(
                model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                name="Qwen2.5 Coder 7B",
                size_mb=13500,
                params_millions=7200,
                min_ram_gb=14.0,
                min_vram_gb=6.0,
                recommended_for=["desktop-gpu", "professional", "production"],
                license="Apache-2.0",
                description="Professionelles Code-Generierungsmodell"
            ),
            ModelInfo(
                model_id="deepseek-coder-6.7b-instruct",
                name="DeepSeek Coder 6.7B",
                size_mb=12800,
                params_millions=6700,
                min_ram_gb=13.0,
                min_vram_gb=6.0,
                recommended_for=["desktop-gpu", "coding", "enterprise"],
                license="Apache-2.0",
                description="Leistungsstarkes Coding-Modell"
            ),
        ],
        
        # Vision-Language Models
        "vision-language": [
            ModelInfo(
                model_id="microsoft/Florence-2-base",
                name="Florence-2 Base",
                size_mb=2300,
                params_millions=230,
                min_ram_gb=4.0,
                min_vram_gb=2.5,
                recommended_for=["desktop", "vision", "multimodal"],
                license="MIT",
                description="Vision-Language Modell von Microsoft"
            ),
            ModelInfo(
                model_id="Salesforce/blip-image-captioning-base",
                name="BLIP Base",
                size_mb=770,
                params_millions=170,
                min_ram_gb=2.5,
                min_vram_gb=1.5,
                recommended_for=["desktop", "captioning"],
                license="BSD-3-Clause",
                description="Bildbeschriftung in Echtzeit"
            ),
        ],
        
        # Embedding Models
        "embeddings": [
            ModelInfo(
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                name="MiniLM L6 v2",
                size_mb=80,
                params_millions=22,
                min_ram_gb=0.5,
                min_vram_gb=0.3,
                recommended_for=["all", "mobile", "edge", "search"],
                license="Apache-2.0",
                description="Ultraleichtes Embedding-Modell"
            ),
            ModelInfo(
                model_id="sentence-transformers/all-mpnet-base-v2",
                name="MPNet Base v2",
                size_mb=420,
                params_millions=110,
                min_ram_gb=2.0,
                min_vram_gb=1.0,
                recommended_for=["desktop", "search", "clustering"],
                license="Apache-2.0",
                description="High-Quality Embeddings"
            ),
            ModelInfo(
                model_id="BAAI/bge-small-en-v1.5",
                name="BGE Small v1.5",
                size_mb=130,
                params_millions=33,
                min_ram_gb=1.0,
                min_vram_gb=0.5,
                recommended_for=["mobile", "edge", "retrieval"],
                license="MIT",
                description="Effizientes Retrieval-Modell"
            ),
        ],
        
        # Speech Models
        "speech-recognition": [
            ModelInfo(
                model_id="openai/whisper-tiny",
                name="Whisper Tiny",
                size_mb=150,
                params_millions=39,
                min_ram_gb=1.0,
                min_vram_gb=0.5,
                recommended_for=["mobile", "edge", "real-time"],
                license="MIT",
                description="Echtzeit-Spracherkennung"
            ),
            ModelInfo(
                model_id="openai/whisper-base",
                name="Whisper Base",
                size_mb=280,
                params_millions=74,
                min_ram_gb=1.5,
                min_vram_gb=0.8,
                recommended_for=["desktop", "transcription"],
                license="MIT",
                description="Ausgewogene Spracherkennung"
            ),
            ModelInfo(
                model_id="openai/whisper-small",
                name="Whisper Small",
                size_mb=960,
                params_millions=244,
                min_ram_gb=2.5,
                min_vram_gb=1.5,
                recommended_for=["desktop-gpu", "production"],
                license="MIT",
                description="Produktionsreife Spracherkennung"
            ),
        ],
        
        # Community Favorites (No Auth Required)
        "community-favorites": [
            ModelInfo(
                model_id="HuggingFaceH4/zephyr-7b-beta",
                name="Zephyr 7B Beta",
                size_mb=14000,
                params_millions=7300,
                min_ram_gb=14.0,
                min_vram_gb=6.0,
                recommended_for=["desktop-gpu", "chat", "community"],
                license="Apache-2.0",
                description="Community-favorisiertes Chat-Modell"
            ),
            ModelInfo(
                model_id="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                name="Dolphin Mixtral 8x7B",
                size_mb=26000,
                params_millions=46700,
                min_ram_gb=52.0,
                min_vram_gb=20.0,
                recommended_for=["cluster", "multi-gpu", "uncensored"],
                license="Apache-2.0",
                description="Unzensiertes Mixture-of-Experts Modell"
            ),
            ModelInfo(
                model_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                name="TinyLlama GGUF",
                size_mb=650,
                params_millions=1100,
                min_ram_gb=2.0,
                min_vram_gb=0.0,
                recommended_for=["cpu-only", "mobile", "quantized"],
                license="Apache-2.0",
                description="Quantisierte Version für CPU"
            ),
        ]
    }
    
    def __init__(self, hardware_specs: HardwareSpecs):
        self.specs = hardware_specs
    
    def suggest_models(self, task_type: str = "text-generation", 
                       limit: int = 5) -> List[ModelInfo]:
        """
        Schlägt passende Modelle basierend auf Hardware vor
        
        Args:
            task_type: Art der Aufgabe (text-generation, classification, etc.)
            limit: Maximale Anzahl zurückgegebener Vorschläge
            
        Returns:
            Liste von empfohlenen Modellen
        """
        if task_type not in self.AVAILABLE_MODELS:
            task_type = "text-generation"
        
        all_models = self.AVAILABLE_MODELS[task_type]
        compatible_models = []
        
        for model in all_models:
            if self._is_model_compatible(model):
                score = self._calculate_compatibility_score(model)
                compatible_models.append((score, model))
        
        # Nach Kompatibilität sortieren (höchster Score zuerst)
        compatible_models.sort(key=lambda x: x[0], reverse=True)
        
        return [model for score, model in compatible_models[:limit]]
    
    def _is_model_compatible(self, model: ModelInfo) -> bool:
        """Prüft ob ein Modell auf der aktuellen Hardware laufen kann"""
        # Spezialfall: GGUF/CPU-only Modelle können immer auf CPU laufen
        if "cpu-only" in model.recommended_for or "quantized" in model.recommended_for:
            if self.specs.ram_available >= model.min_ram_gb:
                return True
        
        # RAM Check
        if self.specs.ram_available < model.min_ram_gb:
            return False
        
        # GPU VRAM Check (wenn GPU verfügbar)
        if self.specs.gpu_available:
            if self.specs.gpu_vram < model.min_vram_gb:
                # CPU Fallback möglich?
                if self.specs.ram_available < model.min_ram_gb * 1.5:
                    return False
        else:
            # Nur CPU - benötigt mehr RAM für unquantisierte Modelle
            if "quantized" not in model.recommended_for and "cpu-only" not in model.recommended_for:
                if self.specs.ram_available < model.min_ram_gb * 1.5:
                    return False
        
        return True
    
    def _calculate_compatibility_score(self, model: ModelInfo) -> float:
        """Berechnet einen Kompatibilitäts-Score (höher = besser)"""
        score = 0.0
        
        # RAM Headroom
        ram_ratio = self.specs.ram_available / model.min_ram_gb
        score += min(ram_ratio, 3.0) * 10
        
        # GPU Bonus
        if self.specs.gpu_available:
            if self.specs.gpu_vram >= model.min_vram_gb:
                vram_ratio = self.specs.gpu_vram / model.min_vram_gb
                score += min(vram_ratio, 2.0) * 15
        
        # Platform recommendations
        if self.specs.is_mobile and "mobile" in model.recommended_for:
            score += 20
        elif self.specs.is_colab and "colab-free" in model.recommended_for:
            score += 20
        elif not self.specs.is_mobile and "desktop" in model.recommended_for:
            score += 15
        
        # Size preference (kleinere Modelle bevorzugt für schnellere Experimente)
        size_penalty = model.params_millions / 1000
        score -= size_penalty
        
        return score
    
    def get_best_model(self, task_type: str = "text-generation") -> Optional[ModelInfo]:
        """Gibt das beste kompatible Modell zurück"""
        suggestions = self.suggest_models(task_type, limit=1)
        return suggestions[0] if suggestions else None
    
    def print_suggestions(self, task_type: str = "text-generation"):
        """Druckt Modellauswahl-Vorschläge"""
        print("\n" + "=" * 70)
        print(f"UARF MODEL SUGGESTIONS für {task_type}")
        print("=" * 70)
        
        suggestions = self.suggest_models(task_type)
        
        if not suggestions:
            print("Keine kompatiblen Modelle gefunden!")
            print("Versuchen Sie:")
            print("  - Mehr RAM freigeben")
            print("  - Kleinere Modelle")
            print("  - Cloud-Ressourcen (Colab, Cluster)")
            return
        
        for i, model in enumerate(suggestions, 1):
            print(f"\n{i}. {model.name}")
            print(f"   ID: {model.model_id}")
            print(f"   Größe: {model.size_mb:.0f} MB ({model.params_millions}M Parameter)")
            print(f"   Mindestanforderungen: {model.min_ram_gb:.1f} GB RAM, {model.min_vram_gb:.1f} GB VRAM")
            print(f"   Lizenz: {model.license}")
            print(f"   Beschreibung: {model.description}")
            print(f"   Empfohlen für: {', '.join(model.recommended_for)}")
        
        print("\n" + "=" * 70)
        print(f"Tipp: Verwenden Sie 'uarf run --model {suggestions[0].model_id}'")
        print("=" * 70)
