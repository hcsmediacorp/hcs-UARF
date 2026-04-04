"""
UARF Test Dataset Generator
Erstellt synthetische Trainingsdaten für Tests und Demos
"""

import os
import random
import json
from pathlib import Path
from typing import List, Dict, Any


class TestDatasetGenerator:
    """Generiert synthetische Text-Datensätze für UARF Tests"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # Einfache Templates für synthetischen Text
        self.templates = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models can learn from data.",
            "Python is a popular programming language.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret images.",
            "Data science combines statistics and programming.",
            "Algorithms solve problems step by step.",
            "Neural networks are inspired by the human brain.",
            "Training models requires large amounts of data.",
            "Gradient descent optimizes model parameters.",
            "Overfitting occurs when models memorize training data.",
            "Regularization helps prevent overfitting.",
            "Batch normalization speeds up training.",
            "Attention mechanisms focus on relevant information.",
            "Transformers revolutionized natural language processing.",
            "Language models predict the next word in a sequence.",
            "Embeddings represent words as vectors.",
            "Tokenization splits text into smaller units.",
        ]
        
        # Sätze für Variation
        self.subjects = [
            "The researcher", "The model", "The algorithm", "The system",
            "The network", "The dataset", "The experiment", "The team",
            "Scientists", "Engineers", "Developers", "Users"
        ]
        
        self.verbs = [
            "analyzes", "processes", "generates", "optimizes",
            "evaluates", "trains", "tests", "improves",
            "develops", "implements", "designs", "creates"
        ]
        
        self.objects = [
            "the data", "the results", "the predictions", "the output",
            "the features", "the patterns", "the trends", "the insights",
            "new solutions", "better models", "efficient algorithms"
        ]
    
    def generate_sentence(self) -> str:
        """Generiert einen zufälligen Satz"""
        if random.random() < 0.5:
            return random.choice(self.templates)
        else:
            subject = random.choice(self.subjects)
            verb = random.choice(self.verbs)
            obj = random.choice(self.objects)
            return f"{subject} {verb} {obj}."
    
    def generate_paragraph(self, num_sentences: int = 3) -> str:
        """Generiert einen Absatz mit mehreren Sätzen"""
        sentences = [self.generate_sentence() for _ in range(num_sentences)]
        return " ".join(sentences)
    
    def generate_dataset(
        self,
        num_samples: int = 1000,
        min_sentences: int = 2,
        max_sentences: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generiert einen kompletten Datensatz
        
        Args:
            num_samples: Anzahl der Textproben
            min_sentences: Mindestanzahl Sätze pro Probe
            max_sentences: Maximalanzahl Sätze pro Probe
            
        Returns:
            Liste von Dictionarys mit 'text' Schlüssel
        """
        dataset = []
        
        for i in range(num_samples):
            num_sentences = random.randint(min_sentences, max_sentences)
            text = self.generate_paragraph(num_sentences)
            
            dataset.append({
                'id': i,
                'text': text,
                'length': len(text),
                'sentences': num_sentences
            })
        
        return dataset
    
    def save_dataset(
        self,
        output_path: str,
        num_samples: int = 1000,
        format: str = 'json'
    ):
        """
        Speichert den generierten Datensatz
        
        Args:
            output_path: Pfad zur Ausgabedatei oder Verzeichnis
            num_samples: Anzahl der Textproben
            format: Ausgabeformat ('json', 'txt', 'jsonl')
        """
        dataset = self.generate_dataset(num_samples)
        output_path = Path(output_path)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        elif format == 'jsonl':
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        elif format == 'txt':
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(item['text'] + '\n\n')
        
        print(f"✅ Datensatz gespeichert: {output_path}")
        print(f"   Samples: {num_samples}")
        print(f"   Format: {format}")
        
        return dataset


def create_test_datasets(output_dir: str = "./test_data"):
    """
    Erstellt verschiedene Test-Datensätze für UARF
    
    Args:
        output_dir: Basisverzeichnis für Testdaten
    """
    generator = TestDatasetGenerator(seed=42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {}
    
    # Mini Dataset (sehr klein, für schnelle Tests)
    mini_data = generator.generate_dataset(num_samples=100)
    mini_path = output_dir / "mini_dataset.json"
    with open(mini_path, 'w') as f:
        json.dump(mini_data, f, indent=2)
    datasets['mini'] = str(mini_path)
    print(f"✓ Mini-Dataset: {len(mini_data)} Samples")
    
    # Small Dataset (für Demo-Training)
    small_data = generator.generate_dataset(num_samples=500)
    small_path = output_dir / "small_dataset.json"
    with open(small_path, 'w') as f:
        json.dump(small_data, f, indent=2)
    datasets['small'] = str(small_path)
    print(f"✓ Small-Dataset: {len(small_data)} Samples")
    
    # Medium Dataset (für ausführliche Tests)
    medium_data = generator.generate_dataset(num_samples=2000)
    medium_path = output_dir / "medium_dataset.json"
    with open(medium_path, 'w') as f:
        json.dump(medium_data, f, indent=2)
    datasets['medium'] = str(medium_path)
    print(f"✓ Medium-Dataset: {len(medium_data)} Samples")
    
    # JSONL Format (für effizientes Laden)
    jsonl_data = generator.generate_dataset(num_samples=1000)
    jsonl_path = output_dir / "dataset.jsonl"
    with open(jsonl_path, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')
    datasets['jsonl'] = str(jsonl_path)
    print(f"✓ JSONL-Dataset: {len(jsonl_data)} Samples")
    
    # Train/Validation Split
    train_data = generator.generate_dataset(num_samples=800)
    val_data = generator.generate_dataset(num_samples=200)
    
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    datasets['train'] = str(train_path)
    datasets['val'] = str(val_path)
    print(f"✓ Train/Val Split: {len(train_data)}/{len(val_data)} Samples")
    
    # Zusammenfassung speichern
    summary = {
        'created_by': 'UARF TestDatasetGenerator',
        'seed': 42,
        'datasets': datasets,
        'total_samples': sum(len(d) if isinstance(d, list) else 0 
                           for d in [mini_data, small_data, medium_data, jsonl_data, train_data, val_data])
    }
    
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Alle Datensätze gespeichert in: {output_dir.absolute()}")
    print(f"   Gesamte Samples: {summary['total_samples']}")
    
    return datasets


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./test_data"
    create_test_datasets(output_dir)
