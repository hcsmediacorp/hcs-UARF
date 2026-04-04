"""
UARF Local Dataset Loader
Lädt lokale JSON/JSONL-Datensätze ohne HuggingFace-Abhängigkeit
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset


def load_local_dataset(
    path: str,
    split: str = "train",
    text_column: str = "text"
) -> Dataset:
    """
    Lädt einen lokalen JSON/JSONL-Datensatz
    
    Args:
        path: Pfad zur Datei oder Verzeichnis
        split: Dataset-Split ('train', 'validation', 'test')
        text_column: Name der Text-Spalte
        
    Returns:
        HuggingFace Dataset Objekt
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    
    # JSONL Datei
    if path.suffix == '.jsonl':
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    
    # JSON Datei
    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # Verzeichnis mit JSON Dateien
    elif path.is_dir():
        data = []
        # Suche nach JSON Dateien für den gewünschten Split
        split_files = [
            path / f"{split}.json",
            path / f"{split}s.json",  # trains, validations, etc.
        ]
        
        for file_path in split_files:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        data.extend(loaded)
                    else:
                        data.append(loaded)
                break
        
        # Falls keine Split-spezifische Datei gefunden, lade alle JSON Dateien
        if not data:
            for json_file in path.glob("*.json"):
                if json_file.name != 'dataset_summary.json':
                    with open(json_file, 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                        if isinstance(loaded, list):
                            data.extend(loaded)
                        else:
                            data.append(loaded)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    # Stelle sicher, dass Daten im richtigen Format sind
    if not data:
        raise ValueError(f"No data loaded from {path}")
    
    # Konvertiere zu HuggingFace Dataset
    dataset = Dataset.from_list(data)
    
    return dataset


def create_dataset_from_files(
    train_path: str,
    val_path: Optional[str] = None,
    text_column: str = "text"
) -> tuple:
    """
    Erstellt Trainings- und Validierungs-Datasets aus lokalen Dateien
    
    Args:
        train_path: Pfad zum Trainings-Dataset
        val_path: Optionaler Pfad zum Validierungs-Dataset
        text_column: Name der Text-Spalte
        
    Returns:
        Tuple von (train_dataset, val_dataset)
    """
    train_dataset = load_local_dataset(train_path, split="train", text_column=text_column)
    
    if val_path:
        val_dataset = load_local_dataset(val_path, split="validation", text_column=text_column)
    else:
        val_dataset = None
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test des Loaders
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        dataset = load_local_dataset(path)
        print(f"✅ Dataset geladen: {len(dataset)} Samples")
        print(f"   Spalten: {dataset.column_names}")
        if len(dataset) > 0:
            print(f"   Beispiel: {dataset[0]}")
    else:
        print("Usage: python -m uarf.data.local_loader <path_to_dataset>")
