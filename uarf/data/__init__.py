"""
UARF Data Package - Dataset Loading and Processing Utilities

Provides unified interface for loading datasets from various sources:
- Local files (JSON, JSONL, TXT)
- HuggingFace Hub
- Custom data generators
"""

from .local_loader import load_local_dataset, create_dataset_from_files
from .test_dataset import create_test_datasets

# Backward compatibility aliases
def create_test_dataset(output_dir: str = "./test_data"):
    """Alias for create_test_datasets for backward compatibility."""
    return create_test_datasets(output_dir=output_dir)

def generate_sample_data(output_dir: str = "./test_data", **kwargs):
    """Alias for create_test_datasets for backward compatibility."""
    return create_test_datasets(output_dir=output_dir, **kwargs)

__all__ = [
    "load_local_dataset",
    "create_dataset_from_files",
    "create_test_dataset",
    "generate_sample_data",
    "create_test_datasets",
]