"""
UARF Data Package - Dataset Loading and Processing Utilities

Provides unified interface for loading datasets from various sources:
- Local files (JSON, JSONL, TXT)
- HuggingFace Hub
- Custom data generators
"""

from .local_loader import load_local_dataset, create_dataset_from_files
from .test_dataset import create_test_dataset, generate_sample_data

__all__ = [
    "load_local_dataset",
    "create_dataset_from_files",
    "create_test_dataset",
    "generate_sample_data",
]