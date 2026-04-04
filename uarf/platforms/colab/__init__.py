"""
UARF Google Colab Platform Adapter

Provides Colab-specific functionality for UARF:
- GPU/TPU detection and configuration
- Drive mounting
- Colab-optimized settings
- Session management
"""

from .adapter import ColabAdapter

__all__ = ["ColabAdapter"]