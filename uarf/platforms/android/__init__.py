"""
UARF Android Platform Adapter

Provides Android-specific functionality for UARF:
- Termux environment detection
- Android hardware capabilities
- Mobile-optimized configurations
- ADB deployment support
"""

from .adapter import AndroidAdapter

__all__ = ["AndroidAdapter"]