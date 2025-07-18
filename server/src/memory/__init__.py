"""
Memory Systems - Unified Memory Architecture

This package contains all memory-related systems:
- Universal memory for any modality
"""

from .pattern_memory import UniversalMemorySystem, MemoryPattern

__all__ = [
    'UniversalMemorySystem',
    'MemoryPattern'
]