"""
Persistence System for Dynamic Brain Architecture

This module provides cross-session learning continuity through:
- Automatic state saves during operation
- Recovery on startup
- Crash resilience
- Session tracking
"""

# Integrated persistence for dynamic brains
from .integrated_persistence import (
    IntegratedPersistence,
    initialize_persistence,
    get_persistence
)
from .dynamic_persistence_adapter import (
    DynamicPersistenceAdapter,
    DynamicBrainState
)

__all__ = [
    'IntegratedPersistence',
    'initialize_persistence', 
    'get_persistence',
    'DynamicPersistenceAdapter',
    'DynamicBrainState'
]