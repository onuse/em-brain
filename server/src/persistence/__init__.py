"""
Persistence System for Dynamic Brain Architecture

This module provides cross-session learning continuity through:
- Automatic state saves during operation
- Recovery on startup
- Crash resilience
- Session tracking

Two implementations available:
1. Legacy system (PersistenceManager) - for old BrainFactory architecture
2. Integrated system (IntegratedPersistence) - for new dynamic brain architecture
"""

# Legacy persistence (kept for compatibility)
from .persistence_manager import PersistenceManager
from .brain_serializer import BrainSerializer
from .recovery_manager import RecoveryManager
from .persistence_config import PersistenceConfig, ConsolidationPolicy

# New integrated persistence for dynamic brains
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
    # Legacy
    'PersistenceManager',
    'BrainSerializer', 
    'RecoveryManager',
    'PersistenceConfig',
    'ConsolidationPolicy',
    # New
    'IntegratedPersistence',
    'initialize_persistence', 
    'get_persistence',
    'DynamicPersistenceAdapter',
    'DynamicBrainState'
]