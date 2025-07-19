"""
Production-Grade Brain Persistence Subsystem

A comprehensive system for maintaining brain state across sessions with:
- Incremental saves containing full brain content
- Automatic consolidation of incremental files
- Robust recovery from consolidated + incremental state
- Crash resistance and corruption handling

Main API:
- PersistenceManager: Central coordinator
- BrainSerializer: Extract/restore patterns from vector streams
- RecoveryManager: Boot sequence and crash recovery
"""

from .persistence_manager import PersistenceManager
from .brain_serializer import BrainSerializer
from .recovery_manager import RecoveryManager
from .persistence_config import PersistenceConfig, ConsolidationPolicy

__all__ = [
    'PersistenceManager',
    'BrainSerializer', 
    'RecoveryManager',
    'PersistenceConfig',
    'ConsolidationPolicy'
]