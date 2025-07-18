"""
Organized Brain Architecture

A constraint-based artificial brain with modular architecture:
- Vector streams for core intelligence (sensory, motor, temporal)
- Attention systems for selective focus and binding
- Memory systems for pattern storage and retrieval
- Parameter systems for emergent constraint-based behavior
- Maintenance systems for system health and optimization

Everything emerges from hardware constraints and biological principles.
"""

__version__ = "0.3.0"
__author__ = "Robot Brain Project"

# Core brain systems
from .brain import MinimalBrain
from .vector_stream.vector_stream_brain import MinimalVectorStreamBrain
from .vector_stream.goldilocks_brain import GoldilocksBrain

# Modular systems
from .attention import UniversalAttentionSystem, CrossModalAttentionSystem
from .memory import UniversalMemorySystem, MemoryPattern
from .parameters import EmergentParameterSystem, ConstraintType

__all__ = [
    # Core brain systems
    "MinimalBrain",
    "MinimalVectorStreamBrain", 
    "GoldilocksBrain",
    
    # Attention systems
    "UniversalAttentionSystem",
    "CrossModalAttentionSystem",
    
    # Memory systems
    "UniversalMemorySystem",
    "MemoryPattern",
    
    # Parameter systems
    "EmergentParameterSystem",
    "ConstraintType"
]