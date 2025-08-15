"""
Parameter Systems - Emergent Parameter Management

This package contains all parameter-related systems:
- Emergent parameters derived from constraints
- Cognitive constants and capacity limits
"""

from .constraint_parameters import EmergentParameterSystem, HardwareProfile, BiologicalConstraints, ConstraintType
from .cognitive_constants import *

__all__ = [
    'EmergentParameterSystem',
    'HardwareProfile',
    'BiologicalConstraints', 
    'ConstraintType'
]