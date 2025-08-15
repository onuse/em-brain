"""
Hardware Interface Definitions

Abstract interfaces that define the contract between brainstem control
and hardware implementations. All hardware drivers implement these interfaces.
"""

from .vocal_interface import VocalInterface, VocalParameters, VocalSafetyConstraints, EmotionalVocalMapper

__all__ = [
    'VocalInterface',
    'VocalParameters', 
    'VocalSafetyConstraints',
    'EmotionalVocalMapper'
]