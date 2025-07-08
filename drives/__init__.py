"""
Drives package - Multi-drive motivation system for emergent robot behavior.

This package provides a clean, extensible architecture for robot motivations
where multiple drives compete and cooperate to guide behavior.
"""

from .base_drive import BaseDrive, DriveContext, ActionEvaluation
from .motivation_system import MotivationSystem, MotivationResult, create_default_motivation_system
from .curiosity_drive import CuriosityDrive
from .survival_drive import SurvivalDrive
from .exploration_drive import ExplorationDrive

__all__ = [
    'BaseDrive',
    'DriveContext', 
    'ActionEvaluation',
    'MotivationSystem',
    'MotivationResult',
    'create_default_motivation_system',
    'CuriosityDrive',
    'SurvivalDrive', 
    'ExplorationDrive'
]