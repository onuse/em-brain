"""
Drives package - Multi-drive motivation system for emergent robot behavior.

This package provides a clean, extensible architecture for robot motivations
where multiple drives compete and cooperate to guide behavior.
"""

from .base_motivator import BaseMotivator, MotivatorContext, ActionEvaluation
from .motivation_system import MotivationSystem, MotivationResult, create_default_motivation_system
from .curiosity_motivator import CuriosityMotivator
from .mastery_motivator import MasteryMotivator
from .survival_motivator import SurvivalMotivator

__all__ = [
    'BaseMotivator',
    'MotivatorContext', 
    'ActionEvaluation',
    'MotivationSystem',
    'MotivationResult',
    'create_default_motivation_system',
    'CuriosityMotivator',
    'MasteryMotivator',
    'SurvivalMotivator'
]