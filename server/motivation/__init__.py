"""
Motivation System - Action Selection Layer

Provides competitive action selection on top of the 4-system minimal brain.
Motivations propose actions; best proposal wins execution.
"""

from .base import BaseMotivation, ActionProposal
from .system import MotivationSystem
from .standard import CuriosityMotivation, EnergyMotivation, SafetyMotivation

__all__ = [
    'BaseMotivation',
    'ActionProposal', 
    'MotivationSystem',
    'CuriosityMotivation',
    'EnergyMotivation',
    'SafetyMotivation'
]