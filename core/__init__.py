"""
Core module for the emergent intelligence robot brain.
Contains the fundamental data structures and algorithms for learning and experience management.
"""

from .experience_node import ExperienceNode
from .world_graph import WorldGraph
from .mental_context import MentalContext
from .communication import PredictionPacket, SensoryPacket

__all__ = [
    'ExperienceNode',
    'WorldGraph', 
    'MentalContext',
    'PredictionPacket',
    'SensoryPacket'
]