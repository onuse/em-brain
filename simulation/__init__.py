"""
Grid world simulation module for testing the emergent intelligence brain.
Provides a virtual environment that simulates sensors and actuators.
"""

from .grid_world import GridWorldSimulation, Robot
from .brainstem_sim import GridWorldBrainstem

__all__ = [
    'GridWorldSimulation',
    'Robot', 
    'GridWorldBrainstem'
]