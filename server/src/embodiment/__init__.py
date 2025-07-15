"""
Embodied Free Energy System

A biologically accurate implementation of the Free Energy Principle for robotic systems.
Replaces hardcoded motivations with physics-grounded action selection that emerges
from minimizing prediction error across embodied priors.

Core Components:
- EmbodiedFreeEnergySystem: Main action selection system
- EmbodiedPriorSystem: Manages hardware-grounded priors
- HardwareInterface: Abstract interface for robot hardware
- EmbodiedBrainAdapter: Integrates with 4-system brain

Usage:
    from embodied_free_energy import EmbodiedFreeEnergySystem, EmbodiedBrainAdapter
    
    # Create brain adapter
    brain_adapter = EmbodiedBrainAdapter(your_4_system_brain)
    
    # Create embodied system
    embodied_system = EmbodiedFreeEnergySystem(brain_adapter)
    
    # Use for action selection
    action = embodied_system.select_action(current_sensory_input)
"""

from .base import (
    EmbodiedPrior,
    HardwareTelemetry,
    HardwareInterface,
    MockHardwareInterface,
    ActionProposal,
    EmbodiedPriorSystem
)

from .system import EmbodiedFreeEnergySystem

from .brain_adapter import (
    EmbodiedBrainAdapter,
    PredictionOutcome,
    StateExtractor
)

__all__ = [
    # Core system
    'EmbodiedFreeEnergySystem',
    
    # Brain integration
    'EmbodiedBrainAdapter',
    'PredictionOutcome',
    'StateExtractor',
    
    # Hardware interface
    'HardwareInterface',
    'MockHardwareInterface',
    'HardwareTelemetry',
    
    # Embodied priors
    'EmbodiedPrior',
    'EmbodiedPriorSystem',
    'ActionProposal'
]

# Version info
__version__ = "1.0.0"
__author__ = "Robot Brain Project"
__description__ = "Embodied Free Energy Principle for robotic intelligence"