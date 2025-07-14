"""
Minimal Brain Implementation

A distilled implementation of the core brain following MINIMAL.md principles:
- Experience Storage: Raw sensory-motor experiences
- Similarity Search: Fast nearest-neighbor retrieval
- Activation Dynamics: Neural spreading activation  
- Prediction Engine: Pattern-based action generation

Everything else emerges from these 4 systems.
"""

__version__ = "0.1.0"
__author__ = "Robot Brain Project"

from .brain import MinimalBrain
from .experience import Experience, ExperienceStorage
from .similarity import SimilarityEngine
from .activation import ActivationDynamics
from .prediction import PredictionEngine

__all__ = [
    "MinimalBrain",
    "Experience", 
    "ExperienceStorage",
    "SimilarityEngine",
    "ActivationDynamics", 
    "PredictionEngine"
]