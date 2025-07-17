"""
Minimal Vector Stream Brain Implementation

A biologically-realistic implementation using vector streams:
- Continuous processing replaces discrete experience packages
- Modular streams (sensory, motor, temporal) with cross-stream learning
- Organic temporal metronome for biological timing
- Emergent intelligence from stream dynamics

Everything emerges from continuous vector flow.
"""

__version__ = "0.2.0"
__author__ = "Robot Brain Project"

from .brain import MinimalBrain
from .vector_stream.minimal_brain import MinimalVectorStreamBrain

__all__ = [
    "MinimalBrain",
    "MinimalVectorStreamBrain"
]