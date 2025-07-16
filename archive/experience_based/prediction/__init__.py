"""
Prediction Engine Subsystem

Generates next action by following patterns in activated memories:
- Look at what happened next in similar past situations
- Weight by activation levels and prediction accuracy
- Return consensus action from multiple similar experiences
- Store new experience when outcome arrives
"""

from .engine import PredictionEngine

__all__ = ["PredictionEngine"]