"""
Activation Dynamics Subsystem

Neural-like spreading activation through related memories:
- Recently accessed experiences stay "hot"
- Activation spreads to connected experiences
- Natural decay creates working memory effects
- Most activated experiences influence decisions
"""

from .dynamics import ActivationDynamics

__all__ = ["ActivationDynamics"]