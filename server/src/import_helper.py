"""
Import Helper for Minimal Brain Demos

Handles the relative import issues when running demos directly.
"""

import sys
import os

# Add the parent directory to sys.path so we can import minimal as a package
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import the minimal brain systems
from minimal.brain import MinimalBrain
from minimal.communication.client import MinimalBrainClient

__all__ = ['MinimalBrain', 'MinimalBrainClient']