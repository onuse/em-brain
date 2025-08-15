"""Visualization components."""
from .renderer import SimulationRenderer

# Try to import 3D renderer (requires PyOpenGL)
try:
    from .renderer3d import Renderer3D
    __all__ = ['SimulationRenderer', 'Renderer3D']
except ImportError:
    __all__ = ['SimulationRenderer']
    print("Note: 3D renderer not available (requires PyOpenGL)")