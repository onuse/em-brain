"""
Simple Dimension Calculator

Radical simplification: All properties emerge from a unified 4D field.
No semantic encoding in tensor structure - just pure capacity.
"""

from typing import List, Dict, Any, Tuple
import torch


class SimpleDimensionCalculator:
    """
    Ultra-simple dimension calculator for GPU-optimized performance.
    
    Instead of complex semantic mappings, we use a simple 4D tensor where
    all cognitive properties emerge from the dynamics.
    """
    
    def __init__(self):
        # Standard spatial resolution for all brains
        self.spatial_resolution = 32  # Reduced for better performance
        
    def calculate_tensor_shape(self, sensory_dim: int, motor_dim: int) -> Tuple[List[int], int]:
        """
        Calculate simple 4D tensor shape optimized for GPU.
        
        Args:
            sensory_dim: Number of sensors (ignored - we use fixed capacity)
            motor_dim: Number of motors (ignored - we use fixed capacity)
            
        Returns:
            Tuple of (tensor_shape, conceptual_dimensions)
        """
        # Simple 4D tensor that works efficiently on all hardware
        # Total elements: 32^3 * 64 = 2,097,152 (8MB in float32)
        tensor_shape = [
            self.spatial_resolution,  # Spatial X
            self.spatial_resolution,  # Spatial Y  
            self.spatial_resolution,  # Spatial Z
            64  # Feature/pattern dimension
        ]
        
        # We still report this as "26D conceptual" for compatibility
        # but it's really just a 4D tensor where properties emerge
        conceptual_dimensions = 26
        
        print(f"📐 Simple tensor: {sensory_dim}D sensors → {tensor_shape} tensor")
        print(f"   Total elements: {self._calculate_elements(tensor_shape):,}")
        print(f"   Memory usage: {self._calculate_memory_mb(tensor_shape):.1f}MB")
        
        return tensor_shape, conceptual_dimensions
    
    def create_dimension_mapping(self) -> Dict[str, Any]:
        """
        Create simplified dimension mapping.
        
        Since we're not encoding semantics in structure, this is mostly
        for compatibility with existing code.
        """
        return {
            'conceptual_to_tensor': {},  # No complex mapping needed
            'family_tensor_ranges': {},  # No family separation
            'tensor_shape': None  # Will be set by brain
        }
    
    def _calculate_elements(self, shape: List[int]) -> int:
        """Calculate total elements in tensor."""
        total = 1
        for dim in shape:
            total *= dim
        return total
    
    def _calculate_memory_mb(self, shape: List[int]) -> float:
        """Calculate memory usage in MB (assuming float32)."""
        return (self._calculate_elements(shape) * 4) / (1024 * 1024)
    
    def get_device_recommendation(self, tensor_shape: List[int]) -> torch.device:
        """
        Recommend device based on tensor shape.
        
        Our 4D tensors should work well on all devices!
        """
        # Check if MPS is available (Apple Silicon)
        if torch.backends.mps.is_available():
            return torch.device('mps')
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            return torch.device('cuda')
        
        # Fallback to CPU
        return torch.device('cpu')