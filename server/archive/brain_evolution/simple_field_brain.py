"""
Simple Field Brain - A clean implementation for dynamic dimensions.

This is a simplified field brain that properly implements the IBrain interface
and supports truly dynamic dimensions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import time

from .interfaces import IBrain


class SimpleFieldBrain(IBrain):
    """
    A simple field brain that supports dynamic dimensions.
    
    This implementation focuses on the core concept of field dynamics
    without the complexity of the original UnifiedFieldBrain.
    """
    
    def __init__(self, field_dimensions: int, spatial_resolution: int,
                 sensory_dim: int, motor_dim: int):
        """
        Initialize a simple field brain with dynamic dimensions.
        
        Args:
            field_dimensions: Number of field dimensions
            spatial_resolution: Spatial resolution for the field
            sensory_dim: Expected sensory input dimensions
            motor_dim: Expected motor output dimensions
        """
        self.field_dimensions = field_dimensions
        self.spatial_resolution = spatial_resolution
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Create the unified field - simple 4D tensor
        # (field_dimensions, x, y, z)
        self.field = torch.zeros(
            (field_dimensions, spatial_resolution, spatial_resolution, spatial_resolution),
            device=self.device,
            dtype=torch.float32
        )
        
        # Initialize with small random values
        self.field += torch.randn_like(self.field) * 0.01
        
        # Field dynamics parameters
        self.decay_rate = 0.99
        self.diffusion_rate = 0.1
        self.activation_threshold = 0.1
        
        # Statistics
        self.total_cycles = 0
        self.creation_time = time.time()
    
    def process_field_dynamics(self, field_input: torch.Tensor) -> torch.Tensor:
        """
        Process field dynamics and return field output.
        
        This is the core method that evolves the field based on input.
        """
        # Ensure input is on correct device
        if not isinstance(field_input, torch.Tensor):
            field_input = torch.tensor(field_input, device=self.device, dtype=torch.float32)
        else:
            field_input = field_input.to(self.device)
        
        # 1. Apply input to field (simple additive influence)
        # Distribute input across spatial dimensions
        influence = field_input.view(-1, 1, 1, 1)
        
        # Create spatial distribution (center-weighted)
        center = self.spatial_resolution // 2
        spatial_weight = torch.zeros_like(self.field[0])
        spatial_weight[center, center, center] = 1.0
        
        # Smooth the spatial weight
        for _ in range(2):
            spatial_weight = self._smooth_3d(spatial_weight)
        
        # Apply influence to field
        for i in range(min(len(field_input), self.field_dimensions)):
            self.field[i] += influence[i] * spatial_weight * 0.1
        
        # 2. Field evolution dynamics
        
        # Decay
        self.field *= self.decay_rate
        
        # Diffusion
        for i in range(self.field_dimensions):
            self.field[i] = self._smooth_3d(self.field[i])
        
        # Nonlinear activation
        self.field = torch.tanh(self.field)
        
        # 3. Extract output from field
        # Take the maximum activation from each field dimension
        field_output = torch.zeros(self.field_dimensions, device=self.device)
        for i in range(self.field_dimensions):
            field_output[i] = self.field[i].max()
        
        self.total_cycles += 1
        
        return field_output
    
    def _smooth_3d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply 3D smoothing (simple diffusion)."""
        # Create a simple 3x3x3 averaging kernel
        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device) / 27.0
        
        # Add batch and channel dimensions
        x = tensor.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution with padding
        smoothed = torch.nn.functional.conv3d(x, kernel, padding=1)
        
        # Remove extra dimensions
        return smoothed.squeeze(0).squeeze(0)
    
    def get_field_dimensions(self) -> int:
        """Get number of field dimensions."""
        return self.field_dimensions
    
    def get_state(self) -> Dict[str, Any]:
        """Get brain state for persistence."""
        return {
            'field_dimensions': self.field_dimensions,
            'spatial_resolution': self.spatial_resolution,
            'sensory_dim': self.sensory_dim,
            'motor_dim': self.motor_dim,
            'field': self.field.cpu().numpy().tolist(),
            'total_cycles': self.total_cycles,
            'creation_time': self.creation_time
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load brain state from persistence."""
        if 'field' in state:
            field_data = torch.tensor(state['field'], device=self.device)
            if field_data.shape == self.field.shape:
                self.field = field_data
        
        if 'total_cycles' in state:
            self.total_cycles = state['total_cycles']
    
    def perform_maintenance(self):
        """Perform field maintenance operations."""
        # Decay old activations slightly
        self.field *= 0.995
        
        # Clear very small activations to save computation
        self.field[self.field < 0.001] = 0.0
        
    def cleanup_memory(self):
        """Clean up memory usage."""
        # Force gradient cleanup if not training
        if not self.field.requires_grad:
            self.field = self.field.detach()
            
        # Clear gradient cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()