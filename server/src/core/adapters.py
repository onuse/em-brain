"""
Adapter implementations for robot-brain translation.

These adapters handle the conversion between robot-specific sensory/motor
spaces and the abstract unified field space of the brain.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

from .interfaces import (
    ISensoryAdapter, IMotorAdapter, IAdapterFactory,
    Robot, SensorChannel, MotorChannel
)


class SensoryAdapter(ISensoryAdapter):
    """
    Translates robot-specific sensors to abstract field space.
    
    This adapter creates a learnable mapping from the robot's sensory
    channels to the brain's unified field representation.
    """
    
    def __init__(self, robot: Robot, field_dimensions: int):
        self.robot = robot
        self.field_dimensions = field_dimensions
        self.sensory_dim = len(robot.sensory_channels)
        
        # Create projection matrix
        self.projection = self._create_projection()
        
        # Normalization parameters for each sensor channel
        self.normalizers = self._create_normalizers()
    
    def to_field_space(self, sensory: List[float]) -> torch.Tensor:
        """Convert robot sensory input to field coordinates."""
        
        # Validate input size
        if len(sensory) != self.sensory_dim:
            raise ValueError(f"Expected {self.sensory_dim} sensory values, got {len(sensory)}")
        
        # Normalize sensor values to [-1, 1] range
        normalized = torch.zeros(self.sensory_dim)
        for i, (value, channel) in enumerate(zip(sensory, self.robot.sensory_channels)):
            # Clamp to channel range
            value = max(channel.range_min, min(channel.range_max, value))
            
            # Normalize to [-1, 1]
            range_span = channel.range_max - channel.range_min
            if range_span > 0:
                normalized[i] = 2.0 * (value - channel.range_min) / range_span - 1.0
            else:
                normalized[i] = 0.0
        
        # Apply projection to field space
        field_coords = self.projection(normalized)
        
        # Apply activation to encourage sparsity and bounded values
        field_coords = torch.tanh(field_coords)
        
        return field_coords
    
    def get_field_dimensions(self) -> int:
        """Get expected field dimensions."""
        return self.field_dimensions
    
    def _create_projection(self) -> nn.Module:
        """
        Create projection from sensory to field space.
        
        This creates a structured mapping that preserves locality
        and semantic relationships where possible.
        """
        
        # For now, use a simple linear projection
        # In future, this could be more sophisticated (e.g., multi-layer)
        projection = nn.Linear(self.sensory_dim, self.field_dimensions, bias=True)
        
        # Initialize with structured weights
        with torch.no_grad():
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(projection.weight)
            
            # Group similar sensors to nearby field dimensions
            # This is a heuristic that can be improved
            if self.sensory_dim <= self.field_dimensions:
                # Spread sensors across field dimensions
                for i in range(self.sensory_dim):
                    field_idx = int(i * self.field_dimensions / self.sensory_dim)
                    projection.weight[field_idx, i] *= 2.0  # Strengthen direct mapping
            
            # Small bias initialization
            nn.init.constant_(projection.bias, 0.0)
        
        return projection
    
    def _create_normalizers(self) -> Dict[int, Dict[str, float]]:
        """Create normalization parameters for each sensor."""
        normalizers = {}
        
        for i, channel in enumerate(self.robot.sensory_channels):
            normalizers[i] = {
                'min': channel.range_min,
                'max': channel.range_max,
                'scale': 2.0 / (channel.range_max - channel.range_min) if channel.range_max > channel.range_min else 1.0
            }
        
        return normalizers


class MotorAdapter(IMotorAdapter):
    """
    Translates abstract field space to robot-specific motors.
    
    This adapter extracts motor commands from the brain's unified field
    representation and maps them to the robot's actuators.
    """
    
    def __init__(self, robot: Robot, field_dimensions: int):
        self.robot = robot
        self.field_dimensions = field_dimensions
        self.motor_dim = len(robot.motor_channels)
        
        # Create extraction matrix
        self.extraction = self._create_extraction()
        
        # Denormalization parameters for each motor channel
        self.denormalizers = self._create_denormalizers()
    
    def from_field_space(self, field_state: torch.Tensor) -> List[float]:
        """Convert field output to motor commands."""
        
        # Validate input size
        if field_state.shape[0] != self.field_dimensions:
            raise ValueError(f"Expected {self.field_dimensions}D field state, got {field_state.shape[0]}D")
        
        # Ensure field_state is on CPU for the linear layer
        if field_state.is_cuda or field_state.device.type == 'mps':
            field_state = field_state.cpu()
        
        # Extract motor values from field
        motor_normalized = self.extraction(field_state)
        
        # Apply activation to ensure bounded outputs
        motor_normalized = torch.tanh(motor_normalized)
        
        # Denormalize to motor ranges
        motor_commands = []
        for i, channel in enumerate(self.robot.motor_channels):
            # Convert from [-1, 1] to motor range
            normalized_value = motor_normalized[i].item()
            range_span = channel.range_max - channel.range_min
            value = channel.range_min + (normalized_value + 1.0) * range_span / 2.0
            
            # Clamp to valid range
            value = max(channel.range_min, min(channel.range_max, value))
            motor_commands.append(value)
        
        return motor_commands
    
    def get_motor_dimensions(self) -> int:
        """Get expected motor dimensions."""
        return self.motor_dim
    
    def _create_extraction(self) -> nn.Module:
        """
        Create extraction from field space to motor space.
        
        This creates a mapping that extracts relevant motor commands
        from the unified field representation.
        """
        
        # Linear extraction layer
        extraction = nn.Linear(self.field_dimensions, self.motor_dim, bias=True)
        
        # Initialize for stable motor outputs
        with torch.no_grad():
            # Small weights to start with gentle motor commands
            nn.init.uniform_(extraction.weight, -0.1, 0.1)
            
            # Zero bias for neutral starting position
            nn.init.constant_(extraction.bias, 0.0)
        
        return extraction
    
    def _create_denormalizers(self) -> Dict[int, Dict[str, float]]:
        """Create denormalization parameters for each motor."""
        denormalizers = {}
        
        for i, channel in enumerate(self.robot.motor_channels):
            denormalizers[i] = {
                'min': channel.range_min,
                'max': channel.range_max,
                'center': (channel.range_min + channel.range_max) / 2.0,
                'scale': (channel.range_max - channel.range_min) / 2.0
            }
        
        return denormalizers


class AdapterFactory(IAdapterFactory):
    """Creates adapters for robot-brain translation."""
    
    def create_sensory_adapter(self, robot: Robot, field_dimensions: int) -> ISensoryAdapter:
        """Create sensory adapter for robot."""
        return SensoryAdapter(robot, field_dimensions)
    
    def create_motor_adapter(self, robot: Robot, field_dimensions: int) -> IMotorAdapter:
        """Create motor adapter for robot."""
        return MotorAdapter(robot, field_dimensions)