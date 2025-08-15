"""
Unified Field Brain Adapters

Direct adapters for the unified field brain that handle
normalization without complex mappings.
"""

import torch
from typing import List
from .interfaces import ISensoryAdapter, IMotorAdapter, Robot


class UnifiedSensoryAdapter(ISensoryAdapter):
    """
    Direct sensory adapter for unified field brain.
    Simple normalization to [-1, 1] range.
    """
    
    def __init__(self, robot: Robot):
        self.robot = robot
        self.sensory_dim = len(robot.sensory_channels)
    
    def to_field_space(self, sensory: List[float]) -> torch.Tensor:
        """Simply normalize sensory input to [-1, 1] range."""
        if len(sensory) != self.sensory_dim:
            raise ValueError(f"Expected {self.sensory_dim} sensory values, got {len(sensory)}")
        
        # Normalize each sensor to [-1, 1]
        normalized = []
        for value, channel in zip(sensory, self.robot.sensory_channels):
            # Clamp to channel range
            value = max(channel.range_min, min(channel.range_max, value))
            
            # Normalize
            range_span = channel.range_max - channel.range_min
            if range_span > 0:
                norm_value = 2.0 * (value - channel.range_min) / range_span - 1.0
            else:
                norm_value = 0.0
            normalized.append(norm_value)
        
        return normalized  # SimplifiedBrain expects list, not tensor
    
    def get_field_dimensions(self) -> int:
        """For compatibility - not used by simplified brain."""
        return self.sensory_dim


class UnifiedMotorAdapter(IMotorAdapter):
    """
    Direct motor adapter for unified field brain.
    Maps field gradients to motor commands.
    """
    
    def __init__(self, robot: Robot):
        self.robot = robot
        self.motor_dim = len(robot.motor_channels)
    
    def from_field_space(self, motor_commands) -> List[float]:
        """
        Motor commands from simplified brain are already in [-1, 1] range.
        Just validate and return them.
        """
        # Handle both tensor and list inputs
        if isinstance(motor_commands, torch.Tensor):
            motor_list = motor_commands.cpu().tolist()
        else:
            motor_list = motor_commands
            
        if len(motor_list) != self.motor_dim:
            # SimplifiedBrain returns 3 motor commands, but robot expects 4
            # Add a default confidence value if missing
            if len(motor_list) == self.motor_dim - 1:
                motor_list = motor_list + [0.5]  # Default confidence
            else:
                raise ValueError(f"Expected {self.motor_dim} motor values, got {len(motor_list)}")
        
        # Validate each motor command is in valid range
        validated_commands = []
        for i, (value, channel) in enumerate(zip(motor_list, self.robot.motor_channels)):
            # The brain outputs in [-1, 1], but let's ensure it's clamped
            value = max(-1.0, min(1.0, float(value)))
            validated_commands.append(value)
        
        return validated_commands
    
    def get_motor_dimensions(self) -> int:
        """Get motor dimensions."""
        return self.motor_dim


class UnifiedAdapterFactory:
    """Factory for creating simplified adapters."""
    
    def create_sensory_adapter(self, robot: Robot, field_dimensions: int = None) -> ISensoryAdapter:
        """Create simplified sensory adapter."""
        return UnifiedSensoryAdapter(robot)
    
    def create_motor_adapter(self, robot: Robot, field_dimensions: int = None) -> IMotorAdapter:
        """Create simplified motor adapter."""
        return UnifiedMotorAdapter(robot)