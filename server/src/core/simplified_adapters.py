"""
Simplified Adapters for Direct Brain-Robot Communication

Clean adapters that work directly with the simplified brain's
motor commands without workarounds or padding.
"""

import torch
from typing import List
from .interfaces import ISensoryAdapter, IMotorAdapter, Robot


class SimplifiedSensoryAdapter(ISensoryAdapter):
    """
    Direct sensory adapter for simplified brain.
    No complex field space mapping - just normalization.
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


class SimplifiedMotorAdapter(IMotorAdapter):
    """
    Direct motor adapter for simplified brain.
    No field space extraction - motor commands are already in correct format.
    """
    
    def __init__(self, robot: Robot):
        self.robot = robot
        self.motor_dim = len(robot.motor_channels)
    
    def from_field_space(self, motor_commands: List[float]) -> List[float]:
        """
        Motor commands from simplified brain are already in [-1, 1] range.
        Just validate and return them.
        """
        if len(motor_commands) != self.motor_dim:
            raise ValueError(f"Expected {self.motor_dim} motor values, got {len(motor_commands)}")
        
        # Validate each motor command is in valid range
        validated_commands = []
        for i, (value, channel) in enumerate(zip(motor_commands, self.robot.motor_channels)):
            # The brain outputs in [-1, 1], but let's ensure it's clamped
            value = max(-1.0, min(1.0, value))
            validated_commands.append(value)
        
        return validated_commands
    
    def get_motor_dimensions(self) -> int:
        """Get motor dimensions."""
        return self.motor_dim