# -*- coding: utf-8 -*-
"""
Unified Field Brain Package

A single brain implementation with automatic algorithm selection
based on tensor size for optimal performance.
"""

from .unified_field_brain import UnifiedFieldBrain, MinimalUnifiedBrain, TrulyMinimalBrain

# Export main interfaces
__all__ = [
    'MinimalUnifiedBrain',
    'UnifiedFieldBrain',  # Alias for compatibility
    'create_brain',  # Factory function
]


def create_brain(
    sensory_dim: int = 16,
    motor_dim: int = 5,
    spatial_resolution: int = 32,
    device=None,
    quiet_mode: bool = False
):
    """
    Factory function to create minimal brain
    
    Args:
        sensory_dim: Number of sensors
        motor_dim: Number of motors
        spatial_resolution: Spatial resolution (32 = 32³×64 field)
        device: Specific device to use (auto-selects if None)
        quiet_mode: Suppress output
    
    Returns:
        MinimalUnifiedBrain instance
    """
    return MinimalUnifiedBrain(
        sensory_dim=sensory_dim,
        motor_dim=motor_dim,
        spatial_resolution=spatial_resolution,
        device=device,
        quiet_mode=quiet_mode
    )