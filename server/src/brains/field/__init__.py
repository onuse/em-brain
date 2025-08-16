# -*- coding: utf-8 -*-
"""
Truly Minimal Field Brain Package

Reduced to absolute essentials - ~250 lines total.
Everything complex emerges from simple physics + intrinsic tensions.
"""

from .truly_minimal_brain import TrulyMinimalBrain, MinimalUnifiedBrain, UnifiedFieldBrain

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