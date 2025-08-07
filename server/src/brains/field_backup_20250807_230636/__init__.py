# -*- coding: utf-8 -*-
"""
Field Brain Package - GPU Optimization Integration
"""

from .unified_field_brain import UnifiedFieldBrain

# Import GPU optimization components
try:
    from .optimized_unified_field_brain import OptimizedUnifiedFieldBrain
    from .gpu_performance_integration import (
        GPUBrainFactory, 
        create_optimized_brain,
        quick_performance_test
    )
    GPU_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    GPU_OPTIMIZATION_AVAILABLE = False
    print(f"GPU optimizations unavailable: {e}")

# Export main interfaces
__all__ = [
    'UnifiedFieldBrain',
    'create_brain',  # Factory function
    'GPU_OPTIMIZATION_AVAILABLE'
]

# Add GPU exports if available
if GPU_OPTIMIZATION_AVAILABLE:
    __all__.extend([
        'OptimizedUnifiedFieldBrain',
        'GPUBrainFactory',
        'create_optimized_brain',
        'quick_performance_test'
    ])


def create_brain(
    sensory_dim: int = 16,
    motor_dim: int = 5,
    spatial_resolution: int = 32,
    device=None,
    prefer_gpu: bool = True,
    quiet_mode: bool = False
):
    """
    Factory function to create the best available brain implementation
    
    Args:
        sensory_dim: Number of sensors
        motor_dim: Number of motors
        spatial_resolution: Spatial resolution
        device: Specific device to use
        prefer_gpu: Use GPU optimization if available
        quiet_mode: Suppress output
    
    Returns:
        Brain instance (optimized if possible)
    """
    if GPU_OPTIMIZATION_AVAILABLE and prefer_gpu:
        try:
            return create_optimized_brain(
                sensory_dim=sensory_dim,
                motor_dim=motor_dim,
                spatial_resolution=spatial_resolution,
                device=device,
                quiet_mode=quiet_mode
            )
        except Exception as e:
            if not quiet_mode:
                print(f"WARNING: GPU optimization failed, using standard brain: {e}")
    
    # Fallback to standard brain
    import torch
    if device is None:
        device = torch.device('cpu')
    
    return UnifiedFieldBrain(
        sensory_dim=sensory_dim,
        motor_dim=motor_dim,
        spatial_resolution=spatial_resolution,
        device=device,
        quiet_mode=quiet_mode
    )