#!/usr/bin/env python3
"""
Debug motor generation in dynamic brain.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np
import torch


def debug_motor_generation():
    """Debug why motors aren't being generated."""
    
    print("🔍 Debugging Motor Generation")
    print("=" * 50)
    
    # Create brain
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': False
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    print("\n📊 Initial Field State:")
    print(f"  Field mean: {torch.mean(brain.unified_field):.6f}")
    print(f"  Field max: {torch.max(brain.unified_field):.6f}")
    print(f"  Field min: {torch.min(brain.unified_field):.6f}")
    
    # Process several cycles with strong inputs
    print("\n🏃 Processing cycles with strong sensory input...")
    
    for i in range(10):
        # Create strong, varying sensory input
        sensory_input = [0.9 if j < 12 else 0.1 for j in range(25)]
        sensory_input[0] = 0.9  # Strong X
        sensory_input[1] = 0.1  # Weak Y
        sensory_input[24] = 0.8  # High reward
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        print(f"\nCycle {i+1}:")
        print(f"  Motor output: {[f'{m:.4f}' for m in motor_output]}")
        print(f"  Field energy: {brain_state['field_energy']:.6f}")
        print(f"  Max activation: {brain_state['max_activation']:.4f}")
    
    # Check field state after activation
    print("\n📊 Field State After Activation:")
    print(f"  Field mean: {torch.mean(brain.unified_field):.6f}")
    print(f"  Field max: {torch.max(brain.unified_field):.6f}")
    print(f"  Field std: {torch.std(brain.unified_field):.6f}")
    
    # Manually check gradient calculation
    print("\n🔬 Checking Gradient Calculation:")
    
    # Get a test position
    test_position = brain._conceptual_to_tensor_indices(
        torch.tensor([0.5] * brain.total_dimensions, device=brain.device)
    )
    print(f"  Test position indices: {test_position[:3]}")
    
    # Calculate gradients manually
    spatial_dims = list(range(min(3, len(brain.tensor_shape))))
    region_center = tuple(test_position[i] for i in range(min(3, len(test_position))))
    
    gradients = brain.gradient_calculator.calculate_gradients(
        field=brain.unified_field,
        dimensions_to_compute=spatial_dims,
        local_region_only=True,
        region_center=region_center,
        region_size=3
    )
    
    print(f"  Gradients calculated: {list(gradients.keys())}")
    for dim, grad in gradients.items():
        print(f"    Dim {dim} gradient shape: {grad.shape}")
        print(f"    Dim {dim} gradient range: [{torch.min(grad):.6f}, {torch.max(grad):.6f}]")
    
    # Check motor mapping
    print("\n🎮 Checking Motor Mapping:")
    print(f"  Gradient following strength: {brain.gradient_following_strength}")
    print(f"  Motor smoothing factor: {brain.motor_smoothing_factor}")
    print(f"  Previous motor commands: {brain.previous_motor_commands}")


if __name__ == "__main__":
    debug_motor_generation()