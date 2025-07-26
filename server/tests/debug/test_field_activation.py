#!/usr/bin/env python3
"""
Test field activation patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np
import torch


def test_field_activation():
    """Test if field is actually getting activated properly."""
    
    print("🔍 Testing Field Activation")
    print("=" * 50)
    
    # Create brain
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': True
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    print(f"Brain created: {brain.total_dimensions}D conceptual")
    print(f"Tensor shape: {brain.tensor_shape}")
    
    # Process cycles at different positions
    print("\n📍 Processing at different positions...")
    
    positions = [
        (0.1, 0.1),
        (0.3, 0.1),
        (0.5, 0.1),
        (0.7, 0.1),
        (0.9, 0.1),
    ]
    
    for x, y in positions:
        sensory_input = [0.5] * 25
        sensory_input[0] = x
        sensory_input[1] = y
        sensory_input[24] = 0.9  # High reward
        
        _, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Check field state at this position
        field_coords = torch.zeros(brain.total_dimensions, device=brain.device)
        field_coords[0] = x * 2.0 - 1.0  # Map to [-1,1]
        field_coords[1] = y * 2.0 - 1.0
        
        tensor_indices = brain._conceptual_to_tensor_indices(field_coords)
        
        # Get activation at this position
        activation = brain.unified_field[tuple(tensor_indices)]
        
        print(f"\nPosition ({x:.1f}, {y:.1f}):")
        print(f"  Tensor indices: {tensor_indices[:3]}")
        print(f"  Activation at position: {float(activation):.6f}")
        print(f"  Field max: {float(torch.max(brain.unified_field)):.6f}")
    
    # Check field variation along X axis
    print("\n📊 Field variation along X axis (Y=0.1, Z=0):")
    
    for x_idx in range(brain.tensor_shape[0]):
        # Sample at center of other dimensions
        sample_indices = [x_idx, 2, 2] + [s//2 for s in brain.tensor_shape[3:]]
        activation = brain.unified_field[tuple(sample_indices)]
        print(f"  X={x_idx}: {float(activation):.6f}")
    
    # Calculate actual gradients manually
    print("\n📐 Manual gradient calculation:")
    
    # Get a slice of the field
    field_slice = brain.unified_field[:, 2, 2, 5, 7, 1, 1, 1, 1, 1, 1]
    print(f"Field slice shape: {field_slice.shape}")
    print(f"Field slice values: {field_slice}")
    
    # Calculate gradient
    if field_slice.shape[0] > 2:
        grad = torch.zeros_like(field_slice)
        grad[1:-1] = (field_slice[2:] - field_slice[:-2]) / 2.0
        print(f"Gradient: {grad}")
        print(f"Gradient magnitude: {torch.abs(grad).max():.6f}")
    
    # Test gradient calculator directly on a simple field
    print("\n🧪 Testing gradient calculator on simple field:")
    
    # Create a simple field with clear gradient
    test_field = torch.zeros([4, 4, 4], device=brain.device)
    for x in range(4):
        test_field[x, :, :] = float(x)
    
    print(f"Test field X values: {test_field[:, 0, 0]}")
    
    # Calculate gradients
    grad_calc = brain.gradient_calculator
    gradients = grad_calc.calculate_gradients(
        field=test_field,
        dimensions_to_compute=[0],
        local_region_only=False
    )
    
    if 'gradient_dim_0' in gradients:
        grad = gradients['gradient_dim_0']
        print(f"Gradient shape: {grad.shape}")
        print(f"Gradient values: {grad[:, 0, 0]}")
    else:
        print("No gradient returned!")


if __name__ == "__main__":
    test_field_activation()