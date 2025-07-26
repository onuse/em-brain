#!/usr/bin/env python3
"""
Debug gradient calculation in dynamic brain.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np
import torch


def test_gradient_calculation():
    """Test gradient calculation in detail."""
    
    print("🔍 Testing Gradient Calculation")
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
    
    # Manually create some activation in the field
    print("\n📊 Creating test activation pattern...")
    
    # Create a gradient pattern in the field
    for x in range(brain.tensor_shape[0]):
        for y in range(brain.tensor_shape[1]):
            for z in range(brain.tensor_shape[2]):
                # Create increasing activation along X axis
                brain.unified_field[x, y, z, 0, 0, 0, 0, 0, 0, 0, 0] = float(x) / brain.tensor_shape[0]
    
    print(f"Field stats after manual activation:")
    print(f"  Mean: {torch.mean(brain.unified_field):.6f}")
    print(f"  Max: {torch.max(brain.unified_field):.6f}")
    print(f"  Min: {torch.min(brain.unified_field):.6f}")
    
    # Test gradient calculation directly
    print("\n🔬 Testing Gradient Calculator:")
    
    # Test 1: Full field gradient
    spatial_dims = [0, 1, 2]
    region_center = (2, 2, 2)
    
    gradients = brain.gradient_calculator.calculate_gradients(
        field=brain.unified_field,
        dimensions_to_compute=spatial_dims,
        local_region_only=True,
        region_center=region_center,
        region_size=3
    )
    
    print(f"\nLocal gradient calculation at {region_center}:")
    for dim, grad in gradients.items():
        print(f"  {dim}: shape={grad.shape}")
        if grad.numel() > 0:
            print(f"    Range: [{torch.min(grad):.6f}, {torch.max(grad):.6f}]")
            print(f"    Mean: {torch.mean(torch.abs(grad)):.6f}")
        else:
            print(f"    Empty gradient!")
    
    # Test 2: Manual gradient calculation
    print("\n📐 Manual gradient calculation (for comparison):")
    
    # Calculate gradient in X direction manually
    field_slice = brain.unified_field[:, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    manual_grad_x = torch.zeros_like(field_slice)
    manual_grad_x[1:-1] = (field_slice[2:] - field_slice[:-2]) / 2.0
    
    print(f"Manual X gradient: {manual_grad_x}")
    print(f"  Range: [{torch.min(manual_grad_x):.6f}, {torch.max(manual_grad_x):.6f}]")
    
    # Test 3: Test the _discover_field_constraints method
    print("\n🧪 Testing _discover_field_constraints method:")
    
    # Look at the gradient calculation there
    gradients_discover = {}
    spatial_range = brain.dimension_mapping['family_tensor_ranges'].get(
        brain.field_dimensions[0].family, (0, 3)
    )
    
    print(f"Spatial tensor range: {spatial_range}")
    
    for dim in range(spatial_range[0], min(spatial_range[1], len(brain.tensor_shape))):
        # This is the buggy gradient calculation!
        grad = torch.zeros_like(brain.unified_field)
        grad[1:-1] = (brain.unified_field[2:] - brain.unified_field[:-2]) / 2.0
        gradients_discover[f'grad_dim_{dim}'] = grad
        
        print(f"\nGradient dim {dim} (buggy method):")
        print(f"  Shape: {grad.shape}")
        print(f"  All zeros? {torch.all(grad == 0)}")
    
    # Test 4: Correct gradient calculation
    print("\n✅ Correct gradient calculation:")
    
    for dim in range(min(3, len(brain.tensor_shape))):
        grad = torch.zeros_like(brain.unified_field)
        
        # Correct way: roll along specific dimension
        if dim == 0:
            grad[1:-1, :, :] = (brain.unified_field[2:, :, :] - brain.unified_field[:-2, :, :]) / 2.0
        elif dim == 1:
            grad[:, 1:-1, :] = (brain.unified_field[:, 2:, :] - brain.unified_field[:, :-2, :]) / 2.0
        elif dim == 2:
            grad[:, :, 1:-1] = (brain.unified_field[:, :, 2:] - brain.unified_field[:, :, :-2]) / 2.0
        
        print(f"\nGradient dim {dim} (correct):")
        print(f"  Non-zero elements: {torch.sum(grad != 0)}")
        print(f"  Mean abs gradient: {torch.mean(torch.abs(grad)):.6f}")
    
    # Test 5: Test motor generation with corrected gradients
    print("\n🎮 Testing motor generation with manual gradient:")
    
    # Create a mock experience
    field_coords = torch.tensor([0.5] * brain.total_dimensions, device=brain.device)
    tensor_indices = brain._conceptual_to_tensor_indices(field_coords)
    print(f"Test position indices: {tensor_indices[:3]}")
    
    # Manually set some gradients
    test_gradients = [0.5, -0.3, 0.1]  # X, Y, Z gradients
    motor_commands = torch.zeros(4, device=brain.device)
    
    # Apply gradient following
    motor_commands[0] = test_gradients[0] * brain.gradient_following_strength
    motor_commands[1] = test_gradients[1] * brain.gradient_following_strength
    
    print(f"Test motor commands: {motor_commands}")
    print(f"Expected: [0.25, -0.15, 0.0, 0.0]")


if __name__ == "__main__":
    test_gradient_calculation()