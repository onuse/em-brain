#!/usr/bin/env python3
"""
Debug gradient extraction in motor generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np
import torch


def test_gradient_extraction():
    """Debug gradient extraction step by step."""
    
    print("🔍 Debugging Gradient Extraction")
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
    
    # Create a strong gradient pattern manually
    print("\n📊 Creating manual gradient pattern...")
    
    # Create increasing activation along X axis
    for x in range(brain.tensor_shape[0]):
        brain.unified_field[x, :, :, 0, 0, 0, 0, 0, 0, 0, 0] = float(x + 1) / brain.tensor_shape[0]
    
    print(f"Field stats:")
    print(f"  Shape: {brain.tensor_shape}")
    print(f"  Mean: {torch.mean(brain.unified_field):.6f}")
    print(f"  Max: {torch.max(brain.unified_field):.6f}")
    
    # Test gradient calculation at different positions
    print("\n🎯 Testing gradient extraction at different positions:")
    
    test_positions = [
        [0.0, 0.0, 0.0],  # Start
        [0.5, 0.0, 0.0],  # Middle
        [1.0, 0.0, 0.0],  # End
    ]
    
    for pos in test_positions:
        print(f"\n📍 Position {pos}:")
        
        # Create mock field coordinates
        field_coords = torch.tensor(pos * (brain.total_dimensions // 3) + [0.0] * (brain.total_dimensions % 3), device=brain.device)
        
        # Get tensor indices
        tensor_indices = brain._conceptual_to_tensor_indices(field_coords)
        print(f"  Tensor indices: {tensor_indices[:3]}")
        
        # Calculate gradients using the brain's method
        spatial_dims = list(range(min(3, len(brain.tensor_shape))))
        region_center = tuple(tensor_indices[i] for i in range(min(3, len(tensor_indices))))
        
        print(f"  Region center: {region_center}")
        print(f"  Calculating gradients for dims: {spatial_dims}")
        
        # Calculate gradients
        all_gradients = brain.gradient_calculator.calculate_gradients(
            field=brain.unified_field,
            dimensions_to_compute=spatial_dims,
            local_region_only=True,
            region_center=region_center,
            region_size=3
        )
        
        print(f"  Gradients returned: {list(all_gradients.keys())}")
        
        # Extract gradients at position
        gradients = []
        for dim in spatial_dims:
            if dim in all_gradients:
                grad_tensor = all_gradients[dim]
                print(f"\n  Gradient dim {dim}:")
                print(f"    Shape: {grad_tensor.shape}")
                
                # Build index for extraction
                idx_list = []
                for i in range(len(grad_tensor.shape)):
                    if i < len(tensor_indices):
                        idx = min(tensor_indices[i], grad_tensor.shape[i] - 1)
                        idx_list.append(idx)
                    else:
                        idx_list.append(0)
                
                print(f"    Index for extraction: {idx_list}")
                
                try:
                    grad_val = float(grad_tensor[tuple(idx_list)])
                    print(f"    Extracted value: {grad_val:.6f}")
                    gradients.append(grad_val)
                except Exception as e:
                    print(f"    ERROR extracting: {e}")
                    gradients.append(0.0)
            else:
                print(f"  No gradient for dim {dim}")
                gradients.append(0.0)
        
        # Calculate motor commands
        motor_commands = torch.zeros(4, device=brain.device)
        if len(gradients) >= 2:
            motor_commands[0] = gradients[0] * brain.gradient_following_strength
            motor_commands[1] = gradients[1] * brain.gradient_following_strength
        
        print(f"\n  Final motor commands: {motor_commands.tolist()}")
    
    # Test with actual brain cycle
    print("\n\n🚀 Testing with actual brain cycle:")
    
    sensory_input = [0.8, 0.5, 0.5] + [0.5] * 21 + [0.8]  # High X, high reward
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    
    print(f"Motor output: {motor_output}")
    print(f"Field energy: {brain_state['field_energy']:.6f}")


if __name__ == "__main__":
    test_gradient_extraction()