#!/usr/bin/env python3
"""
Detailed debug of motor generation pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np
import torch


def test_motor_generation_detailed():
    """Debug motor generation step by step."""
    
    print("🔍 Detailed Motor Generation Debug")
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
    
    # Create activation pattern
    print("\n📊 Creating activation pattern...")
    
    positions = [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2)]
    for x, y in positions:
        sensory_input = [0.5] * 25
        sensory_input[0] = x
        sensory_input[1] = y
        sensory_input[24] = 0.9
        brain.process_robot_cycle(sensory_input)
    
    # Now test from a position between activations
    print("\n🎯 Testing motor generation from position (0.35, 0.2)...")
    
    sensory_input = [0.5] * 25
    sensory_input[0] = 0.35
    sensory_input[1] = 0.2
    sensory_input[24] = 0.0  # No reward
    
    # Create experience manually to debug
    experience = brain._robot_sensors_to_field_experience(sensory_input)
    print(f"\nExperience created:")
    print(f"  Field coords (first 3): {experience.field_coordinates[:3]}")
    
    # Get tensor indices
    tensor_indices = brain._conceptual_to_tensor_indices(experience.field_coordinates)
    print(f"  Tensor indices: {tensor_indices[:3]}")
    
    # Calculate gradients
    spatial_dims = list(range(min(3, len(brain.tensor_shape))))
    region_center = tuple(tensor_indices[i] for i in range(min(3, len(tensor_indices))))
    
    print(f"\nGradient calculation:")
    print(f"  Spatial dims: {spatial_dims}")
    print(f"  Region center: {region_center}")
    
    # Get gradients
    all_gradients = brain.gradient_calculator.calculate_gradients(
        field=brain.unified_field,
        dimensions_to_compute=spatial_dims,
        local_region_only=True,
        region_center=region_center,
        region_size=3
    )
    
    print(f"  Gradients calculated: {list(all_gradients.keys())}")
    
    # Extract gradients manually
    gradients = []
    for dim in spatial_dims:
        grad_key = f'gradient_dim_{dim}'
        if grad_key in all_gradients:
            grad_tensor = all_gradients[grad_key]
            print(f"\n  Gradient dim {dim}:")
            print(f"    Shape: {grad_tensor.shape}")
            print(f"    Non-zero elements: {torch.count_nonzero(grad_tensor)}")
            
            # Sample gradient values around the position
            if dim < len(tensor_indices):
                idx = tensor_indices[dim]
                print(f"    Sampling at index {idx}:")
                
                # Try to extract gradient value
                idx_list = tensor_indices.copy()
                
                # Show gradient values in local region
                for offset in [-1, 0, 1]:
                    test_idx = idx + offset
                    if 0 <= test_idx < grad_tensor.shape[dim]:
                        idx_list[dim] = test_idx
                        # Truncate index list to match gradient tensor dimensions
                        truncated_idx = idx_list[:len(grad_tensor.shape)]
                        try:
                            val = grad_tensor[tuple(truncated_idx)]
                            print(f"      idx[{dim}]={test_idx}: {float(val):.6f}")
                        except:
                            print(f"      idx[{dim}]={test_idx}: ERROR accessing")
    
    # Now run actual motor generation
    print("\n🚀 Running actual motor generation...")
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    
    print(f"\nMotor output: {motor_output}")
    print(f"Prediction confidence: {brain_state['prediction_confidence']:.3f}")
    
    # Check brain parameters
    print(f"\nBrain parameters:")
    print(f"  Gradient following strength: {brain.gradient_following_strength}")
    print(f"  Motor smoothing factor: {brain.motor_smoothing_factor}")
    print(f"  Previous motor commands: {brain.previous_motor_commands}")


if __name__ == "__main__":
    test_motor_generation_detailed()