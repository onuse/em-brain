#!/usr/bin/env python3
"""
Simple test with direct field manipulation to ensure gradients work.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np
import torch


def test_simple_gradient():
    """Test with manually created gradient pattern."""
    
    print("🔍 Simple Gradient Test")
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
    
    # Manually create a strong gradient in the field
    print("\n📊 Creating manual gradient pattern...")
    
    # Create gradient along X axis
    for x in range(brain.tensor_shape[0]):
        for y in range(brain.tensor_shape[1]):
            for z in range(brain.tensor_shape[2]):
                # Create strong gradient: 0 at x=0, 1 at x=3
                value = float(x) / (brain.tensor_shape[0] - 1)
                # Set for all higher dimensions
                brain.unified_field[x, y, z, :, :, :, :, :, :, :, :] = value
    
    print(f"Field stats:")
    print(f"  Min: {torch.min(brain.unified_field):.3f}")
    print(f"  Max: {torch.max(brain.unified_field):.3f}")
    
    # Test motor generation at different X positions
    print("\n🎮 Testing motor generation at different positions...")
    
    test_positions = [
        (0.25, 0.5, "Low X - should move positive"),
        (0.5, 0.5, "Mid X - neutral"),
        (0.75, 0.5, "High X - should move negative"),
    ]
    
    for x, y, desc in test_positions:
        sensory_input = [0.5] * 25
        sensory_input[0] = x
        sensory_input[1] = y
        
        # Don't use process_robot_cycle as it modifies the field
        # Instead, call motor generation directly
        experience = brain._robot_sensors_to_field_experience(sensory_input)
        action = brain._unified_field_to_robot_action(experience)
        
        print(f"\nPosition ({x}, {y}) - {desc}:")
        print(f"  Tensor indices: {brain._conceptual_to_tensor_indices(experience.field_coordinates)[:3]}")
        print(f"  Motor X: {action.motor_commands[0]:.4f}")
        print(f"  Motor Y: {action.motor_commands[1]:.4f}")
        print(f"  Gradient strength: {action.gradient_strength:.4f}")
    
    # Create Y gradient and test
    print("\n\n📊 Creating Y gradient pattern...")
    
    # Reset field
    brain.unified_field.fill_(0.0001)
    
    # Create gradient along Y axis
    for x in range(brain.tensor_shape[0]):
        for y in range(brain.tensor_shape[1]):
            for z in range(brain.tensor_shape[2]):
                value = float(y) / (brain.tensor_shape[1] - 1)
                brain.unified_field[x, y, z, :, :, :, :, :, :, :, :] = value
    
    # Test at low Y position
    sensory_input = [0.5] * 25
    sensory_input[0] = 0.5
    sensory_input[1] = 0.25  # Low Y
    
    experience = brain._robot_sensors_to_field_experience(sensory_input)
    action = brain._unified_field_to_robot_action(experience)
    
    print(f"\nPosition (0.5, 0.25) - Low Y:")
    print(f"  Motor X: {action.motor_commands[0]:.4f}")
    print(f"  Motor Y: {action.motor_commands[1]:.4f}")
    print(f"  Gradient strength: {action.gradient_strength:.4f}")


if __name__ == "__main__":
    test_simple_gradient()