#!/usr/bin/env python3
"""
Test motor generation with simulated movement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np
import torch


def test_motor_with_movement():
    """Test motor generation with changing positions."""
    
    print("🏃 Testing Motor Generation with Movement")
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
    print(f"Gradient following strength: {brain.gradient_following_strength}")
    
    # Simulate movement through space
    print("\n🚶 Simulating movement through space...")
    
    positions = [
        (0.2, 0.2),  # Start position
        (0.3, 0.2),
        (0.4, 0.2),
        (0.5, 0.2),
        (0.6, 0.2),
        (0.7, 0.2),  # Move along X
        (0.7, 0.3),
        (0.7, 0.4),
        (0.7, 0.5),  # Then move along Y
        (0.6, 0.5),
        (0.5, 0.5),  # Move back
    ]
    
    # First, lay down a reward trail
    print("\n📍 Laying down reward trail...")
    for i, (x, y) in enumerate(positions):
        sensory_input = [0.5] * 25
        sensory_input[0] = x  # X position
        sensory_input[1] = y  # Y position
        sensory_input[24] = 0.9  # High reward
        
        _, brain_state = brain.process_robot_cycle(sensory_input)
        
        if i % 3 == 0:
            print(f"  Position ({x:.1f}, {y:.1f}): field_energy={brain_state['field_energy']:.4f}")
    
    # Now test motor generation from different positions
    print("\n🎮 Testing motor generation from various positions...")
    
    test_positions = [
        (0.15, 0.2, "Near start of trail"),
        (0.5, 0.15, "Below middle of trail"),
        (0.8, 0.5, "Beyond end of trail"),
        (0.5, 0.5, "At end position"),
    ]
    
    for x, y, desc in test_positions:
        sensory_input = [0.5] * 25
        sensory_input[0] = x
        sensory_input[1] = y
        sensory_input[24] = 0.0  # No reward now
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        print(f"\nPosition ({x:.2f}, {y:.2f}) - {desc}:")
        print(f"  Motor output: X={motor_output[0]:.4f}, Y={motor_output[1]:.4f}")
        print(f"  Field energy: {brain_state['field_energy']:.6f}")
    
    # Test exploration behavior (no reward trail)
    print("\n\n🔍 Testing exploration behavior (fresh brain)...")
    
    # Create fresh brain
    brain_wrapper2 = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    brain2 = brain_wrapper2.brain
    
    # Process with neutral input
    motor_outputs = []
    for i in range(20):
        sensory_input = [0.5] * 25
        sensory_input[0] = 0.5 + 0.1 * np.sin(i * 0.3)  # Slight movement
        sensory_input[1] = 0.5 + 0.1 * np.cos(i * 0.3)
        
        motor_output, _ = brain2.process_robot_cycle(sensory_input)
        motor_outputs.append(motor_output)
    
    # Calculate exploration score
    motor_variance = np.var([m[0] for m in motor_outputs]) + np.var([m[1] for m in motor_outputs])
    print(f"Motor variance (exploration): {motor_variance:.6f}")
    print(f"Average motor magnitude: {np.mean([np.linalg.norm(m[:2]) for m in motor_outputs]):.4f}")
    
    # Show last few motor outputs
    print("\nLast 5 motor outputs:")
    for i, m in enumerate(motor_outputs[-5:]):
        print(f"  {i+16}: X={m[0]:.4f}, Y={m[1]:.4f}")


if __name__ == "__main__":
    test_motor_with_movement()