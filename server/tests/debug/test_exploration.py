#!/usr/bin/env python3
"""
Debug exploration behavior.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np


def test_exploration():
    """Test exploration behavior."""
    
    print("🔍 Testing Exploration Behavior")
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
    
    print(f"Brain parameters:")
    print(f"  Field decay rate: {brain.field_decay_rate}")
    print(f"  Field diffusion rate: {brain.field_diffusion_rate}")
    print(f"  Gradient following strength: {brain.gradient_following_strength}")
    
    # Test with varying sensory input
    print("\n📊 Testing with varying sensory input...")
    
    motor_outputs = []
    for i in range(50):
        # Create varying sensory input
        sensory_input = []
        for j in range(24):
            # Add noise to create variation
            value = 0.5 + 0.2 * np.sin(i * 0.1 + j * 0.5) + 0.1 * np.random.randn()
            sensory_input.append(np.clip(value, 0, 1))
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        motor_outputs.append(motor_output)
        
        if i % 10 == 0:
            print(f"\nCycle {i}:")
            print(f"  Motor output: {[f'{m:.4f}' for m in motor_output]}")
            print(f"  Field energy: {brain_state['field_energy']:.6f}")
            print(f"  Max activation: {brain_state['max_activation']:.4f}")
    
    # Calculate exploration metrics
    print("\n📈 Exploration metrics:")
    
    # Motor variance
    motor_x = [m[0] for m in motor_outputs]
    motor_y = [m[1] for m in motor_outputs]
    
    print(f"  Motor X variance: {np.var(motor_x):.6f}")
    print(f"  Motor Y variance: {np.var(motor_y):.6f}")
    print(f"  Motor X range: [{min(motor_x):.4f}, {max(motor_x):.4f}]")
    print(f"  Motor Y range: [{min(motor_y):.4f}, {max(motor_y):.4f}]")
    
    # Check if motors change over time
    motor_changes = []
    for i in range(1, len(motor_outputs)):
        change = np.linalg.norm(np.array(motor_outputs[i]) - np.array(motor_outputs[i-1]))
        motor_changes.append(change)
    
    print(f"  Average motor change: {np.mean(motor_changes):.6f}")
    print(f"  Max motor change: {np.max(motor_changes):.6f}")
    
    # Test without reward influence
    print("\n\n🔄 Testing pure exploration (no rewards)...")
    
    # Create fresh brain
    brain_wrapper2 = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    brain2 = brain_wrapper2.brain
    
    motor_outputs2 = []
    for i in range(30):
        # Random sensory input without reward
        sensory_input = [0.5 + 0.1 * np.random.randn() for _ in range(24)]
        sensory_input = [np.clip(v, 0, 1) for v in sensory_input]
        
        motor_output, _ = brain2.process_robot_cycle(sensory_input)
        motor_outputs2.append(motor_output)
    
    # Show some outputs
    print("\nSample motor outputs:")
    for i in range(0, 30, 5):
        print(f"  Cycle {i}: {[f'{m:.4f}' for m in motor_outputs2[i]]}")
    
    # Check for any non-zero outputs
    non_zero_count = sum(1 for m in motor_outputs2 if any(abs(v) > 0.001 for v in m))
    print(f"\nNon-zero motor outputs: {non_zero_count}/30")


if __name__ == "__main__":
    test_exploration()