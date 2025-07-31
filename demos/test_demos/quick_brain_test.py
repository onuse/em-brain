#!/usr/bin/env python3
"""Quick test of the simplified unified brain."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import numpy as np
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Quick Brain Test\n")
print("="*60)

# Create a brain with 16 sensors
SENSORY_DIM = 16
MOTOR_DIM = 4

brain = SimplifiedUnifiedBrain(
    sensory_dim=SENSORY_DIM,
    motor_dim=MOTOR_DIM,
    spatial_resolution=32,
    quiet_mode=False
)

print("\nRunning 10 cycles with simple sensory input...\n")

# Run some cycles
for i in range(10):
    # Create varying sensory input
    sensory_input = [0.5 + 0.2 * np.sin(i * 0.5 + j * 0.1) for j in range(SENSORY_DIM)]
    
    # Process the input
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    
    # Print results
    print(f"Cycle {i+1}:")
    print(f"  Motor output: {[f'{m:.3f}' for m in motor_output]}")
    print(f"  Energy: {brain_state.get('energy', 0):.3f}")
    print(f"  Confidence: {brain_state.get('confidence', 0):.3f}")
    print(f"  Information: {brain_state.get('information', 0):.3f}")
    
    # Give small rewards occasionally
    if i % 3 == 0 and i > 0:
        # Append reward as last element of sensory input
        sensory_with_reward = sensory_input + [0.5]
        brain.process_robot_cycle(sensory_with_reward)
        print(f"  -> Reward given!")

print("\n" + "="*60)
print("Brain test complete!")
print(f"Total cycles: {brain.brain_cycles}")
print(f"Field dynamics evolution count: {brain.field_dynamics.evolution_count}")
print(f"Active topology regions: {len(brain.topology_region_system.regions)}")
print(f"Self-modification strength: {brain.field_dynamics.self_modification_strength:.3f}")
print(f"Smoothed confidence: {brain.field_dynamics.smoothed_confidence:.3f}")