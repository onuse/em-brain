#!/usr/bin/env python3
"""
Test motor output variation to verify it's working correctly.
"""

import sys
import os
# Add paths to find modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'server'))

import numpy as np
import torch
from src.core.simplified_brain_factory import SimplifiedBrainFactory

def test_motor_variation(cycles=100):
    """Test that motor outputs show proper variation."""
    print("Testing motor output variation...")
    
    # Create brain
    brain_factory = SimplifiedBrainFactory()
    brain = brain_factory.create(sensory_dim=25, motor_dim=5)
    
    # Standard dimensions for picarx
    sensory_dim = 25  # 3 distance + 3 light + 9 color + 4 touch + 5 proprioception + 1 reward
    motor_dim = 5     # left, right, arm, pan, tilt
    
    print(f"Sensory dimensions: {sensory_dim}")
    print(f"Motor dimensions: {motor_dim}")
    
    # Collect motor outputs
    motor_history = []
    
    for cycle in range(cycles):
        # Create varied sensory input
        sensory_input = []
        
        # Distance sensors (3) - vary with cycle
        for i in range(3):
            distance = 0.5 + 0.3 * np.sin(cycle * 0.1 + i)
            sensory_input.append(distance)
            
        # Light sensors (3) - some noise
        for i in range(3):
            light = 0.3 + 0.2 * np.random.random()
            sensory_input.append(light)
            
        # Color sensors (9) - pattern with variation
        for i in range(9):
            color = 0.5 + 0.1 * np.sin(cycle * 0.05 + i) + 0.1 * np.random.random()
            sensory_input.append(color)
            
        # Touch sensors (4) - occasionally active
        for i in range(4):
            touch = 1.0 if np.random.random() > 0.9 else 0.0
            sensory_input.append(touch)
            
        # Proprioception (5) - motor feedback
        if motor_history:
            for i in range(min(5, len(motor_history[-1]))):
                sensory_input.append(motor_history[-1][i] * 0.5)
        else:
            sensory_input.extend([0.0] * 5)
            
        # Reward - occasional
        reward = 1.0 if cycle % 20 == 0 else 0.0
        sensory_input.append(reward)
        
        # Process cycle
        motor_tensor = brain.process_field_dynamics(sensory_input)
        motor_output = motor_tensor.tolist()
        brain_state = brain.get_brain_state()
        motor_history.append(motor_output)
        
        if cycle % 20 == 0:
            print(f"Cycle {cycle}: motors={[f'{m:.3f}' for m in motor_output]}")
            print(f"  Information: {brain_state.get('field_information', 0):.3f}")
            print(f"  Confidence: {brain_state.get('prediction_confidence', 0):.3f}")
            print(f"  Exploration: {brain_state.get('information_state', {}).get('exploration_drive', 0):.3f}")
    
    # Analyze variation
    motor_array = np.array(motor_history)
    
    print("\n=== Motor Variation Analysis ===")
    for i in range(motor_array.shape[1]):
        motor_values = motor_array[:, i]
        variance = np.var(motor_values)
        std = np.std(motor_values)
        range_val = np.max(motor_values) - np.min(motor_values)
        
        print(f"\nMotor {i}:")
        print(f"  Mean: {np.mean(motor_values):.3f}")
        print(f"  Std: {std:.3f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Range: {range_val:.3f} [{np.min(motor_values):.3f}, {np.max(motor_values):.3f}]")
        
        # Check if motor is stuck
        unique_values = len(np.unique(np.round(motor_values, 3)))
        if unique_values < 10:
            print(f"  WARNING: Only {unique_values} unique values!")
    
    # Check correlation between motors
    print("\n=== Motor Correlations ===")
    for i in range(min(3, motor_array.shape[1])):
        for j in range(i+1, min(3, motor_array.shape[1])):
            corr = np.corrcoef(motor_array[:, i], motor_array[:, j])[0, 1]
            print(f"Motor {i} vs Motor {j}: {corr:.3f}")
    
    # Check if motors respond to exploration drive
    print("\n=== Testing exploration response ===")
    
    # Force high exploration
    brain.brain.field_dynamics.learning_plateau_cycles = 200  # Trigger exploration
    
    high_exploration_motors = []
    for cycle in range(20):
        sensory_input = [0.5] * sensory_dim
        motor_tensor = brain.process_field_dynamics(sensory_input)
        motor_output = motor_tensor.tolist()
        high_exploration_motors.append(motor_output)
        
    high_exp_array = np.array(high_exploration_motors)
    high_exp_variance = np.mean([np.var(high_exp_array[:, i]) for i in range(high_exp_array.shape[1])])
    
    print(f"Normal variance: {np.mean([np.var(motor_array[:, i]) for i in range(motor_array.shape[1])]):.6f}")
    print(f"High exploration variance: {high_exp_variance:.6f}")
    
    if high_exp_variance > np.mean([np.var(motor_array[:, i]) for i in range(motor_array.shape[1])]) * 1.5:
        print("✓ Motors show increased variation during exploration")
    else:
        print("✗ Motors do not respond strongly to exploration")

if __name__ == "__main__":
    test_motor_variation()