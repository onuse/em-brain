#!/usr/bin/env python3
"""
Test how exploration drive translates to motor output
"""

import sys
import os
from pathlib import Path

# Add brain root to path
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
import numpy as np
import matplotlib.pyplot as plt


def test_exploration_motor_relationship():
    """Test relationship between exploration drive and motor variability"""
    print("\n=== Testing Exploration → Motor Relationship ===")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=4,
        motor_dim=3,  # Need at least 3 for differential drive (2 motors + confidence)
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    exploration_values = []
    motor_outputs = []
    motor_variance = []
    
    # Force different exploration levels by manipulating energy/novelty
    for i in range(100):
        # Create different sensory patterns to vary novelty
        if i % 20 == 0:
            sensory_input = [np.random.randn() for _ in range(4)]
        else:
            sensory_input = [0.1, 0.1, 0.1, 0.1]
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        exploration = brain_state['energy_state'].get('exploration_drive', 0)
        exploration_values.append(exploration)
        motor_outputs.append(motor_output)
        
        # Calculate motor variance over last 5 outputs
        if len(motor_outputs) >= 5:
            recent_motors = np.array(motor_outputs[-5:])
            variance = np.var(recent_motors, axis=0).mean()
            motor_variance.append(variance)
        else:
            motor_variance.append(0)
        
        if i % 10 == 0:
            print(f"Cycle {i}: Exploration={exploration:.3f}, Motors={motor_output}, Variance={motor_variance[-1]:.4f}")
    
    # Analyze correlation
    # Compare high vs low exploration periods
    high_exp_indices = [i for i, exp in enumerate(exploration_values) if exp > 0.5]
    low_exp_indices = [i for i, exp in enumerate(exploration_values) if exp < 0.3]
    
    if high_exp_indices and low_exp_indices:
        high_exp_variance = np.mean([motor_variance[i] for i in high_exp_indices if i < len(motor_variance)])
        low_exp_variance = np.mean([motor_variance[i] for i in low_exp_indices if i < len(motor_variance)])
        
        print(f"\nHigh exploration motor variance: {high_exp_variance:.4f}")
        print(f"Low exploration motor variance: {low_exp_variance:.4f}")
        print(f"Variance ratio: {high_exp_variance/low_exp_variance:.2f}x")
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(exploration_values)
    plt.ylabel('Exploration Drive')
    plt.title('Exploration Drive Over Time')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    motor_array = np.array(motor_outputs)
    if len(motor_array.shape) > 1 and motor_array.shape[1] >= 2:
        plt.plot(motor_array[:, 0], label='Left Motor', alpha=0.7)
        plt.plot(motor_array[:, 1], label='Right Motor', alpha=0.7)
    plt.ylabel('Motor Commands')
    plt.title('Motor Commands Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(motor_variance)
    plt.ylabel('Motor Variance')
    plt.xlabel('Cycle')
    plt.title('Motor Output Variance (5-cycle window)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('exploration_motor_relationship.png', dpi=150)
    print("\nPlot saved as exploration_motor_relationship.png")


def test_motor_noise_scaling():
    """Test if motor noise is properly scaled with exploration"""
    print("\n=== Testing Motor Noise Scaling ===")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=4,
        motor_dim=3,  # Need at least 3 for differential drive
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    # Collect motor outputs at different exploration levels
    exploration_levels = []
    motor_ranges = []
    
    for i in range(50):
        # Vary input to create different exploration levels
        if i < 10:
            # Novel inputs - high exploration
            sensory_input = [np.random.randn() * 2 for _ in range(4)]
        elif i < 30:
            # Familiar inputs - low exploration  
            sensory_input = [0.1, 0.1, 0.1, 0.1]
        else:
            # Novel again
            sensory_input = [np.random.randn() * 3 for _ in range(4)]
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        exploration = brain_state['energy_state'].get('exploration_drive', 0)
        motor_range = max(motor_output) - min(motor_output)
        
        exploration_levels.append(exploration)
        motor_ranges.append(motor_range)
        
        if i % 10 == 0:
            print(f"Cycle {i}: Exploration={exploration:.3f}, Motor range={motor_range:.3f}")
    
    # Check if motor range increases with exploration
    correlation = np.corrcoef(exploration_levels, motor_ranges)[0, 1]
    print(f"\nCorrelation between exploration and motor range: {correlation:.3f}")
    
    if correlation > 0.3:
        print("✓ Motor output properly varies with exploration drive")
    else:
        print("⚠️  Motor output may not be responding to exploration drive")


if __name__ == "__main__":
    print("Testing exploration to motor translation...")
    
    test_exploration_motor_relationship()
    test_motor_noise_scaling()
    
    print("\n✅ Exploration-motor tests complete!")