#!/usr/bin/env python3
"""
Investigate why energy is so high and what the brain is experiencing
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


def test_energy_dynamics():
    """Test what drives energy high"""
    print("\n=== Testing Energy Dynamics ===")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=16,
        motor_dim=3,
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    energy_values = []
    exploration_values = []
    reward_values = []
    field_stats = []
    
    # Test different input patterns
    for i in range(100):
        if i < 20:
            # No input
            sensory_input = [0.0] * 16
            print(f"\nCycle {i}: No input")
        elif i < 40:
            # Weak input
            sensory_input = [0.1] * 16
            print(f"\nCycle {i}: Weak input") if i == 20 else None
        elif i < 60:
            # Strong input
            sensory_input = [0.5] * 16
            print(f"\nCycle {i}: Strong input") if i == 40 else None
        elif i < 80:
            # High reward
            sensory_input = [0.1] * 15 + [1.0]  # Last value is reward
            print(f"\nCycle {i}: High reward") if i == 60 else None
        else:
            # Negative reward
            sensory_input = [0.1] * 15 + [-1.0]
            print(f"\nCycle {i}: Negative reward") if i == 80 else None
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Extract metrics
        energy = brain_state['energy_state']['energy']
        exploration = brain_state['energy_state']['exploration_drive']
        reward = sensory_input[-1] if len(sensory_input) > 15 else 0.0
        
        energy_values.append(energy)
        exploration_values.append(exploration)
        reward_values.append(reward)
        
        # Get field statistics
        field_energy = brain_state.get('field_energy', 0)
        max_activation = brain_state.get('max_activation', 0)
        field_stats.append({
            'field_energy': field_energy,
            'max_activation': max_activation,
            'evolution_cycles': brain_state['evolution_state']['evolution_cycles']
        })
        
        if i % 10 == 0:
            print(f"  Energy: {energy:.3f}, Exploration: {exploration:.3f}, Field energy: {field_energy:.3f}")
    
    # Analyze what drives energy
    print("\n=== Analysis ===")
    print(f"Energy range: {min(energy_values):.3f} - {max(energy_values):.3f}")
    print(f"Average energy: {np.mean(energy_values):.3f}")
    
    # Check correlation with reward
    reward_correlation = np.corrcoef(reward_values, energy_values)[0, 1]
    print(f"Energy-reward correlation: {reward_correlation:.3f}")
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(energy_values, label='Normalized Energy')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('Energy')
    plt.title('Energy Dynamics')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(exploration_values)
    plt.axhline(y=0.15, color='r', linestyle='--', label='Min floor')
    plt.ylabel('Exploration')
    plt.title('Exploration Drive')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot([fs['field_energy'] for fs in field_stats], label='Field Energy')
    plt.plot([fs['max_activation'] for fs in field_stats], label='Max Activation')
    plt.ylabel('Field Metrics')
    plt.title('Raw Field Statistics')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(reward_values)
    plt.ylabel('Reward')
    plt.xlabel('Cycle')
    plt.title('Input Reward Signal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('energy_investigation.png', dpi=150)
    print("\nPlot saved as energy_investigation.png")
    
    return energy_values, field_stats


def test_timescale_mismatch():
    """Test if the brain expects different timescales"""
    print("\n\n=== Testing Timescale Expectations ===")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=16,
        motor_dim=3,
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    # Simulate at different rates
    print("\nFast cycling (robot perspective - 10Hz):")
    for i in range(10):
        sensory_input = [0.1] * 16
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        if i % 5 == 0:
            energy = brain_state['energy_state']['energy']
            print(f"  Cycle {i}: Energy={energy:.3f}")
    
    print("\nWhat if we only process every 10th sensory frame (1Hz)?")
    # This might be more biological - integrating over time
    
    # Look at the energy calculation
    print("\n=== Energy Calculation Investigation ===")
    print("Energy is calculated as: field_energy / 2.0")
    print("Field energy is: mean(abs(content))")
    print("So high field activations → high energy → low exploration")
    print("\nThis might be backwards from biology where high energy → more exploration!")


if __name__ == "__main__":
    print("Investigating energy dynamics...")
    
    energy_values, field_stats = test_energy_dynamics()
    test_timescale_mismatch()
    
    print("\n✅ Investigation complete!")