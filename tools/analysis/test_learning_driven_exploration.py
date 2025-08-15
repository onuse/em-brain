#!/usr/bin/env python3
"""
Test learning-driven exploration
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


def test_learning_plateau_exploration():
    """Test that exploration increases when learning plateaus"""
    print("\n=== Testing Learning-Driven Exploration ===")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=8,
        motor_dim=3,
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    exploration_values = []
    plateau_cycles = []
    prediction_errors = []
    
    # Phase 1: Novel pattern that becomes predictable
    print("\nPhase 1: Learning a new pattern")
    pattern = [0.5, -0.3, 0.7, -0.2, 0.4, -0.6, 0.8, -0.1]
    
    for i in range(150):
        motor_output, brain_state = brain.process_robot_cycle(pattern)
        
        exploration = brain_state['energy_state']['exploration_drive']
        plateau = brain.field_dynamics.learning_plateau_cycles
        error = brain._last_prediction_error
        
        exploration_values.append(exploration)
        plateau_cycles.append(plateau)
        prediction_errors.append(error)
        
        if i % 30 == 0:
            print(f"  Cycle {i}: Exploration={exploration:.3f}, "
                  f"Plateau cycles={plateau}, Prediction error={error:.3f}")
    
    # Phase 2: Continue with same pattern (should plateau)
    print("\nPhase 2: Continuing with same pattern (expecting plateau)")
    for i in range(150, 300):
        motor_output, brain_state = brain.process_robot_cycle(pattern)
        
        exploration = brain_state['energy_state']['exploration_drive']
        plateau = brain.field_dynamics.learning_plateau_cycles
        error = brain._last_prediction_error
        
        exploration_values.append(exploration)
        plateau_cycles.append(plateau)
        prediction_errors.append(error)
        
        if i % 30 == 0 or (plateau > 50 and plateau % 10 == 0):
            mode = brain_state.get('cognitive_mode', 'unknown')
            print(f"  Cycle {i}: Exploration={exploration:.3f}, "
                  f"Plateau cycles={plateau}, Mode={mode}")
    
    # Analyze results
    print("\n=== Analysis ===")
    print(f"Initial exploration: {np.mean(exploration_values[:20]):.3f}")
    print(f"Final exploration: {np.mean(exploration_values[-20:]):.3f}")
    print(f"Max plateau cycles: {max(plateau_cycles)}")
    print(f"Exploration at max plateau: {exploration_values[plateau_cycles.index(max(plateau_cycles))]:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(exploration_values)
    plt.ylabel('Exploration Drive')
    plt.title('Learning-Driven Exploration')
    plt.grid(True)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Exploring threshold')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(plateau_cycles)
    plt.ylabel('Learning Plateau Cycles')
    plt.title('Cycles Without Prediction Improvement')
    plt.grid(True)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.3, label='High exploration trigger')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(prediction_errors)
    plt.ylabel('Prediction Error')
    plt.xlabel('Cycle')
    plt.title('Prediction Error Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_driven_exploration.png', dpi=150)
    print("\nPlot saved as learning_driven_exploration.png")
    
    # Verify behavior
    if max(plateau_cycles) > 50:
        print("\n✓ Learning plateau detected!")
    if exploration_values[-1] > exploration_values[0] + 0.1:
        print("✓ Exploration increased due to learning plateau!")
    else:
        print("⚠️  Exploration did not increase significantly")


def test_metabolic_baseline():
    """Test metabolic baseline exploration"""
    print("\n\n=== Testing Metabolic Baseline ===")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=4,
        motor_dim=3,
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    # No input at all - should still have some exploration
    no_input = [0.0] * 4
    
    min_exploration = 1.0
    for i in range(50):
        motor_output, brain_state = brain.process_robot_cycle(no_input)
        exploration = brain_state['energy_state']['exploration_drive']
        min_exploration = min(min_exploration, exploration)
        
        if i % 10 == 0:
            print(f"  Cycle {i}: Exploration={exploration:.3f}")
    
    print(f"\nMinimum exploration: {min_exploration:.3f}")
    if min_exploration >= 0.1:
        print("✓ Metabolic baseline ensures minimum exploration!")
    else:
        print("⚠️  Exploration dropped below metabolic baseline")


if __name__ == "__main__":
    print("Testing learning-driven exploration...")
    
    test_learning_plateau_exploration()
    test_metabolic_baseline()
    
    print("\n✅ Learning-driven exploration tests complete!")