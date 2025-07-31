#!/usr/bin/env python3
"""
Test predictive sensory gating
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


def test_predictive_gating():
    """Test how predictive gating affects energy accumulation"""
    print("\n=== Testing Predictive Sensory Gating ===")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=16,
        motor_dim=3,
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    energy_values = []
    prediction_confidence = []
    imprint_strengths = []
    exploration_values = []
    
    # Test with repetitive input (should become predictable)
    print("\nPhase 1: Repetitive sensory pattern")
    pattern_a = [0.5] * 16
    pattern_b = [0.3] * 16
    
    for i in range(100):
        # Alternate between two patterns
        if i % 10 < 5:
            sensory_input = pattern_a
        else:
            sensory_input = pattern_b
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Track metrics
        energy = brain_state['energy_state']['energy']
        confidence = brain._current_prediction_confidence
        imprint = brain._last_imprint_strength
        exploration = brain_state['energy_state']['exploration_drive']
        
        energy_values.append(energy)
        prediction_confidence.append(confidence)
        imprint_strengths.append(imprint)
        exploration_values.append(exploration)
        
        if i % 20 == 0:
            print(f"  Cycle {i}: Energy={energy:.3f}, Confidence={confidence:.3f}, "
                  f"Imprint={imprint:.3f}, Exploration={exploration:.3f}")
    
    # Introduce novel pattern
    print("\nPhase 2: Novel pattern (surprise)")
    novel_pattern = [np.random.randn() * 0.5 for _ in range(16)]
    
    for i in range(100, 120):
        sensory_input = novel_pattern if i == 100 else pattern_a
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Track metrics
        energy = brain_state['energy_state']['energy']
        confidence = brain._current_prediction_confidence
        imprint = brain._last_imprint_strength
        exploration = brain_state['energy_state']['exploration_drive']
        
        energy_values.append(energy)
        prediction_confidence.append(confidence)
        imprint_strengths.append(imprint)
        exploration_values.append(exploration)
        
        if i == 100 or i == 101:
            print(f"  Cycle {i}: Energy={energy:.3f}, Confidence={confidence:.3f}, "
                  f"Imprint={imprint:.3f}, Exploration={exploration:.3f}")
    
    # Analyze results
    print("\n=== Analysis ===")
    print(f"Energy before gating: {np.mean(energy_values[:20]):.3f}")
    print(f"Energy after learning: {np.mean(energy_values[80:100]):.3f}")
    print(f"Confidence after learning: {np.mean(prediction_confidence[80:100]):.3f}")
    print(f"Imprint strength with good predictions: {np.mean(imprint_strengths[80:100]):.3f}")
    print(f"Imprint strength on surprise: {imprint_strengths[100]:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(energy_values)
    plt.axvline(x=100, color='r', linestyle='--', alpha=0.5, label='Novel input')
    plt.ylabel('Energy')
    plt.title('Field Energy with Predictive Gating')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(prediction_confidence)
    plt.axvline(x=100, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('Prediction Confidence')
    plt.title('Prediction Confidence Over Time')
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(imprint_strengths)
    plt.axvline(x=100, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('Imprint Strength')
    plt.title('Sensory Imprint Strength (Gated by Prediction)')
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(exploration_values)
    plt.axvline(x=100, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('Exploration')
    plt.xlabel('Cycle')
    plt.title('Exploration Drive')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('predictive_gating.png', dpi=150)
    print("\nPlot saved as predictive_gating.png")


def test_exploration_recovery():
    """Test if predictive gating allows exploration to recover"""
    print("\n\n=== Testing Exploration Recovery ===")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=4,
        motor_dim=3,
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    # Simulate being near a light (predictable environment)
    light_pattern = [0.8, 0.8, 0.8, 0.8]
    
    exploration_track = []
    energy_track = []
    
    for i in range(200):
        motor_output, brain_state = brain.process_robot_cycle(light_pattern)
        
        exploration = brain_state['energy_state']['exploration_drive']
        energy = brain_state['energy_state']['energy']
        
        exploration_track.append(exploration)
        energy_track.append(energy)
        
        if i % 50 == 0:
            print(f"Cycle {i}: Energy={energy:.3f}, Exploration={exploration:.3f}")
    
    print(f"\nFinal state: Energy={energy:.3f}, Exploration={exploration:.3f}")
    print(f"Average exploration last 50 cycles: {np.mean(exploration_track[-50:]):.3f}")
    
    if np.mean(exploration_track[-50:]) > 0.4:
        print("✓ Exploration recovered with predictive gating!")
    else:
        print("⚠️  Exploration still suppressed")


if __name__ == "__main__":
    print("Testing predictive sensory gating...")
    
    test_predictive_gating()
    test_exploration_recovery()
    
    print("\n✅ Predictive gating tests complete!")