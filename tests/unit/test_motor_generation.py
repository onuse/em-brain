#!/usr/bin/env python3
"""Test motor generation from field with tensions."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from server.src.brains.field.unified_field_brain import UnifiedFieldBrain


def test_motor_generation_with_low_confidence():
    """Test that low confidence creates exploration behavior."""
    print("Testing motor generation with low confidence...")
    
    # Create brain
    brain = UnifiedFieldBrain(
        sensory_dim=24,
        motor_dim=4,
        device='cpu',  # Use CPU for testing
        quiet_mode=True
    )
    
    # Enable strategic planning
    brain.enable_strategic_planning(True)
    
    # Simulate low confidence scenario
    brain._current_prediction_confidence = 0.1  # 10% confidence
    brain.modulation = {'exploration_drive': 0.5}  # Normal exploration
    
    # Create a field with some gradients
    brain.unified_field = torch.zeros(32, 32, 32, 64)
    # Add gradient in X direction
    for i in range(32):
        brain.unified_field[i, :, :, :32] = i / 32.0 - 0.5
    
    # Generate motor action
    motor_action = brain._generate_motor_action()
    
    print(f"Motor action: {motor_action}")
    print(f"Action magnitude: {np.linalg.norm(motor_action):.3f}")
    
    # Check that we get non-zero movement
    assert abs(motor_action[0]) > 0.1 or abs(motor_action[1]) > 0.1, \
        f"Motor action too weak: {motor_action}"
    
    print("✓ Low confidence creates movement\n")


def test_tension_driven_pattern_discovery():
    """Test that high tensions trigger pattern discovery."""
    print("Testing tension-driven pattern discovery...")
    
    # Create brain
    brain = UnifiedFieldBrain(
        sensory_dim=24,
        motor_dim=4,
        device='cpu',
        quiet_mode=False  # Want to see pattern discovery messages
    )
    
    # Enable strategic planning
    brain.enable_strategic_planning(True)
    
    # Create a low-energy field (high information tension)
    brain.unified_field = torch.zeros(32, 32, 32, 64)
    brain.unified_field[:, :, :, :32] = 0.1  # Very low energy
    
    # Set low confidence
    brain._current_prediction_confidence = 0.05
    
    # Process a sensory input to trigger pattern discovery
    sensory_input = [0.5] * 24  # Dummy sensory input
    
    print("Processing with high tensions...")
    motor_output = brain.process(sensory_input)
    
    print(f"Motor output: {motor_output}")
    
    # Check tensions
    if brain.strategic_planner:
        tensions = brain.strategic_planner._measure_field_tensions(brain.unified_field)
        print(f"\nField tensions:")
        for k, v in tensions.items():
            print(f"  {k}: {v:.3f}")
    
    print("\n✓ Tension measurement working")


def test_gradient_amplification():
    """Test that gradients are properly amplified for movement."""
    print("\nTesting gradient amplification...")
    
    # Create brain
    brain = UnifiedFieldBrain(
        sensory_dim=24,
        motor_dim=4,
        device='cpu',
        quiet_mode=True
    )
    
    # Create field with small gradient
    brain.unified_field = torch.zeros(32, 32, 32, 64)
    # Small gradient: values from 0.4 to 0.6
    for i in range(32):
        brain.unified_field[i, :, :, :32] = 0.4 + (i / 32.0) * 0.2
    
    # Extract motor tendencies
    motor_tendencies = brain._extract_motor_tendencies_from_field()
    
    print(f"Motor tendencies: {motor_tendencies}")
    print(f"Forward tendency: {motor_tendencies[0]:.3f}")
    
    # With amplification of 5.0 and gradient of ~0.2/32, 
    # we should get meaningful movement
    assert abs(motor_tendencies[0]) > 0.01, \
        f"Motor tendency too weak: {motor_tendencies[0]}"
    
    print("✓ Gradients properly amplified\n")


if __name__ == "__main__":
    test_motor_generation_with_low_confidence()
    test_tension_driven_pattern_discovery()
    test_gradient_amplification()