#!/usr/bin/env python3
"""Debug prediction system to understand why confidence is 0%."""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'server'))

import numpy as np
import torch
from src.core.simplified_brain_factory import SimplifiedBrainFactory

print("Testing prediction system...")

# Create brain
brain_factory = SimplifiedBrainFactory()
brain_wrapper = brain_factory.create(sensory_dim=25, motor_dim=5)
brain = brain_wrapper.brain  # Get actual brain instance

# Run a few cycles and track predictions
for i in range(10):
    # Create sensory input with pattern
    sensory_input = []
    for j in range(24):
        # Create a predictable pattern
        sensory_input.append(0.5 + 0.3 * np.sin(i * 0.1 + j * 0.2))
    sensory_input.append(0.0)  # No reward
    
    # Before processing, check predictions
    print(f"\n=== Cycle {i} ===")
    if brain._predicted_sensory is not None:
        print(f"Predicted sensory shape: {brain._predicted_sensory.shape}")
        print(f"Predicted values (first 5): {brain._predicted_sensory[:5].tolist()}")
        print(f"Actual values (first 5): {sensory_input[:5]}")
        
        # Calculate error manually
        actual = torch.tensor(sensory_input[:len(brain._predicted_sensory)], dtype=torch.float32, device=brain.device)
        error = torch.mean(torch.abs(actual - brain._predicted_sensory)).item()
        print(f"Prediction error: {error:.4f}")
        print(f"Expected confidence: {1.0 - min(1.0, error * 2.0):.3f}")
    else:
        print("No prediction yet")
    
    # Process
    motor_tensor = brain_wrapper.process_field_dynamics(sensory_input)
    
    # After processing, check state
    print(f"Last prediction error: {brain._last_prediction_error:.4f}")
    print(f"Current confidence: {brain._current_prediction_confidence:.3f}")
    
    # Check if patterns were extracted
    patterns = brain.pattern_system.extract_patterns(brain.unified_field, n_patterns=1)
    if patterns:
        print(f"Dominant pattern energy: {patterns[0].energy:.4f}")
    else:
        print("No patterns extracted")

# Check confidence tracking in field dynamics
print("\n=== Field Dynamics Confidence ===")
print(f"Smoothed confidence: {brain.field_dynamics.smoothed_confidence:.3f}")
print(f"Prediction errors: {list(brain.field_dynamics.prediction_errors)[-5:] if brain.field_dynamics.prediction_errors else 'None'}")

# Test with constant input
print("\n=== Testing with constant input ===")
constant_input = [0.5] * 25
for i in range(5):
    motor_tensor = brain_wrapper.process_field_dynamics(constant_input)
    print(f"Cycle {i}: confidence={brain._current_prediction_confidence:.3f}, error={brain._last_prediction_error:.4f}")