#!/usr/bin/env python3
"""Test that field information is non-zero after recent fixes."""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'server'))

import numpy as np
import torch
from src.core.simplified_brain_factory import SimplifiedBrainFactory

print("Testing field information levels...")

# Create brain
brain_factory = SimplifiedBrainFactory()
brain = brain_factory.create(sensory_dim=25, motor_dim=5)

# Check initial field state
initial_field_info = brain.brain._create_brain_state()['field_information']
print(f"Initial field information: {initial_field_info:.6f}")

# Run some cycles with sensory input
information_history = []
confidence_history = []

for i in range(100):
    # Create varied sensory input
    sensory_input = []
    for j in range(24):
        sensory_input.append(0.5 + 0.3 * np.sin(i * 0.1 + j * 0.5))
    sensory_input.append(0.0)  # No reward
    
    # Process
    motor_tensor = brain.process_field_dynamics(sensory_input)
    
    # Get state
    state = brain.brain._create_brain_state()
    info = state['field_information']
    conf = state['prediction_confidence']
    
    information_history.append(info)
    confidence_history.append(conf)
    
    if i % 20 == 0:
        print(f"Cycle {i}: information={info:.6f}, confidence={conf:.3f}")

# Analysis
print("\n=== Field Information Analysis ===")
print(f"Initial: {information_history[0]:.6f}")
print(f"Final: {information_history[-1]:.6f}")
print(f"Average: {np.mean(information_history):.6f}")
print(f"Min: {min(information_history):.6f}")
print(f"Max: {max(information_history):.6f}")

print("\n=== Confidence Analysis ===") 
print(f"Initial: {confidence_history[0]:.3f}")
print(f"Final: {confidence_history[-1]:.3f}")
print(f"Average: {np.mean(confidence_history):.3f}")
print(f"Min: {min(confidence_history):.3f}")
print(f"Max: {max(confidence_history):.3f}")

# Check if values are reasonable
if information_history[-1] < 0.001:
    print("\n❌ WARNING: Field information is still near zero!")
else:
    print(f"\n✅ Field information is healthy: {information_history[-1]:.6f}")

if max(confidence_history) == 0.0:
    print("❌ WARNING: Confidence is stuck at 0%!")
elif min(confidence_history) == 1.0:
    print("❌ WARNING: Confidence is stuck at 100%!")
else:
    print(f"✅ Confidence shows variation: {min(confidence_history):.3f} - {max(confidence_history):.3f}")