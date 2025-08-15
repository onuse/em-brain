#!/usr/bin/env python3
"""Quick test of adaptive motor cortex."""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'server'))

import numpy as np
import torch
from src.core.simplified_brain_factory import SimplifiedBrainFactory

# Create brain
print("Creating brain with adaptive motor cortex...")
brain_factory = SimplifiedBrainFactory()
brain = brain_factory.create(sensory_dim=25, motor_dim=5)

# Run a few cycles
print("\nRunning test cycles...")
motor_outputs = []

for i in range(20):
    # Varied sensory input
    sensory_input = [0.5 + 0.3 * np.sin(i * 0.2 + j) for j in range(25)]
    
    # Process
    motor_tensor = brain.process_field_dynamics(sensory_input)
    motor_output = motor_tensor.tolist()
    motor_outputs.append(motor_output)
    
    print(f"Cycle {i}: {[f'{m:.3f}' for m in motor_output]}")

# Analyze
motor_array = np.array(motor_outputs)
print("\nAnalysis:")
for i in range(motor_array.shape[1]):
    values = motor_array[:, i]
    unique = len(np.unique(np.round(values, 3)))
    print(f"Motor {i}: {unique} unique values, range [{values.min():.3f}, {values.max():.3f}]")