#!/usr/bin/env python3
"""Debug a single cycle step by step."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import torch
from brains.field.core_brain import UnifiedFieldBrain

# Monkey patch to add debug output
original_apply = UnifiedFieldBrain._apply_multidimensional_imprint_new
def debug_apply(self, indices, intensity):
    print(f"  Applying imprint at {indices} with intensity {intensity}")
    original_apply(self, indices, intensity)
    
UnifiedFieldBrain._apply_multidimensional_imprint_new = debug_apply

# Create brain
brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)
print("Brain created")

# Also debug experience creation
original_create = UnifiedFieldBrain._robot_sensors_to_field_experience
def debug_create(self, raw_input):
    exp = original_create(self, raw_input)
    print(f"\nExperience created:")
    print(f"  Intensity: {exp.field_intensity}")
    print(f"  Field coords norm: {torch.norm(exp.field_coordinates).item():.4f}")
    return exp

UnifiedFieldBrain._robot_sensors_to_field_experience = debug_create

# Run one cycle
print("\nRunning cycle...")
action, state = brain.process_robot_cycle([0.9] * 24)

print(f"\nFinal field state:")
print(f"  Min: {brain.unified_field.min():.4f}")
print(f"  Max: {brain.unified_field.max():.4f}")
print(f"  Unique values: {len(torch.unique(brain.unified_field))}")