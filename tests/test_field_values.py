#!/usr/bin/env python3
"""Test why field values stay at baseline."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

from brains.field.core_brain import UnifiedFieldBrain

# Create brain and track field changes
brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)

print("1. Initial state:")
print(f"   Field: min={brain.unified_field.min():.4f}, max={brain.unified_field.max():.4f}")

# Send strong input
input_data = [0.9] * 24
print("\n2. Processing strong input (0.9 x 24)...")

# Step through the process
action, state = brain.process_robot_cycle(input_data)

print(f"\n3. After processing:")
print(f"   Field: min={brain.unified_field.min():.4f}, max={brain.unified_field.max():.4f}")
print(f"   Non-baseline values: {(brain.unified_field > 0.011).sum().item()}")
print(f"   Max value location: {torch.where(brain.unified_field == brain.unified_field.max())}")

# Try multiple cycles
print("\n4. Running 5 more cycles...")
for i in range(5):
    action, state = brain.process_robot_cycle([0.7] * 24)
    max_val = brain.unified_field.max().item()
    above_baseline = (brain.unified_field > 0.011).sum().item()
    print(f"   Cycle {i+1}: max={max_val:.4f}, above_baseline={above_baseline}")

# Check if experiences are being stored
print(f"\n5. Experiences stored: {len(brain.field_experiences)}")
if brain.field_experiences:
    exp = brain.field_experiences[0]
    print(f"   First experience intensity: {exp.field_intensity}")

import torch