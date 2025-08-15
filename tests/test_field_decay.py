#!/usr/bin/env python3
"""Test field decay issue."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

from brains.field.core_brain import UnifiedFieldBrain

# Create brain with verbose output
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=False)

print("\n=== Testing field decay ===")

# Strong pattern
strong_pattern = [0.9] * 24

print("\nBefore cycle:")
print(f"Field at [1,1,1]: {brain.unified_field[1,1,1,5,7,1,1,1,0,1,0].item():.4f}")

print("\nCycle 1:")
action, state = brain.process_robot_cycle(strong_pattern)

print("\nAfter cycle:")
# Check the whole field max
print(f"Field max: {brain.unified_field.max().item():.4f}")
print(f"Field mean: {brain.unified_field.mean().item():.4f}")

# Check if any values are above baseline
above_baseline = (brain.unified_field > 0.02).sum().item()
print(f"Values above 0.02: {above_baseline}")

# Find the maximum value location
max_val, max_idx = brain.unified_field.max(dim=-1)
for i in range(10):
    max_val, max_idx = max_val.max(dim=-1)
    
max_indices = []
remaining = max_idx.item()
for dim in reversed(brain.unified_field.shape[1:]):
    max_indices.append(remaining % dim)
    remaining //= dim
max_indices.append(remaining)
max_indices = list(reversed(max_indices))

print(f"Max value {brain.unified_field.max().item():.4f} at indices: {max_indices}")

# Check the exact location where we imprinted
imprint_indices = [1, 1, 1, 4, 9, 0, 2, 0, 0, 0, 1]
val_at_imprint = brain.unified_field[tuple(imprint_indices)].item()
print(f"\nValue at imprint location {imprint_indices}: {val_at_imprint:.4f}")

# Check neighbor values
for offset in [-1, 0, 1]:
    neighbor_indices = imprint_indices.copy()
    neighbor_indices[0] += offset
    if 0 <= neighbor_indices[0] < 3:
        val = brain.unified_field[tuple(neighbor_indices)].item()
        print(f"Value at {neighbor_indices[:3]}: {val:.4f}")