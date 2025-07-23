#!/usr/bin/env python3
"""Check field value distribution."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import torch
from brains.field.core_brain import UnifiedFieldBrain

brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)

# Run a few cycles
for _ in range(3):
    brain.process_robot_cycle([0.9] * 24)

# Find where the max value is
max_val = brain.unified_field.max()
max_location = torch.where(brain.unified_field == max_val)

print(f"Max value: {max_val.item():.4f}")
print(f"Max location: {[loc[0].item() for loc in max_location]}")

# Check how sparse the activation is
above_baseline = (brain.unified_field > 0.011).sum().item()
total_elements = brain.unified_field.numel()
print(f"\nActivation sparsity:")
print(f"  Elements above baseline: {above_baseline} / {total_elements} ({100*above_baseline/total_elements:.2f}%)")

# Check the neighborhood of max
if max_location[0].numel() > 0:
    x, y, z = max_location[0][0].item(), max_location[1][0].item(), max_location[2][0].item()
    
    # Sample 3x3x3 region around max
    x_min, x_max = max(0, x-1), min(brain.spatial_resolution, x+2)
    y_min, y_max = max(0, y-1), min(brain.spatial_resolution, y+2)
    z_min, z_max = max(0, z-1), min(brain.spatial_resolution, z+2)
    
    region = brain.unified_field[x_min:x_max, y_min:y_max, z_min:z_max, :, :]
    
    print(f"\n3x3x3 region around max:")
    print(f"  Region mean: {region.mean().item():.4f}")
    print(f"  Region std: {region.std().item():.4f}")
    print(f"  Region max: {region.max().item():.4f}")
    
    # Just the spatial cube
    spatial_cube = brain.unified_field[x_min:x_max, y_min:y_max, z_min:z_max, 
                                      max_location[3][0], max_location[4][0],
                                      max_location[5][0], max_location[6][0],
                                      max_location[7][0], max_location[8][0],
                                      max_location[9][0], max_location[10][0]]
    print(f"\nSpatial cube at max indices:")
    print(f"  Values: {spatial_cube.flatten().tolist()}")