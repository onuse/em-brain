#!/usr/bin/env python3
"""Test topology discovery."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

from brains.field.core_brain import UnifiedFieldBrain

brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)
print(f"Topology threshold: {brain.topology_stability_threshold}")

# Run several cycles with same input
input_pattern = [0.9] * 24
for i in range(10):
    action, state = brain.process_robot_cycle(input_pattern)
    max_val = brain.unified_field.max().item()
    mean_val = brain.unified_field.mean().item()
    regions = len(brain.topology_regions)
    print(f"Cycle {i+1}: max={max_val:.4f}, mean={mean_val:.4f}, regions={regions}")

# Check field statistics at a specific location
if max_val > brain.topology_stability_threshold:
    print(f"\nField max ({max_val:.4f}) exceeds threshold ({brain.topology_stability_threshold})")
    print("Topology discovery should be triggered!")
    
# Sample local region like discovery does
x_idx = brain.spatial_resolution // 2
y_idx = brain.spatial_resolution // 2 
z_idx = brain.spatial_resolution // 2

local_region = brain.unified_field[
    max(0, x_idx-1):min(brain.spatial_resolution, x_idx+2),
    max(0, y_idx-1):min(brain.spatial_resolution, y_idx+2),
    max(0, z_idx-1):min(brain.spatial_resolution, z_idx+2),
    :, :
]

import torch
region_mean = torch.mean(local_region).item()
region_std = torch.std(local_region.float()).item()

print(f"\nLocal region stats:")
print(f"  Mean: {region_mean:.4f}")
print(f"  Std: {region_std:.4f}")
print(f"  Stable? mean>{brain.topology_stability_threshold} and std<0.1: {region_mean > brain.topology_stability_threshold and region_std < 0.1}")