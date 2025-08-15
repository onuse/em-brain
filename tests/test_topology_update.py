#!/usr/bin/env python3
"""Debug topology update issue."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

from brains.field.core_brain import UnifiedFieldBrain

# Create brain 
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)

# Process one cycle to create a region
action, state = brain.process_robot_cycle([0.9] * 24)

print(f"\nAfter cycle 1:")
print(f"Topology regions: {len(brain.topology_regions)}")

if brain.topology_regions:
    for key, region in brain.topology_regions.items():
        print(f"\n{key}:")
        print(f"  center: {region['center'][:3].tolist()}")
        print(f"  field_indices: {region['field_indices']}")
        print(f"  activation: {region['activation']:.4f}")
        
        # Check actual field value at those indices
        actual_val = brain.unified_field[tuple(region['field_indices'])].item()
        print(f"  actual field value: {actual_val:.4f}")