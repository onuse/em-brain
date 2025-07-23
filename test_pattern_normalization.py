#!/usr/bin/env python3
"""Test pattern normalization for memory."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

from brains.field.core_brain import UnifiedFieldBrain
import numpy as np

# Create brain with quiet mode
brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)

print("\n=== Testing Pattern Normalization ===")

# Test that loud and quiet versions of same pattern create similar memories
patterns = [
    # Same pattern at different volumes
    ([0.9, 0.1, 0.9, 0.1] * 6, "Loud alternating"),
    ([0.6, 0.4, 0.6, 0.4] * 6, "Medium alternating"),  
    ([0.55, 0.45, 0.55, 0.45] * 6, "Quiet alternating"),
    
    # Different pattern
    ([0.1, 0.1, 0.9, 0.9] * 6, "Different pattern"),
]

# Process each pattern
for pattern, name in patterns:
    print(f"\n--- {name} ---")
    action, _ = brain.process_robot_cycle(pattern)
    
    # Check topology regions
    print(f"Topology regions: {len(brain.topology_regions)}")
    
    if brain.topology_regions:
        # Show region centers (normalized patterns)
        for key, region in list(brain.topology_regions.items())[-2:]:  # Last 2 regions
            center = region['center'][:5].numpy()  # First 5 dims
            print(f"  {key}: center={center}")

print("\n✅ Pattern normalization test complete!")
print("\nThe first three patterns should create similar topology regions")
print("(same normalized pattern despite different volumes).")