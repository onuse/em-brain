#!/usr/bin/env python3
"""Debug topology discovery."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

from brains.field.core_brain import UnifiedFieldBrain

# Create brain with verbose output
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=False)

print("\n=== Testing topology discovery ===")

# Strong pattern to hopefully trigger topology
strong_pattern = [0.9] * 24

print("\nCycle 1:")
action, state = brain.process_robot_cycle(strong_pattern)

print("\nCycle 2 (same pattern):")
action, state = brain.process_robot_cycle(strong_pattern)

print("\nCycle 3 (same pattern):")
action, state = brain.process_robot_cycle(strong_pattern)

print(f"\n🔍 Final topology regions: {len(brain.topology_regions)}")
if brain.topology_regions:
    for key, region in brain.topology_regions.items():
        print(f"  {key}: activation={region['activation']:.3f}, importance={region['importance']:.1f}")
        
# Check field values
print(f"\n🔍 Field stats:")
print(f"  Max: {brain.unified_field.max().item():.4f}")
print(f"  Mean: {brain.unified_field.mean().item():.4f}")
print(f"  Min: {brain.unified_field.min().item():.4f}")