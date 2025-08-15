#!/usr/bin/env python3
"""Test topology discovery with verbose output."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

from brains.field.core_brain import UnifiedFieldBrain

brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=False)

print("\nRunning cycle with verbose output...")
action, state = brain.process_robot_cycle([0.9] * 24)

print(f"\nTopology regions: {len(brain.topology_regions)}")
if brain.topology_regions:
    for key, region in brain.topology_regions.items():
        print(f"  {key}: activation={region['activation']:.3f}, importance={region['importance']:.1f}")