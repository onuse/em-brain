#!/usr/bin/env python3
"""Test a single brain cycle."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import time
from brains.field.core_brain import UnifiedFieldBrain

brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)
print(f"Brain created. Field shape: {list(brain.unified_field.shape)}")
print(f"Field min: {brain.unified_field.min():.4f}, max: {brain.unified_field.max():.4f}")

print("\nRunning one cycle...")
t0 = time.perf_counter()
action, state = brain.process_robot_cycle([0.5] * 24)
elapsed = (time.perf_counter() - t0) * 1000
print(f"Cycle completed in {elapsed:.1f}ms")

print(f"Field min: {brain.unified_field.min():.4f}, max: {brain.unified_field.max():.4f}")
print(f"Topology regions: {len(brain.topology_regions)}")