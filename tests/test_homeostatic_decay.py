#!/usr/bin/env python3
"""Quick test of homeostatic decay."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import time
from brains.field.core_brain import UnifiedFieldBrain

print("Creating brain with homeostatic decay...")
brain = UnifiedFieldBrain(spatial_resolution=8, quiet_mode=False)

print(f"\nField stats before input:")
print(f"  Min: {brain.unified_field.min().item():.4f}")
print(f"  Max: {brain.unified_field.max().item():.4f}")
print(f"  Mean: {brain.unified_field.mean().item():.4f}")

# Send a strong pattern
pattern = [0.9] * 24
print("\nSending strong pattern...")
t0 = time.perf_counter()
action, state = brain.process_robot_cycle(pattern)
elapsed = (time.perf_counter() - t0) * 1000
print(f"Cycle time: {elapsed:.1f}ms")

print(f"\nField stats after input:")
print(f"  Min: {brain.unified_field.min().item():.4f}")
print(f"  Max: {brain.unified_field.max().item():.4f}")
print(f"  Mean: {brain.unified_field.mean().item():.4f}")

print(f"\nTopology regions: {len(brain.topology_regions)}")

# Test a few more cycles
print("\nTesting 5 more cycles...")
for i in range(5):
    t0 = time.perf_counter()
    action, state = brain.process_robot_cycle([0.5] * 24)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Cycle {i+1}: {elapsed:.1f}ms, regions: {len(brain.topology_regions)}")