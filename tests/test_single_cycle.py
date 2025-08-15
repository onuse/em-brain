#!/usr/bin/env python3
"""Test a single brain cycle to see actual timing."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import time
from brains.field.core_brain import create_unified_field_brain

print("Creating brain (this might take a moment)...")
t0 = time.perf_counter()
brain = create_unified_field_brain(spatial_resolution=15, quiet_mode=False)
t1 = time.perf_counter()
print(f"Brain creation took: {(t1-t0):.2f} seconds\n")

print("Running ONE cycle...")
print("Starting cycle at:", time.strftime("%H:%M:%S"))

start = time.perf_counter()
action, state = brain.process_robot_cycle([0.5] * 24)
elapsed = time.perf_counter() - start

print(f"\nCycle completed at:", time.strftime("%H:%M:%S"))
print(f"Cycle took: {elapsed:.2f} seconds ({elapsed*1000:.0f}ms)")
print(f"Action: {action}")

if elapsed > 1.0:
    print(f"\n⚠️  WARNING: Cycle took {elapsed:.1f} seconds! Way too slow for real-time.")
elif elapsed > 0.1:
    print(f"\n⚠️  WARNING: Cycle took {elapsed*1000:.0f}ms, target is <100ms")
else:
    print(f"\n✅ Good: Cycle took {elapsed*1000:.0f}ms")