#!/usr/bin/env python3
"""Test performance with fixed field size."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import time
from brains.field.core_brain import create_unified_field_brain

# Calculate new field size
shape = [15, 15, 15, 10, 15, 3, 3, 2, 2, 2, 2]
total = 1
for s in shape:
    total *= s
print(f"New field shape: {shape}")
print(f"Total elements: {total:,}")
print(f"Memory: {(total * 4) / (1024 * 1024):.2f} MB")

print("\nCreating brain...")
t0 = time.perf_counter()
brain = create_unified_field_brain(spatial_resolution=15, quiet_mode=True)
print(f"Creation took: {(time.perf_counter()-t0):.2f}s")

print("\nTiming 5 cycles:")
for i in range(5):
    t0 = time.perf_counter()
    action, _ = brain.process_robot_cycle([0.5] * 24)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Cycle {i+1}: {elapsed:.1f}ms")
    
    if elapsed < 100:
        print("    ✅ Under 100ms!")
    else:
        print(f"    ⚠️  Still too slow")