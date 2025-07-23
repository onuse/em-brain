#!/usr/bin/env python3
"""
Quick check if brain is ready for robot integration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import time
from brains.field.core_brain import create_unified_field_brain


print("Quick Integration Readiness Check")
print("=" * 40)

# Create brain
print("\n1. Creating brain...")
start = time.perf_counter()
brain = create_unified_field_brain(
    spatial_resolution=15,
    quiet_mode=True
)
create_time = (time.perf_counter() - start) * 1000
print(f"   Brain created in {create_time:.1f}ms")

# Test single cycle
print("\n2. Testing single cycle...")
start = time.perf_counter()
action, state = brain.process_robot_cycle([0.5] * 24)
cycle_time = (time.perf_counter() - start) * 1000
print(f"   Cycle time: {cycle_time:.1f}ms")

# Check output
print(f"\n3. Checking output...")
print(f"   Action: {[f'{a:.3f}' for a in action]}")
print(f"   Non-zero actions: {sum(1 for a in action if abs(a) > 0.001)}/4")

# Quick performance test
print(f"\n4. Quick performance test (10 cycles)...")
times = []
for i in range(10):
    start = time.perf_counter()
    brain.process_robot_cycle([0.5 + 0.1 * i for _ in range(24)])
    times.append((time.perf_counter() - start) * 1000)

avg_time = sum(times) / len(times)
print(f"   Average: {avg_time:.1f}ms")

# Summary
print(f"\n" + "=" * 40)
if avg_time < 100 and sum(1 for a in action if abs(a) > 0.001) > 0:
    print("✅ READY FOR ROBOT INTEGRATION!")
else:
    print("⚠️  Issues detected:")
    if avg_time >= 100:
        print(f"   - Cycle time too slow: {avg_time:.1f}ms")
    if sum(1 for a in action if abs(a) > 0.001) == 0:
        print("   - No motor output generated")