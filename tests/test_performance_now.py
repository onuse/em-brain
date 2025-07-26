#!/usr/bin/env python3
"""Simple performance test - will it run under 100ms?"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import time
from brains.field.core_brain import create_unified_field_brain

print("Creating brain...")
brain = create_unified_field_brain(spatial_resolution=15, quiet_mode=True)

print("\nTiming 10 cycles:")
times = []
for i in range(10):
    start = time.perf_counter()
    action, _ = brain.process_robot_cycle([0.5] * 24)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"  Cycle {i+1}: {elapsed:.1f}ms")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.1f}ms")
print("✅ PASS" if avg < 100 else "❌ FAIL")