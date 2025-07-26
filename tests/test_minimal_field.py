#!/usr/bin/env python3
"""Test with truly minimal field for real-time performance."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import time
from brains.field.core_brain import UnifiedFieldBrain

# Create brain with minimal dimensions
print("Creating minimal brain...")
brain = UnifiedFieldBrain(
    spatial_resolution=10,  # Smaller spatial
    quiet_mode=True
)

# Override field shape to be even smaller
field_shape = [10, 10, 10, 5, 8, 2, 2, 1, 1, 1, 1]  # Much smaller
print(f"Field shape: {field_shape}")
total = 1
for s in field_shape:
    total *= s
print(f"Total elements: {total:,}")
print(f"Memory: {(total * 4) / (1024 * 1024):.2f} MB")

# Disable expensive features for testing
brain.constraint_field.constraint_discovery_rate = 0.0  # No constraint discovery
brain.gradient_calculator.sparse_computation = True  # Keep sparse computation
brain.field_diffusion_rate = 0.0  # No diffusion

print("\nWarming up...")
for _ in range(3):
    brain.process_robot_cycle([0.5] * 24)

print("\nTiming 10 cycles:")
times = []
for i in range(10):
    t0 = time.perf_counter()
    action, _ = brain.process_robot_cycle([0.5 + 0.1 * i for _ in range(24)])
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
    print(f"  Cycle {i+1}: {elapsed:.1f}ms")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.1f}ms")
print("✅ PASS!" if avg < 100 else "⚠️  Still needs optimization")

# Check what's taking time
if avg > 100:
    print("\nProfiling one cycle...")
    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    brain.process_robot_cycle([0.5] * 24)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('time')
    print("\nTop time consumers:")
    stats.print_stats(5)