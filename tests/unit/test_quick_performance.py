#!/usr/bin/env python3
"""Quick performance test to verify optimizations."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

import time
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Quick Performance Test\n")

# Create brain
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

# Enable features
brain.enable_hierarchical_prediction(True)
brain.enable_action_prediction(True)

# Warmup
print("Warming up...")
for i in range(10):
    sensory_input = [0.5] * 24
    brain.process_robot_cycle(sensory_input)

# Measure 20 cycles
print("\nMeasuring 20 cycles...")
times = []

for i in range(20):
    sensory_input = [0.5 + i*0.01] * 24
    
    start = time.perf_counter()
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    elapsed = time.perf_counter() - start
    times.append(elapsed * 1000)  # Convert to ms
    
    if i % 5 == 0:
        print(f"  Cycle {i}: {elapsed*1000:.2f}ms")

# Results
avg_time = sum(times) / len(times)
min_time = min(times)
max_time = max(times)

print(f"\nResults:")
print(f"  Average: {avg_time:.2f}ms")
print(f"  Min: {min_time:.2f}ms")
print(f"  Max: {max_time:.2f}ms")

# Compare field evolution speed
print("\nField evolution only (10 iterations):")
field = brain.unified_field
evo_times = []

for i in range(10):
    start = time.perf_counter()
    field = brain.field_dynamics.evolve_field(field)
    elapsed = time.perf_counter() - start
    evo_times.append(elapsed * 1000)

print(f"  Average: {sum(evo_times)/len(evo_times):.2f}ms")

print("\n✅ Performance test complete!")