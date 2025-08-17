#!/usr/bin/env python3
"""Quick performance test with smaller brain."""

import sys
import os
sys.path.append('/mnt/c/Users/glimm/Documents/Projects/em-brain/server/src')

import torch
import time
from brains.field.gpu_optimized_brain import GPUOptimizedFieldBrain

# Test with smaller size first
brain = GPUOptimizedFieldBrain(
    sensory_dim=16,
    motor_dim=5,
    spatial_size=16,  # Small size for testing
    channels=32,      # Small channels
    quiet_mode=False
)

print("\nTesting single cycle...")
start = time.perf_counter()
motor_output, telemetry = brain.process([0.5] * 16)
end = time.perf_counter()

print(f"Time: {(end-start)*1000:.1f}ms")
print(f"Motivation: {telemetry['motivation']}")
print(f"Exploring: {telemetry['exploring']}")

print("\nTesting 10 cycles...")
times = []
for i in range(10):
    start = time.perf_counter()
    motor_output, telemetry = brain.process([0.5 + 0.1*i] * 16)
    end = time.perf_counter()
    times.append((end-start)*1000)
    print(f"  Cycle {i+1}: {times[-1]:.1f}ms")

print(f"\nAverage: {sum(times)/len(times):.1f}ms")