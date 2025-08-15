#!/usr/bin/env python3
"""Minimal test to check brain cycle performance."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import time
import torch

# Check device first
print("Checking compute device...")
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"✓ Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ Using CUDA GPU")
else:
    device = torch.device('cpu')
    print(f"⚠️  Using CPU (will be slower)")

# Import brain
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("\nCreating brain...")
start = time.perf_counter()
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)
init_time = (time.perf_counter() - start) * 1000
print(f"✓ Brain initialized in {init_time:.0f}ms")

print(f"\nRunning single cycle...")
sensory = [0.5] * 24

start = time.perf_counter()
motor, state = brain.process_robot_cycle(sensory)
cycle_time = (time.perf_counter() - start) * 1000

print(f"✓ Cycle completed in {cycle_time:.0f}ms")
print(f"  Reported cycle time: {state.get('cycle_time_ms', 0):.0f}ms")
print(f"  Motor output dimensions: {len(motor)}")

# Quick check of what phase features are enabled
print(f"\nEnabled features:")
print(f"  Hierarchical prediction: {brain.predictive_field.use_hierarchical}")
print(f"  Action prediction: {brain.use_action_prediction}")
print(f"  Active vision: {brain.use_active_vision}")

# Do a few more cycles to see if it speeds up
print(f"\nRunning 5 more cycles...")
times = []
for i in range(5):
    start = time.perf_counter()
    motor, state = brain.process_robot_cycle(sensory)
    cycle_time = (time.perf_counter() - start) * 1000
    times.append(cycle_time)
    print(f"  Cycle {i+1}: {cycle_time:.0f}ms")

avg_time = sum(times) / len(times)
print(f"\nAverage: {avg_time:.0f}ms")
print(f"Status: {'NORMAL' if avg_time < 1000 else 'SLOW'}")