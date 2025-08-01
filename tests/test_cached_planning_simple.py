#!/usr/bin/env python3
"""Simple test to verify cached planning is working."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import time
import torch
import numpy as np
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Simple Cached Planning Test\n")

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
brain.enable_future_simulation(True, n_futures=4, horizon=5)  # Smaller for faster test
brain.enable_cached_planning(True)

print("Testing reactive action (no simulation)...")
# First test reactive action speed
brain.use_future_simulation = False
start = time.perf_counter()
motor, state = brain.process_robot_cycle([0.5] * 24)
reactive_time = time.perf_counter() - start
print(f"Reactive action time: {reactive_time*1000:.0f}ms")

print("\nTesting with background planning...")
# Re-enable future simulation
brain.use_future_simulation = True

# Run a few cycles with similar inputs to build cache
for i in range(5):
    # Use very similar inputs to increase cache hits
    sensory_input = [0.5] * 24  # Same input every time
    start = time.perf_counter()
    motor, state = brain.process_robot_cycle(sensory_input)
    cycle_time = time.perf_counter() - start
    
    cache_stats = state.get('cached_planning', {})
    print(f"Cycle {i+1}: {cycle_time*1000:.0f}ms | "
          f"Cache hits: {cache_stats.get('cache_hits', 0)}, "
          f"misses: {cache_stats.get('cache_misses', 0)}, "
          f"size: {cache_stats.get('cache_size', 0)}")

# Check if caching is helping
if cache_stats.get('cache_hits', 0) > 0:
    print("\n✅ Cache is being used!")
else:
    print("\n⚠️  No cache hits yet - plans may still be computing")

# Give background planning time to complete
print("\nWaiting 10s for background planning to complete...")
time.sleep(10)

# Try again after waiting
print("\nTesting after background planning:")
for i in range(3):
    sensory_input = [0.5] * 24
    start = time.perf_counter()
    motor, state = brain.process_robot_cycle(sensory_input)
    cycle_time = time.perf_counter() - start
    
    cache_stats = state.get('cached_planning', {})
    print(f"Cycle {i+1}: {cycle_time*1000:.0f}ms | "
          f"Cache hits: {cache_stats.get('cache_hits', 0)}, "
          f"misses: {cache_stats.get('cache_misses', 0)}, "
          f"size: {cache_stats.get('cache_size', 0)}")