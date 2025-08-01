#!/usr/bin/env python3
"""Debug test for cached planning to understand the flow."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import time
import torch
import numpy as np
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Cached Planning Debug Test\n")

# Create brain with minimal features for faster testing
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=16,  # Smaller for faster computation
    quiet_mode=True
)

# Enable only necessary features
brain.enable_action_prediction(True)
brain.enable_future_simulation(True, n_futures=2, horizon=3)  # Very small for debugging
brain.enable_cached_planning(True)

print("Configuration:")
print(f"  Future simulation: {brain.use_future_simulation}")
print(f"  Cached planning: {brain.use_cached_planning}")
print(f"  Cache system exists: {brain.cached_plan_system is not None}")

# Test the flow
print("\nTesting cache flow...")

# First cycle - should be a cache miss
print("\nCycle 1:")
sensory_input = [0.5] * 24
start = time.perf_counter()
motor, state = brain.process_robot_cycle(sensory_input)
cycle_time = time.perf_counter() - start

cache_stats = state.get('cached_planning', {})
print(f"  Time: {cycle_time*1000:.0f}ms")
print(f"  Cache stats: {cache_stats}")
print(f"  Background planning active: {cache_stats.get('background_planning', False)}")

# Wait a bit for background planning
print("\nWaiting 3 seconds for background planning...")
time.sleep(3)

# Second cycle - might get a cache hit if planning completed
print("\nCycle 2:")
start = time.perf_counter()
motor, state = brain.process_robot_cycle(sensory_input)
cycle_time = time.perf_counter() - start

cache_stats = state.get('cached_planning', {})
print(f"  Time: {cycle_time*1000:.0f}ms")
print(f"  Cache stats: {cache_stats}")

# Check the cache system directly
if brain.cached_plan_system:
    print(f"\nDirect cache inspection:")
    print(f"  Cached plans: {len(brain.cached_plan_system.cached_plans)}")
    print(f"  Background future exists: {brain.cached_plan_system.background_future is not None}")
    if brain.cached_plan_system.background_future:
        print(f"  Background future done: {brain.cached_plan_system.background_future.done()}")
    
    # Try to manually check cache
    brain.cached_plan_system.cleanup_stale_plans()
    print(f"  Plans after cleanup: {len(brain.cached_plan_system.cached_plans)}")

# Let's manually trigger the reactive path to compare
print("\nTesting reactive-only path (no GPU simulation):")
brain.use_future_simulation = False
start = time.perf_counter()
motor, state = brain.process_robot_cycle(sensory_input)
reactive_time = time.perf_counter() - start
print(f"  Reactive time: {reactive_time*1000:.0f}ms")