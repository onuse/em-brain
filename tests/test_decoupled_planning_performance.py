#!/usr/bin/env python3
"""
Test to demonstrate decoupled planning performance improvement.
Shows how the brain remains responsive while planning in background.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import time
import torch
import numpy as np
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Decoupled Planning Performance Demonstration")
print("="*60)

# Create brain with realistic settings
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

# Enable all features
brain.enable_hierarchical_prediction(True)
brain.enable_action_prediction(True)
brain.enable_active_vision(True)
brain.enable_future_simulation(True, n_futures=8, horizon=10)

print("\nTest 1: Blocking Mode (Original Implementation)")
print("-"*40)
brain.enable_cached_planning(False)  # Disable caching

blocking_times = []
for i in range(3):
    sensory_input = [0.5 + 0.1 * np.sin(i * 0.5)] * 24
    start = time.perf_counter()
    motor, state = brain.process_robot_cycle(sensory_input)
    cycle_time = time.perf_counter() - start
    blocking_times.append(cycle_time)
    print(f"Cycle {i+1}: {cycle_time*1000:.0f}ms (blocking on GPU simulation)")

avg_blocking = np.mean(blocking_times) * 1000

print("\nTest 2: Decoupled Mode (With Cached Planning)")
print("-"*40)
brain.enable_cached_planning(True)  # Enable caching

# First cycle will be slow (cache miss)
sensory_input = [0.5] * 24
start = time.perf_counter()
motor, state = brain.process_robot_cycle(sensory_input)
first_cycle_time = time.perf_counter() - start
print(f"Cycle 1: {first_cycle_time*1000:.0f}ms (cache miss, using reactive action)")

# Subsequent cycles should be fast
decoupled_times = []
for i in range(5):
    # Small variations in input
    sensory_input = [0.5 + 0.01 * i] * 24
    start = time.perf_counter()
    motor, state = brain.process_robot_cycle(sensory_input)
    cycle_time = time.perf_counter() - start
    decoupled_times.append(cycle_time)
    
    # Check cache directly
    cache_info = ""
    if brain.cached_plan_system:
        cache_size = len(brain.cached_plan_system.cached_plans)
        cache_hits = brain.cached_plan_system.cache_hits
        cache_misses = brain.cached_plan_system.cache_misses
        cache_info = f" | Cache: {cache_size} plans, {cache_hits} hits, {cache_misses} misses"
    
    print(f"Cycle {i+2}: {cycle_time*1000:.0f}ms (responsive){cache_info}")

avg_decoupled = np.mean(decoupled_times) * 1000

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Blocking mode average:    {avg_blocking:.0f}ms per cycle")
print(f"Decoupled mode average:   {avg_decoupled:.0f}ms per cycle")
print(f"Performance improvement:  {avg_blocking/avg_decoupled:.1f}x faster")
print(f"Responsiveness achieved:  {'✅ YES' if avg_decoupled < 2000 else '⚠️  PARTIAL'}")

# Final cache statistics
if brain.cached_plan_system:
    stats = brain.cached_plan_system.get_statistics()
    print(f"\nCache Performance:")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Total hits: {stats['cache_hits']}")
    print(f"  Total misses: {stats['cache_misses']}")
    print(f"  Plans cached: {stats['cache_size']}")

print("\n" + "="*60)
print("Key Insights:")
print("- First cycle uses reactive action (fast but simple)")
print("- Background planning runs while robot continues")
print("- Future cycles can use sophisticated plans from cache")
print("- Brain remains responsive throughout")
print("="*60)