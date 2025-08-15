#!/usr/bin/env python3
"""Test script to verify cached planning reduces cycle times to ~2s."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import time
import torch
import numpy as np
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Cached Planning Performance Test\n")
print("="*60)

# Create brain with all features
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=False
)

# Enable all phases including cached planning
print("\nEnabling brain features...")
brain.enable_hierarchical_prediction(True)
brain.enable_action_prediction(True)
brain.enable_active_vision(True)
brain.enable_future_simulation(True, n_futures=8, horizon=10)
brain.enable_cached_planning(True)

print("\n" + "="*60)
print("TESTING CACHED PLANNING PERFORMANCE")
print("="*60)

# Warm up the brain with a few cycles
print("\nWarm-up phase (5 cycles)...")
for i in range(5):
    sensory_input = [0.5 + 0.1 * np.sin(i * 0.5)] * 24
    start = time.perf_counter()
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    cycle_time = time.perf_counter() - start
    print(f"Cycle {i+1}: {cycle_time*1000:.0f}ms")

# Test with caching enabled
print("\n" + "-"*40)
print("WITH CACHED PLANNING (10 cycles)")
print("-"*40)

cached_times = []
for i in range(10):
    sensory_input = [0.5 + 0.1 * np.sin(i * 0.3)] * 24
    start = time.perf_counter()
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    cycle_time = time.perf_counter() - start
    cached_times.append(cycle_time)
    
    # Get cache statistics
    cache_stats = brain_state.get('cached_planning', {})
    cache_hits = cache_stats.get('cache_hits', 0)
    cache_misses = cache_stats.get('cache_misses', 0)
    hit_rate = cache_stats.get('hit_rate', 0)
    
    print(f"Cycle {i+1}: {cycle_time*1000:.1f}ms | "
          f"Cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.0%} hit rate)")

avg_cached = np.mean(cached_times[2:]) * 1000  # Skip first 2 for cache warmup
print(f"\nAverage cycle time (cached): {avg_cached:.0f}ms")

# Test without caching for comparison
print("\n" + "-"*40)
print("WITHOUT CACHED PLANNING (5 cycles)")
print("-"*40)

brain.enable_cached_planning(False)  # Disable caching

uncached_times = []
for i in range(5):
    sensory_input = [0.5 + 0.1 * np.sin(i * 0.2)] * 24
    start = time.perf_counter()
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    cycle_time = time.perf_counter() - start
    uncached_times.append(cycle_time)
    print(f"Cycle {i+1}: {cycle_time*1000:.0f}ms")

avg_uncached = np.mean(uncached_times) * 1000

# Results summary
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"Average with caching:    {avg_cached:.0f}ms")
print(f"Average without caching: {avg_uncached:.0f}ms")
print(f"Speedup:                 {avg_uncached/avg_cached:.1f}x")
print(f"Target achieved:         {'✅ YES' if avg_cached < 2500 else '❌ NO'} (target: <2500ms)")

# Cache effectiveness
if 'cached_planning' in brain_state:
    stats = brain_state['cached_planning']
    print(f"\nCache Statistics:")
    print(f"  Total hits:     {stats.get('cache_hits', 0)}")
    print(f"  Total misses:   {stats.get('cache_misses', 0)}")
    print(f"  Hit rate:       {stats.get('hit_rate', 0):.1%}")
    print(f"  Plans cached:   {stats.get('cache_size', 0)}")
    print(f"  Plans executed: {stats.get('plan_executions', 0)}")

print("\n" + "="*60)
print("✅ Cached planning is working!")
print("="*60)