#!/usr/bin/env python3
"""Profile brain performance to identify bottlenecks."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import time
import cProfile
import pstats
from brains.field.core_brain import UnifiedFieldBrain

def profile_brain_cycle():
    """Profile a single brain cycle."""
    print("Creating brain with spatial resolution 8...")
    brain = UnifiedFieldBrain(spatial_resolution=8, quiet_mode=True)
    
    print(f"Field shape: {list(brain.unified_field.shape)}")
    print(f"Total elements: {brain.unified_field.numel():,}")
    print(f"Memory: {(brain.unified_field.numel() * 4) / (1024**2):.1f} MB")
    
    # Warm up
    print("\nWarming up...")
    for _ in range(2):
        brain.process_robot_cycle([0.5] * 24)
    
    # Profile
    print("\nProfiling brain cycle...")
    profiler = cProfile.Profile()
    
    t0 = time.perf_counter()
    profiler.enable()
    brain.process_robot_cycle([0.5] * 24)
    profiler.disable()
    elapsed = (time.perf_counter() - t0) * 1000
    
    print(f"\nCycle time: {elapsed:.1f}ms")
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    print("\n=== Top 20 time-consuming functions ===")
    stats.print_stats(20)
    
    # Also show by total time
    print("\n=== Top 10 by total time ===")
    stats.sort_stats('time')
    stats.print_stats(10)

if __name__ == "__main__":
    profile_brain_cycle()