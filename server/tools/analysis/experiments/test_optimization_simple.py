#!/usr/bin/env python3
"""
Simple Optimization Test

Tests specific optimizations in isolation.
"""

import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
from src.brains.field.pattern_cache_pool import PatternCachePool


def test_pattern_cache():
    """Test pattern cache pool performance."""
    print("Testing Pattern Cache Pool")
    print("-" * 50)
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True,
        use_optimized=False  # We'll test cache separately
    )
    
    # Create cache pool
    cache_pool = PatternCachePool(
        field_shape=brain.tensor_shape,
        max_patterns=50,
        device=brain.device
    )
    
    # Time regular pattern extraction
    print("\nRegular pattern extraction:")
    times = []
    for _ in range(20):
        start = time.perf_counter()
        patterns = brain.pattern_system.extract_patterns(brain.unified_field, n_patterns=10)
        times.append((time.perf_counter() - start) * 1000)
    avg_regular = sum(times) / len(times)
    print(f"  Average time: {avg_regular:.2f}ms")
    
    # Time cached pattern extraction
    print("\nCached pattern extraction:")
    times = []
    for _ in range(20):
        start = time.perf_counter()
        patterns = cache_pool.extract_patterns_fast(brain.unified_field, n_patterns=10)
        times.append((time.perf_counter() - start) * 1000)
    avg_cached = sum(times) / len(times)
    print(f"  Average time: {avg_cached:.2f}ms")
    
    print(f"\nSpeedup: {avg_regular / avg_cached:.2f}x")
    
    # Test different pattern counts
    print("\nPattern count scaling:")
    for n_patterns in [1, 5, 10, 20]:
        start = time.perf_counter()
        for _ in range(10):
            patterns = cache_pool.extract_patterns_fast(brain.unified_field, n_patterns=n_patterns)
        elapsed = (time.perf_counter() - start) * 100  # ms per call
        print(f"  {n_patterns} patterns: {elapsed:.2f}ms")


def test_optimized_cycle():
    """Test optimized brain cycle."""
    print("\n\nTesting Optimized Brain Cycle")
    print("-" * 50)
    
    # Create regular brain
    brain_normal = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True,
        use_optimized=False
    )
    
    # Create optimized brain
    brain_opt = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True,
        use_optimized=True
    )
    
    # Disable predictive actions for faster test
    brain_opt.enable_predictive_actions(False)
    
    sensory_input = [0.1] * 24
    
    # Warmup
    for _ in range(5):
        brain_normal.process_robot_cycle(sensory_input)
        brain_opt.process_robot_cycle(sensory_input)
    
    # Benchmark normal
    print("\nNormal brain:")
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _, state = brain_normal.process_robot_cycle(sensory_input)
        times.append((time.perf_counter() - start) * 1000)
    avg_normal = sum(times) / len(times)
    print(f"  Average cycle: {avg_normal:.2f}ms")
    print(f"  Min/Max: {min(times):.2f}ms / {max(times):.2f}ms")
    
    # Benchmark optimized
    print("\nOptimized brain:")
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _, state = brain_opt.process_robot_cycle(sensory_input)
        times.append((time.perf_counter() - start) * 1000)
    avg_opt = sum(times) / len(times)
    print(f"  Average cycle: {avg_opt:.2f}ms")
    print(f"  Min/Max: {min(times):.2f}ms / {max(times):.2f}ms")
    
    print(f"\nSpeedup: {avg_normal / avg_opt:.2f}x")
    
    # Test with different settings
    print("\nOptimization settings impact:")
    
    # With predictive actions
    brain_opt.enable_predictive_actions(True)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        brain_opt.process_robot_cycle(sensory_input)
        times.append((time.perf_counter() - start) * 1000)
    print(f"  With predictive actions: {sum(times) / len(times):.2f}ms")
    
    # Minimal patterns
    brain_opt.enable_predictive_actions(False)
    brain_opt.set_pattern_extraction_limit(1)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        brain_opt.process_robot_cycle(sensory_input)
        times.append((time.perf_counter() - start) * 1000)
    print(f"  With 1 pattern only: {sum(times) / len(times):.2f}ms")


if __name__ == "__main__":
    test_pattern_cache()
    test_optimized_cycle()