#!/usr/bin/env python3
"""
Test Optimized Brain Performance

Compares optimized vs non-optimized brain performance.
"""

import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.core.simplified_brain_factory import SimplifiedBrainFactory
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def benchmark_brain(brain, cycles: int = 100, warmup: int = 10):
    """Benchmark brain performance."""
    sensory_input = [0.1] * 24
    
    # Warmup
    for _ in range(warmup):
        brain.process_robot_cycle(sensory_input)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    cycle_times = []
    for _ in range(cycles):
        cycle_start = time.perf_counter()
        _, state = brain.process_robot_cycle(sensory_input)
        cycle_times.append((time.perf_counter() - cycle_start) * 1000)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = (time.perf_counter() - start_time) * 1000
    
    return {
        'total_ms': total_time,
        'avg_cycle_ms': total_time / cycles,
        'min_cycle_ms': min(cycle_times),
        'max_cycle_ms': max(cycle_times),
        'cycles': cycles
    }


def compare_optimizations():
    """Compare optimized vs non-optimized performance."""
    print("Performance Optimization Comparison")
    print("=" * 50)
    
    # Create non-optimized brain
    print("\nCreating non-optimized brain...")
    brain_normal = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True,
        use_optimized=False
    )
    
    # Create optimized brain
    print("Creating optimized brain...")
    brain_optimized = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True,
        use_optimized=True
    )
    
    print(f"\nDevice: {brain_normal.device}")
    print(f"Field shape: {brain_normal.unified_field.shape}")
    print(f"Field size: {brain_normal.unified_field.numel() * 4 / 1024 / 1024:.2f}MB")
    
    # Test different optimization settings
    optimization_configs = [
        ("Default optimized", lambda b: None),
        ("No predictive actions", lambda b: b.enable_predictive_actions(False)),
        ("Reduced patterns (3)", lambda b: b.set_pattern_extraction_limit(3)),
        ("Minimal patterns (1)", lambda b: b.set_pattern_extraction_limit(1))
    ]
    
    # Benchmark non-optimized
    print("\nBenchmarking non-optimized brain...")
    baseline = benchmark_brain(brain_normal, cycles=100)
    print(f"  Average cycle time: {baseline['avg_cycle_ms']:.2f}ms")
    
    print("\nOptimization Results:")
    print("-" * 60)
    print(f"{'Configuration':<30} {'Avg(ms)':>10} {'Speedup':>10} {'Min(ms)':>10}")
    print("-" * 60)
    
    for config_name, apply_config in optimization_configs:
        # Reset and apply configuration
        brain_optimized = SimplifiedUnifiedBrain(
            sensory_dim=24,
            motor_dim=4,
            spatial_resolution=32,
            quiet_mode=True,
            use_optimized=True
        )
        apply_config(brain_optimized)
        
        # Benchmark
        result = benchmark_brain(brain_optimized, cycles=100)
        speedup = baseline['avg_cycle_ms'] / result['avg_cycle_ms']
        
        print(f"{config_name:<30} {result['avg_cycle_ms']:>10.2f} {speedup:>10.2f}x {result['min_cycle_ms']:>10.2f}")
    
    # Test batch processing potential
    print("\n\nBatch Processing Analysis:")
    print("-" * 50)
    
    # Single brain baseline
    single_time = baseline['avg_cycle_ms']
    print(f"Single brain: {single_time:.2f}ms per cycle")
    
    # Simulate multiple brains
    batch_sizes = [2, 4, 8]
    for batch_size in batch_sizes:
        brains = [
            SimplifiedUnifiedBrain(
                sensory_dim=24,
                motor_dim=4,
                spatial_resolution=32,
                quiet_mode=True,
                use_optimized=True
            ) for _ in range(batch_size)
        ]
        
        # Time sequential processing
        start = time.perf_counter()
        for _ in range(10):
            for brain in brains:
                brain.process_robot_cycle([0.1] * 24)
        sequential_time = (time.perf_counter() - start) * 1000 / 10
        
        print(f"\nBatch size {batch_size}:")
        print(f"  Sequential: {sequential_time:.2f}ms total, {sequential_time/batch_size:.2f}ms per brain")
        print(f"  Efficiency: {(single_time * batch_size) / sequential_time:.1%}")
    
    # Test pattern cache effectiveness
    if hasattr(brain_optimized, 'pattern_cache_pool'):
        print("\n\nPattern Cache Statistics:")
        print("-" * 50)
        stats = brain_optimized.pattern_cache_pool.get_cache_stats()
        print(f"Cache hits: {stats['cache_hits']:.0f}")
        print(f"Cache misses: {stats['cache_misses']:.0f}")
        print(f"Hit rate: {stats['hit_rate']:.1%}")


if __name__ == "__main__":
    compare_optimizations()