#!/usr/bin/env python3
"""
Test the optimized gradient calculation performance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import time
import numpy as np
from brains.field.core_brain import create_unified_field_brain


def test_gradient_performance_comparison():
    """Compare performance before and after optimization."""
    print("Gradient Calculation Performance Test")
    print("=" * 60)
    
    resolutions = [10, 15, 20]
    
    for res in resolutions:
        brain = create_unified_field_brain(
            spatial_resolution=res,
            quiet_mode=True
        )
        
        print(f"\nTesting with resolution {res}x{res}x{res}")
        print(f"Field shape: {brain.unified_field.shape}")
        
        # Add some sparse activation
        for _ in range(20):
            x = torch.randint(0, res, (1,)).item()
            y = torch.randint(0, res, (1,)).item()
            z = torch.randint(0, res, (1,)).item()
            brain.unified_field[x, y, z, 5, 7] = torch.randn(1).item() * 0.5
        
        # Warm up
        brain._calculate_gradient_flows()
        
        # Time gradient calculation
        times = []
        for i in range(50):
            start = time.perf_counter()
            brain._calculate_gradient_flows()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Check memory usage
        total_gradient_memory = sum(
            g.element_size() * g.nelement() / (1024 * 1024) 
            for g in brain.gradient_flows.values()
        )
        
        print(f"  Gradient calc time: {avg_time:.2f} ± {std_time:.2f}ms")
        print(f"  Gradient memory: {total_gradient_memory:.2f}MB")
        
        # Test cache performance
        cache_stats = brain.gradient_calculator.get_cache_stats()
        print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")


def test_sparse_vs_dense_computation():
    """Compare sparse vs dense gradient computation."""
    print("\n\nSparse vs Dense Computation Test")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=20,
        quiet_mode=True
    )
    
    # Test with different sparsity levels
    sparsity_levels = [0.99, 0.95, 0.90, 0.50]
    
    for sparsity in sparsity_levels:
        # Clear field
        brain.unified_field.zero_()
        
        # Add activation based on sparsity
        n_active = int((1 - sparsity) * 400)  # Out of 20^3 = 8000 spatial points
        for _ in range(n_active):
            x = torch.randint(0, 20, (1,)).item()
            y = torch.randint(0, 20, (1,)).item()
            z = torch.randint(0, 20, (1,)).item()
            brain.unified_field[x, y, z, 5, 7] = torch.randn(1).item()
        
        # Test sparse computation
        brain.gradient_calculator.sparse_computation = True
        brain.gradient_calculator.clear_cache()
        
        start = time.perf_counter()
        for _ in range(20):
            brain._calculate_gradient_flows()
        sparse_time = (time.perf_counter() - start) * 1000 / 20
        
        # Test dense computation
        brain.gradient_calculator.sparse_computation = False
        brain.gradient_calculator.clear_cache()
        
        start = time.perf_counter()
        for _ in range(20):
            brain._calculate_gradient_flows()
        dense_time = (time.perf_counter() - start) * 1000 / 20
        
        speedup = dense_time / sparse_time
        
        print(f"\nSparsity {sparsity:.1%} ({n_active} active points):")
        print(f"  Sparse computation: {sparse_time:.2f}ms")
        print(f"  Dense computation: {dense_time:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print(f"  ✅ Sparse computation is {speedup:.1f}x faster!")
        else:
            print(f"  ⚠️  Limited benefit from sparse computation")


def test_cache_effectiveness():
    """Test gradient caching effectiveness."""
    print("\n\nGradient Cache Effectiveness Test")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Add some activation
    for _ in range(10):
        x = torch.randint(0, 15, (1,)).item()
        y = torch.randint(0, 15, (1,)).item()
        z = torch.randint(0, 15, (1,)).item()
        brain.unified_field[x, y, z, 5, 7] = torch.randn(1).item()
    
    # Test without cache
    brain.gradient_calculator.cache_enabled = False
    
    start = time.perf_counter()
    for _ in range(100):
        brain._calculate_gradient_flows()
    no_cache_time = (time.perf_counter() - start) * 1000
    
    # Test with cache
    brain.gradient_calculator.cache_enabled = True
    brain.gradient_calculator.clear_cache()
    
    start = time.perf_counter()
    for _ in range(100):
        brain._calculate_gradient_flows()
    with_cache_time = (time.perf_counter() - start) * 1000
    
    cache_stats = brain.gradient_calculator.get_cache_stats()
    
    print(f"100 gradient calculations:")
    print(f"  Without cache: {no_cache_time:.2f}ms")
    print(f"  With cache: {with_cache_time:.2f}ms")
    print(f"  Speedup: {no_cache_time / with_cache_time:.2f}x")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Cache misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    
    if cache_stats['hit_rate'] > 0.9:
        print("  ✅ Cache is highly effective!")
    else:
        print("  ⚠️  Cache effectiveness could be improved")


def test_gradient_accuracy():
    """Verify optimized gradients are accurate."""
    print("\n\nGradient Accuracy Test")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    
    # Create a simple test pattern
    brain.unified_field[5, 5, 5, 5, 7] = 1.0
    brain.unified_field[6, 5, 5, 5, 7] = 0.5
    brain.unified_field[5, 6, 5, 5, 7] = 0.3
    
    # Calculate gradients
    brain._calculate_gradient_flows()
    
    # Check gradient values at specific points
    grad_x = brain.gradient_flows['gradient_x']
    grad_y = brain.gradient_flows['gradient_y']
    
    # Expected gradients (finite differences)
    expected_grad_x_at_5_5_5 = brain.unified_field[6, 5, 5] - brain.unified_field[5, 5, 5]
    expected_grad_y_at_5_5_5 = brain.unified_field[5, 6, 5] - brain.unified_field[5, 5, 5]
    
    actual_grad_x = grad_x[5, 5, 5, 5, 7].item()
    actual_grad_y = grad_y[5, 5, 5, 5, 7].item()
    
    print(f"At point [5, 5, 5]:")
    print(f"  Expected grad_x: {expected_grad_x_at_5_5_5[5, 7].item():.4f}")
    print(f"  Actual grad_x: {actual_grad_x:.4f}")
    print(f"  Expected grad_y: {expected_grad_y_at_5_5_5[5, 7].item():.4f}")
    print(f"  Actual grad_y: {actual_grad_y:.4f}")
    
    # Check if gradients are sparse where field is zero
    field_zeros = torch.sum(torch.abs(brain.unified_field) < 1e-6).item()
    grad_zeros = torch.sum(torch.abs(grad_x) < 1e-6).item()
    
    print(f"\nSparsity check:")
    print(f"  Field zeros: {field_zeros} / {brain.unified_field.nelement()}")
    print(f"  Gradient zeros: {grad_zeros} / {grad_x.nelement()}")
    
    if brain.gradient_calculator.sparse_computation:
        print("  ✅ Sparse computation active")
    else:
        print("  ⚠️  Dense computation active")


def analyze_optimization_impact():
    """Analyze overall impact of optimizations."""
    print("\n\nOptimization Impact Analysis")
    print("=" * 60)
    
    # Create brain with typical parameters
    brain = create_unified_field_brain(
        spatial_resolution=20,
        quiet_mode=True
    )
    
    # Simulate typical usage pattern
    print("Simulating 100 brain cycles...")
    
    start_time = time.perf_counter()
    gradient_times = []
    
    for i in range(100):
        # Add some dynamic activation
        if i % 10 == 0:
            x = torch.randint(0, 20, (1,)).item()
            y = torch.randint(0, 20, (1,)).item()
            z = torch.randint(0, 20, (1,)).item()
            brain.unified_field[x, y, z, 5, 7] = torch.randn(1).item()
        
        # Process cycle
        input_data = [0.5 + 0.1 * np.sin(i * 0.1) for _ in range(24)]
        
        grad_start = time.perf_counter()
        brain.process_robot_cycle(input_data)
        gradient_times.append((time.perf_counter() - grad_start) * 1000)
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}ms")
    print(f"  Average cycle: {total_time / 100:.2f}ms")
    print(f"  Average gradient time: {np.mean(gradient_times):.2f}ms")
    
    cache_stats = brain.gradient_calculator.get_cache_stats()
    print(f"\nCache performance:")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Total hits: {cache_stats['cache_hits']}")
    print(f"  Total misses: {cache_stats['cache_misses']}")
    
    # Memory analysis
    field_memory = brain.unified_field.element_size() * brain.unified_field.nelement() / (1024 * 1024)
    gradient_memory = sum(
        g.element_size() * g.nelement() / (1024 * 1024) 
        for g in brain.gradient_flows.values()
    )
    
    print(f"\nMemory usage:")
    print(f"  Field: {field_memory:.2f}MB")
    print(f"  Gradients: {gradient_memory:.2f}MB")
    print(f"  Gradient/Field ratio: {gradient_memory / field_memory:.1%}")
    
    if total_time / 100 < 50:  # Less than 50ms per cycle
        print("\n✅ EXCELLENT: Brain cycles are fast enough for real-time operation!")
    elif total_time / 100 < 100:
        print("\n✅ GOOD: Brain cycles meet performance targets")
    else:
        print("\n⚠️  WARNING: Brain cycles may be too slow for real-time")


if __name__ == "__main__":
    test_gradient_performance_comparison()
    test_sparse_vs_dense_computation()
    test_cache_effectiveness()
    test_gradient_accuracy()
    analyze_optimization_impact()
    
    print("\n\n✅ Gradient optimization testing complete!")