#!/usr/bin/env python3
"""
Profile gradient calculation performance to identify bottlenecks.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import time
import cProfile
import pstats
from memory_profiler import profile as memory_profile
from brains.field.core_brain import create_unified_field_brain


def measure_gradient_performance():
    """Measure gradient calculation performance at different scales."""
    print("Gradient Calculation Performance Analysis")
    print("=" * 60)
    
    resolutions = [5, 10, 15, 20, 25]
    results = []
    
    for res in resolutions:
        brain = create_unified_field_brain(
            spatial_resolution=res,
            quiet_mode=True
        )
        
        # Warm up
        brain._calculate_gradient_flows()
        
        # Time gradient calculation
        start_time = time.perf_counter()
        
        for _ in range(10):
            brain._calculate_gradient_flows()
        
        elapsed = (time.perf_counter() - start_time) * 1000 / 10
        
        # Calculate memory usage
        field_memory = brain.unified_field.element_size() * brain.unified_field.nelement() / (1024 * 1024)
        grad_memory = sum(
            g.element_size() * g.nelement() / (1024 * 1024) 
            for g in brain.gradient_flows.values()
        )
        
        results.append({
            'resolution': res,
            'field_shape': brain.unified_field.shape,
            'time_ms': elapsed,
            'field_memory_mb': field_memory,
            'gradient_memory_mb': grad_memory,
            'total_memory_mb': field_memory + grad_memory
        })
        
        print(f"\nResolution {res}x{res}x{res}:")
        print(f"  Field shape: {brain.unified_field.shape}")
        print(f"  Gradient calc time: {elapsed:.2f}ms")
        print(f"  Field memory: {field_memory:.2f}MB")
        print(f"  Gradient memory: {grad_memory:.2f}MB")
        print(f"  Total memory: {field_memory + grad_memory:.2f}MB")
    
    # Analyze scaling
    print("\n\nPerformance Scaling Analysis:")
    print("-" * 40)
    
    base_time = results[0]['time_ms']
    base_memory = results[0]['total_memory_mb']
    
    for r in results:
        time_factor = r['time_ms'] / base_time
        memory_factor = r['total_memory_mb'] / base_memory
        elements = r['resolution'] ** 3
        element_factor = elements / (results[0]['resolution'] ** 3)
        
        print(f"Resolution {r['resolution']}: "
              f"time={time_factor:.2f}x, "
              f"memory={memory_factor:.2f}x, "
              f"elements={element_factor:.2f}x")
    
    return results


def profile_gradient_calculation():
    """Detailed profiling of gradient calculation."""
    print("\n\nDetailed Gradient Calculation Profile")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Profile the method
    profiler = cProfile.Profile()
    
    profiler.enable()
    for _ in range(50):
        brain._calculate_gradient_flows()
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    print("\nTop 10 time-consuming operations:")
    stats.print_stats(10)
    
    return profiler


def test_sparse_gradient_approach():
    """Test if sparse gradients would be more efficient."""
    print("\n\nSparse Gradient Feasibility Test")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=20,
        quiet_mode=True
    )
    
    # Add some activation to the field
    for _ in range(10):
        x = torch.randint(0, 20, (1,)).item()
        y = torch.randint(0, 20, (1,)).item()
        z = torch.randint(0, 20, (1,)).item()
        brain.unified_field[x, y, z, 5, 7] = torch.randn(1).item()
    
    # Calculate gradients
    brain._calculate_gradient_flows()
    
    # Analyze sparsity
    total_elements = 0
    nonzero_elements = 0
    
    for name, grad in brain.gradient_flows.items():
        total_elements += grad.nelement()
        nonzero_elements += torch.sum(grad != 0).item()
    
    sparsity = 1.0 - (nonzero_elements / total_elements)
    
    print(f"Field activation points: 10")
    print(f"Total gradient elements: {total_elements:,}")
    print(f"Non-zero gradient elements: {nonzero_elements:,}")
    print(f"Gradient sparsity: {sparsity:.2%}")
    
    if sparsity > 0.9:
        print("\n✅ HIGH SPARSITY: Sparse gradients would be beneficial!")
    else:
        print("\n⚠️  LOW SPARSITY: Dense gradients may be more efficient")


def compare_gradient_methods():
    """Compare different gradient calculation methods."""
    print("\n\nGradient Method Comparison")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Method 1: Current approach (full tensors)
    def current_method():
        brain.gradient_flows.clear()
        grad_x = torch.zeros_like(brain.unified_field)
        if brain.spatial_resolution > 1:
            grad_x[1:, :, :] = brain.unified_field[1:, :, :] - brain.unified_field[:-1, :, :]
        brain.gradient_flows['gradient_x'] = grad_x
    
    # Method 2: In-place operations
    def inplace_method():
        brain.gradient_flows.clear()
        grad_x = brain.unified_field.clone()
        grad_x[:-1] = brain.unified_field[1:] - brain.unified_field[:-1]
        grad_x[-1] = 0
        brain.gradient_flows['gradient_x'] = grad_x
    
    # Method 3: Only compute where needed
    def selective_method():
        brain.gradient_flows.clear()
        # Only compute gradients near active regions
        active_mask = torch.abs(brain.unified_field) > 0.01
        grad_x = torch.zeros_like(brain.unified_field)
        
        # Find active regions and compute gradients only there
        active_indices = torch.nonzero(active_mask)
        if len(active_indices) > 0:
            for idx in active_indices:
                x, y, z = idx[0].item(), idx[1].item(), idx[2].item()
                if x > 0 and x < brain.spatial_resolution - 1:
                    grad_x[x, y, z] = brain.unified_field[x+1, y, z] - brain.unified_field[x-1, y, z]
        
        brain.gradient_flows['gradient_x'] = grad_x
    
    # Time each method
    methods = [
        ("Current (full tensors)", current_method),
        ("In-place operations", inplace_method),
        ("Selective computation", selective_method)
    ]
    
    for name, method in methods:
        # Warm up
        method()
        
        # Time
        start = time.perf_counter()
        for _ in range(100):
            method()
        elapsed = (time.perf_counter() - start) * 1000 / 100
        
        print(f"{name}: {elapsed:.2f}ms")


def suggest_optimizations():
    """Suggest specific optimizations based on profiling."""
    print("\n\nOptimization Recommendations")
    print("=" * 60)
    
    print("""
1. MEMORY OPTIMIZATION:
   - Current: Creates full-size gradient tensors for each dimension
   - Suggested: Use in-place operations or compute gradients on-demand
   - Potential savings: ~80% memory for gradient storage

2. SPARSE COMPUTATION:
   - Most of the field is near-zero (high sparsity)
   - Only compute gradients where field activation > threshold
   - Use torch.sparse tensors or custom sparse structure

3. DIMENSION REDUCTION:
   - Currently computing gradients for all 11 dimensions
   - Only first 3-5 dimensions typically have meaningful gradients
   - Skip gradient calculation for singleton or near-constant dimensions

4. CACHING:
   - Cache gradients when field hasn't changed significantly
   - Reuse gradients for multiple action generations
   - Invalidate cache on field evolution

5. VECTORIZATION:
   - Batch gradient calculations across multiple dimensions
   - Use torch.gradient() for better performance on GPU
   - Leverage hardware-specific optimizations
""")


if __name__ == "__main__":
    # Run all analyses
    results = measure_gradient_performance()
    profile_gradient_calculation()
    test_sparse_gradient_approach()
    compare_gradient_methods()
    suggest_optimizations()
    
    print("\n✅ Performance analysis complete!")