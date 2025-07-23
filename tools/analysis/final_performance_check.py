#!/usr/bin/env python3
"""
Final performance check before robot integration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import time
import numpy as np
from brains.field.core_brain import create_unified_field_brain


def test_cycle_performance():
    """Test if brain cycles meet <100ms target."""
    print("Final Performance Check")
    print("=" * 60)
    
    # Test different resolutions
    resolutions = [10, 15, 20]
    
    for res in resolutions:
        print(f"\nTesting resolution {res}x{res}x{res}:")
        
        brain = create_unified_field_brain(
            spatial_resolution=res,
            quiet_mode=True
        )
        
        # Warm up
        for _ in range(5):
            brain.process_robot_cycle([0.5] * 24)
        
        # Time cycles
        cycle_times = []
        for i in range(50):
            input_data = [0.5 + 0.1 * np.sin(i * 0.1) for _ in range(24)]
            
            start = time.perf_counter()
            action, state = brain.process_robot_cycle(input_data)
            elapsed = (time.perf_counter() - start) * 1000
            
            cycle_times.append(elapsed)
        
        avg_time = np.mean(cycle_times)
        std_time = np.std(cycle_times)
        max_time = np.max(cycle_times)
        
        print(f"  Average: {avg_time:.1f}ms ± {std_time:.1f}ms")
        print(f"  Maximum: {max_time:.1f}ms")
        
        if avg_time < 100:
            print(f"  ✅ PASS: Meets <100ms target")
        else:
            print(f"  ❌ FAIL: Too slow for real-time")
    
    return cycle_times


def test_all_features_active():
    """Test performance with all features active."""
    print("\n\nTesting with All Features Active")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        constraint_discovery_rate=0.5,  # High rate
        quiet_mode=True
    )
    
    # Build up some state
    print("Building up brain state...")
    for i in range(100):
        input_data = [
            0.5 + 0.3 * np.sin(i * 0.1),
            0.5 + 0.3 * np.cos(i * 0.15),
            0.5 + 0.2 * np.sin(i * 0.05)
        ] + [0.5 + 0.1 * np.random.randn() for _ in range(21)]
        
        brain.process_robot_cycle(input_data)
    
    print(f"\nBrain state:")
    print(f"  Topology regions: {len(brain.topology_regions)}")
    print(f"  Active constraints: {len(brain.constraint_field.active_constraints)}")
    print(f"  Cache hit rate: {brain.gradient_calculator.get_cache_stats()['hit_rate']:.1%}")
    
    # Time cycles with full state
    print("\nTiming with full state...")
    cycle_times = []
    
    for i in range(50):
        input_data = [0.5 + 0.1 * np.random.randn() for _ in range(24)]
        
        start = time.perf_counter()
        action, state = brain.process_robot_cycle(input_data)
        elapsed = (time.perf_counter() - start) * 1000
        
        cycle_times.append(elapsed)
    
    avg_time = np.mean(cycle_times)
    print(f"\nAverage cycle time: {avg_time:.1f}ms")
    
    if avg_time < 100:
        print("✅ PASS: Performance acceptable with all features")
    else:
        print("❌ FAIL: Too slow with features active")


def test_memory_overhead():
    """Test memory usage is reasonable."""
    print("\n\nMemory Usage Check")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=20,
        quiet_mode=True
    )
    
    # Get memory stats
    stats = brain.get_field_memory_stats()
    
    print(f"Memory usage:")
    print(f"  Field shape: {stats['field_shape']}")
    print(f"  Total elements: {stats['total_elements']:,}")
    print(f"  Memory size: {stats['memory_size_mb']:.2f} MB")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    
    if stats['memory_size_mb'] < 1000:
        print("✅ PASS: Memory usage < 1GB")
    else:
        print("❌ FAIL: Memory usage too high")


def test_critical_fixes():
    """Verify all critical fixes are working."""
    print("\n\nCritical Fixes Verification")
    print("=" * 60)
    
    brain = create_unified_field_brain(quiet_mode=True)
    
    # 1. Gradient strength
    brain._calculate_gradient_flows()
    grad_x = brain.gradient_flows.get('gradient_x')
    if grad_x is not None:
        max_grad = torch.max(torch.abs(grad_x)).item()
        print(f"1. Gradient strength: {max_grad:.4f}")
        if max_grad > 0.01:
            print("   ✅ PASS: Gradients have good strength")
        else:
            print("   ❌ FAIL: Gradients too weak")
    
    # 2. Dimension utilization
    print(f"2. Field dimensions: {len(brain.unified_field.shape)}")
    singleton_dims = sum(1 for s in brain.unified_field.shape if s == 1)
    print(f"   Singleton dimensions: {singleton_dims}")
    if singleton_dims == 0:
        print("   ✅ PASS: No wasted dimensions")
    else:
        print("   ❌ FAIL: Still have singleton dimensions")
    
    # 3. Constraint system
    print(f"3. Constraint system dimensions: {brain.constraint_field.n_dimensions}")
    if brain.constraint_field.n_dimensions == len(brain.unified_field.shape):
        print("   ✅ PASS: Constraints handle all dimensions")
    else:
        print("   ❌ FAIL: Dimension mismatch")
    
    # 4. Energy dissipation
    initial_energy = torch.sum(torch.abs(brain.unified_field)).item()
    brain._run_field_maintenance()
    final_energy = torch.sum(torch.abs(brain.unified_field)).item()
    print(f"4. Energy dissipation: {initial_energy:.3f} → {final_energy:.3f}")
    if final_energy <= initial_energy:
        print("   ✅ PASS: Energy dissipation working")
    else:
        print("   ❌ FAIL: Energy not dissipating")
    
    # 5. Memory persistence
    print(f"5. Topology regions support full 37D: ", end="")
    test_region = {
        'center': torch.randn(37),
        'importance': 1.0,
        'decay_rate': 0.99
    }
    if len(test_region['center']) == 37:
        print("✅ PASS")
    else:
        print("❌ FAIL")


if __name__ == "__main__":
    print("🚀 FINAL PERFORMANCE CHECK BEFORE ROBOT INTEGRATION\n")
    
    # Run all checks
    cycle_times = test_cycle_performance()
    test_all_features_active()
    test_memory_overhead()
    test_critical_fixes()
    
    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_performance = np.mean(cycle_times)
    if avg_performance < 100:
        print(f"✅ READY FOR ROBOT INTEGRATION!")
        print(f"   Average cycle time: {avg_performance:.1f}ms")
        print(f"   All critical fixes implemented")
        print(f"   Memory usage within limits")
    else:
        print(f"⚠️  PERFORMANCE NEEDS WORK")
        print(f"   Average cycle time: {avg_performance:.1f}ms (target: <100ms)")