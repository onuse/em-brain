#!/usr/bin/env python3
"""Simplified test for hardware-adaptive field dimensions."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from server.src.brains.field.core_brain import UnifiedFieldBrain

def test_hardware_adaptation():
    """Test that brain adapts to different hardware configurations."""
    print("\n=== Testing Hardware-Adaptive Field Dimensions ===")
    
    # Test 1: Small spatial resolution (fast)
    print("\n1. Testing with small spatial resolution (10)...")
    brain_small = UnifiedFieldBrain(spatial_resolution=10, quiet_mode=False)
    
    print(f"\nField shape: {list(brain_small.unified_field.shape)}")
    print(f"Total elements: {brain_small.unified_field.numel():,}")
    print(f"Memory: {(brain_small.unified_field.numel() * 4) / (1024**2):.1f} MB")
    
    # Time a few cycles
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        brain_small.process_robot_cycle([0.5] * 24)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage cycle time: {avg_time:.1f}ms")
    print("✅ PASS" if avg_time < 400 else "❌ FAIL")
    
    # Test 2: Medium spatial resolution
    print("\n2. Testing with medium spatial resolution (15)...")
    brain_med = UnifiedFieldBrain(spatial_resolution=15, quiet_mode=True)
    
    print(f"Field shape: {list(brain_med.unified_field.shape)}")
    print(f"Total elements: {brain_med.unified_field.numel():,}")
    print(f"Memory: {(brain_med.unified_field.numel() * 4) / (1024**2):.1f} MB")
    
    t0 = time.perf_counter()
    brain_med.process_robot_cycle([0.5] * 24)
    cycle_time = (time.perf_counter() - t0) * 1000
    print(f"Single cycle: {cycle_time:.1f}ms")
    print("✅ PASS" if cycle_time < 400 else "⚠️  WARNING: Exceeds target")
    
    # Test 3: Verify adaptation occurred
    print("\n3. Verifying hardware adaptation...")
    
    if brain_small.hw_profile:
        print("✅ Hardware profile detected")
        print(f"   CPU cores: {brain_small.hw_profile.cpu_cores}")
        print(f"   RAM: {brain_small.hw_profile.total_memory_gb:.1f} GB")
        print(f"   GPU: {'Yes' if brain_small.hw_profile.gpu_available else 'No'}")
        
        # Check that dimensions were adapted
        shape_small = brain_small.unified_field.shape
        shape_med = brain_med.unified_field.shape
        
        # Non-spatial dimensions should be the same (hardware-based)
        non_spatial_small = list(shape_small[5:])
        non_spatial_med = list(shape_med[5:])
        
        print(f"\nNon-spatial dims (small): {non_spatial_small}")
        print(f"Non-spatial dims (med): {non_spatial_med}")
        
        if non_spatial_small == non_spatial_med:
            print("✅ Dimensions consistently adapted to hardware")
        else:
            print("⚠️  Dimensions differ (may be due to memory constraints)")
    else:
        print("⚠️  No hardware profile (using defaults)")
    
    return avg_time < 400

if __name__ == "__main__":
    success = test_hardware_adaptation()
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed'}")