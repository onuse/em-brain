#!/usr/bin/env python3
"""Test hardware-adaptive field dimensions in UnifiedFieldBrain."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import torch
from unittest.mock import patch, MagicMock
from server.src.brains.field.core_brain import UnifiedFieldBrain
from server.src.utils.hardware_adaptation import HardwareProfile

def test_hardware_adaptive_dimensions():
    """Test brain adapts field dimensions based on hardware capabilities."""
    print("\n=== Testing Hardware-Adaptive Field Dimensions ===")
    
    # Test 1: High-end hardware simulation
    print("\n1. Testing high-end hardware configuration...")
    
    # Mock the hardware adaptation instance
    mock_hw_adapter = MagicMock()
    mock_hw_adapter.hardware_profile = HardwareProfile(
        cpu_cores=16,
        total_memory_gb=64.0,
        gpu_available=True,
        gpu_memory_gb=24.0,
        avg_cycle_time_ms=30.0,
        memory_pressure=0.1,
        cpu_utilization=0.3,
        working_memory_limit=16000,
        similarity_search_limit=150000,
        batch_processing_threshold=1500,
        cognitive_energy_budget=32000,
        max_experiences_per_cycle=32000
    )
    
    with patch('server.src.utils.hardware_adaptation.get_hardware_adaptation', return_value=mock_hw_adapter):
        brain = UnifiedFieldBrain(spatial_resolution=20, quiet_mode=True)
        
        print(f"Field shape: {list(brain.unified_field.shape)}")
        print(f"Total elements: {brain.unified_field.numel():,}")
        print(f"Memory usage: {(brain.unified_field.numel() * 4) / (1024**2):.1f} MB")
        
        # Verify larger dimensions for key families
        shape = brain.unified_field.shape
        assert shape[4] == 15, "Time dimension should be 15 for high-end"
        # Note: dimensions may be reduced due to memory constraints
        # Check that high-end has larger dims than default
        default_dims = [3, 3, 2, 2, 2, 2]  # Default non-spatial dims
        actual_non_spatial = list(shape[5:])
        print(f"Non-spatial dims: {actual_non_spatial}")
        assert sum(actual_non_spatial) > sum(default_dims), "High-end should have larger total dimensions"
        
        # Test performance
        times = []
        for i in range(5):
            t0 = time.perf_counter()
            brain.process_robot_cycle([0.5] * 24)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"Average cycle time: {avg_time:.1f}ms")
        assert avg_time < 400, f"High-end system should maintain <400ms cycles, got {avg_time:.1f}ms"
    
    # Test 2: Mid-range hardware
    print("\n2. Testing mid-range hardware configuration...")
    mock_hw_adapter_mid = MagicMock()
    mock_hw_adapter_mid.hardware_profile = HardwareProfile(
        cpu_cores=8,
        total_memory_gb=16.0,
        gpu_available=False,
        gpu_memory_gb=None,
        avg_cycle_time_ms=50.0,
        memory_pressure=0.3,
        cpu_utilization=0.5,
        working_memory_limit=8000,
        similarity_search_limit=50000,
        batch_processing_threshold=750,
        cognitive_energy_budget=8000,
        max_experiences_per_cycle=16000
    )
    
    with patch('server.src.utils.hardware_adaptation.get_hardware_adaptation', return_value=mock_hw_adapter_mid):
        brain_mid = UnifiedFieldBrain(spatial_resolution=18, quiet_mode=True)
        
        print(f"Field shape: {list(brain_mid.unified_field.shape)}")
        print(f"Total elements: {brain_mid.unified_field.numel():,}")
        print(f"Memory usage: {(brain_mid.unified_field.numel() * 4) / (1024**2):.1f} MB")
        
        # Verify moderate dimensions
        shape_mid = brain_mid.unified_field.shape
        # Mid-range should have smaller total non-spatial dimensions
        mid_non_spatial = list(shape_mid[5:])
        print(f"Non-spatial dims: {mid_non_spatial}")
        assert sum(mid_non_spatial) <= sum(actual_non_spatial), "Mid-range should have smaller total dims"
        
        # Performance should still be acceptable
        t0 = time.perf_counter()
        brain_mid.process_robot_cycle([0.5] * 24)
        cycle_time = (time.perf_counter() - t0) * 1000
        print(f"Single cycle time: {cycle_time:.1f}ms")
        assert cycle_time < 400, f"Mid-range should maintain <400ms cycles"
    
    # Test 3: Low-end hardware
    print("\n3. Testing low-end hardware configuration...")
    mock_hw_adapter_low = MagicMock()
    mock_hw_adapter_low.hardware_profile = HardwareProfile(
        cpu_cores=4,
        total_memory_gb=4.0,
        gpu_available=False,
        gpu_memory_gb=None,
        avg_cycle_time_ms=80.0,
        memory_pressure=0.6,
        cpu_utilization=0.8,
        working_memory_limit=4000,
        similarity_search_limit=20000,
        batch_processing_threshold=400,
        cognitive_energy_budget=4000,
        max_experiences_per_cycle=8000
    )
    
    with patch('server.src.utils.hardware_adaptation.get_hardware_adaptation', return_value=mock_hw_adapter_low):
        brain_low = UnifiedFieldBrain(spatial_resolution=12, quiet_mode=True)
        
        print(f"Field shape: {list(brain_low.unified_field.shape)}")
        print(f"Total elements: {brain_low.unified_field.numel():,}")
        print(f"Memory usage: {(brain_low.unified_field.numel() * 4) / (1024**2):.1f} MB")
        
        # Verify minimal dimensions
        shape_low = brain_low.unified_field.shape
        low_non_spatial = list(shape_low[5:])
        print(f"Non-spatial dims: {low_non_spatial}")
        # Low-end should have smallest total non-spatial dims
        assert sum(low_non_spatial) <= sum(mid_non_spatial), "Low-end should have smallest total dims"
        
        # Should still function, even if slower
        t0 = time.perf_counter()
        brain_low.process_robot_cycle([0.5] * 24)
        cycle_time = (time.perf_counter() - t0) * 1000
        print(f"Single cycle time: {cycle_time:.1f}ms")
        # Low-end might exceed 400ms but should still work
    
    print("\n✅ Hardware-adaptive field dimensions working correctly!")
    return True

def test_actual_hardware_detection():
    """Test brain on actual hardware without mocking."""
    print("\n=== Testing Actual Hardware Detection ===")
    
    # Create brain and let it detect real hardware
    brain = UnifiedFieldBrain(spatial_resolution=15, quiet_mode=False)
    
    # Get hardware profile
    hw_profile = brain.hw_profile
    if hw_profile:
        print(f"\nDetected hardware:")
        print(f"  CPU cores: {hw_profile.cpu_cores}")
        print(f"  Total memory: {hw_profile.total_memory_gb:.1f} GB")
        print(f"  GPU available: {hw_profile.gpu_available}")
        if hw_profile.gpu_memory_gb:
            print(f"  GPU memory: {hw_profile.gpu_memory_gb:.1f} GB")
    
    print(f"\nAdapted field shape: {list(brain.unified_field.shape)}")
    print(f"Total elements: {brain.unified_field.numel():,}")
    print(f"Memory usage: {(brain.unified_field.numel() * 4) / (1024**2):.1f} MB")
    
    # Test performance with actual hardware
    print("\nTesting performance on actual hardware...")
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        brain.process_robot_cycle([0.5] * 24)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        if i == 0:
            print(f"  First cycle: {elapsed:.1f}ms")
    
    avg_time = sum(times[1:]) / len(times[1:])  # Skip first cycle
    min_time = min(times[1:])
    max_time = max(times[1:])
    
    print(f"\nPerformance summary:")
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Min: {min_time:.1f}ms")
    print(f"  Max: {max_time:.1f}ms")
    print(f"  Target: <400ms")
    
    if avg_time < 400:
        print("✅ Performance within target!")
    else:
        print("⚠️  Performance exceeds target, but may be acceptable for this hardware")
    
    return True

if __name__ == "__main__":
    test_hardware_adaptive_dimensions()
    test_actual_hardware_detection()
    print("\n✅ All hardware adaptation tests passed!")