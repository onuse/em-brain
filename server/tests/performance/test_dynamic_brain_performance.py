#!/usr/bin/env python3
"""
Realistic performance test for dynamic brain.
Tests with actual workload similar to behavioral framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import time
import numpy as np


def test_realistic_performance():
    """Test performance with realistic robot behavior patterns."""
    
    print("🏃 Dynamic Brain Performance Test")
    print("=" * 50)
    
    # Create brain with default PiCar-X profile
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': True
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    print(f"Brain created: {brain.total_dimensions}D conceptual, {len(brain.tensor_shape)}D tensor")
    print(f"Memory usage: {brain._calculate_memory_usage():.1f}MB")
    print(f"Device: {brain.device}")
    
    # Warmup cycles
    print("\nWarming up...")
    for i in range(10):
        sensory_input = [0.5 + 0.1 * np.random.randn() for _ in range(24)]
        brain.process_robot_cycle(sensory_input)
    
    # Test different scenarios
    scenarios = [
        ("Random exploration", lambda i: [0.5 + 0.2 * np.random.randn() for _ in range(24)]),
        ("Pattern learning", lambda i: [0.5 + 0.3 * np.sin(i * 0.1 + j * 0.5) for j in range(24)]),
        ("Goal seeking", lambda i: [0.8 if j < 12 else 0.2 for j in range(24)]),
        ("High activity", lambda i: [0.9 + 0.1 * np.random.randn() for _ in range(24)]),
    ]
    
    print("\nRunning performance scenarios:")
    print("-" * 50)
    
    all_times = []
    
    for scenario_name, input_generator in scenarios:
        cycle_times = []
        
        # Run 100 cycles per scenario
        for i in range(100):
            sensory_input = input_generator(i)
            
            start = time.perf_counter()
            _, brain_state = brain.process_robot_cycle(sensory_input)
            elapsed = time.perf_counter() - start
            
            cycle_times.append(elapsed * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(cycle_times)
        min_time = np.min(cycle_times)
        max_time = np.max(cycle_times)
        p50_time = np.percentile(cycle_times, 50)
        p95_time = np.percentile(cycle_times, 95)
        p99_time = np.percentile(cycle_times, 99)
        
        print(f"\n{scenario_name}:")
        print(f"  Average: {avg_time:.1f}ms")
        print(f"  Min/Max: {min_time:.1f}ms / {max_time:.1f}ms")
        print(f"  Percentiles - 50th: {p50_time:.1f}ms, 95th: {p95_time:.1f}ms, 99th: {p99_time:.1f}ms")
        
        all_times.extend(cycle_times)
    
    # Overall statistics
    print("\n" + "=" * 50)
    print("Overall Performance (400 cycles):")
    print(f"  Average: {np.mean(all_times):.1f}ms")
    print(f"  Median: {np.median(all_times):.1f}ms")
    print(f"  95th percentile: {np.percentile(all_times, 95):.1f}ms")
    print(f"  99th percentile: {np.percentile(all_times, 99):.1f}ms")
    
    # Check brain state
    print(f"\nFinal brain state:")
    print(f"  Cycles processed: {brain.brain_cycles}")
    print(f"  Active constraints: {len(brain.constraint_field.active_constraints)}")
    print(f"  Topology regions: {len(brain.topology_regions)}")
    print(f"  Field energy: {brain_state['field_energy']:.4f}")
    
    # Throughput calculation
    total_time = sum(all_times) / 1000  # Convert to seconds
    throughput = 400 / total_time
    print(f"\nThroughput: {throughput:.1f} cycles/second")


def test_scaling_performance():
    """Test how performance scales with robot complexity."""
    
    print("\n\n🔬 Performance Scaling Test")
    print("=" * 50)
    
    robot_configs = [
        (8, 2, "Simple"),
        (24, 4, "Medium"),
        (64, 8, "Complex"),
        (128, 12, "Advanced")
    ]
    
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': True
    })
    
    for sensors, motors, label in robot_configs:
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=sensors,
            motor_dim=motors
        )
        brain = brain_wrapper.brain
        
        # Run 50 cycles
        cycle_times = []
        for i in range(50):
            sensory_input = [0.5 + 0.1 * np.random.randn() for _ in range(sensors)]
            
            start = time.perf_counter()
            brain.process_robot_cycle(sensory_input)
            elapsed = time.perf_counter() - start
            
            cycle_times.append(elapsed * 1000)
        
        avg_time = np.mean(cycle_times[10:])  # Skip warmup
        
        print(f"\n{label} robot ({sensors} sensors, {motors} motors):")
        print(f"  Conceptual dims: {brain.total_dimensions}D")
        print(f"  Tensor shape: {brain.tensor_shape}")
        print(f"  Memory: {brain._calculate_memory_usage():.1f}MB")
        print(f"  Avg cycle time: {avg_time:.1f}ms")


if __name__ == "__main__":
    test_realistic_performance()
    test_scaling_performance()
    
    print("\n\n📊 Performance Summary:")
    print("These are realistic cycle times with full feature processing.")
    print("CUDA would typically be 5-10x faster than these CPU times.")