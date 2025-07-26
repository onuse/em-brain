#!/usr/bin/env python3
"""
Verify that performance optimizations haven't damaged brain functionality.
This tests all key capabilities that could be affected by our changes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import torch
import numpy as np
import time
from typing import Dict, List, Tuple

from brain_factory import BrainFactory


def test_spatial_navigation():
    """Test that spatial gradient navigation still works."""
    print("\n📍 Testing Spatial Navigation...")
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,  # Use conservative resolution
            'target_cycle_time_ms': 150
        }
    }
    
    factory = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Simulate navigation task
    positions = []
    for i in range(20):
        # Create sensory input with position and target
        current_pos = [np.sin(i * 0.1), np.cos(i * 0.1), 0.0]
        target_pos = [1.0, 0.0, 0.0]
        
        sensory_input = current_pos + target_pos + [0.0] * 10  # Pad to 16D
        
        action, state = factory.process_sensory_input(sensory_input)
        positions.append(action[:3])  # First 3 outputs are movement
    
    # Check if brain generates varied movement patterns
    positions = np.array(positions)
    movement_variance = np.var(positions, axis=0)
    
    success = np.mean(movement_variance) > 0.01
    print(f"   Movement variance: {np.mean(movement_variance):.4f}")
    print(f"   ✅ Spatial navigation: {'INTACT' if success else 'DAMAGED'}")
    
    factory.shutdown()
    return success


def test_memory_formation():
    """Test that topology regions (memories) still form."""
    print("\n🧠 Testing Memory Formation...")
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        }
    }
    
    factory = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Present repeated pattern to form memory
    pattern = [1.0, 0.5, -0.5, 0.0] * 4  # 16D pattern
    
    for i in range(30):
        action, state = factory.process_sensory_input(pattern)
    
    # Check if topology regions formed
    stats = factory.get_brain_stats()
    topology_count = stats.get('field_brain', {}).get('topology', {}).get('active_regions', 0)
    
    success = topology_count > 0
    print(f"   Active topology regions: {topology_count}")
    print(f"   ✅ Memory formation: {'INTACT' if success else 'DAMAGED'}")
    
    factory.shutdown()
    return success


def test_constraint_system():
    """Test that constraint discovery and enforcement still works."""
    print("\n🔗 Testing Constraint System...")
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        }
    }
    
    factory = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Run cycles to allow constraint discovery
    for i in range(50):
        sensory_input = [np.sin(i * 0.2) * (j + 1) for j in range(16)]
        action, state = factory.process_sensory_input(sensory_input)
    
    # Check constraint statistics
    stats = factory.get_brain_stats()
    constraints = stats.get('field_brain', {}).get('constraints', {})
    
    discovered = constraints.get('constraints_discovered', 0)
    enforced = constraints.get('constraints_enforced', 0)
    
    success = discovered > 0 and enforced > 0
    print(f"   Constraints discovered: {discovered}")
    print(f"   Constraints enforced: {enforced}")
    print(f"   ✅ Constraint system: {'INTACT' if success else 'DAMAGED'}")
    
    factory.shutdown()
    return success


def test_performance_timing():
    """Test that performance meets biological constraints."""
    print("\n⏱️ Testing Performance...")
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        }
    }
    
    factory = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Warm up
    for i in range(10):
        sensory_input = [0.1] * 16
        factory.process_sensory_input(sensory_input)
    
    # Measure cycle times
    cycle_times = []
    for i in range(50):
        start = time.time()
        sensory_input = [np.random.randn() * 0.1 for _ in range(16)]
        action, state = factory.process_sensory_input(sensory_input)
        cycle_time = (time.time() - start) * 1000
        cycle_times.append(cycle_time)
    
    avg_cycle_time = np.mean(cycle_times)
    max_cycle_time = np.max(cycle_times)
    
    success = avg_cycle_time < 150  # Biological constraint
    print(f"   Average cycle time: {avg_cycle_time:.1f}ms")
    print(f"   Max cycle time: {max_cycle_time:.1f}ms")
    print(f"   ✅ Performance: {'MEETS BIOLOGICAL CONSTRAINTS' if success else 'TOO SLOW'}")
    
    factory.shutdown()
    return success


def test_behavioral_differentiation():
    """Test that brain shows different behaviors for different stimuli."""
    print("\n🎭 Testing Behavioral Differentiation...")
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        }
    }
    
    factory = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    
    # Test with two very different stimuli
    stimulus_a = [1.0] * 8 + [0.0] * 8
    stimulus_b = [0.0] * 8 + [1.0] * 8
    
    responses_a = []
    responses_b = []
    
    for i in range(20):
        # Alternate stimuli
        if i % 2 == 0:
            action, _ = factory.process_sensory_input(stimulus_a)
            responses_a.append(action)
        else:
            action, _ = factory.process_sensory_input(stimulus_b)
            responses_b.append(action)
    
    # Check if responses are different
    avg_response_a = np.mean(responses_a, axis=0)
    avg_response_b = np.mean(responses_b, axis=0)
    
    response_difference = np.linalg.norm(avg_response_a - avg_response_b)
    
    success = response_difference > 0.1
    print(f"   Response difference: {response_difference:.4f}")
    print(f"   ✅ Behavioral differentiation: {'INTACT' if success else 'DAMAGED'}")
    
    factory.shutdown()
    return success


def main():
    """Run all verification tests."""
    print("🔬 Brain Performance Optimization Verification")
    print("=" * 50)
    print("Testing that optimizations haven't damaged intelligence...")
    
    tests = {
        'Spatial Navigation': test_spatial_navigation,
        'Memory Formation': test_memory_formation,
        'Constraint System': test_constraint_system,
        'Performance Timing': test_performance_timing,
        'Behavioral Differentiation': test_behavioral_differentiation
    }
    
    results = {}
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 SUCCESS: Brain intelligence remains intact after optimizations!")
        print("The performance improvements did NOT lobotomize the brain.")
    elif passed_tests >= total_tests - 1:
        print("\n⚠️ MOSTLY INTACT: Brain shows minor degradation but remains functional.")
    else:
        print("\n🚨 WARNING: Significant intelligence degradation detected!")
        print("The optimizations may have damaged core brain functionality.")


if __name__ == "__main__":
    main()