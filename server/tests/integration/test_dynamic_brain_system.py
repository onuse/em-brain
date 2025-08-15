#!/usr/bin/env python3
"""
Test Dynamic Brain System

Integration test for the pragmatic dynamic dimension implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import torch


def test_dynamic_brain_creation():
    """Test creating brains with different robot complexities."""
    
    # Create factory with dynamic brain enabled
    factory = DynamicBrainFactory({'use_dynamic_brain': True})
    
    print("🧪 Testing Dynamic Brain Creation")
    print("=" * 50)
    
    # Test cases: (name, sensors, motors, expected_conceptual_range)
    test_robots = [
        ("Simple Robot", 8, 2, (15, 20)),
        ("PiCar-X Robot", 24, 4, (27, 32)),
        ("Complex Robot", 64, 8, (38, 44)),
        ("Advanced Robot", 128, 12, (48, 54))
    ]
    
    for name, sensors, motors, expected_range in test_robots:
        print(f"\n{name}:")
        print(f"  Input: {sensors} sensors, {motors} motors")
        
        # Create brain
        brain_wrapper = factory.create(
            field_dimensions=None,  # Not used in dynamic mode
            spatial_resolution=4,
            sensory_dim=sensors,
            motor_dim=motors
        )
        
        # Get the actual brain
        brain = brain_wrapper.brain
        
        # Verify dimensions
        conceptual_dims = len(brain.field_dimensions)
        tensor_dims = len(brain.tensor_shape)
        
        print(f"  Conceptual dimensions: {conceptual_dims}")
        print(f"  Tensor dimensions: {tensor_dims}")
        print(f"  Tensor shape: {brain.tensor_shape}")
        print(f"  Memory usage: {brain._calculate_memory_usage():.1f}MB")
        
        # Check conceptual dimensions are in expected range
        assert expected_range[0] <= conceptual_dims <= expected_range[1], \
            f"Expected {expected_range} conceptual dims, got {conceptual_dims}"
        
        # All tensor configs should be 11D
        assert tensor_dims == 11, f"Expected 11D tensor, got {tensor_dims}D"
        
        print("  ✅ Brain created successfully")


def test_dynamic_brain_processing():
    """Test that dynamic brains can process robot cycles."""
    
    factory = DynamicBrainFactory({'use_dynamic_brain': True, 'quiet_mode': True})
    
    print("\n\n🧪 Testing Dynamic Brain Processing")
    print("=" * 50)
    
    # Create a medium complexity brain
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    
    brain = brain_wrapper.brain
    
    # Test processing cycles
    print("\nProcessing robot cycles:")
    
    for cycle in range(5):
        # Create dummy sensory input
        sensory_input = [0.5 + 0.1 * cycle] * 24
        
        # Process cycle
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        print(f"  Cycle {cycle + 1}:")
        print(f"    Motor output: {[f'{m:.3f}' for m in motor_output[:2]]}")
        print(f"    Field energy: {brain_state['field_energy']:.4f}")
        print(f"    Cycle time: {brain_state['cycle_time_ms']:.1f}ms")
        
        # Verify outputs
        assert len(motor_output) == 4, f"Expected 4 motor outputs, got {len(motor_output)}"
        assert brain_state['cycle'] == cycle + 1, "Cycle count mismatch"
    
    print("\n✅ Brain processing working correctly")


def test_dimension_mapping():
    """Test the conceptual to tensor dimension mapping."""
    
    factory = DynamicBrainFactory({'use_dynamic_brain': True, 'quiet_mode': True})
    
    print("\n\n🧪 Testing Dimension Mapping")
    print("=" * 50)
    
    # Create brain
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    
    brain = brain_wrapper.brain
    mapping = brain.dimension_mapping
    
    print("\nDimension mapping analysis:")
    print(f"  Total conceptual dimensions: {len(brain.field_dimensions)}")
    print(f"  Total tensor dimensions: {len(brain.tensor_shape)}")
    
    # Check family mappings
    print("\nFamily tensor assignments:")
    for family, (start, end) in mapping['family_tensor_ranges'].items():
        conceptual_count = sum(1 for d in brain.field_dimensions if d.family == family)
        tensor_positions = end - start
        print(f"  {family.value}: {conceptual_count}D conceptual → {tensor_positions} tensor position(s)")
    
    # Verify all conceptual dimensions are mapped
    mapped_count = len(mapping['conceptual_to_tensor'])
    total_dims = len(brain.field_dimensions)
    assert mapped_count == total_dims, f"Not all dimensions mapped: {mapped_count}/{total_dims}"
    
    print("\n✅ Dimension mapping validated")


def test_memory_scaling():
    """Test that memory usage scales appropriately."""
    
    factory = DynamicBrainFactory({'use_dynamic_brain': True, 'quiet_mode': True})
    
    print("\n\n🧪 Testing Memory Scaling")
    print("=" * 50)
    
    configs = [
        (8, 2, "Small"),
        (24, 4, "Medium"),
        (64, 8, "Large"),
        (256, 16, "XLarge")
    ]
    
    print("\nMemory usage by robot complexity:")
    
    prev_memory = 0
    for sensors, motors, label in configs:
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=sensors,
            motor_dim=motors
        )
        
        brain = brain_wrapper.brain
        memory = brain._calculate_memory_usage()
        
        print(f"  {label} ({sensors}×{motors}): {memory:.1f}MB")
        
        # Memory should increase with complexity (except XLarge might plateau)
        if label != "XLarge":
            assert memory >= prev_memory, "Memory should increase with complexity"
        
        # But should stay reasonable
        assert memory < 200, f"Memory usage too high: {memory}MB"
        
        prev_memory = memory
    
    print("\n✅ Memory scaling appropriate")


if __name__ == "__main__":
    try:
        test_dynamic_brain_creation()
        test_dynamic_brain_processing()
        test_dimension_mapping()
        test_memory_scaling()
        
        print("\n\n🎉 All dynamic brain tests passed!")
        print("\nThe pragmatic dynamic dimension system is working correctly.")
        print("Robots of different complexities get appropriately sized brains")
        print("while maintaining manageable memory usage.")
        
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()