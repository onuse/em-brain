#!/usr/bin/env python3
"""
Test Dynamic Dimension Calculator

Validates that the dimension calculator produces correct field architectures
for different robot complexities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_dimension_calculator import DynamicDimensionCalculator
from src.brains.field.core_brain import FieldDynamicsFamily


def test_dimension_scaling():
    """Test logarithmic scaling of dimensions."""
    calculator = DynamicDimensionCalculator()
    
    # Test simple robot (8 sensors)
    dims_simple = calculator.calculate_dimensions(8, 2)
    print(f"\nSimple robot (8 sensors, 2 motors):")
    print(f"  Total dimensions: {len(dims_simple)}")
    print_dimension_summary(dims_simple)
    
    # Test medium robot (24 sensors - like PiCar-X)
    dims_medium = calculator.calculate_dimensions(24, 4)
    print(f"\nMedium robot (24 sensors, 4 motors):")
    print(f"  Total dimensions: {len(dims_medium)}")
    print_dimension_summary(dims_medium)
    
    # Test complex robot (64 sensors)
    dims_complex = calculator.calculate_dimensions(64, 8)
    print(f"\nComplex robot (64 sensors, 8 motors):")
    print(f"  Total dimensions: {len(dims_complex)}")
    print_dimension_summary(dims_complex)
    
    # Test very complex robot (256 sensors)
    dims_very_complex = calculator.calculate_dimensions(256, 16)
    print(f"\nVery complex robot (256 sensors, 16 motors):")
    print(f"  Total dimensions: {len(dims_very_complex)}")
    print_dimension_summary(dims_very_complex)
    
    # Verify scaling is logarithmic
    assert 15 <= len(dims_simple) <= 20, f"Simple robot should have ~18D, got {len(dims_simple)}"
    assert 27 <= len(dims_medium) <= 32, f"Medium robot should have ~29D, got {len(dims_medium)}"
    assert 38 <= len(dims_complex) <= 44, f"Complex robot should have ~41D, got {len(dims_complex)}"
    assert 54 <= len(dims_very_complex) <= 60, f"Very complex robot should have ~57D, got {len(dims_very_complex)}"
    
    print("\n✅ Dimension scaling tests passed!")


def test_capability_adjustments():
    """Test that capabilities affect dimension distribution."""
    calculator = DynamicDimensionCalculator()
    
    # Base robot
    dims_base = calculator.calculate_dimensions(24, 4)
    
    # Robot with visual processing
    dims_visual = calculator.calculate_dimensions(24, 4, {'visual_processing': True})
    
    # Robot with manipulation
    dims_manipulator = calculator.calculate_dimensions(24, 4, {'manipulation': True})
    
    # Robot with both
    dims_full = calculator.calculate_dimensions(24, 4, {
        'visual_processing': True,
        'manipulation': True
    })
    
    print("\nCapability adjustments:")
    print(f"  Base robot: {len(dims_base)}D")
    print(f"  Visual robot: {len(dims_visual)}D")
    print(f"  Manipulator robot: {len(dims_manipulator)}D")
    print(f"  Full capability robot: {len(dims_full)}D")
    
    # Visual robots should have more oscillatory/flow dimensions
    osc_base = count_family_dims(dims_base, FieldDynamicsFamily.OSCILLATORY)
    osc_visual = count_family_dims(dims_visual, FieldDynamicsFamily.OSCILLATORY)
    assert osc_visual >= osc_base, "Visual robot should have more oscillatory dimensions"
    
    # Manipulators should have more topology/coupling dimensions
    topo_base = count_family_dims(dims_base, FieldDynamicsFamily.TOPOLOGY)
    topo_manip = count_family_dims(dims_manipulator, FieldDynamicsFamily.TOPOLOGY)
    assert topo_manip >= topo_base, "Manipulator robot should have more topology dimensions"
    
    print("✅ Capability adjustment tests passed!")


def test_tensor_configuration_selection():
    """Test tensor configuration selection based on conceptual dimensions."""
    calculator = DynamicDimensionCalculator()
    
    # Test different robot complexities
    test_cases = [
        (8, 2, "Small"),    # Simple robot
        (24, 4, "Medium"),  # PiCar-X like
        (64, 8, "Large"),   # Complex robot
        (256, 16, "XLarge") # Very complex
    ]
    
    print("\nTensor configuration selection:")
    for sensors, motors, label in test_cases:
        dims = calculator.calculate_dimensions(sensors, motors)
        tensor_shape = calculator.select_tensor_configuration(len(dims), spatial_resolution=4)
        
        # Calculate memory usage
        elements = 1
        for dim in tensor_shape:
            elements *= dim
        memory_mb = (elements * 4) / (1024 * 1024)
        
        print(f"  {label} robot ({len(dims)}D conceptual): {tensor_shape}")
        print(f"    Tensor dimensions: {len(tensor_shape)}D")
        print(f"    Memory usage: {memory_mb:.1f}MB")
    
    # Verify all configurations have 11 tensor dimensions
    for sensors, motors, _ in test_cases:
        dims = calculator.calculate_dimensions(sensors, motors)
        shape = calculator.select_tensor_configuration(len(dims), spatial_resolution=4)
        assert len(shape) == 11, f"All tensor configs should have 11 dimensions, got {len(shape)}"
    
    print("✅ Tensor configuration tests passed!")


def test_dimension_mapping():
    """Test mapping between conceptual and tensor dimensions."""
    calculator = DynamicDimensionCalculator()
    
    # Medium robot
    dims = calculator.calculate_dimensions(24, 4)
    tensor_shape = calculator.select_tensor_configuration(len(dims), spatial_resolution=4)
    mapping = calculator.create_dimension_mapping(dims, tensor_shape)
    
    print(f"\nDimension mapping for {len(dims)}D conceptual → {len(tensor_shape)}D tensor:")
    
    # Show family tensor ranges
    print("  Family tensor ranges:")
    for family, (start, end) in mapping['family_tensor_ranges'].items():
        print(f"    {family.value}: tensor indices [{start}:{end})")
    
    # Debug: show unmapped dimensions
    mapped_dims = set(mapping['conceptual_to_tensor'].keys())
    all_dim_names = set(d.name for d in dims)
    unmapped = all_dim_names - mapped_dims
    if unmapped:
        print(f"\n  Unmapped dimensions: {unmapped}")
    
    # Verify all conceptual dimensions are mapped
    mapped_count = len(mapping['conceptual_to_tensor'])
    assert mapped_count == len(dims), f"All {len(dims)} dimensions should be mapped, got {mapped_count}"
    
    # Verify spatial dimensions map to first tensor positions
    spatial_dims = [d for d in dims if d.family == FieldDynamicsFamily.SPATIAL]
    if spatial_dims:
        first_spatial = spatial_dims[0]
        assert mapping['conceptual_to_tensor'][first_spatial.name] == 0, "First spatial dim should map to tensor index 0"
    
    print("✅ Dimension mapping tests passed!")


def print_dimension_summary(dimensions):
    """Print summary of dimension distribution by family."""
    family_counts = {}
    for dim in dimensions:
        if dim.family not in family_counts:
            family_counts[dim.family] = 0
        family_counts[dim.family] += 1
    
    for family, count in family_counts.items():
        print(f"    {family.value}: {count}D")


def count_family_dims(dimensions, family):
    """Count dimensions in a specific family."""
    return sum(1 for dim in dimensions if dim.family == family)


if __name__ == "__main__":
    print("🧪 Testing Dynamic Dimension Calculator")
    print("=" * 50)
    
    test_dimension_scaling()
    print("\n" + "=" * 50)
    
    test_capability_adjustments()
    print("\n" + "=" * 50)
    
    test_tensor_configuration_selection()
    print("\n" + "=" * 50)
    
    test_dimension_mapping()
    
    print("\n🎉 All tests passed!")