#!/usr/bin/env python3
"""
Test the dimension utilization fix.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
from brains.field.core_brain import create_unified_field_brain


def test_dimension_utilization():
    """Test that all 37 dimensions are now being used."""
    print("Testing Dimension Utilization Fix")
    print("=" * 60)
    
    # Create brain with new field structure
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    
    print(f"\nField shape: {brain.unified_field.shape}")
    print(f"Total dimensions: {len(brain.unified_field.shape)}")
    
    # Check for singleton dimensions
    singleton_count = sum(1 for s in brain.unified_field.shape if s == 1)
    print(f"Singleton dimensions: {singleton_count}")
    
    # Show dimension sizes
    print("\nDimension sizes:")
    dim_names = ["X", "Y", "Z", "Scale", "Time", "Oscillatory", "Flow", 
                 "Topology", "Energy", "Coupling", "Emergence"]
    for i, (name, size) in enumerate(zip(dim_names, brain.unified_field.shape)):
        print(f"  Dim {i} ({name}): {size}")
    
    # Test input mapping
    print("\n\nTesting Input Mapping:")
    test_inputs = [
        ("Stationary", [0.5] * 24),
        ("Moving", [0.8, 0.5, 0.5] + [0.5] * 21),
        ("Complex", [0.7, 0.3, 0.2] + [0.8, 0.9, 0.1, 0.2, 0.7] + [0.5] * 16)
    ]
    
    for name, input_data in test_inputs:
        experience = brain._robot_sensors_to_field_experience(input_data)
        coords = experience.field_coordinates
        
        print(f"\n{name}:")
        print(f"  Non-zero coords: {torch.sum(coords != 0).item()}/{len(coords)}")
        print(f"  Coord variance: {torch.var(coords).item():.4f}")
        print(f"  Coord range: [{torch.min(coords).item():.3f}, {torch.max(coords).item():.3f}]")
        
        # Show samples from each dimension family
        print("  Sample values:")
        print(f"    Spatial: {coords[:3].tolist()}")
        print(f"    Oscillatory: {[f'{c:.3f}' for c in coords[5:8]]}")
        print(f"    Flow: {[f'{c:.3f}' for c in coords[11:14]]}")
        print(f"    Energy: {[f'{c:.3f}' for c in coords[25:29]]}")


def test_gradient_calculation_new_structure():
    """Test gradient calculation with the new field structure."""
    print("\n\nTesting Gradient Calculation")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    
    # Create a pattern in the new multi-dimensional field
    # The indexing is now different!
    try:
        # Old way would fail: brain.unified_field[5, 5, 5, :, :, 0, 0, ...]
        # New way with proper dimensions:
        brain.unified_field[5, 5, 5, 3, 7, 2, 3, 2, 1, 2, 1] = 1.0
        print("✓ Field assignment successful with new structure")
    except Exception as e:
        print(f"✗ Field assignment failed: {e}")
        return
    
    # Calculate gradients
    try:
        brain._calculate_gradient_flows()
        print("✓ Gradient calculation successful")
        
        # Check gradient shapes
        for key in ['gradient_x', 'gradient_y', 'gradient_z']:
            if key in brain.gradient_flows:
                grad = brain.gradient_flows[key]
                print(f"  {key} shape: {grad.shape}")
    except Exception as e:
        print(f"✗ Gradient calculation failed: {e}")


def test_action_diversity():
    """Test if actions now vary more with different inputs."""
    print("\n\nTesting Action Diversity")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    
    # Test very different scenarios
    scenarios = [
        ("Rest", [0.5] * 24),
        ("Forward", [0.9, 0.5, 0.5] + [0.5] * 21),
        ("Backward", [0.1, 0.5, 0.5] + [0.5] * 21),
        ("Left", [0.5, 0.1, 0.5] + [0.5] * 21),
        ("Right", [0.5, 0.9, 0.5] + [0.5] * 21),
        ("Danger", [0.3, 0.3, 0.9] + [1.0, 1.0, 1.0] + [0.5] * 18)
    ]
    
    actions = []
    for name, input_data in scenarios:
        # Clear field between tests for clearer results
        brain.unified_field.zero_()
        
        action, state = brain.process_robot_cycle(input_data)
        actions.append((name, action))
        
        print(f"\n{name}: {[f'{a:.3f}' for a in action]}")
    
    # Calculate action diversity
    print("\nAction Diversity Analysis:")
    total_variance = 0
    for i in range(len(actions)):
        for j in range(i+1, len(actions)):
            diff = sum(abs(a - b) for a, b in zip(actions[i][1], actions[j][1]))
            total_variance += diff
            print(f"  {actions[i][0]} vs {actions[j][0]}: {diff:.4f}")
    
    avg_variance = total_variance / (len(actions) * (len(actions) - 1) / 2)
    print(f"\nAverage action difference: {avg_variance:.4f}")
    
    if avg_variance > 0.05:
        print("✓ Good action diversity!")
    else:
        print("✗ Actions still too similar")


if __name__ == "__main__":
    test_dimension_utilization()
    test_gradient_calculation_new_structure()
    test_action_diversity()