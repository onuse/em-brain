#!/usr/bin/env python3
"""
Investigate why the brain doesn't respond differently to varied inputs.

This is critical - if the brain doesn't differentiate between scenarios,
it's not truly processing information.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from brains.field.core_brain import create_unified_field_brain


def analyze_input_to_field_mapping():
    """Analyze how different inputs map to field coordinates."""
    print("Analyzing Input → Field Coordinate Mapping")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=20,
        quiet_mode=True
    )
    
    # Test scenarios that SHOULD produce very different responses
    scenarios = [
        ("Stationary", [0.5, 0.5, 0.5] + [0.5] * 21),
        ("Moving Forward", [0.8, 0.5, 0.5] + [0.5] * 21),
        ("Obstacle Ahead", [0.5, 0.5, 0.5] + [0.9, 0.9, 0.9] + [0.5] * 18),
        ("Turning Left", [0.5, 0.2, 0.5] + [0.7, 0.3, 0.3] + [0.5] * 18),
        ("Complex Scene", [0.7, 0.3, 0.2] + [0.8, 0.9, 0.1, 0.2, 0.7] + [0.5] * 16),
        ("Danger", [0.3, 0.3, 0.9] + [1.0, 1.0, 1.0] + [0.5] * 18)
    ]
    
    field_coords_list = []
    
    for name, input_data in scenarios:
        # Get field coordinates
        experience = brain._robot_sensors_to_field_experience(input_data)
        coords = experience.field_coordinates
        field_coords_list.append(coords)
        
        print(f"\n{name}:")
        print(f"  Raw input (first 6): {input_data[:6]}")
        print(f"  Field coords (first 10):")
        print(f"    Spatial: [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]")
        print(f"    Scale/Time: [{coords[3]:.3f}, {coords[4]:.3f}]")
        print(f"    Oscillatory: [{coords[5]:.3f}, {coords[6]:.3f}, {coords[7]:.3f}]")
        print(f"    Flow: [{coords[11]:.3f}, {coords[12]:.3f}, {coords[13]:.3f}]")
        
    # Calculate distances between scenarios in field space
    print("\n\nField Space Distances Between Scenarios:")
    print("-" * 50)
    for i in range(len(scenarios)):
        for j in range(i+1, len(scenarios)):
            dist = torch.norm(field_coords_list[i] - field_coords_list[j]).item()
            print(f"{scenarios[i][0]} ↔ {scenarios[j][0]}: {dist:.3f}")


def analyze_field_imprint_overlap():
    """Analyze how field imprints overlap and interfere."""
    print("\n\nAnalyzing Field Imprint Patterns")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Apply different experiences and examine field patterns
    test_cases = [
        ("Left Movement", [0.2, 0.5, 0.5] + [0.5] * 21),
        ("Right Movement", [0.8, 0.5, 0.5] + [0.5] * 21),
        ("Forward Movement", [0.5, 0.8, 0.5] + [0.5] * 21),
    ]
    
    for name, input_data in test_cases:
        # Clear field
        brain.unified_field.zero_()
        
        # Apply single experience
        experience = brain._robot_sensors_to_field_experience(input_data)
        brain._apply_field_experience(experience)
        
        # Analyze field pattern
        field_slice = brain.unified_field[:, :, 7, 5, 7]  # Middle slice
        
        print(f"\n{name}:")
        print(f"  Max activation: {torch.max(field_slice).item():.3f}")
        print(f"  Total energy: {torch.sum(field_slice).item():.3f}")
        
        # Find peak location
        max_idx = torch.argmax(field_slice)
        max_idx_tuple = torch.unravel_index(max_idx, field_slice.shape)
        y, x = max_idx_tuple[0].item(), max_idx_tuple[1].item()
        print(f"  Peak at: [{x}, {y}]")
        
        # Check spread
        active_mask = field_slice > 0.01
        active_count = torch.sum(active_mask).item()
        print(f"  Active points: {active_count}")


def test_dimension_utilization():
    """Test which dimensions are actually being used."""
    print("\n\nAnalyzing Dimension Utilization")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    
    # Process several varied inputs
    varied_inputs = [
        [i/10, (10-i)/10, 0.5] + [np.sin(i/2), np.cos(i/2), 0.5] + [0.5] * 18
        for i in range(10)
    ]
    
    for inp in varied_inputs:
        brain.process_robot_cycle(inp)
    
    # Analyze which dimensions have activity
    print("\nDimension Activity Analysis:")
    
    # Check variance along each dimension
    field_flat = brain.unified_field.view(brain.unified_field.shape[0], 
                                          brain.unified_field.shape[1],
                                          brain.unified_field.shape[2],
                                          brain.unified_field.shape[3],
                                          brain.unified_field.shape[4],
                                          -1)  # Flatten remaining dims
    
    # Variance in each of the first 5 dimensions
    for dim, name in enumerate(['X', 'Y', 'Z', 'Scale', 'Time']):
        dim_std = torch.std(brain.unified_field, dim=dim)
        max_std = torch.max(dim_std).item()
        mean_std = torch.mean(dim_std).item()
        print(f"  Dimension {name}: max_std={max_std:.6f}, mean_std={mean_std:.6f}")
    
    # Check the singleton dimensions
    singleton_dims = brain.unified_field.shape[5:]
    print(f"\n  Singleton dimensions: {singleton_dims}")
    print(f"  Total singleton dims: {len(singleton_dims)} (all are size 1)")
    
    # This is the problem - 32 dimensions are singleton!
    print(f"\n  ⚠️  WARNING: {len(singleton_dims)} dimensions are not being used!")
    print(f"  Only using {5} out of {brain.total_dimensions} dimensions")


def test_gradient_diversity():
    """Test if gradients vary across the field."""
    print("\n\nAnalyzing Gradient Diversity")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=20,
        quiet_mode=True
    )
    
    # Create diverse field pattern
    for i in range(5):
        x = int(np.random.rand() * brain.spatial_resolution)
        y = int(np.random.rand() * brain.spatial_resolution)
        brain.unified_field[x, y, 10, 5, 7] = np.random.rand() * 0.5 + 0.5
    
    # Calculate gradients
    brain._calculate_gradient_flows()
    
    # Sample gradients at different positions
    positions = [(5, 5), (5, 15), (15, 5), (15, 15), (10, 10)]
    
    print("\nGradient samples at different positions:")
    for x, y in positions:
        gx = brain.gradient_flows['gradient_x'][x, y, 10, 5, 7].item()
        gy = brain.gradient_flows['gradient_y'][x, y, 10, 5, 7].item()
        gz = brain.gradient_flows['gradient_z'][x, y, 10, 5, 7].item()
        print(f"  Position [{x:2d}, {y:2d}]: gx={gx:7.4f}, gy={gy:7.4f}, gz={gz:7.4f}")
    
    # Test action generation at different "robot positions"
    print("\nActions from different field positions:")
    for i, (x, y) in enumerate([(5, 5), (15, 15)]):
        # Temporarily change "center" for gradient extraction
        original_center = brain.spatial_resolution // 2
        brain.spatial_resolution = x * 2  # Hack to change center
        
        action = brain._field_gradients_to_robot_action()
        
        brain.spatial_resolution = original_center * 2  # Restore
        
        print(f"  From [{x}, {y}]: motor={action.motor_commands}, strength={action.gradient_strength:.6f}")


def propose_solutions():
    """Propose solutions for better input differentiation."""
    print("\n\nProposed Solutions")
    print("=" * 60)
    
    print("""
1. DIMENSION REDUCTION PROBLEM:
   - Currently using 37D but 32 dimensions are singleton (size 1)
   - This wastes computation and prevents meaningful variation
   - Solution: Either use those dimensions or reduce to actual 5D
   
2. INPUT MAPPING ISSUE:
   - Current mapping doesn't create enough variation in field coordinates
   - Similar inputs map to similar field positions
   - Solution: Implement richer input → field transformations
   
3. FIELD OVERLAP:
   - Experiences create overlapping imprints that average out differences
   - Solution: Implement lateral inhibition or contrast enhancement
   
4. GRADIENT AGGREGATION:
   - Current aggregation might be averaging out important differences
   - Solution: Use max pooling or attention mechanisms for gradients
   
5. MISSING TEMPORAL DYNAMICS:
   - No momentum or velocity tracking
   - Solution: Add temporal difference calculations
   """)


if __name__ == "__main__":
    analyze_input_to_field_mapping()
    analyze_field_imprint_overlap()
    test_dimension_utilization()
    test_gradient_diversity()
    propose_solutions()