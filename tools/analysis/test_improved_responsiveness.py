#!/usr/bin/env python3
"""
Test if the dimension fix improves input responsiveness.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
from brains.field.core_brain import create_unified_field_brain


def test_input_responsiveness():
    """Test how different inputs produce different outputs."""
    print("Testing Improved Input Responsiveness")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Very different scenarios
    scenarios = [
        ("Stationary", [0.5, 0.5, 0.5] + [0.5] * 21),
        ("Forward Fast", [0.9, 0.5, 0.5] + [0.5] * 21),
        ("Backward", [0.1, 0.5, 0.5] + [0.5] * 21),
        ("Turn Left", [0.5, 0.1, 0.5] + [0.7, 0.3, 0.5] * 6),
        ("Turn Right", [0.5, 0.9, 0.5] + [0.3, 0.7, 0.5] * 6),
        ("Obstacle", [0.5, 0.5, 0.5] + [0.9, 0.9, 0.9] + [0.5] * 18),
        ("Complex", [0.7, 0.3, 0.2] + [0.1, 0.9, 0.5, 0.8, 0.2] + [0.5] * 16)
    ]
    
    results = []
    
    for name, input_data in scenarios:
        # Get field coordinates to see variation
        experience = brain._robot_sensors_to_field_experience(input_data)
        coords = experience.field_coordinates
        
        # Process input
        action, state = brain.process_robot_cycle(input_data)
        
        results.append({
            'name': name,
            'action': action,
            'coords_variance': torch.var(coords).item(),
            'non_zero_coords': torch.sum(coords != 0).item(),
            'gradient_strength': state.get('last_gradient_strength', 0)
        })
        
        print(f"\n{name}:")
        print(f"  Action: [{', '.join(f'{a:6.3f}' for a in action)}]")
        print(f"  Coord variance: {results[-1]['coords_variance']:.4f}")
        print(f"  Non-zero coords: {results[-1]['non_zero_coords']}/37")
        print(f"  Gradient strength: {results[-1]['gradient_strength']:.6f}")
    
    # Analyze diversity
    print("\n\nDiversity Analysis:")
    print("-" * 40)
    
    # Action diversity
    total_diff = 0
    comparisons = 0
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            action_diff = sum(abs(a - b) for a, b in 
                            zip(results[i]['action'], results[j]['action']))
            total_diff += action_diff
            comparisons += 1
            
            if action_diff > 0.1:  # Significant difference
                print(f"✓ {results[i]['name']} vs {results[j]['name']}: {action_diff:.3f}")
    
    avg_diff = total_diff / comparisons if comparisons > 0 else 0
    print(f"\nAverage action difference: {avg_diff:.4f}")
    
    # Success criteria
    if avg_diff > 0.05:
        print("\n✅ SUCCESS: Actions show good diversity!")
    else:
        print("\n❌ FAIL: Actions still too similar")
    
    # Coordinate diversity
    coord_variances = [r['coords_variance'] for r in results]
    avg_coord_var = sum(coord_variances) / len(coord_variances)
    print(f"\nAverage coordinate variance: {avg_coord_var:.4f}")
    
    if avg_coord_var > 0.1:
        print("✅ SUCCESS: Good coordinate diversity!")
    else:
        print("❌ FAIL: Coordinates lack diversity")


if __name__ == "__main__":
    test_input_responsiveness()