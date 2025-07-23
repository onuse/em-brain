#!/usr/bin/env python3
"""
Test field responsiveness to different inputs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import numpy as np
from brains.field.core_brain import create_unified_field_brain


def test_field_response():
    """Test how field responds to different input positions."""
    print("Testing Field Response to Input Positions")
    print("=" * 50)
    
    # Create brain with clear initial state
    brain = create_unified_field_brain(
        spatial_resolution=15,
        temporal_window=5.0,
        quiet_mode=True
    )
    
    # Test extreme positions
    test_positions = [
        ("Far Left", [0.0, 0.5, 0.5]),
        ("Left", [0.25, 0.5, 0.5]),
        ("Center", [0.5, 0.5, 0.5]),
        ("Right", [0.75, 0.5, 0.5]),
        ("Far Right", [1.0, 0.5, 0.5]),
    ]
    
    print("\nTesting spatial mapping:")
    for name, pos in test_positions:
        # Create input
        test_input = pos + [0.5] * 21
        
        # Clear field
        brain.unified_field.zero_()
        brain.field_experiences.clear()
        
        # Process single input
        action, state = brain.process_robot_cycle(test_input)
        
        # Find where activity is in the field
        field_slice = brain.unified_field[:, :, 7, 5, 7]  # Middle z, scale, time
        active_points = torch.nonzero(field_slice > 0.01)
        
        if len(active_points) > 0:
            # Calculate center of mass of activity
            total_mass = torch.sum(field_slice)
            x_center = torch.sum(active_points[:, 0].float() * field_slice[active_points[:, 0], active_points[:, 1]]) / total_mass
            y_center = torch.sum(active_points[:, 1].float() * field_slice[active_points[:, 0], active_points[:, 1]]) / total_mass
            
            print(f"\n{name} [{pos[0]:.2f}, {pos[1]:.2f}]:")
            print(f"  Activity center: [{x_center:.1f}, {y_center:.1f}]")
            print(f"  Max activity: {torch.max(field_slice).item():.3f}")
            print(f"  Active points: {len(active_points)}")
            print(f"  Motor output: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
        else:
            print(f"\n{name} [{pos[0]:.2f}, {pos[1]:.2f}]: NO FIELD ACTIVITY")
    
    # Test with accumulated field
    print("\n\nTesting with accumulated field:")
    brain.unified_field.zero_()
    
    for i, (name, pos) in enumerate(test_positions):
        test_input = pos + [0.5] * 21
        action, state = brain.process_robot_cycle(test_input)
        
        if i == len(test_positions) - 1:  # After all inputs
            print(f"\nFinal state after all inputs:")
            print(f"  Total field energy: {state['field_total_energy']:.3f}")
            print(f"  Max field activation: {state['field_max_activation']:.3f}")
            
            # Check gradient distribution
            brain._calculate_gradient_flows()
            grad_x = brain.gradient_flows['gradient_x']
            grad_y = brain.gradient_flows['gradient_y']
            
            # Sample gradients at different positions
            positions = [(3, 7), (7, 7), (11, 7)]
            print("\n  Gradients at different positions:")
            for x, y in positions:
                gx = grad_x[x, y, 7, 5, 7].item()
                gy = grad_y[x, y, 7, 5, 7].item()
                print(f"    Position [{x}, {y}]: grad_x={gx:.4f}, grad_y={gy:.4f}")


def test_direct_field_manipulation():
    """Test action generation with manually set field patterns."""
    print("\n\nTesting Direct Field Patterns")
    print("=" * 50)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Create clear directional gradients
    patterns = [
        ("X Gradient", lambda: create_x_gradient(brain)),
        ("Y Gradient", lambda: create_y_gradient(brain)),
        ("Central Peak", lambda: create_central_peak(brain)),
        ("Multiple Peaks", lambda: create_multiple_peaks(brain))
    ]
    
    for name, create_pattern in patterns:
        print(f"\n{name}:")
        
        # Clear and create pattern
        brain.unified_field.zero_()
        create_pattern()
        
        # Calculate gradients and generate action
        brain._calculate_gradient_flows()
        action = brain._field_gradients_to_robot_action()
        
        print(f"  Motor: [{action.motor_commands[0]:.3f}, {action.motor_commands[1]:.3f}, "
              f"{action.motor_commands[2]:.3f}, {action.motor_commands[3]:.3f}]")
        print(f"  Gradient strength: {action.gradient_strength:.6f}")
        print(f"  Confidence: {action.action_confidence:.3f}")


def create_x_gradient(brain):
    """Create gradient in X direction."""
    for x in range(brain.spatial_resolution):
        value = x / (brain.spatial_resolution - 1)
        brain.unified_field[x, :, :, 5, 7] = value * 0.5


def create_y_gradient(brain):
    """Create gradient in Y direction."""
    for y in range(brain.spatial_resolution):
        value = y / (brain.spatial_resolution - 1)
        brain.unified_field[:, y, :, 5, 7] = value * 0.5


def create_central_peak(brain):
    """Create central peak."""
    center = brain.spatial_resolution // 2
    brain.unified_field[center-1:center+2, center-1:center+2, center, 5, 7] = 1.0


def create_multiple_peaks(brain):
    """Create multiple peaks."""
    positions = [(3, 3), (3, 11), (11, 3), (11, 11)]
    for x, y in positions:
        brain.unified_field[x, y, 7, 5, 7] = 0.8


if __name__ == "__main__":
    test_field_response()
    test_direct_field_manipulation()