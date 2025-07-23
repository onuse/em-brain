#!/usr/bin/env python3
"""
Debug why different inputs produce same actions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
from brains.field.core_brain import create_unified_field_brain


def debug_field_mapping():
    """Debug how inputs map to field coordinates."""
    print("Debugging Field Mapping")
    print("=" * 50)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    
    # Test different inputs
    test_cases = [
        ("Centered", [0.5, 0.5, 0.5] + [0.5] * 21),
        ("Left", [0.2, 0.5, 0.5] + [0.5] * 21),
        ("Right", [0.8, 0.5, 0.5] + [0.5] * 21),
        ("Forward", [0.5, 0.8, 0.5] + [0.5] * 21),
        ("Obstacle", [0.5, 0.5, 0.5] + [0.9, 0.9, 0.9] + [0.5] * 18)
    ]
    
    for name, input_data in test_cases:
        print(f"\n{name}: input={input_data[:6]}...")
        
        # Get field coordinates
        experience = brain._robot_sensors_to_field_experience(input_data)
        field_coords = experience.field_coordinates
        
        print(f"  Spatial coords: x={field_coords[0]:.3f}, y={field_coords[1]:.3f}, z={field_coords[2]:.3f}")
        print(f"  Field intensity: {experience.field_intensity:.3f}")
        
        # Map to field indices
        x_idx = int((field_coords[0] + 1) * 0.5 * (brain.spatial_resolution - 1))
        y_idx = int((field_coords[1] + 1) * 0.5 * (brain.spatial_resolution - 1))
        z_idx = int((field_coords[2] + 1) * 0.5 * (brain.spatial_resolution - 1))
        
        print(f"  Field indices: [{x_idx}, {y_idx}, {z_idx}]")
        
        # Apply the experience
        brain._apply_field_experience(experience)
        
        # Check field activity
        field_slice = brain.unified_field[:, :, 5, 5, 5]
        max_activity = torch.max(field_slice).item()
        active_positions = torch.nonzero(field_slice > 0.01)
        
        print(f"  Max field activity: {max_activity:.3f}")
        print(f"  Active positions: {len(active_positions)}")
        
        # Calculate and show gradients
        brain._calculate_gradient_flows()
        
        # Get gradient at center
        center = brain.spatial_resolution // 2
        grad_x_center = brain.gradient_flows['gradient_x'][center, center, center, 5, 5].item()
        grad_y_center = brain.gradient_flows['gradient_y'][center, center, center, 5, 5].item()
        
        print(f"  Gradients at center: x={grad_x_center:.4f}, y={grad_y_center:.4f}")
        
        # Generate action
        action = brain._field_gradients_to_robot_action()
        print(f"  Motor output: {action.motor_commands}")
        
        # Clear field for next test
        brain.unified_field.zero_()


def debug_field_evolution():
    """Debug field evolution over multiple cycles."""
    print("\n\nDebugging Field Evolution")
    print("=" * 50)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        field_decay_rate=0.95,  # Stronger decay for testing
        quiet_mode=True
    )
    
    # Apply several experiences
    print("\nApplying sequence of experiences:")
    
    inputs = [
        [0.3, 0.3, 0.5] + [0.5] * 21,
        [0.7, 0.3, 0.5] + [0.5] * 21,
        [0.7, 0.7, 0.5] + [0.5] * 21,
        [0.3, 0.7, 0.5] + [0.5] * 21,
    ]
    
    for i, input_data in enumerate(inputs):
        motor, state = brain.process_robot_cycle(input_data)
        
        print(f"\nCycle {i+1}:")
        print(f"  Input position: [{input_data[0]:.1f}, {input_data[1]:.1f}]")
        print(f"  Motor output: [{motor[0]:.3f}, {motor[1]:.3f}, {motor[2]:.3f}, {motor[3]:.3f}]")
        print(f"  Field energy: {state['field_total_energy']:.3f}")
        print(f"  Gradient strength: {state['last_gradient_strength']:.6f}")
        
        # Check field pattern
        field_2d = brain.unified_field[:, :, 5, 5, 5]
        max_x, max_y = torch.unravel_index(torch.argmax(field_2d), field_2d.shape)
        print(f"  Peak activity at: [{max_x}, {max_y}], value={field_2d[max_x, max_y]:.3f}")


if __name__ == "__main__":
    debug_field_mapping()
    debug_field_evolution()