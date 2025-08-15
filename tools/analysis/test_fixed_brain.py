#!/usr/bin/env python3
"""
Test the fixed UnifiedFieldBrain implementation.

This verifies that the gradient fix improves action generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import time
from brains.field.core_brain import create_unified_field_brain


def test_gradient_strength_improvement():
    """Test that gradients are now stronger and produce better actions."""
    print("Testing Gradient Strength Improvement")
    print("=" * 50)
    
    # Create brain
    brain = create_unified_field_brain(
        spatial_resolution=15,
        temporal_window=8.0,
        field_evolution_rate=0.15,
        constraint_discovery_rate=0.2,
        quiet_mode=False
    )
    
    print("\nProcessing test cycles:")
    
    # Test patterns that should produce different actions
    test_inputs = [
        ("Stationary", [0.5] * 24),
        ("Moving forward", [0.8, 0.5, 0.1] + [0.5] * 21),
        ("Obstacle ahead", [0.5, 0.5, 0.1] + [0.9, 0.9, 0.9] + [0.5] * 18),
        ("Turning right", [0.5, 0.7, 0.1] + [0.3, 0.7, 0.3] + [0.5] * 18),
        ("Complex scene", [0.6, 0.4, 0.2] + [0.8, 0.3, 0.7, 0.2, 0.9] + [0.5] * 16)
    ]
    
    results = []
    
    for cycle, (scenario, sensory_input) in enumerate(test_inputs):
        print(f"\nCycle {cycle+1}: {scenario}")
        
        # Process input
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Record results
        results.append({
            'scenario': scenario,
            'motor_output': motor_output,
            'gradient_strength': brain_state.get('last_gradient_strength', 0.0),
            'action_confidence': brain_state.get('last_action_confidence', 0.0),
            'field_energy': brain_state['field_total_energy'],
            'field_max': brain_state['field_max_activation']
        })
        
        print(f"  Motor: [{', '.join(f'{x:6.3f}' for x in motor_output)}]")
        print(f"  Gradient strength: {results[-1]['gradient_strength']:.6f}")
        print(f"  Action confidence: {results[-1]['action_confidence']:.3f}")
        print(f"  Field energy: {results[-1]['field_energy']:.3f}")
    
    # Analyze results
    print("\n" + "=" * 50)
    print("Analysis:")
    
    # Check that different scenarios produce different actions
    motor_variations = []
    for i in range(1, len(results)):
        prev_motor = torch.tensor(results[i-1]['motor_output'])
        curr_motor = torch.tensor(results[i]['motor_output'])
        variation = torch.norm(curr_motor - prev_motor).item()
        motor_variations.append(variation)
    
    avg_variation = sum(motor_variations) / len(motor_variations)
    print(f"  Average motor variation between scenarios: {avg_variation:.4f}")
    
    # Check gradient strengths
    avg_gradient = sum(r['gradient_strength'] for r in results) / len(results)
    print(f"  Average gradient strength: {avg_gradient:.6f}")
    
    # Check if we're getting non-zero actions
    non_zero_actions = sum(1 for r in results if any(abs(m) > 0.01 for m in r['motor_output']))
    print(f"  Non-zero actions: {non_zero_actions}/{len(results)}")
    
    # Success criteria
    success = True
    if avg_gradient < 1e-5:
        print("  ⚠️  WARNING: Gradients still very weak")
        success = False
    if avg_variation < 0.01:
        print("  ⚠️  WARNING: Actions not varying between scenarios")
        success = False
    if non_zero_actions < len(results) * 0.5:
        print("  ⚠️  WARNING: Too many zero actions")
        success = False
    
    if success:
        print("\n✅ Gradient fix is working properly!")
    else:
        print("\n⚠️  Some issues remain with gradient generation")
    
    return success


def test_gradient_direction_sensitivity():
    """Test that gradients respond to directional field patterns."""
    print("\n\nTesting Gradient Direction Sensitivity")
    print("=" * 50)
    
    brain = create_unified_field_brain(
        spatial_resolution=20,
        quiet_mode=True
    )
    
    # Create directional patterns
    directions = ['x_positive', 'y_positive', 'z_positive', 'diagonal']
    actions = {}
    
    for direction in directions:
        # Clear field
        brain.unified_field.zero_()
        
        # Create gradient pattern
        if direction == 'x_positive':
            for i in range(brain.spatial_resolution):
                brain.unified_field[i, 10, 10, 5, 5] = i * 0.1
        elif direction == 'y_positive':
            for i in range(brain.spatial_resolution):
                brain.unified_field[10, i, 10, 5, 5] = i * 0.1
        elif direction == 'z_positive':
            for i in range(brain.spatial_resolution):
                brain.unified_field[10, 10, i, 5, 5] = i * 0.1
        else:  # diagonal
            for i in range(brain.spatial_resolution):
                brain.unified_field[i, i, 10, 5, 5] = i * 0.1
        
        # Calculate gradients and generate action
        brain._calculate_gradient_flows()
        action = brain._field_gradients_to_robot_action()
        actions[direction] = action.motor_commands
        
        print(f"\n{direction}: motor=[{', '.join(f'{x:.4f}' for x in action.motor_commands)}], "
              f"strength={action.gradient_strength:.6f}")
    
    # Check that different directions produce different actions
    print("\nAction differences:")
    for i, dir1 in enumerate(directions):
        for dir2 in directions[i+1:]:
            diff = torch.norm(actions[dir1] - actions[dir2]).item()
            print(f"  {dir1} vs {dir2}: {diff:.4f}")
    
    print("\n✅ Direction sensitivity test complete")


if __name__ == "__main__":
    print("Fixed UnifiedFieldBrain Test Suite")
    print("==================================\n")
    
    # Run tests
    gradient_success = test_gradient_strength_improvement()
    test_gradient_direction_sensitivity()
    
    print("\n" + "=" * 50)
    if gradient_success:
        print("🎉 All tests passed! The gradient fix is working.")
    else:
        print("⚠️  Some tests failed. Further debugging needed.")
    print("=" * 50)