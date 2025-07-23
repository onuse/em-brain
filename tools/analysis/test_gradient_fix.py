#!/usr/bin/env python3
"""
Test the gradient extraction fix in a controlled environment.

This script tests the gradient fix without modifying the production brain.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import numpy as np
import time
from typing import List, Tuple

# Import the brain and the fix
from brains.field.core_brain import UnifiedFieldBrain, create_unified_field_brain
from brains.field.gradient_fix import (
    calculate_proper_gradient_flows,
    extract_local_gradients,
    generate_motor_commands_from_gradients,
    patch_unified_field_brain
)


def test_original_vs_fixed_gradients():
    """Compare original broken implementation with fixed implementation."""
    print("=" * 60)
    print("Testing Original vs Fixed Gradient Extraction")
    print("=" * 60)
    
    # Create brain
    brain = create_unified_field_brain(
        spatial_resolution=10,
        temporal_window=5.0,
        quiet_mode=True
    )
    
    # Create test pattern in field
    print("\n1. Creating test field pattern...")
    # Strong gradient in X direction
    for i in range(brain.spatial_resolution):
        brain.unified_field[i, 5, 5, 5, 5] = i * 0.5
    
    print(f"   Field shape: {brain.unified_field.shape}")
    print(f"   Field max: {torch.max(brain.unified_field).item():.3f}")
    
    # Test original implementation
    print("\n2. Testing ORIGINAL gradient extraction...")
    brain._calculate_gradient_flows()
    
    # Original broken extraction
    center_idx = brain.spatial_resolution // 2
    try:
        # This is the problematic line from original
        grad_x_broken = brain.gradient_flows['gradient_x'][
            center_idx, center_idx, center_idx, :, :, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        print(f"   Original extraction shape: {grad_x_broken.shape}")
        print(f"   Original max gradient: {torch.max(torch.abs(grad_x_broken)).item():.6f}")
    except Exception as e:
        print(f"   Original extraction failed: {e}")
    
    # Generate action with original method
    original_action = brain._field_gradients_to_robot_action()
    print(f"   Original motor commands: {original_action.motor_commands}")
    print(f"   Original gradient strength: {original_action.gradient_strength:.8f}")
    print(f"   Original action confidence: {original_action.action_confidence:.3f}")
    
    # Test fixed implementation
    print("\n3. Testing FIXED gradient extraction...")
    
    # Use the brain's own gradient calculation (now fixed)
    brain._calculate_gradient_flows()
    gradient_flows = brain.gradient_flows
    print(f"   Fixed gradient shapes:")
    for key, grad in gradient_flows.items():
        if grad is not None:
            print(f"     {key}: {grad.shape}")
    
    # Extract local gradients
    local_gradients = extract_local_gradients(
        gradient_flows,
        (center_idx, center_idx, center_idx),
        window_size=5
    )
    
    # Generate motor commands
    motor_commands, gradient_strength = generate_motor_commands_from_gradients(
        local_gradients,
        brain.gradient_following_strength,
        brain.total_dimensions
    )
    
    print(f"   Fixed motor commands: {motor_commands}")
    print(f"   Fixed gradient strength: {gradient_strength:.8f}")
    
    # Compare results
    print("\n4. Comparison:")
    print(f"   Gradient strength improvement: {gradient_strength / (original_action.gradient_strength + 1e-10):.1f}x")
    print(f"   Motor command magnitude - Original: {torch.norm(original_action.motor_commands).item():.6f}")
    print(f"   Motor command magnitude - Fixed: {torch.norm(motor_commands).item():.6f}")


def test_patched_brain_behavior():
    """Test brain behavior with patched gradient extraction."""
    print("\n" + "=" * 60)
    print("Testing Patched Brain Behavior")
    print("=" * 60)
    
    # Patch the brain
    patch_unified_field_brain()
    
    # Create new brain instance (will use patched methods)
    brain = create_unified_field_brain(
        spatial_resolution=10,
        temporal_window=5.0,
        quiet_mode=True
    )
    
    print("\n1. Testing patched brain with varied inputs...")
    
    test_scenarios = [
        ("Stationary", [0.5] * 24),
        ("Moving forward", [0.7, 0.5, 0.1] + [0.5] * 21),
        ("Turning", [0.5, 0.8, 0.1] + [0.5] * 21),
        ("Obstacle detected", [0.5, 0.5, 0.1] + [0.9, 0.9, 0.9] + [0.5] * 18)
    ]
    
    for scenario_name, test_input in test_scenarios:
        print(f"\n   Scenario: {scenario_name}")
        action, state = brain.process_robot_cycle(test_input)
        
        print(f"     Motor output: [{', '.join(f'{x:.3f}' for x in action)}]")
        print(f"     Field energy: {state['field_total_energy']:.3f}")
        print(f"     Gradient strength: {state['last_gradient_strength']:.6f}")
        print(f"     Action confidence: {state['last_action_confidence']:.3f}")


def test_gradient_responsiveness():
    """Test that gradients respond appropriately to field changes."""
    print("\n" + "=" * 60)
    print("Testing Gradient Responsiveness")
    print("=" * 60)
    
    # Use patched brain
    brain = create_unified_field_brain(
        spatial_resolution=15,
        temporal_window=5.0,
        quiet_mode=True
    )
    
    print("\n1. Testing gradient response to different patterns...")
    
    patterns = [
        ("Empty field", lambda: brain.unified_field.zero_()),
        ("Uniform field", lambda: brain.unified_field.fill_(0.5)),
        ("X gradient", lambda: [
            brain.unified_field.zero_(),
            [setattr(brain.unified_field[i, :, :, 5, 5], 'data', 
                    torch.ones_like(brain.unified_field[i, :, :, 5, 5]) * i * 0.1)
             for i in range(brain.spatial_resolution)]
        ]),
        ("Central peak", lambda: [
            brain.unified_field.zero_(),
            setattr(brain.unified_field[7:9, 7:9, 7:9, :, :], 'data',
                   torch.ones(2, 2, 2, 10, 15) * 2.0)
        ])
    ]
    
    for pattern_name, setup_func in patterns:
        print(f"\n   Pattern: {pattern_name}")
        
        # Setup pattern
        result = setup_func()
        if isinstance(result, list) and len(result) > 1:
            for r in result[1:]:
                pass  # Setattr calls already executed
        
        # Calculate gradients
        gradient_flows = calculate_proper_gradient_flows(brain.unified_field)
        
        # Get gradient statistics
        grad_x_max = torch.max(torch.abs(gradient_flows['gradient_x'])).item()
        grad_y_max = torch.max(torch.abs(gradient_flows['gradient_y'])).item()
        grad_z_max = torch.max(torch.abs(gradient_flows['gradient_z'])).item()
        
        print(f"     Max gradients - X: {grad_x_max:.6f}, Y: {grad_y_max:.6f}, Z: {grad_z_max:.6f}")
        
        # Generate action
        center = brain.spatial_resolution // 2
        local_grads = extract_local_gradients(gradient_flows, (center, center, center))
        motor_cmds, strength = generate_motor_commands_from_gradients(local_grads)
        
        print(f"     Motor commands: [{', '.join(f'{x:.3f}' for x in motor_cmds)}]")
        print(f"     Gradient strength: {strength:.6f}")


def test_performance_comparison():
    """Compare performance of original vs fixed implementation."""
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=20,  # Larger for performance test
        temporal_window=10.0,
        quiet_mode=True
    )
    
    # Fill with random data
    brain.unified_field = torch.randn_like(brain.unified_field) * 0.5
    
    # Time original implementation
    print("\n1. Original implementation timing...")
    start = time.perf_counter()
    for _ in range(100):
        brain._calculate_gradient_flows()
        _ = brain._field_gradients_to_robot_action()
    original_time = time.perf_counter() - start
    print(f"   Original: {original_time*10:.2f}ms per cycle")
    
    # Time fixed implementation
    print("\n2. Fixed implementation timing...")
    start = time.perf_counter()
    for _ in range(100):
        gradient_flows = calculate_proper_gradient_flows(brain.unified_field)
        local_grads = extract_local_gradients(gradient_flows, (10, 10, 10))
        motor_cmds, _ = generate_motor_commands_from_gradients(local_grads)
    fixed_time = time.perf_counter() - start
    print(f"   Fixed: {fixed_time*10:.2f}ms per cycle")
    
    print(f"\n   Performance ratio: {original_time/fixed_time:.2f}x")


if __name__ == "__main__":
    print("Gradient Extraction Fix Test Suite")
    print("==================================\n")
    
    # Run all tests
    test_original_vs_fixed_gradients()
    test_patched_brain_behavior()
    test_gradient_responsiveness()
    test_performance_comparison()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)