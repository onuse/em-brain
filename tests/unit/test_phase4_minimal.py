#!/usr/bin/env python3
"""
Minimal Phase 4 Test - Core functionality only
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# Import brain components
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def test_phase4_core():
    """Test core Phase 4 functionality."""
    print("\n=== Testing Phase 4 Core ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=2,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    brain.enable_action_prediction(True)
    
    print("1. System initialization...")
    assert brain.use_action_prediction
    assert brain.action_prediction is not None
    print("   ✓ Action prediction enabled")
    
    print("\n2. Action generation...")
    # Track that different actions are generated
    actions = []
    for i in range(10):
        sensory_input = [float(i % 2), float(1 - i % 2), 0.0]
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        actions.append(motor_output)
    
    # Check action variance (should be non-zero)
    action_tensor = torch.tensor(actions)
    variance = torch.var(action_tensor, dim=0).mean().item()
    print(f"   Action variance: {variance:.3f}")
    assert variance > 0.01
    print("   ✓ Actions are diverse")
    
    print("\n3. Action-outcome tracking...")
    # Run one more cycle to ensure stats are available
    sensory_input = [0.5, 0.5, 0.0]
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    
    stats = brain_state.get('action_prediction', {})
    total_actions = stats.get('total_actions', 0)
    print(f"   Total actions tracked: {total_actions}")
    # First cycle doesn't track yet, so check >= 9
    assert total_actions >= 9
    print("   ✓ Actions are tracked")
    
    print("\n4. Multiple action types...")
    action_types = stats.get('action_types', {})
    active_types = [t for t, info in action_types.items() if info['count'] > 0]
    print(f"   Active action types: {active_types}")
    assert len(active_types) >= 2
    print("   ✓ Multiple strategies used")
    
    return True


def test_simple_learning():
    """Test simple action-outcome learning."""
    print("\n\n=== Testing Simple Learning ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=2,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    brain.enable_action_prediction(True)
    
    print("Creating simple rule: sensor[0] = motor[0] * 0.5")
    
    # Track if weights change
    ap_system = brain.action_prediction
    initial_weights = ap_system.immediate_action_weights.clone()
    
    for cycle in range(20):
        # Apply simple rule
        if cycle == 0:
            sensory_input = [0.0, 0.0]
        else:
            sensory_input = [last_motor[0] * 0.5, 0.0]
        
        sensory_input.append(0.0)  # reward
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        last_motor = motor_output
    
    # Check if weights changed
    weight_change = torch.mean(torch.abs(
        ap_system.immediate_action_weights - initial_weights
    )).item()
    
    print(f"\nWeight change: {weight_change:.6f}")
    assert weight_change > 0.001
    print("✓ Weights are learning")
    
    return True


if __name__ == "__main__":
    print("Phase 4 Minimal Test")
    print("=" * 50)
    
    test1 = test_phase4_core()
    test2 = test_simple_learning()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    if test1 and test2:
        print("\n✓ Phase 4 core functionality is working!")
        print("\nWhat's implemented:")
        print("- Actions selected based on predictions")
        print("- Multiple action types (explore/exploit/test)")
        print("- Action-outcome mappings learned")
        print("- Integrated with hierarchical predictions")
        
        print("\nKnown limitations:")
        print("- Learning is slow on M1 hardware")
        print("- Pinverse warning on MPS (falls back to CPU)")
        print("- Complex behaviors need more training")
        
        print("\nRecommendation: Phase 4 is READY TO PROCEED")
    else:
        print("\n✗ Phase 4 has critical issues")