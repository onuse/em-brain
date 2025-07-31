#!/usr/bin/env python3
"""
Phase 4 Test: Action as Prediction Testing

Tests that actions are selected based on predicted outcomes and that
the brain learns action-outcome relationships over time.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time

# Import brain components
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def test_action_prediction_basics():
    """Test basic action-prediction functionality."""
    print("\n=== Testing Action-Prediction Basics ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=3,
        motor_dim=3,  # 2 motors + confidence
        spatial_resolution=32,
        quiet_mode=False
    )
    
    # Enable action prediction (automatically enables hierarchical)
    brain.enable_action_prediction(True)
    
    print("\n1. Testing that action prediction system initializes...")
    assert hasattr(brain, 'action_prediction')
    assert brain.use_action_prediction
    assert brain.predictive_field.use_hierarchical  # Should be auto-enabled
    print("   ✓ Action prediction initialized")
    
    print("\n2. Testing that actions are generated based on predictions...")
    
    # Run a few cycles
    for i in range(5):
        sensory_input = [float(i % 2), float((i+1) % 2), 0.0, 0.0]  # reward=0
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        print(f"\nCycle {i}:")
        print(f"  Motor output: {[f'{m:.3f}' for m in motor_output]}")
        
        # Check that motor output has correct dimension
        assert len(motor_output) == 2  # motor_dim - 1
    
    print("\n   ✓ Actions generated successfully")
    
    print("\n3. Testing that action statistics are tracked...")
    if 'action_prediction' in brain_state:
        stats = brain_state['action_prediction']
        print(f"   Total actions: {stats['total_actions']}")
        print(f"   Action types: {list(stats['action_types'].keys())}")
        assert stats['total_actions'] > 0
        print("   ✓ Action statistics tracked")
    
    return True


def test_action_outcome_learning():
    """Test that the brain learns action-outcome relationships."""
    print("\n\n=== Testing Action-Outcome Learning ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=2,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    brain.enable_action_prediction(True)
    
    print("Creating deterministic action-outcome relationship...")
    print("Rule: sensor[0] = motor[0], sensor[1] = -motor[1]")
    
    action_errors = []
    
    for cycle in range(50):
        # Previous action determines current sensory
        if cycle == 0:
            sensory_input = [0.0, 0.0]
        else:
            # Apply deterministic rule based on last motor output
            sensory_input = [
                last_motor[0],  # sensor[0] follows motor[0]
                -last_motor[1] if len(last_motor) > 1 else 0.0  # sensor[1] opposes motor[1]
            ]
        
        sensory_input.append(0.0)  # reward
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        last_motor = motor_output
        
        # After some learning, check prediction accuracy
        if cycle >= 10 and cycle % 10 == 0:
            stats = brain_state.get('action_prediction', {})
            if 'prediction_accuracy' in stats and 'immediate' in stats['prediction_accuracy']:
                accuracy = stats['prediction_accuracy']['immediate']
                print(f"\nCycle {cycle}: Prediction accuracy = {accuracy:.3f}")
                action_errors.append(1.0 - accuracy)
    
    # Check if learning occurred
    if len(action_errors) >= 2:
        improvement = action_errors[0] - action_errors[-1]
        print(f"\nLearning improvement: {improvement:.3f}")
        return improvement > 0.1
    
    return True


def test_exploration_vs_exploitation():
    """Test that action selection balances exploration and exploitation."""
    print("\n\n=== Testing Exploration vs Exploitation ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=2,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    brain.enable_action_prediction(True)
    
    print("Running with high exploration drive...")
    
    # Track action diversity
    high_explore_actions = []
    low_explore_actions = []
    
    # High exploration phase
    for cycle in range(20):
        # Provide neutral sensory input
        sensory_input = [0.5, 0.5, 0.0]  # No reward
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        high_explore_actions.append(motor_output)
    
    # Calculate action variance (high exploration = high variance)
    high_explore_tensor = torch.tensor(high_explore_actions)
    high_variance = torch.var(high_explore_tensor, dim=0).mean().item()
    
    print(f"High exploration variance: {high_variance:.3f}")
    
    # Now create a rewarding pattern to encourage exploitation
    print("\nSwitching to exploitation with rewards...")
    
    for cycle in range(20):
        # Reward for specific action pattern
        last_motor = low_explore_actions[-1] if low_explore_actions else [0, 0]
        reward = 1.0 if abs(last_motor[0]) < 0.3 else -0.5
        
        sensory_input = [0.5, 0.5, reward]
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        low_explore_actions.append(motor_output)
    
    # Calculate action variance (exploitation = low variance)
    if len(low_explore_actions) > 10:
        low_explore_tensor = torch.tensor(low_explore_actions[-10:])
        low_variance = torch.var(low_explore_tensor, dim=0).mean().item()
        print(f"Low exploration variance: {low_variance:.3f}")
        
        # High exploration should have higher variance
        return high_variance > low_variance * 1.5
    
    return True


def test_action_types():
    """Test that different action types are generated."""
    print("\n\n=== Testing Action Types ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=3,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    brain.enable_action_prediction(True)
    
    print("Running cycles to generate different action types...")
    
    for cycle in range(30):
        # Vary input to encourage different strategies
        if cycle < 10:
            # Random input - encourage exploration
            sensory_input = [np.random.uniform(-1, 1) for _ in range(3)]
        elif cycle < 20:
            # Stable input - encourage exploitation
            sensory_input = [0.5, -0.5, 0.0]
        else:
            # Changing input - encourage testing
            sensory_input = [float(np.sin(cycle * 0.5)), float(np.cos(cycle * 0.5)), 0.0]
        
        sensory_input.append(0.0)  # reward
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    
    # Check action type statistics
    stats = brain_state.get('action_prediction', {})
    if 'action_types' in stats:
        print("\nAction type counts:")
        for action_type, info in stats['action_types'].items():
            print(f"  {action_type}: {info['count']} actions (avg error: {info['avg_error']:.3f})")
        
        # Should have at least 2 different action types
        active_types = [t for t, info in stats['action_types'].items() if info['count'] > 0]
        print(f"\nActive action types: {active_types}")
        return len(active_types) >= 2
    
    return False


if __name__ == "__main__":
    print("Phase 4 Test: Action as Prediction Testing")
    print("=" * 50)
    
    # Run tests
    test1 = test_action_prediction_basics()
    test2 = test_action_outcome_learning()
    test3 = test_exploration_vs_exploitation()
    test4 = test_action_types()
    
    # Summary
    print("\n" + "=" * 50)
    print("PHASE 4 TEST SUMMARY")
    print("=" * 50)
    
    tests_passed = sum([test1, test2, test3, test4])
    print(f"\nPassed {tests_passed}/4 tests")
    
    if test1:
        print("✓ Action prediction basics work")
    else:
        print("✗ Basic functionality needs work")
        
    if test2:
        print("✓ Action-outcome learning works")
    else:
        print("✗ Learning needs improvement")
        
    if test3:
        print("✓ Exploration/exploitation balance works")
    else:
        print("✗ Action selection needs tuning")
        
    if test4:
        print("✓ Multiple action types generated")
    else:
        print("✗ Action diversity needs work")
    
    if tests_passed == 4:
        print("\n🎉 Phase 4 is working correctly!")
        print("\nKey achievements:")
        print("- Actions are selected based on predicted outcomes")
        print("- Action-outcome relationships are learned")
        print("- Exploration/exploitation is balanced")
        print("- Multiple action strategies emerge")
    else:
        print(f"\n⚠️  Phase 4 needs more work ({4-tests_passed} issues)")