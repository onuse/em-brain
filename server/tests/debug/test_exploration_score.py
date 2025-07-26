#!/usr/bin/env python3
"""
Debug exploration scoring.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np


def test_exploration_scoring():
    """Test exploration scoring logic."""
    
    print("🔍 Testing Exploration Scoring")
    print("=" * 50)
    
    # Create brain
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': True
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    # Run the exact test from behavioral framework
    motor_outputs = []
    rewards_given = []
    
    for i in range(100):  # Same as test
        # Basic sensory input
        sensory_input = [0.5] * 25  # Include reward channel
        
        # Give rewards for certain motor patterns
        if i > 0 and len(motor_outputs) > 0:
            last_motor = motor_outputs[-1]
            # Reward movement in positive X direction
            if last_motor[0] > 0.1:
                sensory_input[24] = 0.8  # High reward
                rewards_given.append(1)
            else:
                sensory_input[24] = 0.0  # No reward
                rewards_given.append(0)
        
        # Process cycle
        motor_output, _ = brain.process_robot_cycle(sensory_input)
        motor_outputs.append(motor_output)
        
        if i % 20 == 0:
            print(f"\nCycle {i}:")
            print(f"  Motor output: {[f'{m:.4f}' for m in motor_output]}")
            if rewards_given:
                print(f"  Recent rewards: {np.mean(rewards_given[-10:]):.2f}")
    
    # Calculate scores
    print("\n📊 Scoring analysis:")
    
    # Motor variance (exploration)
    motor_x_values = [m[0] for m in motor_outputs]
    motor_variance = np.var(motor_x_values)
    print(f"  Motor X values range: [{min(motor_x_values):.4f}, {max(motor_x_values):.4f}]")
    print(f"  Motor X variance: {motor_variance:.6f}")
    print(f"  Variance * 10: {motor_variance * 10:.6f}")
    print(f"  min(1.0, variance * 10): {min(1.0, motor_variance * 10):.6f}")
    
    # Reward acquisition (exploitation)
    reward_rate = np.mean(rewards_given) if rewards_given else 0
    print(f"\n  Rewards given: {sum(rewards_given)}/{len(rewards_given)}")
    print(f"  Reward rate: {reward_rate:.3f}")
    
    # Final score calculation
    exploration_component = min(1.0, motor_variance * 10) * 0.5
    exploitation_component = reward_rate * 0.5
    final_score = exploration_component + exploitation_component
    
    print(f"\n  Exploration component: {exploration_component:.3f}")
    print(f"  Exploitation component: {exploitation_component:.3f}")
    print(f"  Final score: {final_score:.3f}")
    
    # Check if motors are responding to rewards
    print("\n🎯 Checking reward response:")
    
    # Find cycles where reward was given
    reward_cycles = [i for i, r in enumerate(rewards_given) if r == 1]
    if reward_cycles:
        print(f"  Reward cycles: {reward_cycles[:10]}...")
        
        # Check motor output after rewards
        for cycle in reward_cycles[:5]:
            if cycle + 1 < len(motor_outputs):
                print(f"  Cycle {cycle} (reward) → Cycle {cycle+1} motor: {motor_outputs[cycle+1][0]:.4f}")


if __name__ == "__main__":
    test_exploration_scoring()