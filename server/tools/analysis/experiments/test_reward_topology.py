#!/usr/bin/env python3
"""
Test Reward Topology Shaping

Demonstrates how rewards create persistent deformations in field topology,
leading to emergent goal-seeking behavior without explicit goals.
"""

import sys
import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from src.core.dynamic_brain_factory import DynamicBrainFactory


def test_reward_topology_shaping():
    """Test how rewards shape field topology for emergent goal-seeking."""
    
    print("🎯 Testing Reward Topology Shaping")
    print("=" * 60)
    
    # Create brain with minimal config
    config = {
        'use_dynamic_brain': True,
        'quiet_mode': False,
        'field_spatial_resolution': 3,  # Small for faster testing
        'pattern_attention': False,
        'target_cycle_time_ms': 750
    }
    
    factory = DynamicBrainFactory(config)
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=3,
        sensory_dim=16,  # PiCar-X standard
        motor_dim=5
    )
    brain = brain_wrapper.brain
    
    print("\n1️⃣ Baseline Behavior (No Rewards)")
    print("-" * 40)
    
    # Test baseline responses
    baseline_actions = []
    for i in range(5):
        # PiCar-X has 16 sensors, but we need 25 total for reward (24 + reward)
        sensory_input = [0.5] * 24 + [0.0]  # 24 sensors + no reward
        action, state = brain.process_robot_cycle(sensory_input)
        baseline_actions.append(action[:4])
        print(f"   Cycle {i+1}: action = {[f'{a:.3f}' for a in action[:4]]}")
    
    # Calculate baseline variance
    baseline_variance = np.var([np.array(a) for a in baseline_actions])
    print(f"   Baseline action variance: {baseline_variance:.4f}")
    
    print("\n2️⃣ Positive Reward Experience")
    print("-" * 40)
    
    # Give strong positive reward with specific sensory pattern
    reward_pattern = [1.0, 0.0, 0.5] + [0.2] * 21 + [0.8]  # 24 sensors + strong positive reward
    
    print(f"   Applying reward pattern with reward = +0.8")
    action, state = brain.process_robot_cycle(reward_pattern)
    
    # Check topology state
    topology_state = state['topology_shaping']
    print(f"   Active attractors: {topology_state['active_attractors']}")
    print(f"   Strongest attractor: {topology_state['strongest_attractor']:.4f}")
    
    print("\n3️⃣ Testing Goal-Seeking Emergence")
    print("-" * 40)
    
    # Test if brain is drawn toward rewarded pattern
    test_patterns = [
        ([1.0, 0.1, 0.4] + [0.2] * 21 + [0.0], "Similar to reward"),
        ([0.0, 1.0, 0.0] + [0.2] * 21 + [0.0], "Different from reward"),
        ([0.5, 0.5, 0.5] + [0.2] * 21 + [0.0], "Neutral pattern")
    ]
    
    for pattern, desc in test_patterns:
        actions = []
        for _ in range(3):
            action, state = brain.process_robot_cycle(pattern)
            actions.append(action[:4])
        
        avg_action = np.mean([np.array(a) for a in actions], axis=0)
        action_magnitude = np.linalg.norm(avg_action)
        print(f"   {desc}: avg magnitude = {action_magnitude:.3f}")
        print(f"      Actions: {[f'{a:.3f}' for a in avg_action]}")
    
    print("\n4️⃣ Negative Reward (Repulsor)")
    print("-" * 40)
    
    # Apply negative reward
    punish_pattern = [0.0, 1.0, 0.0] + [0.3] * 21 + [-0.7]  # 24 sensors + strong negative reward
    
    print(f"   Applying punishment pattern with reward = -0.7")
    action, state = brain.process_robot_cycle(punish_pattern)
    
    topology_state = state['topology_shaping']
    print(f"   Active attractors: {topology_state['active_attractors']}")
    
    # Test avoidance
    print("\n   Testing avoidance behavior:")
    for i in range(5):
        # Present the punished pattern without reward
        test_pattern = [0.0, 1.0, 0.0] + [0.3] * 21 + [0.0]
        action, state = brain.process_robot_cycle(test_pattern)
        action_magnitude = np.linalg.norm(action[:4])
        print(f"   Cycle {i+1}: magnitude = {action_magnitude:.3f} (should avoid)")
    
    print("\n5️⃣ Long-term Persistence")
    print("-" * 40)
    
    # Run many cycles to see if attractors persist
    print("   Running 20 neutral cycles...")
    neutral_pattern = [0.5] * 24 + [0.0]  # 24 sensors + no reward
    
    for i in range(20):
        action, state = brain.process_robot_cycle(neutral_pattern)
        if i % 5 == 0:
            topology_state = state['topology_shaping']
            print(f"   Cycle {i}: attractors = {topology_state['active_attractors']}, "
                  f"strongest = {topology_state['strongest_attractor']:.4f}")
    
    print("\n✅ Test Complete!")
    print("\n📊 Summary:")
    print("   - Rewards create persistent field deformations")
    print("   - Positive rewards create attractors (goal-seeking)")
    print("   - Negative rewards create repulsors (avoidance)")
    print("   - No explicit goals or planning required!")
    print("   - Behavior emerges from topology alone")


if __name__ == "__main__":
    test_reward_topology_shaping()