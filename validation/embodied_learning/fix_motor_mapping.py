#!/usr/bin/env python3
"""
Fix motor mapping for validation experiments.

The field brain outputs gradients that need to be converted to discrete actions.
This script provides proper mapping functions.
"""

import numpy as np
from typing import List
from environments.sensory_motor_world import ActionType


def gradient_to_action(motor_vector: List[float]) -> ActionType:
    """
    Convert field brain motor gradients to discrete robot actions.
    
    The field brain outputs:
    - motor[0]: x_grad (forward/backward gradient)
    - motor[1]: y_grad (left/right gradient)
    - motor[2]: pattern_strength (always positive)
    - motor[3]: z_grad (vertical, usually 0)
    
    This function interprets gradients as movement intentions.
    """
    if len(motor_vector) < 2:
        return ActionType.STOP
    
    # Extract primary gradients
    forward_grad = motor_vector[0] if len(motor_vector) > 0 else 0.0
    turn_grad = motor_vector[1] if len(motor_vector) > 1 else 0.0
    
    # Decision thresholds
    FORWARD_THRESHOLD = 0.2
    TURN_THRESHOLD = 0.3
    STOP_THRESHOLD = 0.1
    
    # Priority-based action selection
    # 1. Check if we should stop (low overall activation)
    total_activation = abs(forward_grad) + abs(turn_grad)
    if total_activation < STOP_THRESHOLD:
        return ActionType.STOP
    
    # 2. Check turning (higher priority for obstacle avoidance)
    if abs(turn_grad) > TURN_THRESHOLD:
        if turn_grad > 0:
            return ActionType.TURN_RIGHT
        else:
            return ActionType.TURN_LEFT
    
    # 3. Check forward movement
    if forward_grad > FORWARD_THRESHOLD:
        return ActionType.MOVE_FORWARD
    
    # 4. If negative forward gradient, stop (obstacle ahead)
    if forward_grad < -FORWARD_THRESHOLD:
        return ActionType.STOP
    
    # 5. Default to slight exploration (alternate turns)
    # This prevents getting stuck
    return ActionType.TURN_LEFT if np.random.random() < 0.5 else ActionType.TURN_RIGHT


def add_reward_signal(sensory_vector: List[float], environment) -> List[float]:
    """
    Add reward signal as 25th element based on environment state.
    
    Reward encourages:
    - Getting closer to light sources
    - Conserving battery
    - Avoiding obstacles
    """
    # Calculate light reward (inverse of distance)
    min_light_dist = min([
        np.linalg.norm([
            environment.robot_state.position[0] - light.x,
            environment.robot_state.position[1] - light.y
        ])
        for light in environment.light_sources
    ])
    light_reward = 1.0 / (1.0 + min_light_dist)
    
    # Battery reward (conservation)
    battery_reward = environment.robot_state.battery / environment.battery_capacity * 0.5
    
    # Movement reward (encourage exploration)
    if hasattr(environment.robot_state, 'last_position'):
        movement = np.linalg.norm(
            environment.robot_state.position - environment.robot_state.last_position
        )
        movement_reward = min(movement * 0.1, 0.3)
    else:
        movement_reward = 0.0
    
    # Combine rewards
    total_reward = np.clip(
        light_reward * 0.5 + battery_reward * 0.3 + movement_reward * 0.2,
        -1.0, 1.0
    )
    
    # Add to sensory vector
    return sensory_vector + [total_reward]


def test_gradient_mapping():
    """Test the gradient to action mapping."""
    test_cases = [
        # (motor_vector, expected_action, description)
        ([0.5, 0.0, 0.1, 0.0], ActionType.MOVE_FORWARD, "Strong forward gradient"),
        ([0.0, 0.5, 0.1, 0.0], ActionType.TURN_RIGHT, "Strong right gradient"),
        ([0.0, -0.5, 0.1, 0.0], ActionType.TURN_LEFT, "Strong left gradient"),
        ([0.05, 0.05, 0.1, 0.0], ActionType.STOP, "Low activation"),
        ([-0.5, 0.0, 0.1, 0.0], ActionType.STOP, "Negative forward (obstacle)"),
        ([0.3, 0.4, 0.1, 0.0], ActionType.TURN_RIGHT, "Turn priority over forward"),
    ]
    
    print("Testing gradient to action mapping:")
    print("-" * 50)
    
    for motor_vector, expected, description in test_cases:
        action = gradient_to_action(motor_vector)
        status = "✓" if action == expected else "✗"
        print(f"{status} {description}")
        print(f"  Input: {motor_vector}")
        print(f"  Output: {action.name} (expected: {expected.name})")
        print()


if __name__ == "__main__":
    test_gradient_mapping()