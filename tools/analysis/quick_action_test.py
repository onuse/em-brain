#!/usr/bin/env python3
"""
Quick test of motor action generation to identify stuck behavior.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'server'))

from src.communication import MinimalBrainClient
import time
import numpy as np

def test_action_distribution():
    """Test action distribution to identify stuck behavior."""
    
    print("Testing Motor Action Distribution")
    
    # Connect to brain with proper handshake
    client = MinimalBrainClient()
    robot_capabilities = [
        1.0,   # Robot version
        24.0,  # Sensory vector size (24D)
        4.0,   # Action vector size (4 actions)
        2.0,   # Hardware type (2.0 = Simulated robot)
        7.0    # Capabilities mask
    ]
    
    if not client.connect(robot_capabilities):
        print("Failed to connect to brain server")
        return None
    
    print("Connected to brain server")
    
    # Test with varied sensory inputs
    actions_taken = {'MOVE_FORWARD': 0, 'TURN_LEFT': 0, 'TURN_RIGHT': 0, 'STOP': 0}
    
    try:
        # Test 1: Random sensory input
        print("\nTest 1: Random sensory input")
        for i in range(20):
            sensory_input = [np.random.random() for _ in range(24)]
            action = client.get_action(sensory_input, timeout=5.0)
            
            if action is None:
                print(f"No response on cycle {i}")
                continue
            
            # Interpret action
            max_idx = np.argmax(action)
            action_names = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']
            action_name = action_names[max_idx]
            actions_taken[action_name] += 1
            
            if i < 5:  # Show first few
                print(f"  Cycle {i}: {action_name} (vector: {[f'{x:.3f}' for x in action]})")
        
        # Test 2: Light source visible (should encourage forward movement)
        print("\nTest 2: Light source directly ahead")
        for i in range(10):
            # Light sensor indicates bright light ahead
            sensory_input = [0.5, 0.5, 0.0, 1.0] + [0.0]*8 + [1.0, 0.0, 0.0, 0.0] + [0.0]*7 + [0.0]
            action = client.get_action(sensory_input, timeout=5.0)
            
            if action is not None:
                max_idx = np.argmax(action)
                action_name = action_names[max_idx]
                actions_taken[action_name] += 1
                
                if i < 3:
                    print(f"  Light ahead cycle {i}: {action_name}")
        
        # Test 3: Obstacle ahead (should encourage turning)
        print("\nTest 3: Obstacle directly ahead")
        for i in range(10):
            # Distance sensor indicates obstacle ahead
            sensory_input = [0.5, 0.5, 0.0, 1.0] + [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0]*12
            action = client.get_action(sensory_input, timeout=5.0)
            
            if action is not None:
                max_idx = np.argmax(action)
                action_name = action_names[max_idx]
                actions_taken[action_name] += 1
                
                if i < 3:
                    print(f"  Obstacle ahead cycle {i}: {action_name}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        client.disconnect()
    
    # Print results
    total_actions = sum(actions_taken.values())
    print(f"\nResults after {total_actions} actions:")
    for action, count in actions_taken.items():
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"   {action}: {count} ({percentage:.1f}%)")
    
    # Analysis
    right_turn_ratio = actions_taken['TURN_RIGHT'] / total_actions if total_actions > 0 else 0
    forward_ratio = actions_taken['MOVE_FORWARD'] / total_actions if total_actions > 0 else 0
    
    print(f"\nAnalysis:")
    print(f"   Right turn ratio: {right_turn_ratio:.1%}")
    print(f"   Forward movement ratio: {forward_ratio:.1%}")
    
    if right_turn_ratio > 0.7:
        print("   ISSUE: Robot is stuck in right turning behavior!")
        print("   This explains why it can't reach light sources.")
    elif forward_ratio < 0.1:
        print("   ISSUE: Robot rarely moves forward!")
        print("   This could prevent reaching light sources.")
    else:
        print("   Action distribution appears normal.")
    
    return actions_taken

if __name__ == "__main__":
    test_action_distribution()