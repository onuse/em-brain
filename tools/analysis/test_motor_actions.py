#!/usr/bin/env python3
"""
Quick test of motor action generation to verify right turns work.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'server'))

from src.communication import MinimalBrainClient
import time
import numpy as np

def test_motor_actions():
    """Test that all 4 motor actions can be generated."""
    
    print("🧪 Testing Motor Action Generation")
    
    # Connect to brain
    client = MinimalBrainClient()
    if not client.connect():
        print("❌ Failed to connect to brain server")
        return
    
    print("✅ Connected to brain server")
    
    # Test multiple cycles to see action distribution
    actions_taken = {'MOVE_FORWARD': 0, 'TURN_LEFT': 0, 'TURN_RIGHT': 0, 'STOP': 0}
    
    try:
        for i in range(50):  # 50 test actions
            # Send random sensory input
            sensory_input = [np.random.random() for _ in range(16)]
            
            # Get action from brain
            action = client.get_action(sensory_input, timeout=5.0)
            
            if action is None:
                print(f"❌ No response from brain on cycle {i}")
                continue
            
            # Interpret action (same logic as SensoryMotorWorld)
            max_idx = np.argmax(action)
            action_names = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']
            action_name = action_names[max_idx]
            
            actions_taken[action_name] += 1
            
            if i % 10 == 0:
                print(f"   Cycle {i}: {action_name} (vector: {[f'{x:.3f}' for x in action]})")
            
            time.sleep(0.1)  # Small delay
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        client.disconnect()
    
    # Print results
    total_actions = sum(actions_taken.values())
    print(f"\n📊 Results after {total_actions} actions:")
    for action, count in actions_taken.items():
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"   {action}: {count} ({percentage:.1f}%)")
    
    # Check if right turns were generated
    if actions_taken['TURN_RIGHT'] > 0:
        print("✅ RIGHT TURNS WORKING!")
    else:
        print("❌ Right turns still not working")
    
    return actions_taken

if __name__ == "__main__":
    test_motor_actions()