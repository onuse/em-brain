#!/usr/bin/env python3
"""Quick motor output test"""

import sys
from pathlib import Path
import numpy as np

# Add paths
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

# Import minimal brain client
from src.communication import MinimalBrainClient

def test_motor_outputs():
    """Test what motor outputs the brain produces"""
    print("🔍 Quick Motor Output Test")
    print("=" * 40)
    
    # Connect to brain
    client = MinimalBrainClient('localhost', 9999)
    if not client.connect():
        print("❌ Failed to connect to brain")
        return
        
    print("✅ Connected to brain")
    
    # Collect motor outputs
    motor_outputs = []
    action_counts = {'FORWARD': 0, 'LEFT': 0, 'RIGHT': 0, 'STOP': 0}
    
    for i in range(50):
        # Create sensory input
        sensory = [np.sin(i * 0.1), np.cos(i * 0.1)] + [np.random.randn() * 0.1 for _ in range(22)]
        
        # Get motor output
        motor = client.process_data(sensory)
        if motor:
            motor_outputs.append(motor[:4])
            
            # Which action would be selected?
            action_idx = np.argmax(motor[:4])
            actions = ['FORWARD', 'LEFT', 'RIGHT', 'STOP']
            action_counts[actions[action_idx]] += 1
            
    # Analyze
    if motor_outputs:
        motor_array = np.array(motor_outputs)
        
        print(f"\n📊 Motor Analysis ({len(motor_outputs)} samples):")
        print(f"Motor channel means:")
        for i in range(4):
            print(f"  {i}: {np.mean(motor_array[:, i]):.3f} ± {np.std(motor_array[:, i]):.3f}")
            
        print(f"\nAction selection:")
        for action, count in action_counts.items():
            print(f"  {action}: {count} ({count/len(motor_outputs)*100:.1f}%)")
            
        # Check if robot would actually move
        forward_selections = action_counts['FORWARD']
        movement_ratio = forward_selections / len(motor_outputs)
        print(f"\nMovement potential: {movement_ratio:.1%} forward actions")
        
        if movement_ratio < 0.2:
            print("⚠️  Very low forward movement - robot likely stuck!")
    
    client.disconnect()

if __name__ == "__main__":
    test_motor_outputs()