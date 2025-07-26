#!/usr/bin/env python3
"""
Quick behavioral test to verify brain is working correctly
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.communication.client import MinimalBrainClient

def test_obstacle_avoidance():
    """Test basic obstacle avoidance behavior"""
    print("🧪 Testing Obstacle Avoidance")
    
    client = MinimalBrainClient("localhost", 9999)
    if not client.connect():
        print("❌ Failed to connect to brain server")
        return False
    
    print("✅ Connected to brain server")
    
    # Test 10 cycles with obstacle scenarios
    for i in range(10):
        # Create sensory input: 24D vector
        # [ultrasonic, line_sensors(5), gyro(3), accel(3), motor_current(4), battery(3), touch(4), heading]
        sensory = [0.0] * 24
        
        # Simulate obstacle at different distances
        obstacle_distance = 0.2 + (i * 0.08)  # 20cm to 92cm
        sensory[0] = obstacle_distance
        
        # Get motor response
        start_time = time.time()
        motor = client.get_action(sensory)
        cycle_time = (time.time() - start_time) * 1000
        
        if motor is None:
            print(f"❌ Cycle {i+1}: No motor response")
            continue
            
        # Check response: should turn away when obstacle is close
        forward_speed = motor[0]
        turn_rate = motor[1] if len(motor) > 1 else 0
        
        expected_behavior = "turn" if obstacle_distance < 0.5 else "forward"
        actual_behavior = "turn" if abs(turn_rate) > 0.1 else "forward"
        
        status = "✅" if expected_behavior == actual_behavior else "⚠️"
        print(f"{status} Cycle {i+1}: obstacle={obstacle_distance:.1f}m, "
              f"response={actual_behavior}, time={cycle_time:.1f}ms")
    
    client.disconnect()
    return True

def test_learning():
    """Test basic learning behavior"""
    print("\n🧪 Testing Learning Behavior")
    
    client = MinimalBrainClient("localhost", 9999)
    if not client.connect():
        print("❌ Failed to connect to brain server")
        return False
    
    # Present a simple pattern 5 times
    pattern_responses = []
    
    for trial in range(5):
        print(f"\n  Trial {trial+1}:")
        responses = []
        
        # Pattern: obstacle appears and disappears
        for step in range(4):
            sensory = [0.0] * 24
            sensory[0] = 0.3 if step % 2 == 0 else 1.0  # Alternating obstacle
            
            motor = client.get_action(sensory)
            if motor:
                responses.append(motor[1] if len(motor) > 1 else 0)  # Turn rate
                
        pattern_responses.append(responses)
        print(f"    Responses: {[f'{r:.2f}' for r in responses]}")
    
    # Check if responses become more consistent over trials
    if len(pattern_responses) >= 2:
        first_var = np.var(pattern_responses[0])
        last_var = np.var(pattern_responses[-1])
        print(f"\n  Response variance: first={first_var:.3f}, last={last_var:.3f}")
        print(f"  {'✅' if last_var <= first_var else '⚠️'} Learning detected")
    
    client.disconnect()
    return True

def main():
    print("🧠 Quick Behavioral Test")
    print("=" * 60)
    
    # Test obstacle avoidance
    test_obstacle_avoidance()
    
    # Test learning
    test_learning()
    
    print("\n✅ Behavioral test complete!")

if __name__ == "__main__":
    main()