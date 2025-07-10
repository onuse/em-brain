#!/usr/bin/env python3
"""
Short demo test to verify wall-sticking behavior is fixed.
"""

import sys
import os
sys.path.append('/Users/jkarlsson/Documents/Projects/robot-project/brain')

from core.brain_interface import BrainInterface
from core.robot_2d_simulator import Robot2DSimulator, SensorPacket
import time

def test_short_demo():
    """Run a short demo to test behavior."""
    
    print("🧠 Testing Short Demo (100 steps)")
    print("=" * 50)
    
    # Create robot simulator
    robot = Robot2DSimulator()
    
    # Create brain interface
    brain = BrainInterface()
    
    # Initialize sensory vector length
    initial_sensory = robot.get_sensor_reading()
    sensor_packet = SensorPacket(initial_sensory, time.time())
    
    # Initialize brain with first sensory input
    brain.predict_next_motor_action(sensor_packet)
    
    print(f"🤖 Robot starting at position: {robot.position}")
    print(f"🧠 Brain initialized with {len(initial_sensory)} sensors")
    
    # Track robot positions to detect oscillation
    position_history = []
    action_history = []
    
    # Run for 100 steps
    for step in range(100):
        # Get current sensory reading
        sensory_reading = robot.get_sensor_reading()
        sensor_packet = SensorPacket(sensory_reading, time.time())
        
        # Get brain prediction
        prediction = brain.predict_next_motor_action(sensor_packet)
        
        if prediction:
            # Execute action
            robot.execute_action(prediction.motor_action)
            
            # Track position and action
            position_history.append(tuple(robot.position))
            action_history.append(prediction.motor_action.copy())
            
            # Check for getting stuck (same position for too long)
            if len(position_history) >= 20:
                recent_positions = position_history[-20:]
                unique_positions = set(recent_positions)
                
                if len(unique_positions) <= 3:  # Very little movement
                    print(f"⚠️  STUCK DETECTED at step {step}")
                    print(f"   Position: {robot.position}")
                    print(f"   Recent positions: {list(unique_positions)}")
                    print(f"   Recent action: {prediction.motor_action}")
                    
                    # Get latest brain statistics
                    brain_stats = brain.get_brain_statistics()
                    if "adaptive_tuning_stats" in brain_stats:
                        adaptive_stats = brain_stats["adaptive_tuning_stats"]
                        current_params = adaptive_stats.get("current_parameters", {})
                        similarity_threshold = current_params.get("similarity_threshold", "unknown")
                        print(f"   Similarity threshold: {similarity_threshold}")
                    
                    break
            
            # Print progress every 20 steps
            if step % 20 == 0:
                print(f"Step {step:3d}: Position {robot.position}, Action {prediction.motor_action}")
        
        else:
            print(f"Step {step}: No prediction from brain")
            break
    
    # Final analysis
    print(f"\\n📊 Final Analysis:")
    print(f"   Final position: {robot.position}")
    print(f"   Total unique positions visited: {len(set(position_history))}")
    
    # Check for oscillation patterns
    if len(position_history) >= 10:
        recent_positions = position_history[-10:]
        unique_recent = set(recent_positions)
        oscillation_ratio = 1.0 - (len(unique_recent) / len(recent_positions))
        
        print(f"   Recent oscillation ratio: {oscillation_ratio:.2f}")
        
        if oscillation_ratio > 0.7:
            print(f"   ❌ HIGH OSCILLATION DETECTED")
            return False
        else:
            print(f"   ✅ Normal movement pattern")
            return True
    
    return True

if __name__ == "__main__":
    success = test_short_demo()
    if success:
        print(f"\\n✅ Test completed successfully!")
    else:
        print(f"\\n❌ Test detected problematic behavior!")