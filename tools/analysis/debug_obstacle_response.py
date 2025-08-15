#!/usr/bin/env python3
"""
Debug obstacle response issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import numpy as np
from brain_factory import BrainFactory


def debug_obstacle_response():
    """Debug why obstacle responses are weak"""
    
    print("Debug Obstacle Response")
    print("=" * 50)
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': False
        }
    }
    
    brain = BrainFactory(config=config, quiet_mode=False, enable_logging=False)
    
    print("\n1. Testing Front Obstacle")
    print("-" * 30)
    
    # Clear any previous state
    for i in range(3):
        sensory_input = [0.1, 0.1, 0.1] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
    
    # Now test front obstacle
    print("\nApplying front obstacle (sensor 0 = 0.8)...")
    for i in range(5):
        sensory_input = [0.8, 0.1, 0.1] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
        print(f"  Cycle {i}: turn={action[0]:+.3f}, speed={action[1]:+.3f}")
    
    print("\n2. Testing Left Obstacle")
    print("-" * 30)
    
    # Clear state
    for i in range(3):
        sensory_input = [0.1, 0.1, 0.1] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
    
    print("\nApplying left obstacle (sensor 1 = 0.8)...")
    for i in range(5):
        sensory_input = [0.1, 0.8, 0.1] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
        print(f"  Cycle {i}: turn={action[0]:+.3f}, speed={action[1]:+.3f}")
    
    print("\n3. Testing Right Obstacle")
    print("-" * 30)
    
    # Clear state
    for i in range(3):
        sensory_input = [0.1, 0.1, 0.1] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
    
    print("\nApplying right obstacle (sensor 2 = 0.8)...")
    for i in range(5):
        sensory_input = [0.1, 0.1, 0.8] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
        print(f"  Cycle {i}: turn={action[0]:+.3f}, speed={action[1]:+.3f}")
    
    print("\n4. Analyzing Sensor Mapping")
    print("-" * 30)
    
    # Check what field coordinates each sensor maps to
    print("\nSensor to field coordinate mapping:")
    test_inputs = [
        ([0.8, 0.1, 0.1], "Front obstacle"),
        ([0.1, 0.8, 0.1], "Left obstacle"),
        ([0.1, 0.1, 0.8], "Right obstacle"),
    ]
    
    for sensors, desc in test_inputs:
        # Manually calculate field coordinates for first 3 dimensions
        coords = []
        for i in range(3):
            raw = sensors[i]
            if raw > 0.7:  # Close obstacle
                coord = (raw - 0.5) * 4  # Strong response
            elif raw > 0.5:  # Medium distance
                coord = (raw - 0.5) * 2  # Moderate response
            else:
                coord = (raw - 0.5) * 2  # Normal scaling
            coords.append(coord)
        
        print(f"  {desc}: field_coords = [{coords[0]:+.2f}, {coords[1]:+.2f}, {coords[2]:+.2f}]")
    
    brain.shutdown()
    
    print("\nEXPECTED BEHAVIOR:")
    print("- Front obstacle → negative speed (back up)")
    print("- Left obstacle → positive turn (turn right)")
    print("- Right obstacle → negative turn (turn left)")


if __name__ == "__main__":
    debug_obstacle_response()