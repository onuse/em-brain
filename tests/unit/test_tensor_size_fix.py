#!/usr/bin/env python3
"""Test that tensor size mismatch is fixed."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Testing tensor size fix...")

# Create brain with specific sensory dimension
# Note: sensory_dim should be the number of actual sensors (excluding reward)
# The brain will expect sensory_dim + 1 values in the input (sensors + reward)
brain = SimplifiedUnifiedBrain(
    sensory_dim=23,  # 23 sensors (reward will be added as 24th element)
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

# Run several cycles with 24-element input (23 sensors + 1 reward)
for i in range(10):
    sensory_input = [0.5] * 24  # 24 total values
    
    try:
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        if i == 0:
            print(f"✅ Cycle {i}: Success! Motor output: {len(motor_output)} values")
    except Exception as e:
        print(f"❌ Cycle {i}: Error - {e}")
        import traceback
        traceback.print_exc()
        break
else:
    print("\n✅ All cycles completed without tensor size errors!")
    print(f"   Brain sensory_dim: {brain.sensory_dim}")
    print(f"   Predictive system sensory_dim: {brain.predictive_field.sensory_dim}")
    print(f"   Prediction confidence: {brain._current_prediction_confidence:.1%}")