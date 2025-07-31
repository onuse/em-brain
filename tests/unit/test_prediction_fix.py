#!/usr/bin/env python3
"""Test that prediction system handles tensor sizes correctly."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Testing prediction system fix...")

# Create brain
brain = SimplifiedUnifiedBrain(
    sensory_dim=23,  # 23 sensors (reward will be added as 24th element)
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

# Enable prediction phases
brain.enable_hierarchical_prediction(True)
brain.enable_action_prediction(True)

# Run enough cycles to trigger prediction system
success_count = 0
for i in range(10):
    sensory_input = [0.5] * 24  # 24 total values
    
    try:
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        success_count += 1
        if i == 0 or i == 5:
            print(f"✅ Cycle {i}: Success! Confidence: {brain._current_prediction_confidence:.1%}")
    except Exception as e:
        print(f"❌ Cycle {i}: Error - {e}")
        if "size of tensor" in str(e).lower():
            print("   This is the tensor size mismatch error we're trying to fix")
        break

if success_count == 10:
    print(f"\n✅ All {success_count} cycles completed successfully!")
    print(f"   Prediction system is working correctly")
else:
    print(f"\n⚠️  Only {success_count}/10 cycles completed")