#!/usr/bin/env python3
"""Test that brain handles sensory dimensions correctly with and without reward."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Testing sensory dimension handling...\n")

# Test 1: 24 sensors, no reward
print("Test 1: 24 sensors without reward")
brain1 = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

success = True
for i in range(5):
    sensory_input = [0.5] * 24  # 24 sensors, no reward
    try:
        motor_output, brain_state = brain1.process_robot_cycle(sensory_input)
        if i == 0:
            print(f"  ✅ Cycle {i}: Success with 24 sensors")
    except Exception as e:
        print(f"  ❌ Cycle {i}: Failed - {e}")
        success = False
        break

# Test 2: 24 sensors + 1 reward
print("\nTest 2: 24 sensors with reward")
brain2 = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

for i in range(5):
    sensory_input = [0.5] * 24 + [0.8]  # 24 sensors + 1 reward
    try:
        motor_output, brain_state = brain2.process_robot_cycle(sensory_input)
        if i == 0:
            print(f"  ✅ Cycle {i}: Success with 24 sensors + reward")
    except Exception as e:
        print(f"  ❌ Cycle {i}: Failed - {e}")
        success = False
        break

# Test 3: Different sensor count
print("\nTest 3: 16 sensors without reward")
brain3 = SimplifiedUnifiedBrain(
    sensory_dim=16,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

for i in range(5):
    sensory_input = [0.5] * 16  # 16 sensors, no reward
    try:
        motor_output, brain_state = brain3.process_robot_cycle(sensory_input)
        if i == 0:
            print(f"  ✅ Cycle {i}: Success with 16 sensors")
    except Exception as e:
        print(f"  ❌ Cycle {i}: Failed - {e}")
        success = False
        break

if success:
    print("\n✅ All tests passed! Brain correctly handles:")
    print("   - Sensory input without reward (sensory_dim values)")
    print("   - Sensory input with reward (sensory_dim + 1 values)")
    print("   - Different sensor configurations")
else:
    print("\n❌ Some tests failed")