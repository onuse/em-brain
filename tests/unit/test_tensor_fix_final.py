#!/usr/bin/env python3
"""Final test for tensor size mismatch fix."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Final tensor size fix verification...\n")

# Test 1: Standard configuration
print("Test 1: Standard configuration (23 sensors + 1 reward)")
brain = SimplifiedUnifiedBrain(
    sensory_dim=23,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

# Enable all prediction phases
brain.enable_hierarchical_prediction(True)
brain.enable_action_prediction(True)

success_count = 0
for i in range(20):  # Run more cycles to test history stacking
    sensory_input = [0.5] * 24
    try:
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        success_count += 1
    except Exception as e:
        print(f"  ❌ Failed at cycle {i}: {e}")
        break

if success_count == 20:
    print(f"  ✅ All {success_count} cycles completed successfully")
else:
    print(f"  ⚠️  Only {success_count}/20 cycles completed")

# Test 2: Different sensor configuration
print("\nTest 2: Different configuration (15 sensors + 1 reward)")
brain2 = SimplifiedUnifiedBrain(
    sensory_dim=15,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)
brain2.enable_action_prediction(True)

success_count2 = 0
for i in range(20):
    sensory_input = [0.3] * 16  # 15 + 1 reward
    try:
        motor_output, brain_state = brain2.process_robot_cycle(sensory_input)
        success_count2 += 1
    except Exception as e:
        print(f"  ❌ Failed at cycle {i}: {e}")
        break

if success_count2 == 20:
    print(f"  ✅ All {success_count2} cycles completed successfully")
else:
    print(f"  ⚠️  Only {success_count2}/20 cycles completed")

# Summary
print("\n" + "="*50)
print("TENSOR SIZE FIX VERIFICATION")
print("="*50)
if success_count == 20 and success_count2 == 20:
    print("✅ Tensor size mismatch is FIXED!")
    print("   - Action prediction system handles varying sizes")
    print("   - History stacking works correctly")
    print("   - No dimension mismatches detected")
else:
    print("❌ Some tests failed - needs more investigation")