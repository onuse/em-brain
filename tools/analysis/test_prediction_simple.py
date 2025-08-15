#!/usr/bin/env python3
"""Simple test of minimal prediction system."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import numpy as np
from brains.field.core_brain import UnifiedFieldBrain

print("=== SIMPLE PREDICTION TEST ===\n")

# Create brain with minimal output
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)

print("1. STABLE INPUT TEST:")
stable_sensors = [0.5] * 24
confidences = []

for i in range(20):
    action, state = brain.process_robot_cycle(stable_sensors)
    conf = state['prediction_confidence']
    confidences.append(conf)
    if i % 5 == 0:
        print(f"   Cycle {i}: confidence={conf:.3f}")

avg_stable = np.mean(confidences[-10:])
print(f"   Average confidence (stable): {avg_stable:.3f}")

print("\n2. CHANGING INPUT TEST:")
for i in range(20):
    # Oscillating input
    changing_sensors = [0.5 + 0.3*np.sin(i*0.5)] * 24
    action, state = brain.process_robot_cycle(changing_sensors)
    conf = state['prediction_confidence']
    confidences.append(conf)
    if i % 5 == 0:
        print(f"   Cycle {i+20}: confidence={conf:.3f}")

avg_changing = np.mean(confidences[-10:])
print(f"   Average confidence (changing): {avg_changing:.3f}")

print("\n3. SURPRISE TEST:")
# Sudden change
surprise_sensors = [0.9] * 24
action, state = brain.process_robot_cycle(surprise_sensors)
surprise_conf = state['prediction_confidence']
print(f"   Surprise confidence: {surprise_conf:.3f}")

# Continue with new pattern
new_pattern_confs = []
for i in range(10):
    action, state = brain.process_robot_cycle(surprise_sensors)
    new_pattern_confs.append(state['prediction_confidence'])

print(f"   Recovery confidence: {new_pattern_confs[-1]:.3f}")

print("\n4. RESULTS:")
print(f"   ✓ Stable input confidence: {avg_stable:.3f} (should be high)")
print(f"   ✓ Changing input confidence: {avg_changing:.3f} (should be lower)")
print(f"   ✓ Surprise drop: {surprise_conf:.3f} (should be low)")
print(f"   ✓ Recovery: {new_pattern_confs[-1]:.3f} (should recover)")

# Check learning modulation
final_state = brain._get_field_brain_state(0.0)
print(f"\n5. LEARNING MODULATION:")
print(f"   Learning modifier: {final_state['learning_addiction_modifier']:.2f}")
print(f"   Intrinsic reward: {final_state['intrinsic_reward']:.3f}")

# Success criteria
success = (avg_stable > 0.7 and 
          avg_changing < avg_stable and 
          surprise_conf < 0.5 and
          new_pattern_confs[-1] > surprise_conf)

print(f"\n{'SUCCESS' if success else 'NEEDS TUNING'}: Prediction-based confidence is {'working!' if success else 'emerging.'}")

brain.shutdown()