#!/usr/bin/env python3
"""Simple demo to test prediction improvement addiction system."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import numpy as np
from brain_factory import BrainFactory

print("=== PREDICTION ADDICTION DEMO ===\n")

# Create brain
config = {'type': 'unified_field', 'spatial_resolution': 5, 'quiet_mode': True}
brain = BrainFactory(config)

print("1. Teaching a pattern sequence...")
# Pattern: obstacle appears → turn left → clear → straight
patterns = [
    ([0.3, 0.3, 0.8] + [0.5] * 21, "Obstacle detected"),
    ([0.8, 0.5, 0.3] + [0.5] * 21, "Turning left"),
    ([0.8, 0.8, 0.8] + [0.5] * 21, "Path clear"),
    ([0.5, 0.5, 0.5] + [0.5] * 21, "Moving straight"),
]

# Teach pattern multiple times
for epoch in range(5):
    for sensors, desc in patterns:
        action, state = brain.process_sensory_input(sensors)
        if epoch == 0 or epoch == 4:
            conf = state['prediction_confidence']
            mod = state['learning_addiction_modifier']
            print(f"  Epoch {epoch+1}, {desc}: conf={conf:.2f}, modifier={mod:.2f}")

print("\n2. Testing learned behavior...")
# Test if brain predicts and acts accordingly
for sensors, desc in patterns:
    action, state = brain.process_sensory_input(sensors)
    turn = action[0] - action[1]
    speed = action[3]
    conf = state['prediction_confidence']
    
    direction = "LEFT" if turn < -0.1 else "RIGHT" if turn > 0.1 else "STRAIGHT"
    print(f"  {desc}: {direction} (conf={conf:.2f}, speed={speed:.2f})")

print("\n3. Introducing surprise...")
surprise = [0.1, 0.9, 0.1] + [0.5] * 21
action, state = brain.process_sensory_input(surprise)
print(f"  SURPRISE input: conf={state['prediction_confidence']:.2f}, "
      f"modifier={state['learning_addiction_modifier']:.2f}")

print("\n4. Summary:")
# BrainFactory wraps the actual brain
if hasattr(brain, 'brain') and hasattr(brain.brain, '_get_field_brain_state'):
    final_state = brain.brain._get_field_brain_state(0.0)
    print(f"  Total cycles: {final_state['brain_cycles']}")
    print(f"  Prediction efficiency: {final_state['prediction_efficiency']:.3f}")
    print(f"  Intrinsic reward: {final_state['intrinsic_reward']:.3f}")
    
    if final_state['prediction_efficiency'] > 0.5:
        print("\n✓ Brain successfully learned to predict patterns!")
    else:
        print("\n~ Brain is still learning...")
else:
    print("  Cannot access internal brain state directly")

brain.shutdown()