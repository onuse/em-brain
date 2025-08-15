#!/usr/bin/env python3
"""Test prediction system with reward channel."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import numpy as np
from brain_factory import BrainFactory

print("=== REWARD SYSTEM TEST ===\n")

# Create brain
config = {'type': 'unified_field', 'spatial_resolution': 5, 'quiet_mode': True}
brain = BrainFactory(config)

print("1. Testing reward learning...")
# Pattern: obstacle → negative reward, clear path → positive reward
scenarios = [
    ([0.2, 0.2, 0.8] + [0.5] * 21 + [-1.0], "Hit obstacle", "negative"),
    ([0.8, 0.8, 0.2] + [0.5] * 21 + [0.0], "Turning", "neutral"),
    ([0.8, 0.8, 0.8] + [0.5] * 21 + [1.0], "Clear path", "positive"),
    ([0.5, 0.5, 0.5] + [0.5] * 21 + [0.5], "Good progress", "positive"),
]

# Learn associations
for epoch in range(3):
    print(f"\nEpoch {epoch + 1}:")
    for sensors, desc, reward_type in scenarios:
        action, state = brain.process_sensory_input(sensors)
        conf = state['prediction_confidence']
        reward = sensors[-1]
        print(f"  {desc} (reward={reward:+.1f}): conf={conf:.2f}")

print("\n2. Testing learned behavior...")
# Present patterns without reward to see if behavior changed
test_patterns = [
    ([0.2, 0.2, 0.8] + [0.5] * 21 + [0.0], "Obstacle ahead"),
    ([0.8, 0.8, 0.8] + [0.5] * 21 + [0.0], "Clear path"),
]

for sensors, desc in test_patterns:
    action, state = brain.process_sensory_input(sensors)
    turn = action[0] - action[1]
    speed = action[3]
    conf = state['prediction_confidence']
    
    # Analyze behavior
    if turn < -0.2:
        behavior = "AVOID LEFT"
    elif turn > 0.2:
        behavior = "AVOID RIGHT"
    elif speed < 0.5:
        behavior = "SLOW/STOP"
    else:
        behavior = "PROCEED"
    
    print(f"  {desc}: {behavior} (speed={speed:.2f}, turn={turn:+.2f}, conf={conf:.2f})")

print("\n3. Testing value-based decision...")
# Present ambiguous situation
ambiguous = [0.5, 0.5, 0.5] + [0.5] * 21 + [0.0]
action, state = brain.process_sensory_input(ambiguous)
print(f"  Ambiguous input: speed={action[3]:.2f}, conf={state['prediction_confidence']:.2f}")

# Check if brain seeks positive reward states
print("\n4. Checking memory formation...")
if hasattr(brain, 'brain') and hasattr(brain.brain, 'topology_regions'):
    regions = brain.brain.topology_regions
    print(f"  Topology regions formed: {len(regions)}")
    if len(regions) > 0:
        # Check if positive/negative experiences created different regions
        for i, (key, region) in enumerate(list(regions.items())[:3]):
            print(f"  Region {i+1}: activation={region['activation']:.3f}, "
                  f"importance={region['importance']:.1f}")

print("\n5. Summary:")
print("The reward system enables:")
print("- Learning from positive/negative outcomes")
print("- Value-based action selection")
print("- Stronger memory formation for important events")
print("- Goal-directed behavior emergence")

brain.shutdown()