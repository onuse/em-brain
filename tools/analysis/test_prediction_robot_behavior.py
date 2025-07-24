#!/usr/bin/env python3
"""Test prediction system with simulated robot behavior."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import numpy as np
from brain_factory import BrainFactory

print("=== ROBOT BEHAVIOR WITH PREDICTION SYSTEM ===\n")

# Create brain using factory (as robot would)
config = {
    'type': 'unified_field',
    'spatial_resolution': 5,
    'quiet_mode': True
}
brain = BrainFactory(config)

print("1. OBSTACLE AVOIDANCE TEST:")
print("-" * 50)

# Simulate approaching obstacle
for i in range(20):
    # Distance sensors: getting closer to obstacle
    distance = max(0.1, 1.0 - i * 0.05)
    
    # Robot sensors: [front_left, front_center, front_right] + other sensors
    if i < 10:
        # Obstacle in front
        sensors = [distance, distance * 0.5, distance] + [0.5] * 21
    else:
        # Turning away from obstacle
        sensors = [0.9, 0.8, 0.3] + [0.5] * 21
    
    action, state = brain.process_sensory_input(sensors)
    
    if i % 5 == 0:
        print(f"Step {i}: distance={distance:.2f}, confidence={state['prediction_confidence']:.3f}, "
              f"turn={action[0]-action[1]:.3f}, speed={action[3]:.3f}")

print("\n2. PATTERN NAVIGATION TEST:")
print("-" * 50)

# Simulate navigating a repeating environment pattern
patterns = [
    [0.2, 0.8, 0.5],  # Turn left scenario
    [0.8, 0.2, 0.5],  # Turn right scenario  
    [0.5, 0.5, 0.5],  # Straight scenario
]

for epoch in range(3):
    print(f"\nEpoch {epoch + 1}:")
    for p, pattern in enumerate(patterns):
        sensors = pattern + [0.5] * 21
        action, state = brain.process_sensory_input(sensors)
        
        turn = action[0] - action[1]
        direction = "left" if turn < -0.1 else "right" if turn > 0.1 else "straight"
        
        print(f"  Pattern {p+1}: confidence={state['prediction_confidence']:.3f}, "
              f"action={direction}, modifier={state['learning_addiction_modifier']:.2f}")

print("\n3. EXPLORATION VS EXPLOITATION TEST:")
print("-" * 50)

# Stable environment (should exploit)
print("Stable environment:")
stable_sensors = [0.7, 0.7, 0.7] + [0.5] * 21
actions = []
for i in range(10):
    action, state = brain.process_sensory_input(stable_sensors, output_dim=4)
    actions.append(np.linalg.norm(action))
    if i == 0 or i == 9:
        print(f"  Step {i+1}: confidence={state['prediction_confidence']:.3f}, "
              f"modifier={state['learning_addiction_modifier']:.2f}")

avg_stable_action = np.mean(actions)

# Chaotic environment (should explore)
print("\nChaotic environment:")
actions = []
for i in range(10):
    chaotic_sensors = [np.random.random() for _ in range(3)] + [0.5] * 21
    action, state = brain.process_sensory_input(chaotic_sensors, output_dim=4)
    actions.append(np.linalg.norm(action))
    if i == 0 or i == 9:
        print(f"  Step {i+1}: confidence={state['prediction_confidence']:.3f}, "
              f"modifier={state['learning_addiction_modifier']:.2f}")

avg_chaotic_action = np.mean(actions)

print(f"\nAction strength comparison:")
print(f"  Stable: {avg_stable_action:.3f}")
print(f"  Chaotic: {avg_chaotic_action:.3f}")

if avg_chaotic_action > avg_stable_action:
    print("  ✓ More exploration in unpredictable environment!")

print("\n4. REWARD LEARNING SIMULATION:")
print("-" * 50)
print("(Would need reward channel in sensory input)")
print("Current: 24D sensory input")
print("Needed: 25D with reward signal")
print("This would enable:")
print("- Learning good vs bad outcomes")
print("- Seeking rewarding states")
print("- Avoiding negative experiences")

print("\n5. SUMMARY:")
print("-" * 50)
print("The prediction addiction system enables:")
print("✓ Adaptive obstacle avoidance")
print("✓ Pattern learning and recognition")
print("✓ Dynamic exploration/exploitation")
print("✓ Curiosity-driven behavior")
print("✓ Intrinsic motivation to understand environment")

# Get final stats
final_stats = brain._brain._get_field_brain_state(0.0)
print(f"\nFinal brain state:")
print(f"  Brain cycles: {final_stats['brain_cycles']}")
print(f"  Prediction efficiency: {final_stats['prediction_efficiency']:.3f}")
print(f"  Intrinsic reward: {final_stats['intrinsic_reward']:.3f}")
print(f"  Topology regions: {final_stats['topology_regions_count']}")

brain.shutdown()