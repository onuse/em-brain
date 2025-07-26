#!/usr/bin/env python3
"""Quick behavioral test to check brain performance."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

from brain_factory import BrainFactory
import time

print("=== QUICK BEHAVIORAL TEST ===\n")

# Create brain
config = {'type': 'unified_field', 'spatial_resolution': 5, 'quiet_mode': True}
brain = BrainFactory(config)

# Test 1: Obstacle avoidance
print("1. OBSTACLE AVOIDANCE TEST:")
obstacle_patterns = [
    ([0.1, 0.5, 0.9] + [0.5] * 21 + [0.0], "Front obstacle"),
    ([0.9, 0.5, 0.1] + [0.5] * 21 + [0.0], "Left clear"),
    ([0.1, 0.9, 0.5] + [0.5] * 21 + [0.0], "Right obstacle"),
]

actions = []
for pattern, desc in obstacle_patterns:
    # Process multiple times to build field
    for _ in range(3):
        action, _ = brain.process_sensory_input(pattern)
    
    turn = action[0] - action[1]
    speed = action[3]
    actions.append((desc, turn, speed))
    print(f"  {desc}: turn={turn:+.3f}, speed={speed:.3f}")

# Check if different patterns produce different behaviors
turn_variance = max(abs(a[1]) for a in actions) - min(abs(a[1]) for a in actions)
print(f"\n  Turn variance: {turn_variance:.3f} (>0.1 is good differentiation)")

# Test 2: Reward learning
print("\n2. REWARD LEARNING TEST:")
# Give rewards for turning right
reward_patterns = [
    ([0.5, 0.5, 0.5] + [0.5] * 21 + [1.0], "Positive reward"),
    ([0.5, 0.5, 0.5] + [0.5] * 21 + [-1.0], "Negative reward"),
]

for pattern, desc in reward_patterns:
    # Process several times
    speeds = []
    for i in range(5):
        action, _ = brain.process_sensory_input(pattern)
        speeds.append(action[3])
    
    avg_speed = sum(speeds) / len(speeds)
    print(f"  {desc}: avg_speed={avg_speed:.3f}")

# Test 3: Memory formation
print("\n3. MEMORY FORMATION TEST:")
field_brain = brain.brain
initial_regions = len(field_brain.topology_regions)

# Create distinct patterns
memory_patterns = [
    [0.9, 0.1, 0.1] + [0.7] * 21 + [0.5],  # Left wall
    [0.1, 0.9, 0.1] + [0.3] * 21 + [0.5],  # Right wall
    [0.1, 0.1, 0.9] + [0.5] * 21 + [0.5],  # Front wall
]

for pattern in memory_patterns:
    for _ in range(2):
        brain.process_sensory_input(pattern)

final_regions = len(field_brain.topology_regions)
print(f"  Initial regions: {initial_regions}")
print(f"  Final regions: {final_regions}")
print(f"  New memories formed: {final_regions - initial_regions}")

# Test 4: Prediction confidence
print("\n4. PREDICTION CONFIDENCE TEST:")
# Check prediction confidence changes
confidences = []
for i in range(10):
    # Alternate patterns
    if i % 2 == 0:
        pattern = [0.9, 0.9, 0.9] + [0.5] * 21 + [0.0]  # Clear
    else:
        pattern = [0.1, 0.1, 0.1] + [0.5] * 21 + [0.0]  # Blocked
    
    brain.process_sensory_input(pattern)
    conf = field_brain._current_prediction_confidence
    confidences.append(conf)

print(f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
print(f"  Confidence variance: {max(confidences) - min(confidences):.3f}")

# Summary
print("\n=== SUMMARY ===")
print(f"✓ Behavioral differentiation: {'YES' if turn_variance > 0.1 else 'WEAK'}")
print(f"✓ Memory formation: {'YES' if final_regions > initial_regions else 'NO'}")
print(f"✓ Prediction learning: {'YES' if max(confidences) - min(confidences) > 0.05 else 'WEAK'}")

brain.shutdown()
print("\nTest complete.")