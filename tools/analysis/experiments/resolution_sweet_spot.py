#!/usr/bin/env python3
"""Find the optimal resolution for robot operation."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))

import time
import numpy as np
from brains.field.core_brain import UnifiedFieldBrain

print("=== RESOLUTION OPTIMIZATION ANALYSIS ===\n")

resolutions = [3, 4, 5, 6]
results = {}

for res in resolutions:
    print(f"\nTesting Resolution {res}³:")
    brain = UnifiedFieldBrain(spatial_resolution=res, quiet_mode=True)
    
    # Performance test
    times = []
    for _ in range(20):
        start = time.perf_counter()
        brain.process_robot_cycle([0.5] * 24)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times[5:])  # Skip warmup
    frequency = 1000 / avg_time
    
    # Behavioral test - obstacle avoidance
    obstacle_responses = []
    for dist in [0.9, 0.7, 0.5, 0.3, 0.1]:
        sensors = [dist] * 3 + [0.0] * 21
        action, _ = brain.process_robot_cycle(sensors)
        # Check if it slows down or reverses for close obstacles
        if dist > 0.7:
            response = -action[3] if action[3] < 0 else 0
        else:
            response = 1.0 - action[3] if action[3] < 0.5 else 0
        obstacle_responses.append(response)
    
    avg_obstacle_response = np.mean(obstacle_responses)
    
    # Turning test
    turn_responses = []
    # Obstacle on right
    sensors = [0.1, 0.5, 0.9] + [0.0] * 21
    for _ in range(5):
        action, _ = brain.process_robot_cycle(sensors)
        turn_responses.append(abs(action[0] - action[1]))
    
    avg_turn = np.mean(turn_responses)
    
    # Memory test
    initial_regions = len(brain.topology_regions)
    pattern = list(np.sin(np.arange(24) * 0.5))
    for _ in range(30):
        brain.process_robot_cycle(pattern)
    memory_formed = len(brain.topology_regions) > initial_regions
    
    results[res] = {
        'frequency': frequency,
        'obstacle_response': avg_obstacle_response,
        'turn_response': avg_turn,
        'memory_formation': memory_formed,
        'field_size': brain.unified_field.numel()
    }
    
    print(f"  Frequency: {frequency:.1f} Hz")
    print(f"  Obstacle response: {avg_obstacle_response:.3f}")
    print(f"  Turn response: {avg_turn:.3f}")
    print(f"  Memory formation: {'Yes' if memory_formed else 'No'}")
    print(f"  Field size: {brain.unified_field.numel():,} elements")
    
    brain.shutdown()

# Analysis
print("\n" + "="*50)
print("RECOMMENDATION:")

# Score each resolution
scores = {}
for res, data in results.items():
    score = 0
    # Real-time capability (>10Hz is good, >20Hz is better)
    if data['frequency'] >= 20:
        score += 2
    elif data['frequency'] >= 10:
        score += 1
    
    # Obstacle avoidance (critical)
    if data['obstacle_response'] > 0.1:
        score += 3
    
    # Turning behavior
    if data['turn_response'] > 0.2:
        score += 2
    elif data['turn_response'] > 0.1:
        score += 1
    
    # Memory formation
    if data['memory_formation']:
        score += 1
    
    scores[res] = score

best_res = max(scores, key=scores.get)
print(f"\nOptimal resolution: {best_res}³")
print(f"Reasoning:")
data = results[best_res]
print(f"- Runs at {data['frequency']:.1f} Hz (good for real-time)")
print(f"- Obstacle avoidance score: {data['obstacle_response']:.3f}")
print(f"- Turning response: {data['turn_response']:.3f}")
print(f"- Memory formation: {'Working' if data['memory_formation'] else 'Not working'}")
print(f"- Field size: {data['field_size']:,} elements (manageable)")

print("\nFull scores:")
for res, score in sorted(scores.items()):
    print(f"  Resolution {res}³: {score}/8 points")