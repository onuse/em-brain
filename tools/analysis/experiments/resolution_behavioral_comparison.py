#!/usr/bin/env python3
"""Compare behavioral effectiveness at different resolutions."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))

import time
import numpy as np
from brains.field.core_brain import UnifiedFieldBrain

print("=== RESOLUTION BEHAVIORAL COMPARISON ===\n")

def test_resolution(res, warmup_cycles=50):
    """Test a specific resolution with proper warmup."""
    brain = UnifiedFieldBrain(spatial_resolution=res, quiet_mode=True)
    
    # Warmup - let the brain stabilize
    for _ in range(warmup_cycles):
        brain.process_robot_cycle([0.5] * 24)
    
    results = {}
    
    # 1. Performance
    times = []
    for _ in range(20):
        start = time.perf_counter()
        brain.process_robot_cycle([0.5] * 24)
        times.append((time.perf_counter() - start) * 1000)
    results['avg_time_ms'] = np.mean(times)
    results['frequency_hz'] = 1000 / results['avg_time_ms']
    
    # 2. Obstacle avoidance (critical test)
    obstacle_scores = []
    test_distances = [0.95, 0.8, 0.6, 0.4, 0.2]
    
    for dist in test_distances:
        # Reset with safe position
        for _ in range(5):
            brain.process_robot_cycle([0.1] * 24)
        
        # Test obstacle response
        responses = []
        for _ in range(10):
            sensors = [dist, dist, dist] + [0.0] * 21
            action, _ = brain.process_robot_cycle(sensors)
            
            # Score based on appropriate response
            if dist > 0.7:  # Close obstacle - should slow/reverse
                score = max(0, 0.5 - action[3])  # Lower speed is better
            else:  # Far obstacle - mild avoidance ok
                score = max(0, 0.3 - action[3]) * 0.5
            
            responses.append(score)
        
        obstacle_scores.append(np.mean(responses))
    
    results['obstacle_avoidance'] = np.mean(obstacle_scores)
    
    # 3. Turning behavior
    turn_scores = []
    
    # Test left turn (obstacle on right)
    for _ in range(5):
        brain.process_robot_cycle([0.1, 0.1, 0.1] + [0.0] * 21)
    
    for _ in range(10):
        sensors = [0.2, 0.5, 0.9] + [0.0] * 21  # Obstacle on right
        action, _ = brain.process_robot_cycle(sensors)
        turn_left = action[0] - action[1]  # Positive = turn left
        turn_scores.append(abs(turn_left))
    
    # Test right turn (obstacle on left)
    for _ in range(5):
        brain.process_robot_cycle([0.1, 0.1, 0.1] + [0.0] * 21)
    
    for _ in range(10):
        sensors = [0.9, 0.5, 0.2] + [0.0] * 21  # Obstacle on left
        action, _ = brain.process_robot_cycle(sensors)
        turn_right = action[1] - action[0]  # Positive = turn right
        turn_scores.append(abs(turn_right))
    
    results['turning_ability'] = np.mean(turn_scores)
    
    # 4. Behavioral range
    all_actions = []
    for _ in range(20):
        sensors = list(np.random.rand(24))
        action, _ = brain.process_robot_cycle(sensors)
        all_actions.extend(action)
    
    results['action_range'] = np.std(all_actions)
    
    # 5. Memory capability
    initial_regions = len(brain.topology_regions)
    pattern = list(np.sin(np.arange(24) * 0.3))
    for _ in range(100):
        brain.process_robot_cycle(pattern)
    results['memory_growth'] = len(brain.topology_regions) - initial_regions
    
    brain.shutdown()
    return results

# Test each resolution
print("Testing each resolution (this will take a minute)...\n")
test_results = {}

for res in [3, 4, 5]:
    print(f"Testing resolution {res}³...")
    test_results[res] = test_resolution(res, warmup_cycles=50)

# Display results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Metric':<20} {'3³':>12} {'4³':>12} {'5³':>12}")
print("-"*60)

metrics = ['frequency_hz', 'obstacle_avoidance', 'turning_ability', 'action_range', 'memory_growth']
metric_names = ['Frequency (Hz)', 'Obstacle Score', 'Turning Score', 'Action Range', 'Memory Growth']

for metric, name in zip(metrics, metric_names):
    print(f"{name:<20}", end="")
    for res in [3, 4, 5]:
        value = test_results[res][metric]
        if metric == 'frequency_hz':
            print(f"{value:>11.1f}", end=" ")
        elif metric == 'memory_growth':
            print(f"{value:>11d}", end=" ")
        else:
            print(f"{value:>11.3f}", end=" ")
    print()

# Scoring
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

scores = {}
for res in [3, 4, 5]:
    score = 0
    data = test_results[res]
    
    # Real-time performance (0-3 points)
    if data['frequency_hz'] >= 25:
        score += 3
    elif data['frequency_hz'] >= 15:
        score += 2
    elif data['frequency_hz'] >= 10:
        score += 1
    
    # Obstacle avoidance (0-3 points) 
    score += min(3, data['obstacle_avoidance'] * 10)
    
    # Turning ability (0-2 points)
    score += min(2, data['turning_ability'] * 10)
    
    # Behavioral diversity (0-1 point)
    if data['action_range'] > 0.1:
        score += 1
    
    # Memory capability (0-1 point)
    if data['memory_growth'] > 0:
        score += 1
    
    scores[res] = score
    print(f"Resolution {res}³: {score:.1f}/10 points")

best_res = max(scores, key=scores.get)
print(f"\n{'='*60}")
print(f"RECOMMENDATION: Standardize on Resolution {best_res}³")
print(f"{'='*60}")

if best_res == 3:
    print("✓ Fastest performance (real-time guaranteed)")
    print("✓ Lowest computational requirements")
    print("✗ May need behavioral tuning for complex tasks")
elif best_res == 4:
    print("✓ Good balance of speed and behavior")
    print("✓ Still real-time capable") 
    print("✓ Better behavioral responses than 3³")
elif best_res == 5:
    print("✓ Best behavioral responses")
    print("✓ Most reliable obstacle avoidance")
    print("✗ Borderline real-time (may need optimization)")
    
print("\nNote: You can always adjust resolution based on:")
print("- Hardware capabilities (faster CPU = higher resolution)")
print("- Task requirements (complex behavior = higher resolution)")
print("- Real-time constraints (strict timing = lower resolution)")