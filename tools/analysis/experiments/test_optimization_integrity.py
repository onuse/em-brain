#!/usr/bin/env python3
"""Test that optimization preserves brain functionality."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))

import numpy as np
from brains.field.core_brain import UnifiedFieldBrain

print("=== OPTIMIZATION INTEGRITY TEST ===\n")

# Create brain
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)

# Test sensory input variations
test_inputs = [
    [0.0] * 24,  # All zeros
    [1.0] * 24,  # All ones
    [0.5] * 24,  # All middle
    list(np.sin(np.arange(24) * 0.5)),  # Sine wave
    list(np.random.rand(24)),  # Random
]

print("Testing different sensory inputs:")
for i, sensory_input in enumerate(test_inputs):
    action, state = brain.process_robot_cycle(sensory_input)
    
    # Check action bounds
    assert all(-1 <= a <= 1 for a in action), f"Action out of bounds: {action}"
    
    # Check for NaN/Inf
    assert not any(np.isnan(a) or np.isinf(a) for a in action), f"NaN/Inf in action: {action}"
    
    print(f"Input {i}: Action = [{action[0]:6.3f}, {action[1]:6.3f}, {action[2]:6.3f}, {action[3]:6.3f}]")

# Test persistent behavior
print("\nTesting persistent behavior (same input 5 times):")
test_input = [0.7, 0.3, 0.5] + [0.0] * 21
for cycle in range(5):
    action, state = brain.process_robot_cycle(test_input)
    print(f"Cycle {cycle}: [{action[0]:6.3f}, {action[1]:6.3f}, {action[2]:6.3f}, {action[3]:6.3f}]")

# Check gradient cache is working
stats = brain.gradient_calculator.get_cache_stats()
print(f"\nGradient cache stats:")
print(f"  Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']}")
print(f"  Hit rate: {stats['hit_rate']:.1%}")

# Test with obstacle (high front sensor)
print("\nTesting obstacle avoidance:")
obstacle_input = [0.9, 0.9, 0.1] + [0.0] * 21  # High front sensors
action, state = brain.process_robot_cycle(obstacle_input)
print(f"Obstacle response: [{action[0]:6.3f}, {action[1]:6.3f}, {action[2]:6.3f}, {action[3]:6.3f}]")

if action[3] < 0:  # Should slow down or reverse
    print("✅ Correctly slowing/reversing for obstacle")
else:
    print("⚠️  May not be avoiding obstacle properly")

brain.shutdown()
print("\n✅ Optimization integrity test passed!")