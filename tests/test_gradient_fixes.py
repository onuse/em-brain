#!/usr/bin/env python3
"""Test gradient generation fixes."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

from brains.field.core_brain import UnifiedFieldBrain
import numpy as np

# Create brain
brain = UnifiedFieldBrain(spatial_resolution=8, quiet_mode=False)

print("\n=== Testing Gradient Generation with Fixes ===")

# Test patterns
patterns = [
    ([0.9] * 24, "Strong uniform"),
    ([0.1] * 24, "Weak uniform"),
    ([0.5] * 24, "Neutral"),
    ([0.9, 0.1] * 12, "Alternating"),
    ([np.random.random() for _ in range(24)], "Random")
]

for pattern, name in patterns:
    print(f"\n--- {name} pattern ---")
    action, state = brain.process_robot_cycle(pattern)
    
    print(f"Motor output: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, {action[3]:.4f}]")
    print(f"Motor magnitude: {np.linalg.norm(action):.4f}")
    
    # Check for NaN
    if any(np.isnan(action)):
        print("⚠️  WARNING: NaN detected in motor output!")
    
# Test edge cases for NaN protection
print("\n--- Edge case: Single value ---")
action, _ = brain.process_robot_cycle([0.5])  # Single sensor

print("\n--- Edge case: Empty input ---") 
try:
    action, _ = brain.process_robot_cycle([])
except Exception as e:
    print(f"Expected error: {e}")

print("\n✅ Test complete!")