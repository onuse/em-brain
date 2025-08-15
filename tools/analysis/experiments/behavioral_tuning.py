#!/usr/bin/env python3
"""Test different parameter settings to improve behaviors."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))

import numpy as np
from brains.field.core_brain import UnifiedFieldBrain

print("=== BEHAVIORAL TUNING EXPERIMENTS ===\n")

# Test different gradient following strengths
print("1. Testing Gradient Following Strength:")
strengths = [5.0, 10.0, 20.0, 50.0]

for strength in strengths:
    brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)
    brain.gradient_following_strength = strength
    
    # Test obstacle response
    obstacle_sensors = [0.9, 0.9, 0.9] + [0.0] * 21
    action, _ = brain.process_robot_cycle(obstacle_sensors)
    
    print(f"   Strength {strength:4.0f}: action=[{action[0]:6.3f}, {action[1]:6.3f}, {action[2]:6.3f}, {action[3]:6.3f}]")
    brain.shutdown()

# Test different field intensity mappings
print("\n2. Testing Field Intensity Scaling:")
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)

# Check current mapping
obstacle_sensors = [0.9, 0.9, 0.9] + [0.0] * 21
exp = brain._robot_sensors_to_field_experience(obstacle_sensors)
print(f"   Current field intensity: {exp.field_intensity:.3f}")
print(f"   Field coordinates norm: {np.linalg.norm(exp.field_coordinates.numpy()):.3f}")

# Recommendation
print("\n3. RECOMMENDATIONS:")
print("   To improve behavioral responses:")
print("   1. Increase gradient_following_strength to 20-50")
print("   2. Scale up sensory input mapping")
print("   3. Consider reducing field decay rate")
print("   4. Add sensory-specific amplification for obstacles")

brain.shutdown()