#!/usr/bin/env python3
"""Diagnose why behaviors are weak after optimization."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))

import numpy as np
import torch
from brains.field.core_brain import UnifiedFieldBrain

print("=== WEAK BEHAVIOR DIAGNOSIS ===\n")

# Create brain
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)

# Check key parameters
print("1. Key Parameters:")
print(f"   Gradient following strength: {brain.gradient_following_strength}")
print(f"   Field evolution rate: {brain.field_evolution_rate}")
print(f"   Field decay rate: {brain.field_decay_rate}")
print(f"   Field diffusion rate: {brain.field_diffusion_rate}")
print(f"   Local gradient region: 3x3x3")

# Test gradient generation
print("\n2. Testing Gradient Generation:")

# Simulate strong obstacle
obstacle_sensors = [0.9, 0.9, 0.9] + [0.0] * 21
field_exp = brain._robot_sensors_to_field_experience(obstacle_sensors)
print(f"   Obstacle field experience intensity: {field_exp.field_intensity:.3f}")

# Apply experience and calculate gradients
brain._apply_field_experience(field_exp)
brain._calculate_gradient_flows()

# Check gradient magnitudes
print("\n3. Gradient Magnitudes:")
for name, grad in brain.gradient_flows.items():
    if grad is not None:
        # Get gradient in local region
        center = grad.shape[0] // 2
        local_grad = grad[center-1:center+2, center-1:center+2, center-1:center+2]
        magnitude = torch.abs(local_grad).max().item()
        print(f"   {name}: max={magnitude:.6f}")

# Test action generation
print("\n4. Action Generation Test:")
actions = []
for i in range(5):
    # Different obstacle distances
    distance = 0.1 + i * 0.2
    sensors = [1.0 - distance] * 3 + [0.0] * 21
    action, _ = brain.process_robot_cycle(sensors)
    actions.append(action)
    print(f"   Distance {distance:.1f}: action=[{action[0]:6.3f}, {action[1]:6.3f}, {action[2]:6.3f}, {action[3]:6.3f}]")

# Check action magnitudes
action_magnitudes = [np.abs(a).max() for a in actions]
avg_magnitude = np.mean(action_magnitudes)
print(f"\n   Average action magnitude: {avg_magnitude:.3f}")

# Test gradient following directly
print("\n5. Direct Gradient Following Test:")
brain._calculate_gradient_flows()
action = brain._field_gradients_to_robot_action()
raw_gradients = []

if 'gradient_x' in brain.gradient_flows:
    grad_x = brain.gradient_flows['gradient_x']
    center = tuple(s//2 for s in grad_x.shape[:3])
    center_grad = grad_x[center].item() if grad_x.ndim == 3 else grad_x[center + tuple(0 for _ in range(grad_x.ndim-3))].item()
    raw_gradients.append(center_grad)
    print(f"   Raw gradient_x at center: {center_grad:.6f}")

# Check motor mapping
print("\n6. Motor Mapping Analysis:")
print(f"   Gradient → Motor scaling: {brain.gradient_following_strength}")
print(f"   Expected motor range: [-1, 1]")
action_values = [action.left_motor, action.right_motor, action.audio_level, action.forward_speed]
print(f"   Actual motor range: [{min(action_values):.3f}, {max(action_values):.3f}]")

# Diagnosis
print("\n7. DIAGNOSIS:")
if avg_magnitude < 0.1:
    print("   ❌ Actions are too weak - likely due to:")
    print("      - Gradient magnitudes too small")
    print("      - Gradient following strength too low")
    print("      - Field decay too aggressive")
    print("\n   RECOMMENDATIONS:")
    print("      1. Increase gradient_following_strength (currently 5.0)")
    print("      2. Reduce field_decay_rate (currently 0.999)")
    print("      3. Increase sensory-to-field mapping strength")
else:
    print("   ✅ Action magnitudes seem reasonable")

brain.shutdown()