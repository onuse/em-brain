#!/usr/bin/env python3
"""Deep dive into why all behaviors are STRAIGHT/PROCEED."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import numpy as np
from brain_factory import BrainFactory

print("=== BEHAVIORAL DIFFERENTIATION ANALYSIS ===\n")

# Create brain
config = {'type': 'unified_field', 'spatial_resolution': 5, 'quiet_mode': True}
brain = BrainFactory(config)

print("1. GRADIENT ANALYSIS:")
print("-" * 50)

# Test different sensory patterns
patterns = [
    ([0.1, 0.1, 0.9] + [0.5] * 21 + [0.0], "Strong obstacle"),
    ([0.9, 0.1, 0.1] + [0.5] * 21 + [0.0], "Left obstacle"),
    ([0.1, 0.9, 0.1] + [0.5] * 21 + [0.0], "Right obstacle"),
    ([0.9, 0.9, 0.9] + [0.5] * 21 + [0.0], "Clear path"),
    ([0.5, 0.5, 0.5] + [0.5] * 21 + [0.0], "Neutral"),
]

# Get access to internal brain
field_brain = brain.brain

for sensors, desc in patterns:
    # Process input
    action, state = brain.process_sensory_input(sensors)
    
    # Analyze gradients
    if hasattr(field_brain, 'gradient_flows') and field_brain.gradient_flows:
        grad_strength = np.mean([torch.abs(g).mean().item() for g in field_brain.gradient_flows.values()])
    else:
        grad_strength = 0.0
    
    # Analyze action
    turn = action[0] - action[1]
    speed = action[3]
    
    print(f"{desc:15} -> turn={turn:+.3f}, speed={speed:.3f}, grad={grad_strength:.6f}")

print("\n2. FIELD ACTIVATION ANALYSIS:")
print("-" * 50)

# Check field response to obstacles
obstacle_sensors = [0.1, 0.1, 0.9] + [0.5] * 21 + [-1.0]  # With negative reward
clear_sensors = [0.9, 0.9, 0.9] + [0.5] * 21 + [1.0]     # With positive reward

print("\nProcessing obstacle with negative reward...")
action1, state1 = brain.process_sensory_input(obstacle_sensors)
field_max1 = torch.max(field_brain.unified_field).item()
field_mean1 = torch.mean(field_brain.unified_field).item()

print("\nProcessing clear path with positive reward...")
action2, state2 = brain.process_sensory_input(clear_sensors)
field_max2 = torch.max(field_brain.unified_field).item()
field_mean2 = torch.mean(field_brain.unified_field).item()

print(f"\nObstacle: field_max={field_max1:.3f}, field_mean={field_mean1:.6f}")
print(f"Clear:    field_max={field_max2:.3f}, field_mean={field_mean2:.6f}")

print("\n3. ACTION GENERATION MECHANISM:")
print("-" * 50)

# Look at the gradient to action mapping
print("Checking _field_gradients_to_robot_action...")

# Test with artificial gradients
if hasattr(field_brain, 'gradient_following_strength'):
    print(f"Gradient following strength: {field_brain.gradient_following_strength}")

# Get center position
center = field_brain.spatial_resolution // 2

print(f"\nCenter position: [{center}, {center}, {center}]")
print(f"Field shape: {field_brain.unified_field.shape}")

print("\n4. GRADIENT CALCULATION:")
print("-" * 50)

# Force some field activation
test_field = field_brain.unified_field.clone()
# Create artificial gradient
test_field[center-1, center, center] = 1.0  # Left high
test_field[center+1, center, center] = 0.1  # Right low
test_field[center, center-1, center] = 0.5  # Back medium
test_field[center, center+1, center] = 0.5  # Front medium

# Calculate what gradient should be
gradient_x = test_field[center+1, center, center] - test_field[center-1, center, center]
gradient_y = test_field[center, center+1, center] - test_field[center, center-1, center]

print(f"Expected gradients: x={gradient_x.item():.3f}, y={gradient_y.item():.3f}")
print("This should turn RIGHT (away from high left activation)")

print("\n5. HYPOTHESIS:")
print("-" * 50)
print("Possible causes of no differentiation:")
print("1. Gradients too weak after field decay/diffusion")
print("2. Local region (3x3x3) too small to capture differences")
print("3. Gradient following strength not properly tuned")
print("4. Field activations not spatially differentiated")
print("5. Motor mapping not sensitive enough")

print("\n6. CHECKING ACTUAL GRADIENTS:")
print("-" * 50)

# Process obstacle again and check gradients
obstacle_sensors = [0.1, 0.1, 0.9] + [0.5] * 21 + [0.0]
action, state = brain.process_sensory_input(obstacle_sensors)

# Try to access gradient calculator
if hasattr(field_brain, 'gradient_calculator'):
    print("Gradient calculator exists")
    
# Check if gradients were actually calculated
if hasattr(field_brain, '_last_computed_gradients'):
    print("Last computed gradients available")
else:
    print("No stored gradients found")

# Check field actions history
if hasattr(field_brain, 'field_actions') and len(field_brain.field_actions) > 0:
    last_action = field_brain.field_actions[-1]
    print(f"\nLast action:")
    print(f"  Gradient strength: {last_action.gradient_strength:.6f}")
    print(f"  Action confidence: {last_action.action_confidence:.3f}")
    print(f"  Motor commands: {last_action.motor_commands.tolist()}")

print("\n7. CONCLUSION:")
print("-" * 50)
print("The issue appears to be that gradients are too weak")
print("or not spatially differentiated enough to create")
print("distinct motor commands for different situations.")

brain.shutdown()