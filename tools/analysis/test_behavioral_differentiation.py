#!/usr/bin/env python3
"""Test behavioral differentiation after gradient fix."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

from brain_factory import BrainFactory

print("=== TESTING BEHAVIORAL DIFFERENTIATION ===\n")

# Create brain
config = {'type': 'unified_field', 'spatial_resolution': 5, 'quiet_mode': True}
brain = BrainFactory(config)

# Test patterns with expected behaviors
test_cases = [
    # (sensors, description, expected_behavior)
    ([0.1, 0.1, 0.9] + [0.5] * 21 + [0.0], "Front obstacle", "Turn left or right"),
    ([0.9, 0.1, 0.1] + [0.5] * 21 + [0.0], "Left obstacle", "Turn right"),
    ([0.1, 0.9, 0.1] + [0.5] * 21 + [0.0], "Right obstacle", "Turn left"),
    ([0.9, 0.9, 0.9] + [0.5] * 21 + [0.0], "Clear path", "Go straight"),
    ([0.1, 0.1, 0.1] + [0.5] * 21 + [-1.0], "All obstacles + punishment", "Back up"),
    ([0.9, 0.9, 0.9] + [0.5] * 21 + [1.0], "Clear + reward", "Go forward fast"),
]

print("Processing different sensory patterns:\n")
for sensors, desc, expected in test_cases:
    # Process twice to build up field
    action1, _ = brain.process_sensory_input(sensors)
    action2, _ = brain.process_sensory_input(sensors)
    
    # Analyze the second action (more established)
    left_motor = action2[0]
    right_motor = action2[1]
    turn = left_motor - right_motor
    speed = action2[3]
    
    # Determine actual behavior
    if abs(turn) < 0.05:
        if speed > 0.1:
            behavior = "FORWARD"
        elif speed < -0.05:
            behavior = "BACKWARD"
        else:
            behavior = "STOPPED"
    elif turn > 0.05:
        behavior = "TURN LEFT"
    else:
        behavior = "TURN RIGHT"
    
    print(f"{desc:25} → {behavior:12} (turn={turn:+.3f}, speed={speed:.3f})")
    print(f"   Expected: {expected}")
    print(f"   Motors: L={left_motor:.3f}, R={right_motor:.3f}")
    print()

# Test gradient strength at different positions
print("\nGradient strength at different field positions:")
field_brain = brain.brain

# Process strong obstacle to create field activation
strong_obstacle = [0.1, 0.1, 0.9] + [0.8] * 21 + [-0.5]
for i in range(3):
    brain.process_sensory_input(strong_obstacle)

# Check if we have non-zero gradients
if hasattr(field_brain, 'gradient_flows') and field_brain.gradient_flows:
    for name, grad in field_brain.gradient_flows.items():
        non_zero = (grad.abs() > 1e-6).sum().item()
        max_val = grad.abs().max().item()
        print(f"   {name}: non_zero_elements={non_zero}, max={max_val:.6f}")
else:
    print("   No gradients found!")

brain.shutdown()
print("\nTest complete.")