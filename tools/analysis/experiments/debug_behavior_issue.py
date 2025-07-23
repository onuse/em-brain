#!/usr/bin/env python3
"""Debug why behaviors aren't showing in tests."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))

from brains.field.core_brain import UnifiedFieldBrain

print("=== DEBUG BEHAVIOR ISSUE ===\n")

# Create brain at resolution 5 (which worked earlier)
brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)

print("Initial state:")
print(f"  Gradient following strength: {brain.gradient_following_strength}")
print(f"  Field shape: {brain.unified_field.shape}")
print(f"  Topology regions: {len(brain.topology_regions)}")

# Test direct obstacle response
print("\nTesting obstacle response:")
for i in range(10):
    sensors = [0.9, 0.9, 0.9] + [0.0] * 21
    action, _ = brain.process_robot_cycle(sensors)
    if i % 3 == 0:
        print(f"  Cycle {i}: action=[{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]")

# Check if gradients are being calculated
print("\nChecking gradient flows:")
brain._calculate_gradient_flows()
for name, grad in brain.gradient_flows.items():
    if grad is not None:
        max_grad = grad.abs().max().item()
        print(f"  {name}: max magnitude = {max_grad:.6f}")

# Check field state
print("\nField state:")
print(f"  Field min: {brain.unified_field.min().item():.6f}")
print(f"  Field max: {brain.unified_field.max().item():.6f}")
print(f"  Field mean: {brain.unified_field.mean().item():.6f}")

# Test with varied inputs
print("\nTesting varied inputs:")
test_inputs = [
    ([0.1, 0.1, 0.1] + [0.0] * 21, "Far"),
    ([0.5, 0.5, 0.5] + [0.0] * 21, "Medium"),
    ([0.9, 0.9, 0.9] + [0.0] * 21, "Close"),
    ([0.1, 0.5, 0.9] + [0.0] * 21, "Right obstacle"),
    ([0.9, 0.5, 0.1] + [0.0] * 21, "Left obstacle"),
]

for sensors, desc in test_inputs:
    action, _ = brain.process_robot_cycle(sensors)
    print(f"  {desc}: speed={action[3]:.3f}, turn={(action[0]-action[1]):.3f}")

brain.shutdown()