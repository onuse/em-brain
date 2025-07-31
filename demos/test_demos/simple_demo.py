#!/usr/bin/env python3
"""Simple demo of the brain's core capabilities."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import numpy as np
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Field-Native Intelligence Demo")
print("="*60)

# Create brain
brain = SimplifiedUnifiedBrain(
    sensory_dim=8,
    motor_dim=3,
    spatial_resolution=32,
    quiet_mode=True
)

print("Brain initialized with:")
print(f"  • 4D tensor field: {brain.unified_field.shape}")
print(f"  • Device: {brain.device}")
print(f"  • Self-modifying dynamics")
print()

# Demo 1: Basic responsiveness
print("1. BASIC RESPONSIVENESS")
print("-" * 40)
print("Random input → Exploring behavior:")

for i in range(3):
    sensory = np.random.rand(8).tolist()
    motor, state = brain.process_robot_cycle(sensory)
    motor_str = ", ".join([f"{m:.2f}" for m in motor])
    print(f"  Cycle {i+1}: Motor = [{motor_str}] (dim={len(motor)})")

# Demo 2: Pattern response
print("\n2. PATTERN RESPONSE")
print("-" * 40)
print("Structured input → Different behavior:")

patterns = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]

for i, pattern in enumerate(patterns):
    motor, state = brain.process_robot_cycle(pattern)
    motor_str = ", ".join([f"{m:.2f}" for m in motor])
    print(f"  Pattern {i+1}: Motor = [{motor_str}]")

# Demo 3: Reward shaping
print("\n3. REWARD RESPONSE")
print("-" * 40)
print("Testing reward influence:")

# Baseline
sensory = [0.5] * 8
motor_before, _ = brain.process_robot_cycle(sensory)
motor_str = ", ".join([f"{m:.2f}" for m in motor_before])
print(f"  Before reward: Motor = [{motor_str}]")

# Give reward
sensory_with_reward = sensory + [1.0]  # Strong positive reward
brain.process_robot_cycle(sensory_with_reward)
print(f"  → Reward given!")

# Check behavior change
motor_after, _ = brain.process_robot_cycle(sensory)
motor_str = ", ".join([f"{m:.2f}" for m in motor_after])
print(f"  After reward:  Motor = [{motor_str}]")

# Demo 4: Field evolution
print("\n4. FIELD EVOLUTION")
print("-" * 40)

# Run more cycles to show evolution
for i in range(10):
    sensory = [0.5 + 0.2 * np.sin(i * 0.3 + j * 0.1) for j in range(8)]
    brain.process_robot_cycle(sensory)

props = brain.field_dynamics.get_emergent_properties()
print(f"  Evolution cycles: {brain.field_dynamics.evolution_count}")
print(f"  Self-modification: {brain.field_dynamics.self_modification_strength:.3f}")
print(f"  Confidence: {props['smoothed_confidence']:.3f}")
print(f"  Active regions: {len(brain.topology_region_system.regions)}")

# Demo 5: Emergent properties
print("\n5. EMERGENT PROPERTIES")
print("-" * 40)

# Check field statistics
field_max = float(brain.unified_field.max().cpu())
field_mean = float(brain.unified_field.mean().cpu())
field_std = float(brain.unified_field.std().cpu())

print(f"  Field statistics:")
print(f"    Max activation: {field_max:.2f}")
print(f"    Mean: {field_mean:.3f}")
print(f"    Std: {field_std:.3f}")

# Check regional parameters if available
if hasattr(brain.field_dynamics, 'regional_parameters') and brain.field_dynamics.regional_parameters:
    print(f"  Regional specialization detected")
else:
    print(f"  Field evolving uniformly (early stage)")

print("\n" + "="*60)
print("Key insights:")
print("  • The brain responds to patterns without pre-programming")
print("  • Rewards shape the field topology")
print("  • Field dynamics evolve through experience")
print("  • Regions specialize based on activity")