#!/usr/bin/env python3
"""Detailed breakdown of where time is spent."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import time
import torch
from brains.field.core_brain import UnifiedFieldBrain

# Create minimal brain
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)
sensory_input = [0.5] * 24

# Warm up
brain.process_robot_cycle(sensory_input)

print("=== PERFORMANCE BREAKDOWN (Resolution 3³) ===\n")

# Time each major step manually
steps = {}

# 1. Sensory to field conversion
start = time.perf_counter()
field_exp = brain._robot_sensors_to_field_experience(sensory_input)
steps['sensory_conversion'] = (time.perf_counter() - start) * 1000

# 2. Field experience application
start = time.perf_counter()
brain._apply_field_experience(field_exp)
steps['apply_experience'] = (time.perf_counter() - start) * 1000

# 3. Field evolution
start = time.perf_counter()
brain._evolve_unified_field()
steps['field_evolution'] = (time.perf_counter() - start) * 1000

# 4. Action generation
start = time.perf_counter()
action = brain._field_gradients_to_robot_action()
steps['action_generation'] = (time.perf_counter() - start) * 1000

# Now break down field evolution further
print("Field Evolution Breakdown:")
brain._evolve_unified_field()  # Reset

# Time each substep
start = time.perf_counter()
brain.unified_field *= brain.field_decay_rate
substep_decay = (time.perf_counter() - start) * 1000

start = time.perf_counter()
baseline = 0.01
weak_mask = brain.unified_field < baseline
brain.unified_field[weak_mask] = baseline
substep_baseline = (time.perf_counter() - start) * 1000

# Test diffusion
start = time.perf_counter()
brain._apply_spatial_diffusion()
substep_diffusion = (time.perf_counter() - start) * 1000

# Test constraint evolution
start = time.perf_counter()
brain._apply_constraint_guided_evolution()
substep_constraints = (time.perf_counter() - start) * 1000

# Test gradient calculation
start = time.perf_counter()
brain._calculate_gradient_flows()
substep_gradients = (time.perf_counter() - start) * 1000

# Print results
print(f"\nMain Steps:")
total = sum(steps.values())
for step, time_ms in sorted(steps.items(), key=lambda x: x[1], reverse=True):
    pct = (time_ms / total) * 100
    print(f"  {step:<25} {time_ms:>8.2f}ms ({pct:>5.1f}%)")

print(f"\nField Evolution Substeps:")
print(f"  Field decay:              {substep_decay:>8.2f}ms")
print(f"  Baseline application:     {substep_baseline:>8.2f}ms")
print(f"  Spatial diffusion:        {substep_diffusion:>8.2f}ms")
print(f"  Constraint evolution:     {substep_constraints:>8.2f}ms")
print(f"  Gradient calculation:     {substep_gradients:>8.2f}ms")

# Check gradient calculator
if hasattr(brain, 'gradient_calculator'):
    print(f"\nGradient Calculator Info:")
    print(f"  Type: {type(brain.gradient_calculator).__name__}")
    stats = brain.gradient_calculator.get_cache_stats()
    print(f"  Cache hit rate: {stats['hit_rate']:.1%}")

# Check constraint field
print(f"\nConstraint Field Info:")
print(f"  Active constraints: {len(brain.constraint_field.constraints)}")
print(f"  Field shape: {brain.constraint_field.field.shape}")

brain.shutdown()