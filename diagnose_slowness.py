#!/usr/bin/env python3
"""Diagnose where the slowness is coming from."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import time
import torch

print("Step 1: Import brain...")
t0 = time.perf_counter()
from brains.field.core_brain import UnifiedFieldBrain
print(f"  Import took: {(time.perf_counter()-t0):.2f}s")

print("\nStep 2: Create minimal brain...")
t0 = time.perf_counter()
brain = UnifiedFieldBrain(
    spatial_resolution=5,  # Very small
    quiet_mode=True
)
print(f"  Creation took: {(time.perf_counter()-t0):.2f}s")

print(f"\nBrain field shape: {brain.unified_field.shape}")
print(f"Field memory: {brain.unified_field.element_size() * brain.unified_field.nelement() / (1024*1024):.2f} MB")

print("\nStep 3: Test field evolution...")
t0 = time.perf_counter()
brain._evolve_unified_field()
print(f"  Evolution took: {(time.perf_counter()-t0):.2f}s")

print("\nStep 4: Test gradient calculation...")
t0 = time.perf_counter()
brain._calculate_gradient_flows()
print(f"  Gradients took: {(time.perf_counter()-t0):.2f}s")

print("\nStep 5: Test constraint discovery...")
t0 = time.perf_counter()
# Set high discovery rate to force constraint operations
brain.constraint_field.constraint_discovery_rate = 1.0
constraints = brain.constraint_field.discover_constraints(brain.unified_field, brain.gradient_flows)
print(f"  Constraint discovery took: {(time.perf_counter()-t0):.2f}s")
print(f"  Found {len(constraints)} constraints")

print("\nStep 6: Single process cycle...")
t0 = time.perf_counter()
action, state = brain.process_robot_cycle([0.5] * 24)
print(f"  Full cycle took: {(time.perf_counter()-t0):.2f}s")

print("\n🔍 ANALYSIS:")
if time.perf_counter() - t0 > 0.1:
    print("The constraint system might be the bottleneck!")