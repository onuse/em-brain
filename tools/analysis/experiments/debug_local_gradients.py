#!/usr/bin/env python3
"""Debug local gradient optimization."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))

import time
import torch
from brains.field.core_brain import UnifiedFieldBrain

print("=== DEBUG LOCAL GRADIENT OPTIMIZATION ===\n")

# Create brain
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)
print(f"Field shape: {brain.unified_field.shape}")

# Test gradient calculation directly
print("\nTesting gradient calculation methods:")

# 1. Full field gradient (for comparison)
start = time.perf_counter()
full_grads = brain.gradient_calculator.calculate_gradients(
    brain.unified_field,
    dimensions_to_compute=[0, 1, 2, 3, 4],
    local_region_only=False
)
time_full = (time.perf_counter() - start) * 1000
print(f"\n1. Full field gradients: {time_full:.1f}ms")
for name, grad in full_grads.items():
    print(f"   {name}: shape={grad.shape}, non-zero={torch.count_nonzero(grad).item()}")

# 2. Local region gradients
start = time.perf_counter()
local_grads = brain.gradient_calculator.calculate_gradients(
    brain.unified_field,
    dimensions_to_compute=[0, 1, 2, 3, 4],
    local_region_only=True,
    region_size=3
)
time_local = (time.perf_counter() - start) * 1000
print(f"\n2. Local region gradients: {time_local:.1f}ms")
print(f"   Speedup: {time_full/time_local:.1f}x")
for name, grad in local_grads.items():
    print(f"   {name}: shape={grad.shape}, non-zero={torch.count_nonzero(grad).item()}")

# Check if gradients are equivalent in the local region
print("\nChecking gradient equivalence in local region:")
center = brain.unified_field.shape[0] // 2
for dim in range(3):
    full = full_grads[f'gradient_dim_{dim}']
    local = local_grads[f'gradient_dim_{dim}']
    
    # Extract center region from both
    full_center = full[center-1:center+2, center-1:center+2, center-1:center+2]
    local_center = local[center-1:center+2, center-1:center+2, center-1:center+2]
    
    diff = torch.abs(full_center - local_center).max().item()
    print(f"   Dimension {dim}: max difference = {diff:.6f}")

# Test cache behavior
print("\nTesting gradient cache:")
brain.gradient_calculator.clear_cache()

# First call - should miss
brain._calculate_gradient_flows()
stats1 = brain.gradient_calculator.get_cache_stats()

# Second call - should hit?
brain._calculate_gradient_flows()
stats2 = brain.gradient_calculator.get_cache_stats()

print(f"After 1st call: hits={stats1['cache_hits']}, misses={stats1['cache_misses']}")
print(f"After 2nd call: hits={stats2['cache_hits']}, misses={stats2['cache_misses']}")

# Check why cache might not be working
print("\nField hash behavior:")
hash1 = brain.gradient_calculator._compute_field_hash(brain.unified_field)
# Slightly modify field
brain.unified_field[0,0,0,0,0,0,0,0,0,0,0] += 0.00001
hash2 = brain.gradient_calculator._compute_field_hash(brain.unified_field)
print(f"Hash before: {hash1}")
print(f"Hash after tiny change: {hash2}")
print(f"Hashes equal: {hash1 == hash2}")

brain.shutdown()
print("\n✅ Debug complete!")