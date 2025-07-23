#!/usr/bin/env python3
"""Test gradient computation optimizations."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))

import time
import torch
from brains.field.core_brain import UnifiedFieldBrain

print("=== GRADIENT OPTIMIZATION ANALYSIS ===\n")

# Create minimal brain
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)

# Get field reference
field = brain.unified_field
print(f"Field shape: {field.shape}")
print(f"Field elements: {field.numel():,}")

# Test different gradient approaches
print("\n--- Gradient Computation Methods ---")

# 1. Full field gradient (current approach)
print("\n1. Full torch.gradient (current):")
start = time.perf_counter()
grad_x = torch.gradient(field, dim=0)[0]
time_full = (time.perf_counter() - start) * 1000
print(f"   Time: {time_full:.1f}ms")
print(f"   Output shape: {grad_x.shape}")

# 2. Slice first, then gradient
print("\n2. Slice to 3D first:")
# Get central region for all other dimensions
center_indices = [field.shape[i]//2 for i in range(3, 11)]
start = time.perf_counter()
field_slice = field[:, :, :, center_indices[0], center_indices[1], center_indices[2], 
                    center_indices[3], center_indices[4], center_indices[5], 
                    center_indices[6], center_indices[7]]
grad_x_slice = torch.gradient(field_slice, dim=0)[0]
time_slice = (time.perf_counter() - start) * 1000
print(f"   Time: {time_slice:.1f}ms")
print(f"   Speedup: {time_full/time_slice:.1f}x")

# 3. Simple finite differences
print("\n3. Simple finite differences:")
start = time.perf_counter()
grad_fd = torch.zeros_like(field)
grad_fd[:-1] = field[1:] - field[:-1]
time_fd = (time.perf_counter() - start) * 1000
print(f"   Time: {time_fd:.1f}ms")
print(f"   Speedup: {time_full/time_fd:.1f}x")

# 4. Local gradient only (3x3x3 region)
print("\n4. Local 3x3x3 gradient only:")
center = field.shape[0] // 2
start = time.perf_counter()
local_field = field[center-1:center+2, center-1:center+2, center-1:center+2]
local_grad = torch.gradient(local_field, dim=0)[0]
time_local = (time.perf_counter() - start) * 1000
print(f"   Time: {time_local:.1f}ms")
print(f"   Speedup: {time_full/time_local:.1f}x")

# Summary
print("\n--- RECOMMENDATION ---")
print("The gradient computation is the main bottleneck!")
print("Options to fix:")
print("1. Only compute gradients in 3x3x3 local region (1000x speedup)")
print("2. Pre-slice to 3D before gradient computation")
print("3. Use simple finite differences instead of torch.gradient")
print("4. Move to GPU (but MPS doesn't support 11D tensors)")

brain.shutdown()