#!/usr/bin/env python3
"""Profile which line in apply_tensions is slow."""

import torch
import time

device = torch.device('cuda')
field_shape = (96, 96, 96, 192)
field = torch.randn(*field_shape, device=device) * 0.01

print("Profiling individual operations on 96³×192 tensor:")
print("-" * 60)

# Test variance computation
start = time.perf_counter()
var_per_channel = torch.var(field, dim=-1, keepdim=True)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000
print(f"torch.var(field, dim=-1): {elapsed:.2f} ms")

# Test expand_as
start = time.perf_counter()
local_variance = var_per_channel.expand_as(field)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000
print(f"expand_as(field): {elapsed:.2f} ms")

# Test mask creation and where
start = time.perf_counter()
boredom_mask = local_variance < 0.01
boredom_noise = torch.randn_like(field) * 0.02
field_new = torch.where(boredom_mask, field + boredom_noise, field)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000
print(f"Mask creation and where: {elapsed:.2f} ms")

# Test gradient computation
start = time.perf_counter()
dx = torch.diff(field, dim=0, prepend=field[:1])
dy = torch.diff(field, dim=1, prepend=field[:, :1])
dz = torch.diff(field, dim=2, prepend=field[:, :, :1])
gradient_magnitude = torch.sqrt(dx**2 + dy**2 + dz**2)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000
print(f"Gradient computation (3 diffs + sqrt): {elapsed:.2f} ms")

# Test sin operation
phase = torch.zeros_like(field)
start = time.perf_counter()
oscillation = 0.01 * torch.sin(phase) * (1 + torch.abs(field))
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000
print(f"Sin oscillation: {elapsed:.2f} ms")