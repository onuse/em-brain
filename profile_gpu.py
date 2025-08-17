#!/usr/bin/env python3
"""Profile GPU operations to find the bottleneck."""

import torch
import time
import torch.profiler as profiler

# Test basic GPU operations at 96³×192 scale
device = torch.device('cuda')
size = 96
channels = 192

print(f"Testing GPU operations at {size}³×{channels} scale...")
print(f"Total parameters: {size**3 * channels:,}")
print(f"Memory: {size**3 * channels * 4 / 1e9:.2f} GB")

# Create test tensor
print("\n1. Tensor creation:")
start = time.time()
field = torch.randn(size, size, size, channels, device=device) * 0.01
torch.cuda.synchronize()
print(f"   {(time.time() - start)*1000:.1f}ms")

# Test basic operations
print("\n2. Basic math operations:")

# Addition
start = time.time()
result = field + 0.1
torch.cuda.synchronize()
print(f"   Addition: {(time.time() - start)*1000:.1f}ms")

# Multiplication  
start = time.time()
result = field * 0.99
torch.cuda.synchronize()
print(f"   Multiplication: {(time.time() - start)*1000:.1f}ms")

# Mean
start = time.time()
mean_val = field.mean()
torch.cuda.synchronize()
print(f"   Mean: {(time.time() - start)*1000:.1f}ms")

# Variance
start = time.time()
var_val = field.var()
torch.cuda.synchronize()
print(f"   Variance: {(time.time() - start)*1000:.1f}ms")

print("\n3. Complex operations:")

# Gradient computation
start = time.time()
dx = torch.diff(field, dim=0, prepend=field[:1])
torch.cuda.synchronize()
print(f"   Gradient (diff): {(time.time() - start)*1000:.1f}ms")

# Random noise generation
start = time.time()
noise = torch.randn_like(field) * 0.02
torch.cuda.synchronize()
print(f"   randn_like: {(time.time() - start)*1000:.1f}ms")

# Where operation (masking)
start = time.time()
mask = field > 0
result = torch.where(mask, field + noise, field)
torch.cuda.synchronize()
print(f"   Where (masking): {(time.time() - start)*1000:.1f}ms")

# Check if it's the Python max/min issue
print("\n4. Python builtin operations on tensor shapes:")
start = time.time()
for i in range(1000):
    x = max(0, 50)
    y = min(field.shape[0], 100)
elapsed = (time.time() - start)*1000
print(f"   1000x max/min on integers: {elapsed:.1f}ms")

# This was the issue before - max on tensor.shape
start = time.time()
for i in range(1000):
    x = max(0, field.shape[0] - 1)
elapsed = (time.time() - start)*1000
print(f"   1000x max with tensor.shape: {elapsed:.1f}ms")