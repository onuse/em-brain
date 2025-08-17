#!/usr/bin/env python3
"""Test just the scatter operation that should be fast."""

import torch
import time

device = torch.device('cuda')
spatial_size = 96
channels = 192
sensory_dim = 16

print(f"Testing scatter_add on {spatial_size}³×{channels} tensor")

# Create field
field = torch.randn(spatial_size, spatial_size, spatial_size, channels, device=device) * 0.01
sensors = torch.randn(sensory_dim, device=device)

# Pre-compute indices
sensor_spots = torch.randint(0, spatial_size, (sensory_dim, 3), device=device)
sensor_x = sensor_spots[:, 0]
sensor_y = sensor_spots[:, 1]  
sensor_z = sensor_spots[:, 2]
sensor_c = torch.arange(sensory_dim, device=device) % 8

# Create flattened indices
sensor_flat_idx = (
    sensor_x * (spatial_size * spatial_size * channels) +
    sensor_y * (spatial_size * channels) +
    sensor_z * channels +
    sensor_c
)

print(f"Max index: {sensor_flat_idx.max().item()}, Field size: {field.numel()}")

# Test scatter_add
injection_values = sensors * 0.3
field_flat = field.view(-1)

# Time the operation
start = time.perf_counter()
field_flat.scatter_add_(0, sensor_flat_idx, injection_values)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000

print(f"Scatter_add took: {elapsed:.2f} ms")

# Reshape back
field = field_flat.view(spatial_size, spatial_size, spatial_size, channels)
print(f"Field shape after: {field.shape}")