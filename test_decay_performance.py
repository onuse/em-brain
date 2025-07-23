#!/usr/bin/env python3
"""Test performance of homeostatic decay operations."""
import torch
import time

# Create test field
shape = [8, 8, 8, 10, 15, 4, 4, 3, 2, 3, 2]
field = torch.rand(shape) * 0.2

print(f"Field shape: {shape}")
print(f"Total elements: {field.numel():,}")

# Test the decay operations
baseline = 0.02
target = 0.1

print("\nTiming decay operations...")

t0 = time.perf_counter()
excess = torch.relu(field - target)
t1 = time.perf_counter()
print(f"Compute excess: {(t1-t0)*1000:.1f}ms")

t0 = time.perf_counter()
deficit = torch.relu(target - field)
t1 = time.perf_counter()
print(f"Compute deficit: {(t1-t0)*1000:.1f}ms")

t0 = time.perf_counter()
decay_rates = 0.999 - 0.01 * torch.tanh(excess * 5) + 0.001 * torch.tanh(deficit * 10)
t1 = time.perf_counter()
print(f"Compute decay rates: {(t1-t0)*1000:.1f}ms")

t0 = time.perf_counter()
decay_rates = torch.clamp(decay_rates, 0.98, 1.001)
t1 = time.perf_counter()
print(f"Clamp rates: {(t1-t0)*1000:.1f}ms")

t0 = time.perf_counter()
field_new = baseline + (field - baseline) * decay_rates
t1 = time.perf_counter()
print(f"Apply decay: {(t1-t0)*1000:.1f}ms")

t0 = time.perf_counter()
field_new = torch.maximum(field_new, torch.tensor(baseline))
t1 = time.perf_counter()
print(f"Apply baseline: {(t1-t0)*1000:.1f}ms")

print(f"\nTotal time for all operations: ~{((t1-t0)*6)*1000:.1f}ms")