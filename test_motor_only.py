#!/usr/bin/env python3
"""Test just the motor extraction to isolate the issue."""

import torch
import time
from server.src.brains.field.optimized_motor import OptimizedMotorExtraction

print("Testing motor extraction in isolation...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create a test field
field = torch.randn(96, 96, 96, 192, device=device) * 0.1
print(f"Field shape: {field.shape}")

# Create motor extractor
motor = OptimizedMotorExtraction(motor_dim=6, device=device, field_size=96)

print("\nTesting ultra-fast version:")
for i in range(3):
    start = time.perf_counter()
    motors = motor.extract_motors_ultra_fast(field)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  {i+1}: {elapsed:.1f}ms")

print("\nTesting standard version:")
try:
    start = time.perf_counter()
    motors = motor.extract_motors(field)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Time: {elapsed:.1f}ms")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone!")