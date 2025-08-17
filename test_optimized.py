#!/usr/bin/env python3
"""Test the optimized brain performance."""

import torch
import time
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

print("=" * 70)
print("OPTIMIZED BRAIN PERFORMANCE TEST")
print("=" * 70)

# Check GPU
if not torch.cuda.is_available():
    print("WARNING: CUDA not available")
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# Create brain with full size
print("\nInitializing 96³×192 brain...")
brain = TrulyMinimalBrain(
    sensory_dim=12,
    motor_dim=6,
    spatial_size=96,
    channels=192,
    device=device,
    quiet_mode=True
)

params = 96**3 * 192
print(f"Parameters: {params:,}")
print(f"Memory: ~{params * 4 / (1024**3):.2f} GB")

sensors = [0.5] * 12

# Warmup
print("\nWarmup (3 cycles):")
for i in range(3):
    start = time.perf_counter()
    motors, telemetry = brain.process(sensors)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  {i+1}: {elapsed:.1f}ms")

# Performance test
print("\nPerformance test (10 cycles):")
times = []
for i in range(10):
    start = time.perf_counter()
    motors, telemetry = brain.process(sensors)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"  {i+1:2}: {elapsed:.1f}ms")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.1f}ms")

if avg < 100:
    print("✅ SUCCESS! Target <100ms achieved!")
elif avg < 200:
    print("⚠️  Good but not optimal (100-200ms)")
else:
    print("❌ Still too slow (>200ms)")

# Test components individually
print("\n" + "=" * 50)
print("COMPONENT TIMING")
print("=" * 50)

# Test motor extraction
print("\nMotor extraction only:")
for i in range(3):
    start = time.perf_counter()
    motors = brain.motor.extract_motors(brain.field)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  {i+1}: {elapsed:.1f}ms")

# Test tensions
print("\nIntrinsic tensions only:")
for i in range(3):
    start = time.perf_counter()
    field = brain.tensions.apply_tensions(brain.field, 0.0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  {i+1}: {elapsed:.1f}ms")

# Test field dynamics
print("\nField dynamics only:")
for i in range(3):
    start = time.perf_counter()
    field = brain.dynamics.evolve(brain.field)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  {i+1}: {elapsed:.1f}ms")

print("\nDone!")