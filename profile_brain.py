#!/usr/bin/env python3
"""
Profile brain processing to find the bottleneck.
"""

import torch
import time
import cProfile
import pstats
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

print("Profiling brain processing...")

# Test with the original large size
brain = TrulyMinimalBrain(
    sensory_dim=12,
    motor_dim=6,
    spatial_size=96,  # Original large size
    channels=192,
    device=torch.device('cuda'),
    quiet_mode=False
)

# Warmup
sensors = [0.5] * 12
print("\nWarmup run...")
motors, telemetry = brain.process(sensors)

# Time individual components
print("\nTiming brain components...")

# Profile one full cycle
print("\nProfiling full cycle...")
profiler = cProfile.Profile()
profiler.enable()

start = time.time()
motors, telemetry = brain.process(sensors)
elapsed = time.time() - start

profiler.disable()

print(f"\nTotal time: {elapsed*1000:.1f}ms")

# Print profile stats
print("\nTop 10 time consumers:")
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)

# Now let's manually time each component
print("\n" + "="*60)
print("MANUAL COMPONENT TIMING")
print("="*60)

# Test field evolution specifically
field = brain.field.clone()
momentum = brain.field_momentum.clone()

print("\n1. Field dynamics evolution:")
for i in range(3):
    start = time.time()
    test_field, test_momentum = brain.dynamics.evolve(field, momentum)
    torch.cuda.synchronize()  # Wait for GPU to finish
    elapsed = time.time() - start
    print(f"   Run {i+1}: {elapsed*1000:.1f}ms")

print("\n2. Intrinsic tensions:")
for i in range(3):
    start = time.time()
    tension = brain.tensions.generate(field)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Run {i+1}: {elapsed*1000:.1f}ms")

print("\n3. Motor extraction:")
for i in range(3):
    start = time.time()
    motors = brain.motor.extract(field)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Run {i+1}: {elapsed*1000:.1f}ms")

print("\n4. Prediction:")
test_sensors = torch.tensor(sensors, dtype=torch.float32, device=brain.device)
for i in range(3):
    start = time.time()
    pred = brain.prediction.predict_next(field, test_sensors)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Run {i+1}: {elapsed*1000:.1f}ms")

# Check if it's a CPU/GPU transfer issue
print("\n5. CPU-GPU transfer test:")
cpu_data = torch.randn(96, 96, 96, 192)
for i in range(3):
    start = time.time()
    gpu_data = cpu_data.to('cuda')
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   CPU->GPU transfer: {elapsed*1000:.1f}ms")
    del gpu_data

print("\nDone!")