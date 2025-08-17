#!/usr/bin/env python3
"""Profile each part of the brain to find the 11-second bottleneck."""

import torch
import time
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

# Create brain
brain = TrulyMinimalBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_size=64,  # Use 64 which we know takes ~127ms total
    channels=128,
    device=torch.device('cuda'),
    quiet_mode=True
)

sensors = [0.5] * 24

# Warmup
brain.process(sensors)

print("Profiling brain components (64³×128)...")
print("Expected total: ~127ms")
print("\nTiming each component:")

# Time full process
start = time.time()
motors, telemetry = brain.process(sensors)
total_time = (time.time() - start) * 1000
print(f"\nTotal process(): {total_time:.1f}ms")

# Now profile each part manually
sensors_tensor = torch.tensor(sensors[:brain.sensory_dim], 
                              dtype=torch.float32, device=brain.device)

# 1. Sensor injection
start = time.time()
for i, value in enumerate(sensors):
    if i >= brain.sensory_dim:
        break
    x, y, z = brain.sensor_spots[i]
    brain.field[x, y, z, i % 8] += value * 0.3
torch.cuda.synchronize()
print(f"  Sensor injection: {(time.time() - start)*1000:.1f}ms")

# 2. Intrinsic tensions
start = time.time()
brain.field = brain.tensions.apply_tensions(brain.field, 0.0)
torch.cuda.synchronize()
print(f"  Intrinsic tensions: {(time.time() - start)*1000:.1f}ms")

# 3. Field dynamics
start = time.time()
brain.field = brain.dynamics.evolve(brain.field, brain.field_momentum)
torch.cuda.synchronize()
print(f"  Field dynamics: {(time.time() - start)*1000:.1f}ms")

# 4. Motor extraction
start = time.time()
motors = brain.motor.extract_motors(brain.field)
torch.cuda.synchronize()
print(f"  Motor extraction: {(time.time() - start)*1000:.1f}ms")

# 5. Prediction - skip (method name issue)

print("\nNow testing 96³×192 (the slow one)...")
# Clean up
del brain
torch.cuda.empty_cache()

# Create large brain
brain = TrulyMinimalBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_size=96,
    channels=192,
    device=torch.device('cuda'),
    quiet_mode=True
)

# Warmup
print("Warming up (this will be slow)...")
start = time.time()
brain.process(sensors)
print(f"Warmup took: {(time.time() - start):.1f}s")

print("\nProfiling components:")
# Time full process
start = time.time()
motors, telemetry = brain.process(sensors)
total_time = (time.time() - start)
print(f"Total process(): {total_time:.1f}s")

if total_time > 5:
    print("\n⚠️ Something is VERY wrong - this should be <200ms on RTX 3070!")