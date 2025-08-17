#!/usr/bin/env python3
"""Detailed profiling to find the exact bottleneck."""

import torch
import time
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

def time_component(name, func, *args, **kwargs):
    """Time a single component."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    result = func(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  {name:30s}: {elapsed:8.1f}ms")
    return result

print("Detailed Component Profiling")
print("=" * 60)

# Create brain
brain = TrulyMinimalBrain(
    sensory_dim=12,
    motor_dim=6,
    spatial_size=96,
    channels=192,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    quiet_mode=True
)

sensors = [0.5] * 12
sensor_tensor = torch.tensor(sensors, dtype=torch.float32, device=brain.device)

# Warmup
print("\nWarmup cycle...")
brain.process(sensors)

print("\nComponent timing breakdown:")
print("-" * 60)

# Test each component
field = brain.field.clone()

# 1. Sensor injection (manual)
print("\n1. SENSOR INJECTION:")
time_component("  Creating sensor tensor", torch.tensor, sensors, dtype=torch.float32, device=brain.device)

# 2. Intrinsic tensions
print("\n2. INTRINSIC TENSIONS:")
time_component("  apply_tensions", brain.tensions.apply_tensions, field, 0.0)

# 3. Field dynamics
print("\n3. FIELD DYNAMICS:")
time_component("  evolve", brain.dynamics.evolve, field)

# 4. Motor extraction
print("\n4. MOTOR EXTRACTION:")
motors = time_component("  extract_motors", brain.motor.extract_motors, field)

# 5. Prediction
print("\n5. PREDICTION:")
if hasattr(brain.prediction, 'predict_next_sensors'):
    time_component("  predict_next_sensors", brain.prediction.predict_next_sensors, field)

# 6. Get comfort metrics
print("\n6. COMFORT METRICS:")
time_component("  get_comfort_metrics", brain.tensions.get_comfort_metrics, field)

# Now time the full process
print("\n" + "=" * 60)
print("FULL PROCESS CYCLE:")
print("-" * 60)

times = []
for i in range(5):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    motors, telemetry = brain.process(sensors)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"  Cycle {i+1}: {elapsed:.1f}ms")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.1f}ms")

# Check if it's the motor extraction
print("\n" + "=" * 60)
print("MOTOR EXTRACTION DEEP DIVE:")
print("-" * 60)

# Check which method is being used
if hasattr(brain.motor, 'impl'):
    print("Using optimized motor extraction")
    
    # Test ultra-fast directly
    print("\nDirect ultra-fast test:")
    for i in range(3):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        motors = brain.motor.impl.extract_motors_ultra_fast(field)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  {i+1}: {elapsed:.1f}ms")
else:
    print("Using original motor extraction")

print("\nDone!")