#!/usr/bin/env python3
"""Quick test to isolate the performance issue."""

import torch
import time
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

print("Quick performance test...")

# Start with smaller size to test
brain = TrulyMinimalBrain(
    sensory_dim=12,
    motor_dim=6,
    spatial_size=16,  # Start small
    channels=32,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    quiet_mode=True
)

sensors = [0.5] * 12

# Test single cycle
start = time.perf_counter()
motors, telemetry = brain.process(sensors)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000

print(f"Small brain (16³×32): {elapsed:.1f}ms")

# Now test with larger size
print("\nTesting larger brain...")
brain2 = TrulyMinimalBrain(
    sensory_dim=12,
    motor_dim=6,
    spatial_size=32,  # Medium size
    channels=64,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    quiet_mode=True
)

start = time.perf_counter()
motors, telemetry = brain2.process(sensors)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000

print(f"Medium brain (32³×64): {elapsed:.1f}ms")

print("\nDone!")