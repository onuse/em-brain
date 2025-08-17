#!/usr/bin/env python3
"""Test actual brain processing time."""

import torch
import time
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

# Create brain with server config
brain = TrulyMinimalBrain(
    sensory_dim=24,  # What the client sends
    motor_dim=4,      # What the client expects
    spatial_size=96,
    channels=192,
    device=torch.device('cuda'),
    quiet_mode=False
)

# Test sensors (24 values like the client sends)
sensors = [0.5] * 24

print("\nTiming 10 cycles...")
times = []
for i in range(10):
    start = time.time()
    motors, telemetry = brain.process(sensors)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Cycle {i+1}: {elapsed*1000:.1f}ms")

avg = sum(times) / len(times)
print(f"\nAverage: {avg*1000:.1f}ms ({1/avg:.1f} Hz)")

if avg > 1.0:
    print(f"⚠️ WARNING: Brain is too slow! {avg:.2f} seconds per cycle")
else:
    print(f"✅ Performance OK")