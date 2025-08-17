#!/usr/bin/env python3
"""
Test brain processing directly (no network).
"""

import torch
import time
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain
from server.src.brains.field.auto_config import get_optimal_config

print("Testing brain processing...")

# Test with larger brain (96x96x96x192)
print(f"Using config: 96³×192")

# Create brain
brain = TrulyMinimalBrain(
    sensory_dim=12,
    motor_dim=6,
    spatial_size=96,
    channels=192,
    device=torch.device('cuda'),
    quiet_mode=False
)

# Test processing
sensors = [0.5] * 12
print(f"\nSending {len(sensors)} sensors...")

try:
    motors, telemetry = brain.process(sensors)
    print(f"✅ Success! Got {len(motors)} motor commands")
    print(f"   Motors: {motors[:3]}..." if len(motors) > 3 else f"   Motors: {motors}")
    print(f"   Telemetry keys: {list(telemetry.keys())}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Try multiple cycles
print("\n\nTesting 10 processing cycles...")
for i in range(10):
    try:
        start = time.time()
        motors, telemetry = brain.process(sensors)
        elapsed = time.time() - start
        print(f"Cycle {i+1}: {elapsed*1000:.1f}ms")
    except Exception as e:
        print(f"Cycle {i+1}: ERROR - {e}")
        break

print("\nDone!")