#!/usr/bin/env python3
"""Test brain performance in Windows (run from PowerShell)."""

import torch
import time
import sys
sys.path.append('.')  # Add current dir to path

from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

print("Testing brain performance in native Windows...")

# Test both sizes
sizes = [
    (64, 128),   # 33M params - works in WSL2
    (96, 192),   # 170M params - slow in WSL2
]

for spatial, channels in sizes:
    print(f"\n{'='*60}")
    print(f"Testing {spatial}³×{channels} ({(spatial**3 * channels):,} params)")
    print('='*60)
    
    brain = TrulyMinimalBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_size=spatial,
        channels=channels,
        device=torch.device('cuda'),
        quiet_mode=True
    )
    
    sensors = [0.5] * 24
    
    # Warmup
    print("Warming up...")
    start = time.time()
    brain.process(sensors)
    warmup_time = time.time() - start
    print(f"  Warmup: {warmup_time*1000:.1f}ms")
    
    # Time 5 cycles
    print("Timing 5 cycles...")
    times = []
    for i in range(5):
        start = time.time()
        motors, telemetry = brain.process(sensors)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Cycle {i+1}: {elapsed*1000:.1f}ms")
    
    avg = sum(times) / len(times)
    print(f"\nAverage: {avg*1000:.1f}ms ({1/avg:.1f} Hz)")
    
    if avg < 0.2:
        print("✅ Good for real-time control")
    elif avg < 0.5:
        print("⚠️ Marginal for real-time")
    else:
        print("❌ Too slow for real-time")
    
    # Clean up
    del brain
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("CONCLUSION:")
if sys.platform == "win32":
    print("Running in native Windows")
else:
    print("Running in WSL2")
print("="*60)