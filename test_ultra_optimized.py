#!/usr/bin/env python3
"""Test the ultra-optimized brain."""

import torch
import time
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

print("ULTRA-OPTIMIZED BRAIN TEST")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# Test with increasing sizes
sizes = [(16, 32), (32, 64), (64, 128), (96, 192)]

for spatial_size, channels in sizes:
    print(f"\n{spatial_size}³×{channels} Brain:")
    print("-" * 40)
    
    brain = TrulyMinimalBrain(
        sensory_dim=12,
        motor_dim=6,
        spatial_size=spatial_size,
        channels=channels,
        device=device,
        quiet_mode=True
    )
    
    sensors = [0.5] * 12
    
    # Warmup
    brain.process(sensors)
    
    # Time 5 cycles
    times = []
    for i in range(5):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        motors, telemetry = brain.process(sensors)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Cycle {i+1}: {elapsed:.1f}ms")
    
    avg = sum(times) / len(times)
    print(f"  Average: {avg:.1f}ms")
    
    if spatial_size == 96:
        print("\n" + "=" * 70)
        print("FINAL RESULT FOR 96³×192:")
        print("=" * 70)
        
        if avg < 100:
            print(f"✅ SUCCESS! {avg:.1f}ms average (target: <100ms)")
        elif avg < 200:
            print(f"⚠️  ACCEPTABLE: {avg:.1f}ms average (target: <100ms)")
        else:
            print(f"❌ TOO SLOW: {avg:.1f}ms average (target: <100ms)")

print("\nDone!")