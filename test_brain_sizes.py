#!/usr/bin/env python3
"""Find the right brain size for real-time performance."""

import torch
import time
from server.src.brains.field.truly_minimal_brain import TrulyMinimalBrain

sizes = [
    (32, 64),   # ~260K params
    (48, 96),   # ~1M params  
    (64, 128),  # ~33M params
]

for spatial, channels in sizes:
    print(f"\nTesting {spatial}³×{channels} ({(spatial**3 * channels):,} params)...")
    
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
    brain.process(sensors)
    
    # Time 5 cycles
    times = []
    for i in range(5):
        start = time.time()
        motors, telemetry = brain.process(sensors)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    print(f"  Average: {avg*1000:.1f}ms ({1/avg:.1f} Hz)")
    
    if avg < 0.2:
        print(f"  ✅ Good for real-time control")
    elif avg < 0.5:
        print(f"  ⚠️ Marginal for real-time")  
    else:
        print(f"  ❌ Too slow!")
    
    # Clean up
    del brain
    torch.cuda.empty_cache()