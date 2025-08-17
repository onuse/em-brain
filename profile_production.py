#!/usr/bin/env python3
"""Profile production-sized brain to find bottlenecks."""

import sys
import os
sys.path.append('/mnt/c/Users/glimm/Documents/Projects/em-brain/server/src')

import torch
import time
from brains.field.gpu_optimized_brain import GPUOptimizedFieldBrain

print("\nTesting different brain sizes to find scaling issue...")

sizes = [16, 32, 48, 64, 80, 96]
channels = [32, 64, 96, 128, 160, 192]

for size, ch in zip(sizes, channels):
    print(f"\n{'='*60}")
    print(f"Testing {size}³×{ch} = {size**3 * ch:,} parameters")
    
    brain = GPUOptimizedFieldBrain(
        sensory_dim=16,
        motor_dim=5,
        spatial_size=size,
        channels=ch,
        quiet_mode=True
    )
    
    # Warm up
    brain.process([0.5] * 16)
    
    # Time 3 cycles
    times = []
    for i in range(3):
        start = time.perf_counter()
        motor_output, telemetry = brain.process([0.5 + 0.1*i] * 16)
        torch.cuda.synchronize()  # Ensure GPU operations complete
        end = time.perf_counter()
        times.append((end-start)*1000)
    
    avg_time = sum(times)/len(times)
    print(f"  Average time: {avg_time:.1f}ms")
    
    if avg_time > 1000:
        print(f"  ⚠️  TOO SLOW! Breaking here to investigate...")
        
        # Profile individual components
        print("\n  Component timing:")
        
        # Test field dynamics
        start = time.perf_counter()
        brain.dynamics.evolve(brain.field)
        torch.cuda.synchronize()
        print(f"    Field dynamics: {(time.perf_counter()-start)*1000:.1f}ms")
        
        # Test tensions
        start = time.perf_counter()
        brain.tensions.apply_tensions(brain.field, 0.0)
        torch.cuda.synchronize()
        print(f"    Tensions: {(time.perf_counter()-start)*1000:.1f}ms")
        
        # Test motor extraction
        start = time.perf_counter()
        brain.motor.extract_motors(brain.field)
        torch.cuda.synchronize()
        print(f"    Motor extraction: {(time.perf_counter()-start)*1000:.1f}ms")
        
        break
    
    # Clean up GPU memory
    del brain
    torch.cuda.empty_cache()