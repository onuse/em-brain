#!/usr/bin/env python3
"""Test just the intrinsic tensions to see if that's the issue."""

import torch
import time
from server.src.brains.field.optimized_intrinsic_tensions import OptimizedIntrinsicTensions

print("Testing intrinsic tensions in isolation...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Test different sizes
sizes = [(16, 32), (32, 64), (64, 128), (96, 192)]

for spatial_size, channels in sizes:
    print(f"\nTesting {spatial_size}³×{channels}:")
    
    # Create field
    field_shape = (spatial_size, spatial_size, spatial_size, channels)
    field = torch.randn(*field_shape, device=device) * 0.01
    
    # Create tensions
    tensions = OptimizedIntrinsicTensions(field_shape, device)
    
    # Time apply_tensions
    times = []
    for i in range(3):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        
        result = tensions.apply_tensions(field, 0.0)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f}ms")
    
    avg = sum(times) / len(times)
    print(f"  Average: {avg:.1f}ms")
    
    if avg > 1000:
        print(f"  ⚠️ TOO SLOW! Breaking here to investigate")
        break

print("\nDone!")