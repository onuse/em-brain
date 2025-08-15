#!/usr/bin/env python3
"""Quick performance check of UnifiedFieldBrain."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import time
import torch
from brains.field.core_brain import UnifiedFieldBrain
import gc

print("=== UNIFIED FIELD BRAIN PERFORMANCE CHECK ===\n")

# Test different resolutions
for resolution in [3, 5, 8]:
    print(f"\n--- Spatial Resolution: {resolution}³ ---")
    
    # Create brain
    start = time.perf_counter()
    brain = UnifiedFieldBrain(spatial_resolution=resolution, quiet_mode=True)
    init_time = (time.perf_counter() - start) * 1000
    
    # Get field info
    field_shape = brain.unified_field.shape
    field_elements = brain.unified_field.numel()
    field_mb = field_elements * 4 / 1024 / 1024
    
    print(f"Initialization: {init_time:.1f}ms")
    print(f"Field shape: {field_shape}")
    print(f"Field elements: {field_elements:,}")
    print(f"Field memory: {field_mb:.1f}MB")
    
    # Time some cycles
    sensory_input = [0.5] * 24
    
    # Warm up
    brain.process_robot_cycle(sensory_input)
    
    # Time 5 cycles
    times = []
    for i in range(5):
        start = time.perf_counter()
        action, state = brain.process_robot_cycle(sensory_input)
        cycle_time = (time.perf_counter() - start) * 1000
        times.append(cycle_time)
    
    avg_time = sum(times) / len(times)
    print(f"\nCycle times: {[f'{t:.1f}ms' for t in times]}")
    print(f"Average: {avg_time:.1f}ms ({1000/avg_time:.1f} Hz)")
    
    # Check key operations
    print("\nKey operation timing:")
    
    # Field decay
    start = time.perf_counter()
    brain.unified_field *= 0.999
    decay_time = (time.perf_counter() - start) * 1000
    print(f"  Field decay: {decay_time:.2f}ms")
    
    # Gradient calculation (simulate)
    start = time.perf_counter()
    if resolution <= 5:
        # Only test gradient on small fields
        try:
            grad = torch.gradient(brain.unified_field[:,:,:,0,0,0,0,0,0,0,0])
            grad_time = (time.perf_counter() - start) * 1000
            print(f"  Gradient calc: {grad_time:.2f}ms")
        except:
            print(f"  Gradient calc: Failed")
    
    # Check maintenance thread
    print(f"\nMaintenance thread: {'Running' if brain._maintenance_thread.is_alive() else 'Not running'}")
    
    # Clean up
    brain.shutdown()
    del brain
    gc.collect()

print("\n" + "="*50)
print("SUMMARY:")
print("- Resolution 3³: Basic testing only")
print("- Resolution 5³: Minimal viable brain") 
print("- Resolution 8³: Standard brain")
print("- Real-time needs: <40ms per cycle")
print("="*50)