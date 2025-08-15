#!/usr/bin/env python3
"""Test optimized brain performance."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import time
import numpy as np
from brains.field.core_brain import UnifiedFieldBrain

print("=== OPTIMIZED BRAIN PERFORMANCE TEST ===\n")

# Test with different resolutions
resolutions = [3, 5, 8]
sensory_input = [0.5] * 24

for res in resolutions:
    print(f"\n--- Resolution {res}³ ---")
    
    # Create brain
    brain = UnifiedFieldBrain(spatial_resolution=res, quiet_mode=True)
    
    # Warm up
    brain.process_robot_cycle(sensory_input)
    
    # Time 10 cycles
    times = []
    for i in range(10):
        start = time.perf_counter()
        action, state = brain.process_robot_cycle(sensory_input)
        cycle_time = (time.perf_counter() - start) * 1000
        times.append(cycle_time)
        
        # Show action for first cycle
        if i == 0:
            print(f"Sample action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Cycle times: mean={avg_time:.1f}ms, std={std_time:.1f}ms")
    print(f"Frequency: {1000/avg_time:.1f} Hz")
    
    # Check if real-time capable
    if avg_time < 40:
        print("✅ Real-time capable!")
    elif avg_time < 100:
        print("⚠️  Near real-time")
    else:
        print("❌ Too slow for real-time")
    
    # Check gradient cache
    stats = brain.gradient_calculator.get_cache_stats()
    print(f"Gradient cache hit rate: {stats['hit_rate']:.1%}")
    
    brain.shutdown()

print("\n" + "="*50)
print("SUMMARY:")
print("With local gradient optimization, the brain should be")
print("10-100x faster while maintaining full functionality.")
print("Future actuators can be added by expanding the local") 
print("region or computing multiple local regions.")
print("="*50)