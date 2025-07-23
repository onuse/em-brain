#!/usr/bin/env python3
"""Test hardware-adaptive resolution selection."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

from brains.field.core_brain import UnifiedFieldBrain

print("=== HARDWARE-ADAPTIVE RESOLUTION TEST ===\n")

# Test 1: Create brain without specifying resolution
print("1. Testing automatic resolution selection:")
brain = UnifiedFieldBrain(quiet_mode=False)  # Let it print hardware info

print(f"\nAutomatic selection:")
print(f"  Spatial resolution: {brain.spatial_resolution}³")
print(f"  Field shape: {brain.unified_field.shape}")
print(f"  Field elements: {brain.unified_field.numel():,}")

# Check which tier this machine is
if hasattr(brain, 'hw_profile') and brain.hw_profile:
    print(f"\nHardware profile:")
    print(f"  CPU cores: {brain.hw_profile.cpu_cores}")
    print(f"  Memory: {brain.hw_profile.total_memory_gb:.1f} GB")
    print(f"  GPU available: {brain.hw_profile.gpu_available}")
    print(f"  Avg cycle time: {brain.hw_profile.avg_cycle_time_ms:.1f}ms")

brain.shutdown()

# Test 2: Override with manual resolution
print("\n2. Testing manual resolution override:")
for res in [3, 4, 5]:
    brain = UnifiedFieldBrain(spatial_resolution=res, quiet_mode=True)
    print(f"  Resolution {res}³: Field has {brain.unified_field.numel():,} elements")
    brain.shutdown()

# Test 3: Performance at adaptive resolution
print("\n3. Testing performance at adaptive resolution:")
brain = UnifiedFieldBrain()  # Use adaptive

import time
times = []
for _ in range(20):
    start = time.perf_counter()
    brain.process_robot_cycle([0.5] * 24)
    times.append((time.perf_counter() - start) * 1000)

avg_time = sum(times[5:]) / len(times[5:])  # Skip warmup
print(f"  Average cycle time: {avg_time:.1f}ms")
print(f"  Frequency: {1000/avg_time:.1f} Hz")

if avg_time < 40:
    print("  ✅ Real-time performance maintained!")
elif avg_time < 70:
    print("  ⚠️  Near real-time")
else:
    print("  ❌ Too slow for real-time")

brain.shutdown()

print("\n" + "="*50)
print("SUMMARY:")
print("The brain now automatically selects spatial resolution")
print("based on hardware capabilities:")
print("- High performance (≤20ms): 5³ for best behaviors")
print("- Medium performance (≤40ms): 4³ for balance")
print("- Low performance (>40ms): 3³ for speed")
print("="*50)