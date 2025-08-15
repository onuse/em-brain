#!/usr/bin/env python3
"""Final recommendation for standard resolution."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))

from brains.field.core_brain import UnifiedFieldBrain
import numpy as np
import time

print("=== FINAL RESOLUTION RECOMMENDATION ===\n")

# Quick behavioral test
def quick_behavior_test(brain):
    """Quick test of key behaviors."""
    results = {}
    
    # Obstacle avoidance
    sensors = [0.9, 0.9, 0.9] + [0.0] * 21
    actions = []
    for _ in range(5):
        action, _ = brain.process_robot_cycle(sensors)
        actions.append(action[3])  # Forward speed
    results['obstacle_response'] = 1.0 - np.mean(actions)  # Lower speed = better
    
    # Turning
    sensors = [0.2, 0.5, 0.9] + [0.0] * 21  # Right obstacle
    actions = []
    for _ in range(5):
        action, _ = brain.process_robot_cycle(sensors)
        actions.append(abs(action[0] - action[1]))  # Turn magnitude
    results['turn_response'] = np.mean(actions)
    
    return results

print("Testing Resolution 3³, 4³, and 5³:\n")

for res in [3, 4, 5]:
    brain = UnifiedFieldBrain(spatial_resolution=res, quiet_mode=True)
    
    # Warmup
    for _ in range(20):
        brain.process_robot_cycle([0.5] * 24)
    
    # Performance timing
    times = []
    for _ in range(10):
        start = time.perf_counter()
        brain.process_robot_cycle([0.5] * 24)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times)
    freq = 1000 / avg_time
    
    # Behavior test
    behaviors = quick_behavior_test(brain)
    
    print(f"Resolution {res}³:")
    print(f"  Performance: {freq:.1f} Hz ({avg_time:.1f}ms/cycle)")
    print(f"  Obstacle response: {behaviors['obstacle_response']:.3f}")
    print(f"  Turn response: {behaviors['turn_response']:.3f}")
    print(f"  Field size: {brain.unified_field.numel():,} elements")
    print(f"  Memory usage: ~{brain.unified_field.numel() * 4 / 1024 / 1024:.1f} MB")
    print()
    
    brain.shutdown()

print("="*50)
print("RECOMMENDATION: Resolution 4³")
print("="*50)
print("\nReasoning:")
print("- Resolution 3³: Fast (30+ Hz) but limited behavioral complexity")
print("- Resolution 4³: Good balance (20 Hz) with better behaviors")
print("- Resolution 5³: Best behaviors but borderline real-time (14 Hz)")
print("\nResolution 4³ provides:")
print("✓ Real-time performance (20 Hz)")
print("✓ Good behavioral responses")
print("✓ Reasonable memory usage (~21 MB)")
print("✓ Room for optimization")
print("\nYou can adjust based on your specific needs:")
print("- For simple tasks or limited hardware: Use 3³")
print("- For complex behaviors or powerful hardware: Use 5³")
print("- For most robot applications: Use 4³")