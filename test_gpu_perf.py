#!/usr/bin/env python3
"""
GPU Performance Test for Field Brain

Tests the actual production-size brain to identify bottlenecks.
"""

import sys
import os
sys.path.append('/mnt/c/Users/glimm/Documents/Projects/em-brain/server/src')

import torch
import time
from brains.field.truly_minimal_brain import TrulyMinimalBrain

def test_performance():
    """Test brain performance with production settings."""
    
    print("\n" + "="*80)
    print("FIELD BRAIN GPU PERFORMANCE TEST")
    print("="*80)
    
    # Create brain with production size
    brain = TrulyMinimalBrain(
        sensory_dim=16,
        motor_dim=5,
        spatial_size=96,  # Production size
        channels=192,     # Production channels
        quiet_mode=False
    )
    
    print("\n🔥 Warming up...")
    for _ in range(3):
        sensory_input = [0.5] * 16
        brain.process(sensory_input)
    
    print("\n📊 Testing 10 cycles...")
    times = []
    
    for i in range(10):
        sensory_input = [0.5 + 0.1 * (i % 3)] * 16
        
        start = time.perf_counter()
        motor_output, telemetry = brain.process(sensory_input)
        end = time.perf_counter()
        
        cycle_time = (end - start) * 1000
        times.append(cycle_time)
        print(f"  Cycle {i+1}: {cycle_time:.1f}ms - {telemetry['motivation']}")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\n📈 Results:")
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Min:     {min_time:.1f}ms")
    print(f"  Max:     {max_time:.1f}ms")
    
    if avg_time > 1000:
        print(f"\n⚠️  WARNING: Brain is {avg_time/200:.1f}x slower than target (200ms)")
    elif avg_time > 200:
        print(f"\n⚡ Brain is {avg_time/200:.1f}x slower than target (200ms)")
    else:
        print("\n✅ Brain meets performance target!")
    
    return brain

if __name__ == "__main__":
    brain = test_performance()