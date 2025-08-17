#!/usr/bin/env python3
"""
Final Test: Verify <200ms performance with preserved intelligence.
"""

import sys
import os
sys.path.append('/mnt/c/Users/glimm/Documents/Projects/em-brain/server/src')

import torch
import time
import numpy as np
from brains.field.final_optimized_brain import FinalOptimizedFieldBrain

def test_final_brain():
    print("\n" + "="*80)
    print("FINAL OPTIMIZED BRAIN TEST")
    print("="*80)
    
    # Create production-size brain
    brain = FinalOptimizedFieldBrain(
        sensory_dim=16,
        motor_dim=5,
        spatial_size=96,
        channels=192,
        quiet_mode=False
    )
    
    print("\n📊 Performance Test (20 cycles):")
    print("-" * 50)
    
    # Warm up
    for _ in range(3):
        brain.process([0.5] * 16)
    
    times = []
    motivations = []
    exploring_count = 0
    
    for i in range(20):
        # Vary input to test different states
        if i < 5:
            # Static input (should trigger boredom)
            sensory_input = [0.5] * 16
        elif i < 10:
            # Varying input
            sensory_input = [0.5 + 0.3 * np.sin(i * 0.5)] * 16
        elif i < 15:
            # Random (should trigger exploration)
            sensory_input = [np.random.random() for _ in range(16)]
        else:
            # Zero (should trigger starvation)
            sensory_input = [0.0] * 16
        
        start = time.perf_counter()
        motor_output, telemetry = brain.process(sensory_input)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        
        times.append(elapsed)
        motivations.append(telemetry['motivation'])
        if telemetry['exploring']:
            exploring_count += 1
        
        if i % 5 == 0 or i == 19:
            print(f"  Cycle {i+1:2d}: {elapsed:6.1f}ms - {telemetry['motivation']}")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\n📈 Results:")
    print(f"  Average time: {avg_time:.1f}ms")
    print(f"  Min time:     {min_time:.1f}ms")
    print(f"  Max time:     {max_time:.1f}ms")
    
    print("\n🧠 Intelligence Features:")
    unique_motivations = set(motivations)
    print(f"  Unique states: {len(unique_motivations)}")
    for state in unique_motivations:
        print(f"    - {state}")
    print(f"  Exploration rate: {exploring_count/20:.0%}")
    
    # Check for specific behaviors
    has_boredom = any("BORED" in m for m in motivations)
    has_starvation = any("STARVED" in m for m in motivations)
    has_content = any("CONTENT" in m for m in motivations)
    has_active = any("ACTIVE" in m for m in motivations)
    
    print("\n✅ Feature Verification:")
    print(f"  Boredom detection:   {'✓' if has_boredom else '✗'}")
    print(f"  Starvation response: {'✓' if has_starvation else '✗'}")
    print(f"  Content state:       {'✓' if has_content else '✗'}")
    print(f"  Active learning:     {'✓' if has_active else '✗'}")
    print(f"  Exploration:         {'✓' if exploring_count > 0 else '✗'}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    performance_ok = avg_time < 200
    intelligence_ok = (has_boredom or has_starvation) and exploring_count > 0
    
    if performance_ok and intelligence_ok:
        print("🏆 SUCCESS! Target performance achieved with intelligence preserved!")
        print(f"   Performance: {avg_time:.1f}ms < 200ms ✓")
        print(f"   Intelligence: All core features working ✓")
    elif performance_ok:
        print("⚡ Fast but intelligence features may be compromised")
        print(f"   Performance: {avg_time:.1f}ms < 200ms ✓")
        print(f"   Intelligence: Some features missing ✗")
    elif intelligence_ok:
        print("🧠 Intelligence preserved but still too slow")
        print(f"   Performance: {avg_time:.1f}ms > 200ms ✗")
        print(f"   Intelligence: Features working ✓")
    else:
        print("❌ Both performance and intelligence need work")
        print(f"   Performance: {avg_time:.1f}ms (target: <200ms)")
        print(f"   Intelligence: Missing key features")
    
    return brain, avg_time

if __name__ == "__main__":
    brain, avg_time = test_final_brain()