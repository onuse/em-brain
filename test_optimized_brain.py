#!/usr/bin/env python3
"""
Test GPU-Optimized Brain Performance

Verifies that the optimized brain:
1. Runs much faster (<200ms per cycle)
2. Preserves ALL intelligence features
3. Shows proper boredom/exploration behavior
"""

import sys
import os
sys.path.append('/mnt/c/Users/glimm/Documents/Projects/em-brain/server/src')

import torch
import time
import numpy as np
from brains.field.gpu_optimized_brain import GPUOptimizedFieldBrain

def test_performance_and_behavior():
    """Test that optimization preserves all behaviors."""
    
    print("\n" + "="*80)
    print("GPU-OPTIMIZED BRAIN TEST")
    print("="*80)
    
    # Create brain with production size
    brain = GPUOptimizedFieldBrain(
        sensory_dim=16,
        motor_dim=5,
        spatial_size=96,  # Production size
        channels=192,     # Production channels
        quiet_mode=False
    )
    
    print("\n🔬 Testing Intelligence Features:")
    print("-" * 50)
    
    # Test 1: Boredom detection with static input
    print("\n1. Testing BOREDOM with static input...")
    static_input = [0.5] * 16
    boredom_detected = False
    
    for i in range(20):
        motor_output, telemetry = brain.process(static_input)
        if "BORED" in telemetry['motivation'] or telemetry['exploring']:
            boredom_detected = True
            print(f"   ✅ Boredom detected at cycle {i+1}: {telemetry['motivation']}")
            break
    
    if not boredom_detected:
        print("   ⚠️  WARNING: Boredom not detected with static input!")
    
    # Test 2: Response to varying input
    print("\n2. Testing RESPONSE to varying input...")
    brain.reset()
    
    motor_responses = []
    for i in range(10):
        varying_input = [0.5 + 0.3 * np.sin(i * 0.5)] * 16
        motor_output, telemetry = brain.process(varying_input)
        motor_responses.append(np.linalg.norm(motor_output))
    
    motor_variance = np.var(motor_responses)
    if motor_variance > 0.001:
        print(f"   ✅ Motor variation detected: {motor_variance:.4f}")
    else:
        print(f"   ⚠️  WARNING: Low motor variance: {motor_variance:.6f}")
    
    # Test 3: Exploration when prediction fails
    print("\n3. Testing EXPLORATION on prediction error...")
    brain.reset()
    
    exploration_count = 0
    for i in range(15):
        # Random input to cause prediction errors
        random_input = [np.random.random() for _ in range(16)]
        motor_output, telemetry = brain.process(random_input)
        if telemetry['exploring']:
            exploration_count += 1
    
    exploration_rate = exploration_count / 15
    print(f"   Exploration rate: {exploration_rate:.1%}")
    if exploration_rate > 0.3:
        print(f"   ✅ Healthy exploration behavior")
    else:
        print(f"   ⚠️  WARNING: Low exploration rate")
    
    # Test 4: Intrinsic tension mechanisms
    print("\n4. Testing INTRINSIC TENSIONS...")
    brain.reset()
    
    # Starve the brain of input
    zero_input = [0.0] * 16
    starvation_detected = False
    
    for i in range(10):
        motor_output, telemetry = brain.process(zero_input)
        if "STARVED" in telemetry['motivation']:
            starvation_detected = True
            print(f"   ✅ Starvation detected at cycle {i+1}: {telemetry['motivation']}")
            break
    
    if not starvation_detected:
        print("   ⚠️  WARNING: Starvation response not detected!")
    
    # Performance Test
    print("\n" + "="*80)
    print("PERFORMANCE TEST")
    print("="*80)
    
    print("\n🔥 Warming up...")
    for _ in range(3):
        brain.process([0.5] * 16)
    
    print("\n📊 Testing 20 cycles with varying inputs...")
    times = []
    motivations = set()
    
    for i in range(20):
        # Vary input to test different brain states
        if i < 5:
            sensory_input = [0.5] * 16  # Static
        elif i < 10:
            sensory_input = [0.5 + 0.3 * np.sin(i)] * 16  # Varying
        elif i < 15:
            sensory_input = [np.random.random() for _ in range(16)]  # Random
        else:
            sensory_input = [0.0] * 16  # Zero
        
        start = time.perf_counter()
        motor_output, telemetry = brain.process(sensory_input)
        end = time.perf_counter()
        
        cycle_time = (end - start) * 1000
        times.append(cycle_time)
        motivations.add(telemetry['motivation'])
        
        if i % 5 == 0:
            print(f"  Cycle {i+1:2d}: {cycle_time:6.1f}ms - {telemetry['motivation']}")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\n📈 Performance Results:")
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Min:     {min_time:.1f}ms")
    print(f"  Max:     {max_time:.1f}ms")
    
    print(f"\n🧠 Behavioral Diversity:")
    print(f"  Unique motivational states: {len(motivations)}")
    for motivation in sorted(motivations):
        print(f"    - {motivation}")
    
    # Final verdict
    print("\n" + "="*80)
    print("OPTIMIZATION VERDICT")
    print("="*80)
    
    if avg_time < 200:
        print(f"✅ PERFORMANCE: {avg_time:.1f}ms average (TARGET: <200ms)")
    else:
        print(f"⚠️  PERFORMANCE: {avg_time:.1f}ms average (TARGET: <200ms)")
        print(f"   Still {avg_time/200:.1f}x slower than target")
    
    intelligence_score = 0
    if boredom_detected:
        intelligence_score += 1
        print("✅ BOREDOM: Detection working")
    else:
        print("❌ BOREDOM: Not detected")
    
    if motor_variance > 0.001:
        intelligence_score += 1
        print("✅ MOTORS: Responsive to input variation")
    else:
        print("❌ MOTORS: Not responsive")
    
    if exploration_rate > 0.3:
        intelligence_score += 1
        print("✅ EXPLORATION: Active when uncertain")
    else:
        print("❌ EXPLORATION: Not active enough")
    
    if starvation_detected:
        intelligence_score += 1
        print("✅ STARVATION: Intrinsic tension working")
    else:
        print("❌ STARVATION: Not detected")
    
    if len(motivations) >= 3:
        intelligence_score += 1
        print(f"✅ DIVERSITY: {len(motivations)} unique states observed")
    else:
        print(f"❌ DIVERSITY: Only {len(motivations)} states observed")
    
    print(f"\n🎯 Intelligence Score: {intelligence_score}/5")
    
    if intelligence_score == 5 and avg_time < 200:
        print("\n🏆 SUCCESS: Full intelligence preserved with target performance!")
    elif intelligence_score == 5:
        print("\n⚡ Intelligence preserved but needs more speed optimization")
    elif avg_time < 200:
        print("\n⚠️  Fast but intelligence features compromised!")
    else:
        print("\n❌ Both performance and intelligence need work")
    
    return brain

if __name__ == "__main__":
    brain = test_performance_and_behavior()