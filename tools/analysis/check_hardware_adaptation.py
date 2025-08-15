#!/usr/bin/env python3
"""
Check hardware adaptation settings
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import time
from brain_factory import BrainFactory


def check_hardware_adaptation():
    """Check current hardware adaptation settings"""
    
    print("Hardware Adaptation Check")
    print("=" * 60)
    
    config = {
        'brain': {
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': False
        }
    }
    
    # Create brain to see hardware detection
    print("\n1. Creating brain to detect hardware...")
    brain = BrainFactory(config=config, quiet_mode=False, enable_logging=False)
    
    print(f"\n2. Configured settings:")
    print(f"   Spatial resolution: {brain.brain.spatial_resolution}³")
    print(f"   Gradient following: {brain.brain.gradient_following_strength}")
    print(f"   Motor smoothing: {brain.brain.motor_smoothing_factor}")
    
    print(f"\n3. Testing actual cycle times...")
    cycle_times = []
    
    for i in range(10):
        sensory_input = [0.2] * 25  # 25 is expected sensory dimension
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_time = (time.time() - start) * 1000
        cycle_times.append(cycle_time)
        print(f"   Cycle {i}: {cycle_time:.1f}ms")
    
    avg_time = sum(cycle_times) / len(cycle_times)
    max_time = max(cycle_times)
    min_time = min(cycle_times)
    
    print(f"\n4. Performance summary:")
    print(f"   Average cycle time: {avg_time:.1f}ms")
    print(f"   Min cycle time: {min_time:.1f}ms")
    print(f"   Max cycle time: {max_time:.1f}ms")
    
    # Recommend settings
    print(f"\n5. Recommendations:")
    
    if avg_time > 1000:
        print("   ⚠️ VERY SLOW PERFORMANCE DETECTED")
        print("   - Current resolution may be too high")
        print("   - Consider reducing to 3³ or enabling more aggressive optimization")
        print("   - Socket timeouts should be at least 10 seconds")
    elif avg_time > 500:
        print("   ⚠️ SLOW PERFORMANCE DETECTED")
        print("   - Performance is below biological timescale target")
        print("   - Socket timeouts of 10 seconds should work")
        print("   - Consider reducing computational load")
    elif avg_time > 150:
        print("   ⚠️ MARGINAL PERFORMANCE")
        print("   - Just meeting biological timescale requirements")
        print("   - Socket timeouts of 10 seconds are appropriate")
    else:
        print("   ✅ GOOD PERFORMANCE")
        print("   - Meeting biological timescale targets")
        print("   - Current settings are appropriate")
    
    # Check if we should reduce resolution
    if avg_time > 500 and brain.brain.spatial_resolution > 3:
        print(f"\n   💡 SUGGESTION: Reduce spatial resolution from {brain.brain.spatial_resolution}³ to 3³")
        print("      This would reduce field size by ~50% and improve performance")
    
    brain.shutdown()
    
    return avg_time


if __name__ == "__main__":
    check_hardware_adaptation()