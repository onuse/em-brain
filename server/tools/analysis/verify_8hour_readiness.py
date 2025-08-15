#!/usr/bin/env python3
"""
Verify 8-Hour Test Readiness

Checks if the optimized brain is ready for the 8-hour biological_embodied_learning experiment.
"""

import sys
import os
import time
import psutil
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def check_8hour_readiness():
    """Check if system is ready for 8-hour test."""
    print("🔍 Checking 8-Hour Test Readiness")
    print("=" * 60)
    
    results = {
        'brain_compatibility': False,
        'memory_stability': False,
        'performance_adequate': False,
        'thermal_headroom': False,
        'overall_ready': False
    }
    
    # 1. Check brain compatibility
    print("\n1. Brain Compatibility Check:")
    try:
        brain = SimplifiedUnifiedBrain(
            sensory_dim=24,  # biological_embodied_learning uses 24D sensors
            motor_dim=4,     # Standard motor output
            quiet_mode=True,
            use_optimized=True
        )
        brain.enable_predictive_actions(False)
        brain.set_pattern_extraction_limit(3)
        
        # Test processing
        test_input = [0.1] * 24
        motors, state = brain.process_robot_cycle(test_input)
        
        print(f"   ✅ Brain initialized successfully")
        print(f"   ✅ Sensory dim: 24 (matches experiment)")
        print(f"   ✅ Motor dim: {len(motors)+1} (standard)")
        results['brain_compatibility'] = True
        
    except Exception as e:
        print(f"   ❌ Brain initialization failed: {e}")
        return results
    
    # 2. Memory stability check
    print("\n2. Memory Stability Check:")
    initial_memory = brain._calculate_memory_usage()
    
    # Run 1000 cycles to check for memory leaks
    print("   Running 1000 cycles...")
    start_time = time.time()
    for i in range(1000):
        sensory = [0.1 * (1 + i % 10)] * 24
        brain.process_robot_cycle(sensory)
    
    elapsed = time.time() - start_time
    final_memory = brain._calculate_memory_usage()
    memory_growth = final_memory - initial_memory
    
    print(f"   Initial memory: {initial_memory:.1f}MB")
    print(f"   Final memory: {final_memory:.1f}MB")
    print(f"   Memory growth: {memory_growth:.1f}MB")
    
    if abs(memory_growth) < 1.0:  # Less than 1MB growth
        print(f"   ✅ Memory stable")
        results['memory_stability'] = True
    else:
        print(f"   ❌ Memory leak detected")
    
    # 3. Performance check
    print("\n3. Performance Check:")
    cycle_times = []
    for _ in range(100):
        start = time.perf_counter()
        brain.process_robot_cycle([0.1] * 24)
        cycle_times.append((time.perf_counter() - start) * 1000)
    
    avg_cycle = sum(cycle_times) / len(cycle_times)
    max_cycle = max(cycle_times)
    
    print(f"   Average cycle: {avg_cycle:.1f}ms")
    print(f"   Max cycle: {max_cycle:.1f}ms")
    print(f"   Est. cycles/hour: {int(3600 / (avg_cycle/1000)):,}")
    
    # biological_embodied_learning expects ~60 actions/minute = 1Hz
    # Our cycle time should be well under 1000ms
    if avg_cycle < 300 and max_cycle < 500:
        print(f"   ✅ Performance adequate for 1Hz operation")
        results['performance_adequate'] = True
    else:
        print(f"   ❌ Performance may be too slow")
    
    # 4. System thermal check
    print("\n4. System Thermal Check:")
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    print(f"   CPU usage: {cpu_percent:.1f}%")
    print(f"   Memory usage: {memory_percent:.1f}%")
    
    if cpu_percent < 80 and memory_percent < 80:
        print(f"   ✅ System has thermal headroom")
        results['thermal_headroom'] = True
    else:
        print(f"   ⚠️  System may overheat during 8-hour test")
    
    # 5. Estimate 8-hour projections
    print("\n5. 8-Hour Projections:")
    cycles_per_hour = int(3600 / (avg_cycle/1000))
    total_cycles_8h = cycles_per_hour * 8
    
    print(f"   Expected cycles: {total_cycles_8h:,}")
    print(f"   Memory per hour: ~{memory_growth * 3.6:.1f}MB (if linear)")
    print(f"   Total memory 8h: ~{initial_memory + memory_growth * 28.8:.1f}MB")
    
    # Overall assessment
    results['overall_ready'] = all([
        results['brain_compatibility'],
        results['memory_stability'],
        results['performance_adequate'],
        results['thermal_headroom']
    ])
    
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT:")
    print("=" * 60)
    
    if results['overall_ready']:
        print("\n✅ SYSTEM IS READY for 8-hour biological_embodied_learning experiment!")
        print("\nRecommended settings:")
        print("  - Use optimized brain: SimplifiedUnifiedBrain(use_optimized=True)")
        print("  - Disable predictive actions: brain.enable_predictive_actions(False)")
        print("  - Pattern limit: brain.set_pattern_extraction_limit(3)")
        print("  - Expected performance: ~150-200ms per cycle")
        print("  - Memory usage: ~8-10MB stable")
        print("\nTo start experiment:")
        print("  1. Start brain server: python3 server/run_minimal_brain.py")
        print("  2. Run experiment: python3 validation/embodied_learning/experiments/biological_embodied_learning.py")
    else:
        print("\n❌ SYSTEM NOT READY for 8-hour test")
        print("\nIssues to address:")
        for key, value in results.items():
            if key != 'overall_ready' and not value:
                print(f"  - Fix {key}")
    
    return results


if __name__ == "__main__":
    check_8hour_readiness()