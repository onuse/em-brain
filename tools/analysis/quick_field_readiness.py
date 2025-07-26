#!/usr/bin/env python3
"""
Quick Field Readiness Check
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import time
import numpy as np
from brain_factory import BrainFactory


def quick_field_check():
    """Quick check of field deployment readiness"""
    
    print("Quick Field Readiness Check")
    print("=" * 50)
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': False
        }
    }
    
    brain = BrainFactory(config=config, quiet_mode=True, enable_logging=False)
    
    # Check 1: Basic performance
    print("\n1. Performance check...")
    cycle_times = []
    for i in range(20):
        sensory_input = [np.random.rand() * 0.3 for _ in range(16)]
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(cycle_times)
    print(f"   Average cycle time: {avg_time:.1f}ms")
    performance_ok = avg_time < 150
    
    # Check 2: Motor smoothing
    print("\n2. Motor smoothing check...")
    prev_action = None
    max_change = 0
    
    for i in range(10):
        sensory_input = [0.8 if i % 2 == 0 else 0.1, 0.1, 0.1] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
        
        if prev_action is not None:
            change = np.max(np.abs(np.array(action[:4]) - np.array(prev_action[:4])))
            max_change = max(max_change, change)
        
        prev_action = action
    
    print(f"   Max motor change: {max_change:.3f}")
    smoothing_ok = max_change < 0.3
    
    # Check 3: Gradient following
    print("\n3. Exploration behavior check...")
    speeds = []
    for i in range(15):
        sensory_input = [0.1, 0.1, 0.1] + [0.0] * 13  # Clear path
        action, state = brain.process_sensory_input(sensory_input)
        speeds.append(action[1])  # Forward speed
    
    speed_var = np.var(speeds)
    print(f"   Speed variance: {speed_var:.4f}")
    print(f"   Speed values: {[f'{s:.3f}' for s in speeds[:5]]}")  # Show first 5
    print(f"   Speed range: {np.min(speeds):.3f} to {np.max(speeds):.3f}")
    exploration_ok = speed_var > 0.0001 or (np.max(speeds) - np.min(speeds)) > 0.001
    
    # Check 4: Memory formation
    print("\n4. Memory system check...")
    initial_regions = len(brain.brain.topology_regions)
    
    # Create distinct pattern
    for i in range(5):
        sensory_input = [0.7, 0.2, 0.1] + [0.5] * 13
        action, state = brain.process_sensory_input(sensory_input)
    
    final_regions = len(brain.brain.topology_regions)
    print(f"   Regions formed: {final_regions - initial_regions}")
    memory_ok = final_regions > initial_regions
    
    brain.shutdown()
    
    # Summary
    print("\n" + "=" * 50)
    print("QUICK READINESS SUMMARY")
    print("=" * 50)
    
    checks = {
        "Performance (<150ms)": performance_ok,
        "Motor smoothing": smoothing_ok,
        "Exploration behavior": exploration_ok,
        "Memory formation": memory_ok
    }
    
    all_ok = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_ok = False
    
    if all_ok:
        print("\n✅ BRAIN IS READY FOR FIELD TESTING!")
    else:
        print("\n⚠️  Some checks failed - review before deployment")
    
    return all_ok


if __name__ == "__main__":
    quick_field_check()