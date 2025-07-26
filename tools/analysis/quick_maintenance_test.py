#!/usr/bin/env python3
"""
Quick test to show impact of maintenance frequency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import time
import numpy as np
from brain_factory import BrainFactory


def main():
    print("🔬 Quick Maintenance Frequency Test")
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
    
    # Test 1: No maintenance
    print("\n1️⃣ No maintenance:")
    brain1 = BrainFactory(config=config, quiet_mode=True, enable_logging=False)
    
    times1 = []
    for i in range(50):
        sensory_input = [np.sin(i * 0.1), np.cos(i * 0.1), 0.0] + [0.1] * 13
        start = time.time()
        action, state = brain1.process_sensory_input(sensory_input)
        times1.append((time.time() - start) * 1000)
    
    print(f"   First 10 cycles avg: {np.mean(times1[:10]):.1f}ms")
    print(f"   Last 10 cycles avg: {np.mean(times1[-10:]):.1f}ms")
    print(f"   Topology regions: {len(brain1.brain.topology_regions)}")
    brain1.shutdown()
    
    # Test 2: Maintenance every 20 cycles
    print("\n2️⃣ Maintenance every 20 cycles:")
    brain2 = BrainFactory(config=config, quiet_mode=True, enable_logging=False)
    
    times2 = []
    for i in range(50):
        sensory_input = [np.sin(i * 0.1), np.cos(i * 0.1), 0.0] + [0.1] * 13
        start = time.time()
        action, state = brain2.process_sensory_input(sensory_input)
        times2.append((time.time() - start) * 1000)
        
        if i > 0 and i % 20 == 0:
            brain2.brain._perform_field_maintenance()
            print(f"   Maintenance at cycle {i}")
    
    print(f"   First 10 cycles avg: {np.mean(times2[:10]):.1f}ms")
    print(f"   Last 10 cycles avg: {np.mean(times2[-10:]):.1f}ms")
    print(f"   Topology regions: {len(brain2.brain.topology_regions)}")
    brain2.shutdown()
    
    # Test 3: Aggressive maintenance every 10 cycles
    print("\n3️⃣ Maintenance every 10 cycles:")
    brain3 = BrainFactory(config=config, quiet_mode=True, enable_logging=False)
    
    times3 = []
    for i in range(50):
        sensory_input = [np.sin(i * 0.1), np.cos(i * 0.1), 0.0] + [0.1] * 13
        start = time.time()
        action, state = brain3.process_sensory_input(sensory_input)
        times3.append((time.time() - start) * 1000)
        
        if i > 0 and i % 10 == 0:
            brain3.brain._perform_field_maintenance()
            print(f"   Maintenance at cycle {i}")
    
    print(f"   First 10 cycles avg: {np.mean(times3[:10]):.1f}ms")
    print(f"   Last 10 cycles avg: {np.mean(times3[-10:]):.1f}ms")
    print(f"   Topology regions: {len(brain3.brain.topology_regions)}")
    brain3.shutdown()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print("Maintenance | First 10 | Last 10 | Performance")
    print("-" * 50)
    
    perf1 = "🚨 DEGRADED" if np.mean(times1[-10:]) > 150 else "✅ GOOD"
    perf2 = "🚨 DEGRADED" if np.mean(times2[-10:]) > 150 else "✅ GOOD"
    perf3 = "🚨 DEGRADED" if np.mean(times3[-10:]) > 150 else "✅ GOOD"
    
    print(f"None        | {np.mean(times1[:10]):8.1f} | {np.mean(times1[-10:]):7.1f} | {perf1}")
    print(f"Every 20    | {np.mean(times2[:10]):8.1f} | {np.mean(times2[-10:]):7.1f} | {perf2}")
    print(f"Every 10    | {np.mean(times3[:10]):8.1f} | {np.mean(times3[-10:]):7.1f} | {perf3}")


if __name__ == "__main__":
    main()