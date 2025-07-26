#!/usr/bin/env python3
"""
Test different maintenance frequencies to find optimal balance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import time
import numpy as np
from brain_factory import BrainFactory


def test_maintenance_frequency(maintenance_interval: int):
    """Test brain performance with different maintenance intervals"""
    
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
    
    cycle_times = []
    topology_counts = []
    
    print(f"\nTesting maintenance every {maintenance_interval} cycles...")
    
    for i in range(200):
        # Varied sensory input to create topology regions
        sensory_input = [
            np.sin(i * 0.1) * np.random.rand(),
            np.cos(i * 0.1) * np.random.rand(),
            np.random.rand() * 0.1
        ] + [0.1] * 13
        
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_time = (time.time() - start) * 1000
        
        cycle_times.append(cycle_time)
        topology_counts.append(state.get('topology_regions_count', 0))
        
        # Run maintenance at specified interval
        if i > 0 and i % maintenance_interval == 0:
            # Call the internal maintenance method directly
            brain.brain._perform_field_maintenance()
            print(f"   Maintenance at cycle {i}: {len(brain.brain.topology_regions)} regions")
    
    # Calculate statistics
    avg_cycle_time = np.mean(cycle_times)
    max_cycle_time = np.max(cycle_times)
    final_regions = topology_counts[-1]
    
    brain.shutdown()
    
    return {
        'interval': maintenance_interval,
        'avg_cycle_time': avg_cycle_time,
        'max_cycle_time': max_cycle_time,
        'final_regions': final_regions,
        'avg_regions': np.mean(topology_counts)
    }


def main():
    print("🔬 Testing Optimal Maintenance Frequency")
    print("=" * 50)
    
    # Test different intervals
    intervals = [10, 20, 50, 100, 200]  # Never maintenance = 200
    results = []
    
    for interval in intervals:
        result = test_maintenance_frequency(interval)
        results.append(result)
        
        print(f"\nInterval {interval}:")
        print(f"  Avg cycle time: {result['avg_cycle_time']:.1f}ms")
        print(f"  Max cycle time: {result['max_cycle_time']:.1f}ms")
        print(f"  Avg regions: {result['avg_regions']:.1f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print("Interval | Avg Time | Max Time | Avg Regions")
    print("-" * 45)
    
    for r in results:
        status = "✅" if r['avg_cycle_time'] < 150 else "⚠️"
        print(f"{r['interval']:8d} | {r['avg_cycle_time']:8.1f} | {r['max_cycle_time']:8.1f} | {r['avg_regions']:11.1f} {status}")
    
    # Find optimal
    valid_results = [r for r in results if r['avg_cycle_time'] < 150]
    if valid_results:
        optimal = max(valid_results, key=lambda r: r['interval'])
        print(f"\n🎯 Optimal maintenance interval: every {optimal['interval']} cycles")
        print(f"   Maintains {optimal['avg_cycle_time']:.1f}ms average cycle time")
        print(f"   Keeps ~{optimal['avg_regions']:.0f} active regions")


if __name__ == "__main__":
    main()