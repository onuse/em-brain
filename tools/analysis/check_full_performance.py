#!/usr/bin/env python3
"""
Check performance with full features enabled
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import time
from brain_factory import BrainFactory


def check_full_performance():
    """Check performance with all features enabled"""
    
    print("Full Feature Performance Check")
    print("=" * 60)
    
    # Test 1: Minimal configuration
    print("\n1. MINIMAL CONFIGURATION (baseline)")
    print("-" * 30)
    
    minimal_config = {
        'brain': {
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': False
        }
    }
    
    brain = BrainFactory(config=minimal_config, quiet_mode=True, enable_logging=False)
    
    cycle_times_minimal = []
    for i in range(20):
        sensory_input = [0.2] * 25
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_time = (time.time() - start) * 1000
        cycle_times_minimal.append(cycle_time)
    
    avg_minimal = sum(cycle_times_minimal) / len(cycle_times_minimal)
    print(f"   Average cycle time: {avg_minimal:.1f}ms")
    brain.shutdown()
    
    # Test 2: With logging enabled
    print("\n2. WITH LOGGING ENABLED")
    print("-" * 30)
    
    logging_config = {
        'brain': {
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': False
        }
    }
    
    brain = BrainFactory(config=logging_config, quiet_mode=True, enable_logging=True)
    
    cycle_times_logging = []
    for i in range(20):
        sensory_input = [0.2] * 25
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_time = (time.time() - start) * 1000
        cycle_times_logging.append(cycle_time)
    
    avg_logging = sum(cycle_times_logging) / len(cycle_times_logging)
    print(f"   Average cycle time: {avg_logging:.1f}ms")
    print(f"   Overhead: +{avg_logging - avg_minimal:.1f}ms")
    brain.shutdown()
    
    # Test 3: With persistence enabled
    print("\n3. WITH PERSISTENCE ENABLED")
    print("-" * 30)
    
    persistence_config = {
        'brain': {
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': True
        }
    }
    
    brain = BrainFactory(config=persistence_config, quiet_mode=True, enable_logging=False)
    
    cycle_times_persistence = []
    for i in range(20):
        sensory_input = [0.2] * 25
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_time = (time.time() - start) * 1000
        cycle_times_persistence.append(cycle_time)
    
    avg_persistence = sum(cycle_times_persistence) / len(cycle_times_persistence)
    print(f"   Average cycle time: {avg_persistence:.1f}ms")
    print(f"   Overhead: +{avg_persistence - avg_minimal:.1f}ms")
    brain.shutdown()
    
    # Test 4: Full features
    print("\n4. FULL FEATURES (logging + persistence)")
    print("-" * 30)
    
    full_config = {
        'brain': {
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': True
        }
    }
    
    brain = BrainFactory(config=full_config, quiet_mode=True, enable_logging=True)
    
    cycle_times_full = []
    for i in range(20):
        sensory_input = [0.2] * 25
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_time = (time.time() - start) * 1000
        cycle_times_full.append(cycle_time)
    
    avg_full = sum(cycle_times_full) / len(cycle_times_full)
    print(f"   Average cycle time: {avg_full:.1f}ms")
    print(f"   Total overhead: +{avg_full - avg_minimal:.1f}ms")
    brain.shutdown()
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"Minimal:           {avg_minimal:.1f}ms")
    print(f"With logging:      {avg_logging:.1f}ms (+{avg_logging - avg_minimal:.1f}ms)")
    print(f"With persistence:  {avg_persistence:.1f}ms (+{avg_persistence - avg_minimal:.1f}ms)")
    print(f"Full features:     {avg_full:.1f}ms (+{avg_full - avg_minimal:.1f}ms)")
    
    print("\nRECOMMENDATIONS:")
    
    if avg_full > 1000:
        print("⚠️ SEVERE PERFORMANCE ISSUE WITH FULL FEATURES")
        print("- Disable logging and persistence for validation tests")
        print("- Socket timeouts must be >10 seconds")
        print("- Consider reducing spatial resolution to 3³")
    elif avg_full > 500:
        print("⚠️ PERFORMANCE DEGRADED WITH FULL FEATURES")
        print("- Socket timeouts of 10 seconds should work")
        print("- Consider disabling logging for better performance")
        print("- Persistence has minimal impact")
    elif avg_full > 150:
        print("⚠️ MARGINAL PERFORMANCE WITH FULL FEATURES")
        print("- Just meeting biological timescale requirements")
        print("- Current socket timeouts (10s) are appropriate")
    else:
        print("✅ GOOD PERFORMANCE WITH FULL FEATURES")
        print("- All features can be enabled without issues")
        print("- Current settings are appropriate")
    
    return avg_minimal, avg_full


if __name__ == "__main__":
    check_full_performance()