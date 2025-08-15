#!/usr/bin/env python3
"""
Debug Field Evolution Performance
Profile specific parts of the field evolution to find bottlenecks.
"""

import sys
import time
import signal
import traceback
sys.path.append('server/tools/testing')

def timeout_handler(signum, frame):
    print("🚨 TIMEOUT! Stack trace:")
    traceback.print_stack()
    sys.exit(1)

def profile_field_operations():
    """Profile individual field operations to find bottlenecks."""
    print("🔍 Profiling Field Operations")
    print("=" * 50)
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)  # 15 second timeout
        
        from behavioral_test_framework import BehavioralTestFramework
        framework = BehavioralTestFramework(quiet_mode=True)
        
        config = {
            'brain': {
                'type': 'field',
                'sensory_dim': 16,
                'motor_dim': 4,
                'spatial_resolution': 4,  # Even smaller for debugging
            },
            'memory': {'enable_persistence': False}
        }
        brain = framework.create_brain(config)
        pattern = [0.5] * 16
        
        # Profile individual cycle components
        print("Testing single cycle components...")
        
        # First cycle (includes initialization overhead)
        start = time.time()
        action1, state1 = brain.process_sensory_input(pattern)
        first_cycle_time = time.time() - start
        print(f"  First cycle: {first_cycle_time:.3f}s")
        
        # Second cycle (should be faster)
        start = time.time()
        action2, state2 = brain.process_sensory_input(pattern)
        second_cycle_time = time.time() - start
        print(f"  Second cycle: {second_cycle_time:.3f}s")
        
        # Third cycle 
        start = time.time()
        action3, state3 = brain.process_sensory_input(pattern)
        third_cycle_time = time.time() - start
        print(f"  Third cycle: {third_cycle_time:.3f}s")
        
        signal.alarm(0)
        
        if second_cycle_time > 1.0:
            print("🚨 PROBLEM: Even with 4³ field, cycles are >1s")
        elif second_cycle_time > 0.5:
            print("⚠️  WARNING: Cycles are still quite slow")
        else:
            print("✅ Cycle performance looks reasonable")
            
        # Try testing a few more cycles
        signal.alarm(10)  # 10 second timeout for this test
        print("\nTesting 5 more cycles...")
        start = time.time()
        for i in range(5):
            action, state = brain.process_sensory_input(pattern)
        batch_time = time.time() - start
        print(f"  5 cycles: {batch_time:.3f}s ({batch_time/5:.3f}s per cycle)")
        signal.alarm(0)
        
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Error during profiling: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    profile_field_operations()