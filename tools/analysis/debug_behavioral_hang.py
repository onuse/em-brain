#!/usr/bin/env python3
"""
Debug Behavioral Test Framework Hang
Minimal reproduction to identify where the framework hangs.
"""

import sys
import time
import traceback
import signal
sys.path.append('server/tools/testing')

def timeout_handler(signum, frame):
    print("🚨 TIMEOUT! Stack trace at hang:")
    traceback.print_stack()
    sys.exit(1)

def debug_minimal_test():
    """Step through behavioral test framework components to find hang."""
    print("🔍 Debugging Behavioral Test Framework Hang")
    print("=" * 60)
    
    try:
        # Set timeout to catch infinite loops
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        print("Step 1: Import framework...")
        from behavioral_test_framework import BehavioralTestFramework
        print("✅ Import successful")
        
        print("Step 2: Create framework...")
        framework = BehavioralTestFramework(quiet_mode=True)
        print("✅ Framework created")
        
        print("Step 3: Create brain...")
        config = {
            'brain': {
                'type': 'field',
                'sensory_dim': 16,
                'motor_dim': 4,
                'spatial_resolution': 6,  # Very small
            },
            'memory': {'enable_persistence': False}
        }
        brain = framework.create_brain(config)
        print("✅ Brain created")
        
        print("Step 4: Test single cycle...")
        pattern = [0.5] * 16
        action, brain_state = brain.process_sensory_input(pattern)
        print(f"✅ Single cycle: {len(action)}D action, confidence: {brain_state.get('prediction_confidence', 0.0):.3f}")
        
        print("Step 5: Test 10 cycles...")
        start = time.time()
        for i in range(10):
            action, brain_state = brain.process_sensory_input(pattern)
            if i % 5 == 0:
                print(f"   Cycle {i+1}: {time.time() - start:.1f}s elapsed")
        print(f"✅ 10 cycles completed in {time.time() - start:.1f}s")
        
        print("Step 6: Test prediction learning (20 cycles)...")
        start = time.time()
        score = framework.test_prediction_learning(brain, cycles=20)
        elapsed = time.time() - start
        print(f"✅ Prediction test: {score:.3f} score in {elapsed:.1f}s")
        
        signal.alarm(0)  # Cancel timeout
        print("🎉 SUCCESS: No hang detected in minimal test")
        
    except KeyboardInterrupt:
        print("🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)

def debug_full_framework():
    """Try running the actual problematic code to see where it hangs."""
    print("\n🔍 Debugging Full Framework")
    print("=" * 40)
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        print("Running test_paradigm_shifting_experiment...")
        from behavioral_test_framework import test_paradigm_shifting_experiment
        
        print("About to call test function...")
        test_paradigm_shifting_experiment()
        print("✅ test_paradigm_shifting_experiment completed")
        
        signal.alarm(0)
        print("🎉 Full framework test completed")
        
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Full framework error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_minimal_test()
    debug_full_framework()