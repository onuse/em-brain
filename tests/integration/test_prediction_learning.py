#!/usr/bin/env python3
"""
Test Prediction Learning Function
Test the specific function that was hanging in the behavioral framework.
"""

import sys
import time
import signal
import traceback
sys.path.append('server/tools/testing')

def timeout_handler(signum, frame):
    print("🚨 TIMEOUT! Function taking too long")
    traceback.print_stack()
    sys.exit(1)

def test_prediction_learning_directly():
    """Test the prediction learning function that was hanging."""
    print("🔍 Testing Prediction Learning Function")
    print("=" * 50)
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        from behavioral_test_framework import BehavioralTestFramework
        framework = BehavioralTestFramework(quiet_mode=True)
        
        config = {
            'brain': {
                'type': 'field',
                'sensory_dim': 16,
                'motor_dim': 4,
                'spatial_resolution': 6,  # Small field
            },
            'memory': {'enable_persistence': False}
        }
        brain = framework.create_brain(config)
        
        print("Testing prediction learning with different cycle counts...")
        
        # Test with 10 cycles
        start = time.time()
        score_10 = framework.test_prediction_learning(brain, cycles=10)
        time_10 = time.time() - start
        print(f"  10 cycles: {score_10:.3f} score in {time_10:.1f}s")
        
        # Test with 20 cycles
        start = time.time() 
        score_20 = framework.test_prediction_learning(brain, cycles=20)
        time_20 = time.time() - start
        print(f"  20 cycles: {score_20:.3f} score in {time_20:.1f}s")
        
        # Test with 50 cycles
        start = time.time()
        score_50 = framework.test_prediction_learning(brain, cycles=50)
        time_50 = time.time() - start
        print(f"  50 cycles: {score_50:.3f} score in {time_50:.1f}s")
        
        # Test with 100 cycles (this was hanging before)
        print("Testing 100 cycles (the problematic case)...")
        start = time.time()
        score_100 = framework.test_prediction_learning(brain, cycles=100)
        time_100 = time.time() - start
        print(f"  100 cycles: {score_100:.3f} score in {time_100:.1f}s")
        
        signal.alarm(0)
        
        print(f"\n📊 Performance Summary:")
        print(f"  10 cycles: {time_10:.1f}s ({time_10/10:.3f}s per cycle)")
        print(f"  20 cycles: {time_20:.1f}s ({time_20/20:.3f}s per cycle)")  
        print(f"  50 cycles: {time_50:.1f}s ({time_50/50:.3f}s per cycle)")
        print(f"  100 cycles: {time_100:.1f}s ({time_100/100:.3f}s per cycle)")
        
        if time_100 < 30:
            print("✅ SUCCESS: 100-cycle prediction learning now works!")
        else:
            print("⚠️  Still slow but no longer hanging")
            
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction_learning_directly()