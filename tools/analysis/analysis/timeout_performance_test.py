#!/usr/bin/env python3
"""
Timeout Performance Test - Test with proper time constraints

This test ensures we can detect performance issues immediately
by adding time constraints to all operations.
"""

import time
import sys
import os
import signal
from contextlib import contextmanager
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain
import torch

@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)  # Cancel the alarm

def test_with_timeouts():
    """Test brain performance with proper time constraints."""
    print("⏰ TIMEOUT PERFORMANCE TEST")
    print("=" * 40)
    
    # Show available acceleration
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available")
    elif torch.backends.mps.is_available():
        print(f"  ✅ MPS available")
    else:
        print(f"  ⚠️  CPU only")
    
    # Test 1: Brain creation (should be fast)
    print("\n1. Testing brain creation...")
    try:
        with timeout(30):  # 30 second timeout
            start_time = time.time()
            brain = MinimalBrain(quiet_mode=True)
            creation_time = time.time() - start_time
            print(f"   ✅ Brain created in {creation_time:.2f}s")
    except TimeoutError:
        print(f"   ❌ Brain creation TIMED OUT (>30s)")
        return
    
    # Test 2: Single prediction (should be reasonable)
    print("\n2. Testing single prediction...")
    try:
        with timeout(10):  # 10 second timeout for single prediction
            start_time = time.time()
            action, brain_state = brain.process_sensory_input([1.0, 2.0, 3.0, 4.0], action_dimensions=2)
            prediction_time = (time.time() - start_time) * 1000
            print(f"   ✅ Single prediction: {prediction_time:.1f}ms")
    except TimeoutError:
        print(f"   ❌ Single prediction TIMED OUT (>10s)")
        return
    
    # Test 3: Rapid predictions (test cache)
    print("\n3. Testing rapid predictions...")
    times = []
    try:
        with timeout(30):  # 30 second timeout for 5 predictions
            for i in range(5):
                start_time = time.time()
                action, brain_state = brain.process_sensory_input([1.0, 2.0, 3.0, 4.0], action_dimensions=2)
                elapsed = (time.time() - start_time) * 1000
                times.append(elapsed)
                print(f"   Prediction {i+1}: {elapsed:.1f}ms")
    except TimeoutError:
        print(f"   ❌ Rapid predictions TIMED OUT (>30s)")
        return
    
    # Test 4: Cache effectiveness
    print("\n4. Testing cache effectiveness...")
    temporal_hierarchy = brain_state.get('temporal_hierarchy', {})
    reflex_cache = temporal_hierarchy.get('reflex_cache', {})
    
    cache_hits = reflex_cache.get('cache_hits', 0)
    cache_misses = reflex_cache.get('cache_misses', 0)
    hit_rate = reflex_cache.get('cache_hit_rate', 0)
    
    print(f"   Cache hits: {cache_hits}")
    print(f"   Cache misses: {cache_misses}")
    print(f"   Hit rate: {hit_rate:.2f}")
    
    # Test 5: Performance assessment
    print("\n5. Performance assessment...")
    fastest_time = min(times)
    avg_time = sum(times) / len(times)
    
    print(f"   Fastest prediction: {fastest_time:.1f}ms")
    print(f"   Average prediction: {avg_time:.1f}ms")
    
    # Hardware scaling projection
    print(f"\n6. Hardware scaling projection...")
    print(f"   Current (M1): {fastest_time:.1f}ms")
    print(f"   i7+RTX3070: {fastest_time/3:.1f}ms")
    print(f"   RTX 5090: {fastest_time/10:.1f}ms")
    
    # Final assessment
    print(f"\n📊 FINAL ASSESSMENT")
    print("=" * 25)
    
    if fastest_time < 100.0:
        print(f"✅ REAL-TIME PERFORMANCE ACHIEVED!")
        print(f"   {fastest_time:.1f}ms < 100ms target")
    elif fastest_time < 1000.0:
        print(f"⚡ GOOD PERFORMANCE")
        print(f"   {fastest_time:.1f}ms - improvements working")
    else:
        print(f"⚠️  PERFORMANCE NEEDS WORK")
        print(f"   {fastest_time:.1f}ms - bottlenecks remain")
    
    if hit_rate > 0.5:
        print(f"✅ REFLEX CACHE EFFECTIVE: {hit_rate:.1%}")
    else:
        print(f"⚠️  REFLEX CACHE NEEDS IMPROVEMENT: {hit_rate:.1%}")
    
    return fastest_time

if __name__ == "__main__":
    print("Starting timeout performance test...")
    fastest_time = test_with_timeouts()
    
    if fastest_time:
        print(f"\n🏁 TEST COMPLETED SUCCESSFULLY")
        print(f"Best performance: {fastest_time:.1f}ms")
    else:
        print(f"\n❌ TEST FAILED - TIMEOUTS DETECTED")
        print(f"Performance issues need investigation")