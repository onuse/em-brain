#!/usr/bin/env python3
"""
Lifetime Performance Test - Check if 2s delay happens only once

This test checks if the 2-second delay is:
1. One-time initialization cost (happens once per brain lifetime)
2. Per-pattern cost (happens once per unique pattern)
3. Recurring cost (happens multiple times)
"""

import time
import sys
import os
import signal
from contextlib import contextmanager
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain

@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def test_lifetime_performance():
    """Test if the 2s delay happens once or multiple times."""
    print("🧠 BRAIN LIFETIME PERFORMANCE TEST")
    print("=" * 45)
    
    # Create brain
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
        print("✅ Brain created successfully")
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return
    
    # Test patterns
    patterns = [
        [1.0, 2.0, 3.0, 4.0],  # Pattern A
        [5.0, 6.0, 7.0, 8.0],  # Pattern B  
        [9.0, 8.0, 7.0, 6.0],  # Pattern C
        [1.0, 2.0, 3.0, 4.0],  # Pattern A (repeat)
    ]
    
    pattern_names = ["Pattern A", "Pattern B", "Pattern C", "Pattern A (repeat)"]
    
    print("\nTesting multiple patterns and repeats...")
    
    prediction_times = []
    fast_path_usage = []
    
    try:
        with timeout(300):  # 5 minute timeout for all tests
            for i, (pattern, name) in enumerate(zip(patterns, pattern_names)):
                print(f"\n{i+1}. Testing {name}: {pattern}")
                
                start_time = time.time()
                action, brain_state = brain.process_sensory_input(pattern, action_dimensions=2)
                elapsed = (time.time() - start_time) * 1000
                
                fast_path_used = brain_state.get('fast_path_used', False)
                strategy = brain_state.get('strategy', 'unknown')
                
                prediction_times.append(elapsed)
                fast_path_usage.append(fast_path_used)
                
                print(f"   Time: {elapsed:.1f}ms")
                print(f"   Fast path: {fast_path_used}")
                print(f"   Strategy: {strategy}")
                
                if elapsed > 1000:
                    print(f"   🚨 SLOW PREDICTION (>1s)")
                elif elapsed < 10:
                    print(f"   ⚡ VERY FAST PREDICTION (<10ms)")
                elif elapsed < 100:
                    print(f"   ✅ FAST PREDICTION (<100ms)")
    
    except TimeoutError:
        print("❌ Lifetime performance test timed out!")
        return
    
    # Analyze the results
    print(f"\n📊 LIFETIME PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    slow_predictions = [t for t in prediction_times if t > 1000]
    fast_predictions = [t for t in prediction_times if t < 100]
    
    print(f"Total predictions: {len(prediction_times)}")
    print(f"Slow predictions (>1s): {len(slow_predictions)}")
    print(f"Fast predictions (<100ms): {len(fast_predictions)}")
    print(f"Fast path usage: {sum(fast_path_usage)}/{len(fast_path_usage)}")
    
    print(f"\nPrediction times:")
    for i, (name, time_ms, fast_path) in enumerate(zip(pattern_names, prediction_times, fast_path_usage)):
        status = "FAST PATH" if fast_path else "slow path"
        print(f"  {i+1}. {name}: {time_ms:.1f}ms ({status})")
    
    # Test more patterns to see pattern
    print(f"\n🔄 TESTING ADDITIONAL NOVEL PATTERNS")
    print("=" * 45)
    
    novel_patterns = [
        [0.1, 0.2, 0.3, 0.4],  # Novel pattern 1
        [10.0, 11.0, 12.0, 13.0],  # Novel pattern 2
        [0.1, 0.2, 0.3, 0.4],  # Novel pattern 1 (repeat)
    ]
    
    novel_names = ["Novel 1", "Novel 2", "Novel 1 (repeat)"]
    
    try:
        with timeout(180):  # 3 minute timeout
            for i, (pattern, name) in enumerate(zip(novel_patterns, novel_names)):
                print(f"\n{len(patterns)+i+1}. Testing {name}: {pattern}")
                
                start_time = time.time()
                action, brain_state = brain.process_sensory_input(pattern, action_dimensions=2)
                elapsed = (time.time() - start_time) * 1000
                
                fast_path_used = brain_state.get('fast_path_used', False)
                
                print(f"   Time: {elapsed:.1f}ms")
                print(f"   Fast path: {fast_path_used}")
                
                if elapsed > 1000:
                    print(f"   🚨 SLOW (first time processing)")
                else:
                    print(f"   ⚡ FAST (cached or simple)")
    
    except TimeoutError:
        print("❌ Novel pattern testing timed out!")
    
    # Get cache statistics
    temporal_hierarchy = brain_state.get('temporal_hierarchy', {})
    reflex_cache = temporal_hierarchy.get('reflex_cache', {})
    
    print(f"\n📈 CACHE STATISTICS")
    print("=" * 25)
    print(f"Cache size: {reflex_cache.get('cache_size', 0)}")
    print(f"Cache hits: {reflex_cache.get('cache_hits', 0)}")
    print(f"Cache misses: {reflex_cache.get('cache_misses', 0)}")
    print(f"Hit rate: {reflex_cache.get('cache_hit_rate', 0):.2f}")
    
    # Final analysis
    print(f"\n🎯 LIFETIME DELAY ANALYSIS")
    print("=" * 35)
    
    if len(slow_predictions) == 0:
        print("✅ NO SLOW PREDICTIONS DETECTED")
        print("   All predictions were fast - delay may be one-time only")
    elif len(slow_predictions) == 1:
        print("✅ ONLY ONE SLOW PREDICTION")
        print("   The 2s delay appears to be one-time initialization")
        print("   All subsequent predictions use fast path")
    else:
        print(f"⚠️  MULTIPLE SLOW PREDICTIONS ({len(slow_predictions)})")
        print("   The delay may happen once per unique pattern")
        print("   Or there may be other factors causing slowness")
    
    return prediction_times, fast_path_usage

if __name__ == "__main__":
    print("Testing if 2s delay happens once or multiple times...")
    times, fast_path = test_lifetime_performance()
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n🏁 LIFETIME TEST COMPLETE")
        print(f"Average prediction time: {avg_time:.1f}ms")
        print(f"Fast path usage rate: {sum(fast_path)/len(fast_path):.1%}")
        
        slow_count = len([t for t in times if t > 1000])
        if slow_count <= 1:
            print(f"🎉 CONFIRMED: 2s delay is one-time only!")
        else:
            print(f"⚠️  {slow_count} slow predictions detected")
    else:
        print(f"\n❌ LIFETIME TEST FAILED")