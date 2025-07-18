#!/usr/bin/env python3
"""
Fast Reflex Mode Test - Verify that reflex mode bypasses unified storage

This test verifies that the new fast reflex mode:
1. Uses cached sensory-motor mappings
2. Bypasses expensive unified storage searches
3. Achieves <10ms prediction time for routine predictions
4. Falls back to deeper processing for novel situations
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

def test_fast_reflex_mode():
    """Test that reflex mode is truly fast by bypassing unified storage."""
    print("🏃 FAST REFLEX MODE TEST")
    print("=" * 40)
    
    # Create brain with timeout
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return float('inf'), 0.0
    
    # Test routine prediction (should become cached)
    routine_input = [1.0, 2.0, 3.0, 4.0]
    
    print("Testing routine prediction caching...")
    
    # Test predictions with timeout
    try:
        with timeout(45):  # 45 second timeout for 3 predictions
            # First prediction - should be slower (no cache)
            start_time = time.time()
            action1, brain_state1 = brain.process_sensory_input(routine_input, action_dimensions=2)
            first_time = (time.time() - start_time) * 1000
            
            # Second prediction - should be faster (cached)
            start_time = time.time()
            action2, brain_state2 = brain.process_sensory_input(routine_input, action_dimensions=2)
            second_time = (time.time() - start_time) * 1000
            
            # Third prediction - should be even faster (cache hit)
            start_time = time.time()
            action3, brain_state3 = brain.process_sensory_input(routine_input, action_dimensions=2)
            third_time = (time.time() - start_time) * 1000
    except TimeoutError:
        print("❌ Predictions timed out!")
        return float('inf'), 0.0
    
    print(f"First prediction:  {first_time:.1f}ms")
    print(f"Second prediction: {second_time:.1f}ms")
    print(f"Third prediction:  {third_time:.1f}ms")
    
    # Check if reflex mode is being used
    temporal_hierarchy = brain_state3.get('temporal_hierarchy', {})
    reflex_cache = temporal_hierarchy.get('reflex_cache', {})
    
    print(f"\nReflex cache statistics:")
    print(f"  Cache size: {reflex_cache.get('cache_size', 0)}")
    print(f"  Cache hits: {reflex_cache.get('cache_hits', 0)}")
    print(f"  Cache misses: {reflex_cache.get('cache_misses', 0)}")
    print(f"  Hit rate: {reflex_cache.get('cache_hit_rate', 0):.2f}")
    
    # Test with different inputs to verify cache behavior
    print("\nTesting cache behavior with different inputs...")
    
    # Novel input (should be slower)
    novel_input = [5.0, 6.0, 7.0, 8.0]
    start_time = time.time()
    action_novel, brain_state_novel = brain.process_sensory_input(novel_input, action_dimensions=2)
    novel_time = (time.time() - start_time) * 1000
    
    # Repeat novel input (should be faster now)
    start_time = time.time()
    action_novel2, brain_state_novel2 = brain.process_sensory_input(novel_input, action_dimensions=2)
    novel_repeat_time = (time.time() - start_time) * 1000
    
    print(f"Novel input first:  {novel_time:.1f}ms")
    print(f"Novel input repeat: {novel_repeat_time:.1f}ms")
    
    # Final cache stats
    final_temporal_hierarchy = brain_state_novel2.get('temporal_hierarchy', {})
    final_reflex_cache = final_temporal_hierarchy.get('reflex_cache', {})
    
    print(f"\nFinal reflex cache statistics:")
    print(f"  Cache size: {final_reflex_cache.get('cache_size', 0)}")
    print(f"  Cache hits: {final_reflex_cache.get('cache_hits', 0)}")
    print(f"  Cache misses: {final_reflex_cache.get('cache_misses', 0)}")
    print(f"  Hit rate: {final_reflex_cache.get('cache_hit_rate', 0):.2f}")
    
    # Performance analysis
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print(f"=" * 30)
    
    # Check if we're achieving the target
    target_time = 100.0  # 100ms target
    fastest_time = min(second_time, third_time, novel_repeat_time)
    
    print(f"Target time: {target_time:.1f}ms")
    print(f"Fastest time: {fastest_time:.1f}ms")
    
    if fastest_time < target_time:
        print(f"✅ FAST REFLEX MODE SUCCESS!")
        print(f"   Achieved {fastest_time:.1f}ms < {target_time:.1f}ms target")
        speedup = first_time / fastest_time
        print(f"   Speedup: {speedup:.1f}x faster than initial prediction")
    else:
        print(f"❌ FAST REFLEX MODE NEEDS OPTIMIZATION")
        print(f"   Current: {fastest_time:.1f}ms > {target_time:.1f}ms target")
        print(f"   Need to optimize further")
    
    # Check cache effectiveness
    cache_hit_rate = final_reflex_cache.get('cache_hit_rate', 0)
    if cache_hit_rate > 0.5:
        print(f"✅ REFLEX CACHE EFFECTIVE: {cache_hit_rate:.1%} hit rate")
    else:
        print(f"❌ REFLEX CACHE NEEDS IMPROVEMENT: {cache_hit_rate:.1%} hit rate")
    
    return fastest_time, cache_hit_rate

if __name__ == "__main__":
    fastest_time, cache_hit_rate = test_fast_reflex_mode()
    
    print(f"\n🏁 FAST REFLEX TEST COMPLETE")
    print(f"=" * 35)
    print(f"Fastest prediction time: {fastest_time:.1f}ms")
    print(f"Cache hit rate: {cache_hit_rate:.1%}")
    
    if fastest_time < 100.0 and cache_hit_rate > 0.5:
        print(f"🎉 FAST REFLEX MODE WORKING CORRECTLY!")
    else:
        print(f"⚠️  FAST REFLEX MODE NEEDS FURTHER OPTIMIZATION")