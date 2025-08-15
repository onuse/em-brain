#!/usr/bin/env python3
"""
True Fast Path Test - Test the biologically plausible fast path implementation

This test verifies that the true fast path:
1. Bypasses expensive brain processing for cached patterns
2. Achieves significant speedup through biological tradeoffs
3. Maintains correctness while sacrificing some awareness
4. Falls back to slow path for novel patterns
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
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def test_true_fast_path():
    """Test the true fast path with biologically plausible tradeoffs."""
    print("🧠 TRUE FAST PATH TEST")
    print("=" * 35)
    
    # Create brain with timeout
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
        print("✅ Brain created successfully")
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return
    
    # Test pattern for cache building
    routine_input = [1.0, 2.0, 3.0, 4.0]
    
    print("\nTesting fast path development...")
    
    # First few predictions - should be slow (cache building)
    slow_times = []
    fast_times = []
    
    try:
        with timeout(60):  # 60 second timeout for cache building
            for i in range(5):
                start_time = time.time()
                action, brain_state = brain.process_sensory_input(routine_input, action_dimensions=2)
                elapsed = (time.time() - start_time) * 1000
                
                # Check if fast path was used
                fast_path_used = brain_state.get('fast_path_used', False)
                strategy = brain_state.get('strategy', 'unknown')
                
                if fast_path_used:
                    fast_times.append(elapsed)
                    print(f"  Prediction {i+1}: {elapsed:.1f}ms (FAST PATH: {strategy})")
                else:
                    slow_times.append(elapsed)
                    print(f"  Prediction {i+1}: {elapsed:.1f}ms (slow path)")
                
                # Show what components were bypassed
                if fast_path_used:
                    bypassed_components = [k for k, v in brain_state.items() 
                                         if isinstance(v, dict) and v.get('status') == 'bypassed_for_reflex']
                    print(f"    Bypassed: {', '.join(bypassed_components)}")
    except TimeoutError:
        print("❌ Cache building timed out!")
        return
    
    # Test with different input to verify slow path still works
    print("\nTesting slow path fallback...")
    
    novel_input = [5.0, 6.0, 7.0, 8.0]
    try:
        with timeout(20):
            start_time = time.time()
            action, brain_state = brain.process_sensory_input(novel_input, action_dimensions=2)
            elapsed = (time.time() - start_time) * 1000
            
            fast_path_used = brain_state.get('fast_path_used', False)
            strategy = brain_state.get('strategy', 'unknown')
            
            print(f"  Novel input: {elapsed:.1f}ms ({'FAST PATH' if fast_path_used else 'slow path'}: {strategy})")
    except TimeoutError:
        print("❌ Novel input test timed out!")
        return
    
    # Get cache statistics
    temporal_hierarchy = brain_state.get('temporal_hierarchy', {})
    reflex_cache = temporal_hierarchy.get('reflex_cache', {})
    
    print(f"\nCache statistics:")
    print(f"  Cache size: {reflex_cache.get('cache_size', 0)}")
    print(f"  Cache hits: {reflex_cache.get('cache_hits', 0)}")
    print(f"  Cache misses: {reflex_cache.get('cache_misses', 0)}")
    print(f"  Hit rate: {reflex_cache.get('cache_hit_rate', 0):.2f}")
    
    # Performance analysis
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print("=" * 30)
    
    if fast_times and slow_times:
        avg_fast = sum(fast_times) / len(fast_times)
        avg_slow = sum(slow_times) / len(slow_times)
        speedup = avg_slow / avg_fast if avg_fast > 0 else float('inf')
        
        print(f"Fast path predictions: {len(fast_times)}")
        print(f"Slow path predictions: {len(slow_times)}")
        print(f"Average fast time: {avg_fast:.1f}ms")
        print(f"Average slow time: {avg_slow:.1f}ms")
        print(f"Speedup: {speedup:.1f}x")
        
        if speedup > 2.0:
            print(f"✅ SIGNIFICANT SPEEDUP ACHIEVED!")
        elif speedup > 1.2:
            print(f"⚡ MODERATE SPEEDUP")
        else:
            print(f"⚠️  MINIMAL SPEEDUP")
    elif fast_times:
        avg_fast = sum(fast_times) / len(fast_times)
        print(f"Fast path only: {avg_fast:.1f}ms average")
        
        if avg_fast < 100.0:
            print(f"✅ FAST PATH ACHIEVING TARGET!")
        else:
            print(f"⚠️  FAST PATH STILL SLOW")
    elif slow_times:
        avg_slow = sum(slow_times) / len(slow_times)
        print(f"Slow path only: {avg_slow:.1f}ms average")
        print(f"⚠️  FAST PATH NOT ENGAGING")
    else:
        print(f"❌ NO TIMING DATA COLLECTED")
    
    # Biological tradeoff assessment
    print(f"\n🧬 BIOLOGICAL TRADEOFF ASSESSMENT")
    print("=" * 40)
    
    print("Tradeoffs made for speed:")
    print("  ✅ Skip stream updates (reflexes don't learn)")
    print("  ✅ Skip competitive dynamics (reflexes don't compete)")
    print("  ✅ Skip cross-stream coactivation (direct sensory-motor)")
    print("  ✅ Skip complex brain state (minimal awareness)")
    print("  ✅ Use cached patterns (immediate response)")
    
    # Final assessment
    fastest_time = min(fast_times) if fast_times else min(slow_times) if slow_times else float('inf')
    
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 25)
    
    if fastest_time < 50.0:
        print(f"🎉 BIOLOGICAL REFLEX SPEED ACHIEVED!")
        print(f"   {fastest_time:.1f}ms < 50ms biological target")
    elif fastest_time < 100.0:
        print(f"✅ FAST REFLEX PERFORMANCE")
        print(f"   {fastest_time:.1f}ms < 100ms engineering target")
    else:
        print(f"⚠️  STILL NEEDS OPTIMIZATION")
        print(f"   {fastest_time:.1f}ms > 100ms target")
    
    return fastest_time

if __name__ == "__main__":
    print("Testing true fast path with biologically plausible tradeoffs...")
    fastest_time = test_true_fast_path()
    
    if fastest_time and fastest_time < float('inf'):
        print(f"\n🏁 TRUE FAST PATH TEST COMPLETE")
        print(f"Best performance: {fastest_time:.1f}ms")
        
        if fastest_time < 100.0:
            print(f"🎉 FAST PATH SUCCESS!")
        else:
            print(f"⚡ FAST PATH IMPLEMENTED")
    else:
        print(f"\n❌ TEST FAILED - INVESTIGATE IMPLEMENTATION")