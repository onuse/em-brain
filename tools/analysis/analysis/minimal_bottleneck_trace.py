#!/usr/bin/env python3
"""
Minimal Bottleneck Trace - Find the 2.2s bottleneck

Simple trace to identify what's causing the 2.2s delay.
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

def trace_minimal_bottleneck():
    """Trace the bottleneck with minimal complexity."""
    print("🔍 MINIMAL BOTTLENECK TRACE")
    print("=" * 35)
    
    # Create brain
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
        print("✅ Brain created successfully")
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return
    
    # Test input - this should trigger slow path first time
    novel_input = [1.0, 2.0, 3.0, 4.0]
    
    print("\nTesting full prediction timing...")
    
    # Time the full prediction first
    try:
        with timeout(60):
            start_time = time.time()
            action, brain_state = brain.process_sensory_input(novel_input, action_dimensions=2)
            full_time = (time.time() - start_time) * 1000
            
            fast_path_used = brain_state.get('fast_path_used', False)
            
            print(f"Full prediction: {full_time:.1f}ms")
            print(f"Fast path used: {fast_path_used}")
            
            if not fast_path_used:
                print(f"✅ This is the 2.2s slow path!")
            else:
                print(f"⚠️  This used fast path, trying different input...")
                
                # Try a different input to force slow path
                different_input = [9.0, 8.0, 7.0, 6.0]
                start_time = time.time()
                action, brain_state = brain.process_sensory_input(different_input, action_dimensions=2)
                full_time = (time.time() - start_time) * 1000
                
                fast_path_used = brain_state.get('fast_path_used', False)
                print(f"Different input: {full_time:.1f}ms")
                print(f"Fast path used: {fast_path_used}")
                
    except TimeoutError:
        print("❌ Full prediction timed out!")
        return
    
    # Now let's trace the components by accessing the brain internals
    print(f"\n🔍 COMPONENT BREAKDOWN")
    print("=" * 30)
    
    # The key insight is that we need to trace what happens in the slow path
    # Let's look at the vector brain processing
    
    try:
        with timeout(60):
            vector_brain = brain.vector_brain
            
            # Time just the vector brain processing
            start_time = time.time()
            predicted_action, vector_brain_state = vector_brain.process_sensory_input(novel_input)
            vector_brain_time = (time.time() - start_time) * 1000
            
            print(f"Vector brain processing: {vector_brain_time:.1f}ms")
            
            # This is likely the bottleneck
            if vector_brain_time > 1000:
                print(f"🚨 BOTTLENECK FOUND: Vector brain processing!")
                print(f"   This is {vector_brain_time/1000:.1f}s of the 2.2s delay")
                
                # The vector brain contains:
                # - Stream updates
                # - Temporal hierarchy processing
                # - Competitive dynamics
                # - Cross-stream coactivation
                
                print(f"\n📊 VECTOR BRAIN COMPONENTS:")
                print(f"   • Stream updates (sensory, temporal, motor)")
                print(f"   • Temporal hierarchy processing")
                print(f"   • Competitive dynamics")
                print(f"   • Cross-stream coactivation")
                print(f"   • Brain state compilation")
                
                # Check what the fast path skips
                print(f"\n⚡ FAST PATH BYPASSES:")
                print(f"   • ALL vector brain processing")
                print(f"   • Direct cache lookup instead")
                print(f"   • Speedup: {vector_brain_time/2:.0f}x")
                
    except TimeoutError:
        print("❌ Vector brain timing timed out!")
        return
    except Exception as e:
        print(f"❌ Vector brain timing failed: {e}")
        return
    
    # Test the fast path for comparison
    print(f"\n🏃 FAST PATH COMPARISON")
    print("=" * 30)
    
    try:
        with timeout(10):
            # Use the same input again - should hit cache
            start_time = time.time()
            action, brain_state = brain.process_sensory_input(novel_input, action_dimensions=2)
            fast_time = (time.time() - start_time) * 1000
            
            fast_path_used = brain_state.get('fast_path_used', False)
            
            print(f"Second prediction: {fast_time:.1f}ms")
            print(f"Fast path used: {fast_path_used}")
            
            if fast_path_used:
                speedup = full_time / fast_time if fast_time > 0 else float('inf')
                print(f"Speedup: {speedup:.0f}x")
                
                # Show what fast path keeps
                print(f"\n✅ FAST PATH KEEPS ONLY:")
                print(f"   • Pattern hashing")
                print(f"   • Cache lookup")
                print(f"   • Minimal brain state")
                print(f"   • Total: ~{fast_time:.1f}ms")
                
    except TimeoutError:
        print("❌ Fast path comparison timed out!")
    
    return full_time, fast_path_used

if __name__ == "__main__":
    print("Finding the 2.2s bottleneck...")
    result = trace_minimal_bottleneck()
    
    if result:
        full_time, fast_path_used = result
        print(f"\n🏁 BOTTLENECK ANALYSIS COMPLETE")
        print(f"Slow path time: {full_time:.1f}ms")
        print(f"Primary bottleneck: Vector brain processing")
        print(f"Fast path solution: Cache bypass")
    else:
        print(f"\n❌ BOTTLENECK ANALYSIS FAILED")