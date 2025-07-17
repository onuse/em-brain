#!/usr/bin/env python3
"""
Comprehensive profiler to find hidden speedbumps in vector brain processing.
This uses Python's built-in profiling tools to catch everything.
"""
import sys
import os
import time
import cProfile
import pstats
import threading
import gc
from contextlib import contextmanager
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain

@contextmanager
def profile_context(sort_by='cumulative'):
    """Context manager for comprehensive profiling."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        # Print top slowest functions
        stats = pstats.Stats(profiler)
        stats.sort_stats(sort_by)
        stats.print_stats(30)  # Top 30 functions

def monitor_gc_activity():
    """Monitor garbage collection activity."""
    print("🗑️  GC ACTIVITY MONITORING")
    print("=" * 30)
    
    # Enable GC debugging
    gc.set_debug(gc.DEBUG_STATS)
    
    # Get initial GC stats
    initial_counts = gc.get_count()
    initial_stats = gc.get_stats()
    
    print(f"Initial GC counts: {initial_counts}")
    print(f"Initial GC collections: {[s['collections'] for s in initial_stats]}")
    
    return initial_counts, initial_stats

def check_gc_impact(initial_counts, initial_stats):
    """Check if GC had significant impact."""
    final_counts = gc.get_count()
    final_stats = gc.get_stats()
    
    print(f"\n🗑️  GC IMPACT ANALYSIS")
    print("=" * 25)
    print(f"Final GC counts: {final_counts}")
    print(f"Final GC collections: {[s['collections'] for s in final_stats]}")
    
    # Check for collections
    collections = []
    for i, (initial, final) in enumerate(zip(initial_stats, final_stats)):
        diff = final['collections'] - initial['collections']
        if diff > 0:
            collections.append(f"Gen {i}: {diff} collections")
    
    if collections:
        print(f"🚨 GC ACTIVITY: {', '.join(collections)}")
        return True
    else:
        print("✅ No significant GC activity")
        return False

def analyze_thread_activity():
    """Analyze thread activity and potential blocking."""
    print(f"\n🧵 THREAD ANALYSIS")
    print("=" * 20)
    
    # Get all threads
    threads = threading.enumerate()
    print(f"Active threads: {len(threads)}")
    
    for thread in threads:
        print(f"  - {thread.name}: {thread.ident} ({'daemon' if thread.daemon else 'main'})")
    
    # Check for deadlocks or blocking
    main_thread = threading.main_thread()
    print(f"Main thread: {main_thread.name} (alive: {main_thread.is_alive()})")

def find_hidden_speedbumps():
    """Comprehensive analysis to find hidden performance bottlenecks."""
    print("🔍 COMPREHENSIVE SPEEDBUMP ANALYSIS")
    print("=" * 45)
    
    # Create brain
    brain = MinimalBrain(quiet_mode=True)
    
    # Test input with dimension adaptation
    novel_input = [1.0, 2.0, 3.0, 4.0]
    sensory_dim = brain.sensory_dim
    if len(novel_input) < sensory_dim:
        processed_input = novel_input + [0.0] * (sensory_dim - len(novel_input))
    else:
        processed_input = novel_input
    
    print(f"Input: {processed_input[:4]}... (length: {len(processed_input)})")
    
    # Clear cache to force slow path
    brain.vector_brain.emergent_hierarchy.predictor.reflex_cache.clear()
    
    # Monitor GC activity
    initial_gc = monitor_gc_activity()
    
    # Analyze thread activity
    analyze_thread_activity()
    
    print(f"\n🔬 DETAILED PROFILING")
    print("=" * 25)
    
    # Use cProfile to catch everything
    with profile_context('cumulative') as profiler:
        start_time = time.time()
        predicted_action, vector_brain_state = brain.vector_brain.process_sensory_input(processed_input)
        total_time = (time.time() - start_time) * 1000
    
    print(f"\n⏱️  TOTAL TIME: {total_time:.1f}ms")
    
    # Check GC impact
    gc_impact = check_gc_impact(*initial_gc)
    
    # Additional analysis
    print(f"\n📊 ADDITIONAL ANALYSIS")
    print("=" * 25)
    
    # Check for large object creation
    import tracemalloc
    tracemalloc.start()
    
    # Run again with memory tracing
    start_time = time.time()
    predicted_action2, vector_brain_state2 = brain.vector_brain.process_sensory_input(processed_input)
    second_time = (time.time() - start_time) * 1000
    
    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Second call time: {second_time:.1f}ms")
    print(f"Memory usage: {current / 1024 / 1024:.1f}MB current, {peak / 1024 / 1024:.1f}MB peak")
    
    # Check for blocking operations
    if total_time > 1000:  # More than 1 second
        print(f"\n🚨 POTENTIAL BLOCKING DETECTED")
        print("=" * 35)
        print(f"Time too long for CPU-bound operations: {total_time:.1f}ms")
        print("Possible causes:")
        print("  1. Thread synchronization/blocking")
        print("  2. I/O operations")
        print("  3. Memory allocation/GC pressure")
        print("  4. Lock contention")
        print("  5. OS scheduling delays")
    
    return total_time

if __name__ == "__main__":
    print("Finding hidden speedbumps in vector brain processing...")
    total_time = find_hidden_speedbumps()
    
    print(f"\n🏁 ANALYSIS COMPLETE")
    print(f"Total time: {total_time:.1f}ms")
    print(f"Check the profiling output above for the slowest functions!")