#!/usr/bin/env python3
"""
Investigate Performance Discrepancy

The capacity test showed 500k experiences in 1.6ms, but the real brain test timed out.
This suggests our capacity estimates are completely unrealistic for actual brain operation.

Let's investigate exactly where the performance bottleneck is.
"""

import sys
import os
import time
import numpy as np

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


def investigate_performance_discrepancy():
    """Investigate the massive performance discrepancy."""
    print("🔍 INVESTIGATING PERFORMANCE DISCREPANCY")
    print("=" * 50)
    print("Capacity test: 500k experiences in 1.6ms")
    print("Real brain test: Timed out after 5 minutes")
    print("Something is seriously wrong with our estimates!")
    print()
    
    # Start with tiny numbers to see where it breaks
    test_sizes = [10, 50, 100, 500]
    
    for size in test_sizes:
        print(f"🧪 Testing {size} experiences:")
        
        # Test 1: Query performance on pre-loaded brain (what capacity test measured)
        print("   📊 CAPACITY TEST METHOD (query performance only):")
        query_time = test_query_performance_only(size)
        
        # Test 2: Full brain learning (what real brain does)
        print("   🧠 REAL BRAIN METHOD (full learning cycle):")
        learning_time = test_full_brain_learning(size)
        
        # Analysis
        if learning_time > 0 and query_time > 0:
            ratio = learning_time / query_time
            print(f"   ⚡ Results:")
            print(f"      Query-only: {query_time:.1f}ms")
            print(f"      Full learning: {learning_time:.1f}ms") 
            print(f"      Learning overhead: {ratio:.1f}x slower")
            
            if ratio > 100:
                print(f"      🚨 CRITICAL: Learning is {ratio:.0f}x slower than queries!")
            elif ratio > 10:
                print(f"      ⚠️  WARNING: Learning is {ratio:.0f}x slower than queries")
        
        print()
        
        # Stop if learning becomes too slow
        if learning_time > 1000:  # More than 1 second
            print(f"🛑 Stopping at {size} experiences - learning too slow")
            break
    
    print("🎯 DIAGNOSIS:")
    print("-" * 15)
    print("The capacity test only measured QUERY performance on pre-loaded brains.")
    print("It did NOT measure the cost of LEARNING those experiences.")
    print("Real brain operation requires both learning AND querying.")
    print()
    print("Our 500k capacity estimate is therefore completely unrealistic")
    print("for actual brain operation with continuous learning.")


def test_query_performance_only(exp_count):
    """Test only query performance (what the capacity test actually measured)."""
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    
    # Load experiences WITHOUT full brain processing (capacity test method)
    brain.experience_storage.clear()
    
    from src.experience.models import Experience
    for i in range(exp_count):
        sensory = [0.1 + i * 0.01, 0.2 + i * 0.01, 0.3 + i * 0.01, 0.4 + i * 0.01]
        exp = Experience(
            sensory_input=sensory,
            action_taken=[0.1, 0.2, 0.3, 0.4],
            outcome=[0.15, 0.25, 0.35, 0.45],
            prediction_error=0.3,
            timestamp=time.time()
        )
        brain.experience_storage.add_experience(exp)
    
    # Test query performance
    query = [0.5, 0.4, 0.6, 0.3]
    start_time = time.time()
    predicted_action, brain_state = brain.process_sensory_input(query)
    query_time = (time.time() - start_time) * 1000
    
    brain.finalize_session()
    print(f"      Pre-loaded {exp_count} experiences, query took: {query_time:.1f}ms")
    return query_time


def test_full_brain_learning(exp_count):
    """Test full brain learning cycle (what real brain actually does)."""
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    
    print(f"      Learning {exp_count} experiences with full brain processing...")
    
    # Time the LEARNING process
    learning_start = time.time()
    
    for i in range(exp_count):
        # Full brain cycle: sensory input -> prediction -> action -> outcome -> learning
        sensory = [0.1 + i * 0.01, 0.2 + i * 0.01, 0.3 + i * 0.01, 0.4 + i * 0.01]
        
        # STEP 1: Brain processes sensory input (prediction)
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        
        # STEP 2: Simulate outcome
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        
        # STEP 3: Brain learns from experience (THIS IS THE EXPENSIVE PART)
        brain.store_experience(sensory, predicted_action, outcome, predicted_action)
    
    learning_time = (time.time() - learning_start) * 1000
    
    # Now test query performance on the learned brain
    query = [0.5, 0.4, 0.6, 0.3]
    query_start = time.time()
    predicted_action, brain_state = brain.process_sensory_input(query)
    query_time = (time.time() - query_start) * 1000
    
    brain.finalize_session()
    
    print(f"      Learning took: {learning_time:.1f}ms total ({learning_time/exp_count:.1f}ms per experience)")
    print(f"      Final query took: {query_time:.1f}ms")
    
    return learning_time


if __name__ == "__main__":
    investigate_performance_discrepancy()