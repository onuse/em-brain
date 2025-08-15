#!/usr/bin/env python3
"""
Real vs Simplified Performance Test

Compares the performance of:
1. Simplified capacity test (direct experience loading)
2. Real full brain simulation (proper learning and adaptation)

This reveals whether our capacity estimates are realistic for actual brain operation.
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


def test_real_vs_simplified_performance():
    """Compare real brain simulation vs simplified capacity test."""
    print("🔬 REAL vs SIMPLIFIED PERFORMANCE COMPARISON")
    print("=" * 55)
    print("Testing whether capacity estimates reflect real brain performance")
    print()
    
    # Test with smaller numbers first to see the difference
    test_counts = [1000, 2000, 5000, 10000]
    
    for exp_count in test_counts:
        print(f"📊 Testing {exp_count:,} experiences:")
        print()
        
        # Test 1: Simplified method (like our capacity test)
        print("   🚀 SIMPLIFIED METHOD (capacity test approach):")
        simplified_times = test_simplified_method(exp_count)
        
        print("   🧠 REAL BRAIN SIMULATION (full learning):")
        real_times = test_real_brain_simulation(exp_count)
        
        # Compare results
        print("   📈 COMPARISON:")
        simplified_avg = np.mean(simplified_times)
        real_avg = np.mean(real_times)
        slowdown_factor = real_avg / simplified_avg if simplified_avg > 0 else float('inf')
        
        print(f"      Simplified: {simplified_avg:.1f}ms avg")
        print(f"      Real brain: {real_avg:.1f}ms avg")
        print(f"      Slowdown: {slowdown_factor:.1f}x")
        
        if slowdown_factor > 10:
            print(f"      ⚠️  MAJOR DIFFERENCE: Real brain is {slowdown_factor:.1f}x slower!")
        elif slowdown_factor > 3:
            print(f"      ⚠️  SIGNIFICANT DIFFERENCE: Real brain is {slowdown_factor:.1f}x slower")
        elif slowdown_factor > 1.5:
            print(f"      📊 MODERATE DIFFERENCE: Real brain is {slowdown_factor:.1f}x slower")
        else:
            print(f"      ✅ SIMILAR PERFORMANCE: Capacity estimates are realistic")
        
        print()
        
        # Stop if real brain becomes too slow
        if real_avg > 200:  # More than 200ms
            print("   🛑 Stopping test - real brain too slow for larger datasets")
            break
    
    print("🎯 SUMMARY:")
    print("-" * 15)
    print("This test reveals whether our 500k experience capacity estimate")
    print("is realistic for actual brain operation or just for similarity search.")


def test_simplified_method(exp_count):
    """Test the simplified method used in capacity testing."""
    # Create brain
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        quiet_mode=True
    )
    
    # Load experiences directly (simplified method)
    brain.experience_storage.clear()
    
    from src.experience.models import Experience
    for i in range(exp_count):
        sensory = [0.1 + (i % 20) * 0.02, 0.2 + (i % 15) * 0.03, 
                  0.3 + (i % 10) * 0.05, 0.4 + (i % 5) * 0.1]
        action = [0.1, 0.2, 0.3, 0.4]
        outcome = [0.15, 0.25, 0.35, 0.45]
        
        exp = Experience(
            sensory_input=sensory,
            action_taken=action,
            outcome=outcome,
            prediction_error=0.3,
            timestamp=time.time()
        )
        brain.experience_storage.add_experience(exp)
    
    # Test queries
    query_times = []
    for test_i in range(5):
        query = [0.5 + test_i * 0.1, 0.4 + test_i * 0.05, 
                0.6 + test_i * 0.03, 0.3 + test_i * 0.07]
        
        start_time = time.time()
        predicted_action, brain_state = brain.process_sensory_input(query)
        cycle_time = (time.time() - start_time) * 1000
        query_times.append(cycle_time)
    
    brain.finalize_session()
    
    avg_time = np.mean(query_times)
    print(f"      Loaded {exp_count:,} experiences, avg query: {avg_time:.1f}ms")
    
    return query_times


def test_real_brain_simulation(exp_count):
    """Test real brain simulation with full learning and adaptation."""
    # Create brain with full capabilities
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        quiet_mode=True
    )
    
    # REAL brain simulation - full experience loop
    print(f"      Loading {exp_count:,} experiences with full brain simulation...")
    load_start = time.time()
    
    for i in range(exp_count):
        # Create realistic sensory input
        base_sensory = [0.1 + (i % 100) * 0.005, 0.2 + (i % 50) * 0.01, 
                       0.3 + (i % 25) * 0.02, 0.4 + (i % 10) * 0.05]
        noise = np.random.normal(0, 0.02, 4)  # Add noise for realism
        sensory = [max(0, min(1, base + n)) for base, n in zip(base_sensory, noise)]
        
        # FULL BRAIN CYCLE: prediction -> action -> outcome -> learning
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        
        # Simulate realistic outcome with some unpredictability
        outcome_noise = np.random.normal(0, 0.1, 4)
        outcome = [a * 0.85 + 0.1 + n for a, n in zip(predicted_action, outcome_noise)]
        outcome = [max(0, min(1, o)) for o in outcome]  # Clamp to valid range
        
        # Store experience with FULL learning (similarity, activation, etc.)
        brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        # Progress indicator for slower loading
        if (i + 1) % (exp_count // 10) == 0:
            elapsed = time.time() - load_start
            progress = (i + 1) / exp_count
            eta = elapsed / progress - elapsed
            print(f"         {progress:.0%} complete ({elapsed:.1f}s elapsed, {eta:.1f}s remaining)")
    
    load_time = time.time() - load_start
    print(f"      Loaded in {load_time:.1f}s (vs ~{exp_count/1000000:.1f}s for simplified)")
    
    # Test query performance on fully learned brain
    query_times = []
    for test_i in range(5):
        query = [0.5 + test_i * 0.1, 0.4 + test_i * 0.05, 
                0.6 + test_i * 0.03, 0.3 + test_i * 0.07]
        
        start_time = time.time()
        predicted_action, brain_state = brain.process_sensory_input(query)
        cycle_time = (time.time() - start_time) * 1000
        query_times.append(cycle_time)
    
    brain.finalize_session()
    
    avg_time = np.mean(query_times)
    print(f"      Full brain with {exp_count:,} experiences, avg query: {avg_time:.1f}ms")
    
    return query_times


if __name__ == "__main__":
    test_real_vs_simplified_performance()