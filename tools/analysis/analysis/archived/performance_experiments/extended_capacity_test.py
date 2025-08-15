#!/usr/bin/env python3
"""
Extended Capacity Test

Test higher memory capacity since 10k experiences only took 1.5ms.
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


def extended_capacity_test():
    """Test higher memory capacity to find actual 400ms limit."""
    print("🚀 EXTENDED MEMORY CAPACITY TEST")
    print("=" * 45)
    print("Finding actual 400ms limit (10k was only 1.5ms)")
    print()
    
    # Create optimized brain
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        quiet_mode=True
    )
    
    # Higher test levels
    test_levels = [20000, 50000, 100000, 200000, 500000]
    results = []
    
    for exp_count in test_levels:
        print(f"📊 Testing {exp_count:,} experiences...")
        
        # Clear storage
        brain.experience_storage.clear()
        
        # Load experiences directly (fastest method)
        print(f"   Loading {exp_count:,} experiences...", end=" ", flush=True)
        load_start = time.time()
        
        from src.experience.models import Experience
        for i in range(exp_count):
            # Simple diverse experiences
            sensory = [0.1 + (i % 100) * 0.005, 0.2 + (i % 50) * 0.01, 
                      0.3 + (i % 25) * 0.02, 0.4 + (i % 10) * 0.05]
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
        
        load_time = time.time() - load_start
        print(f"loaded in {load_time:.1f}s")
        
        # Test query performance  
        query_times = []
        print(f"   Running performance tests...", end=" ", flush=True)
        
        for test_i in range(3):  # Just 3 tests for speed
            query = [0.5 + test_i * 0.1, 0.4 + test_i * 0.05, 
                    0.6 + test_i * 0.03, 0.3 + test_i * 0.07]
            
            start_time = time.time()
            predicted_action, brain_state = brain.process_sensory_input(query)
            cycle_time = (time.time() - start_time) * 1000
            query_times.append(cycle_time)
        
        print("done")
        
        avg_time = np.mean(query_times)
        max_time = np.max(query_times)
        
        # Check hierarchical performance
        hier_stats = brain.similarity_engine.get_performance_stats().get('hierarchical_indexing', {})
        regions = hier_stats.get('total_regions', 0)
        efficiency = hier_stats.get('search_efficiency', 1.0)
        
        within_limit = max_time <= 400
        status = "✅" if within_limit else "❌"
        
        print(f"   {status} Avg: {avg_time:.1f}ms, Max: {max_time:.1f}ms")
        if regions > 0:
            print(f"      Hierarchical: {regions} regions, {efficiency:.1f}x efficiency")
        
        results.append({
            'experiences': exp_count,
            'avg_time': avg_time,
            'max_time': max_time,
            'load_time': load_time,
            'regions': regions,
            'efficiency': efficiency,
            'within_limit': within_limit
        })
        
        print()
        
        # Stop if we exceed limit or if loading takes too long
        if not within_limit or load_time > 60:
            print(f"Stopping test ({'time limit exceeded' if not within_limit else 'loading too slow'})")
            break
    
    brain.finalize_session()
    
    # Analysis
    print(f"🎯 EXTENDED CAPACITY RESULTS:")
    print("-" * 35)
    
    viable_results = [r for r in results if r['within_limit']]
    if viable_results:
        max_result = viable_results[-1]
        max_experiences = max_result['experiences']
        
        print(f"   ✨ Maximum capacity: {max_experiences:,} experiences")
        print(f"   ⚡ Think time: {max_result['avg_time']:.1f}ms avg, {max_result['max_time']:.1f}ms peak")
        
        if max_result['regions'] > 0:
            print(f"   🧠 Hierarchical: {max_result['regions']} regions, {max_result['efficiency']:.1f}x efficiency")
        
        # Performance analysis
        efficiency_ratio = 400 / max_result['max_time']
        print(f"   📊 Efficiency: {efficiency_ratio:.1f}x under 400ms limit")
        
        # Memory usage estimate
        estimated_mb = max_experiences * 0.5 / 1000  # Rough estimate
        print(f"   💾 Estimated memory: ~{estimated_mb:.1f}MB")
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        if max_experiences >= 100000:
            print(f"   🚀 EXCEPTIONAL: System handles 100k+ experiences (biological brain scale)")
            print(f"   🎯 Suggested limit: {max_experiences:,} experiences")
        elif max_experiences >= 50000:
            print(f"   ✨ EXCELLENT: System handles 50k+ experiences efficiently")
            print(f"   🎯 Suggested limit: {max_experiences:,} experiences")
        elif max_experiences >= 20000:
            print(f"   ✅ VERY GOOD: System handles 20k+ experiences")
            print(f"   🎯 Suggested limit: {max_experiences:,} experiences")
        
        # Conservative recommendation
        conservative = int(max_experiences * 0.7)
        print(f"   🛡️  Conservative limit: {conservative:,} experiences (70% of max)")
        
    else:
        print("   ❌ No viable configuration found")
    
    # Scaling analysis
    print(f"\n📈 PERFORMANCE SCALING:")
    for result in results:
        experiences_per_ms = result['experiences'] / result['avg_time']
        print(f"   {result['experiences']:>7,}: {result['avg_time']:>6.1f}ms "
              f"({experiences_per_ms:>8,.0f} exp/ms)")


if __name__ == "__main__":
    extended_capacity_test()