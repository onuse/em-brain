#!/usr/bin/env python3
"""
Quick Memory Capacity Test

Fast test to find max experiences within 400ms think time limit.
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


def quick_capacity_test():
    """Quick test to find memory capacity within 400ms."""
    print("🧠 QUICK MEMORY CAPACITY TEST")
    print("=" * 40)
    print("Target: 400ms think time limit")
    print()
    
    # Create optimized brain
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        quiet_mode=True
    )
    
    # Test levels (fewer points for speed)
    test_levels = [500, 1000, 2000, 4000, 6000, 8000, 10000]
    results = []
    
    for exp_count in test_levels:
        print(f"📊 Testing {exp_count:,} experiences...")
        
        # Clear and load experiences quickly
        brain.experience_storage.clear()
        
        # Fast experience loading (minimal processing)
        for i in range(exp_count):
            # Simple diverse experiences
            sensory = [0.1 + (i % 20) * 0.02, 0.2 + (i % 15) * 0.03, 
                      0.3 + (i % 10) * 0.05, 0.4 + (i % 5) * 0.1]
            action = [0.1, 0.2, 0.3, 0.4]
            outcome = [0.15, 0.25, 0.35, 0.45]
            
            # Store experience directly (faster than full processing)
            from src.experience.models import Experience
            exp = Experience(
                sensory_input=sensory,
                action_taken=action,
                outcome=outcome,
                prediction_error=0.3,
                timestamp=time.time()
            )
            brain.experience_storage.add_experience(exp)
        
        # Test query performance
        query_times = []
        for test_i in range(5):  # Fewer tests for speed
            query = [0.5, 0.4, 0.6, 0.3]
            
            start_time = time.time()
            predicted_action, brain_state = brain.process_sensory_input(query)
            cycle_time = (time.time() - start_time) * 1000
            query_times.append(cycle_time)
        
        avg_time = np.mean(query_times)
        max_time = np.max(query_times)
        
        # Check hierarchical stats
        hier_stats = brain.similarity_engine.get_performance_stats().get('hierarchical_indexing', {})
        regions = hier_stats.get('total_regions', 0)
        efficiency = hier_stats.get('search_efficiency', 1.0)
        
        within_limit = max_time <= 400
        status = "✅" if within_limit else "❌"
        
        print(f"   {status} Avg: {avg_time:.1f}ms, Max: {max_time:.1f}ms")
        if regions > 0:
            print(f"      Regions: {regions}, Efficiency: {efficiency:.1f}x")
        
        results.append({
            'experiences': exp_count,
            'avg_time': avg_time,
            'max_time': max_time,
            'regions': regions,
            'efficiency': efficiency,
            'within_limit': within_limit
        })
        
        if not within_limit:
            break
    
    brain.finalize_session()
    
    # Find maximum viable capacity
    viable_results = [r for r in results if r['within_limit']]
    if viable_results:
        max_result = viable_results[-1]
        max_experiences = max_result['experiences']
        
        print(f"\n🎯 CAPACITY RESULTS:")
        print(f"   Maximum viable: {max_experiences:,} experiences")
        print(f"   Average think time: {max_result['avg_time']:.1f}ms")
        print(f"   Peak think time: {max_result['max_time']:.1f}ms")
        print(f"   Hierarchical regions: {max_result['regions']}")
        print(f"   Search efficiency: {max_result['efficiency']:.1f}x")
        
        # Recommendations
        conservative = int(max_experiences * 0.8)
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   🎯 Optimal capacity: {max_experiences:,} experiences")
        print(f"   🛡️  Conservative limit: {conservative:,} experiences")
        
        if max_experiences >= 10000:
            print(f"   ✨ Excellent performance - handles 10k+ efficiently")
        elif max_experiences >= 5000:
            print(f"   ✅ Good performance - handles 5k+ well") 
        else:
            print(f"   ⚠️  Limited performance - consider optimizations")
    else:
        print("\n❌ No configuration found within 400ms limit")
    
    # Performance scaling
    print(f"\n📈 SCALING ANALYSIS:")
    for result in results:
        scaling = "Sub-linear" if result['efficiency'] > 2 else "Linear"
        print(f"   {result['experiences']:>6,}: {result['avg_time']:>5.1f}ms ({scaling})")


if __name__ == "__main__":
    quick_capacity_test()