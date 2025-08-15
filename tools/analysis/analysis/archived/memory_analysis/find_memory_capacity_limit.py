#!/usr/bin/env python3
"""
Find Memory Capacity Limit

Determines the maximum number of memory nodes (experiences) this system 
can effectively handle within a 400ms think time limit.

Tests both similarity search performance and full brain cycle performance
to find realistic operational limits.
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


def test_memory_capacity_limit():
    """Find the maximum memory capacity within 400ms think time limit."""
    print("🧠 FINDING MEMORY CAPACITY LIMIT")
    print("=" * 50)
    print("Target: Find max experiences processable within 400ms")
    print()
    
    # Create optimized brain for capacity testing
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,  # No disk I/O overhead
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        quiet_mode=True
    )
    
    print(f"🚀 Testing system: {brain.similarity_engine.__class__.__name__}")
    print(f"   GPU: {brain.similarity_engine.use_gpu}")
    print(f"   Hierarchical indexing: {brain.similarity_engine.use_hierarchical_indexing}")
    print()
    
    # Test parameters
    target_think_time = 400  # ms
    test_levels = [
        100, 200, 500, 1000, 2000, 3000, 4000, 5000, 
        6000, 7000, 8000, 9000, 10000, 12000, 15000, 20000
    ]
    
    results = []
    max_viable_experiences = 0
    
    for experience_count in test_levels:
        print(f"📊 Testing {experience_count:,} experiences...")
        
        # Clear experience storage for clean test
        brain.experience_storage.clear()
        
        # Add experiences progressively
        print(f"   Loading {experience_count:,} experiences...", end=" ", flush=True)
        load_start = time.time()
        
        for i in range(experience_count):
            # Create diverse experiences for realistic testing
            base_sensory = [0.1 + (i % 100) * 0.01, 0.2 + (i % 50) * 0.02, 
                          0.3 + (i % 25) * 0.04, 0.4 + (i % 10) * 0.1]
            noise = np.random.normal(0, 0.05, 4)
            sensory = [max(0, min(1, base + n)) for base, n in zip(base_sensory, noise)]
            
            # Quick brain processing (don't time this part)
            predicted_action, _ = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 + np.random.normal(0, 0.1) for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        load_time = time.time() - load_start
        print(f"loaded in {load_time:.1f}s")
        
        # Test similarity search performance
        similarity_times = []
        brain_cycle_times = []
        
        # Test multiple queries for statistical reliability
        num_tests = 10
        print(f"   Running {num_tests} performance tests...", end=" ", flush=True)
        
        for test_i in range(num_tests):
            # Create test query
            query_sensory = [0.5 + test_i * 0.05, 0.4 + test_i * 0.03, 
                           0.6 + test_i * 0.07, 0.3 + test_i * 0.02]
            
            # Test similarity search only
            similarity_start = time.time()
            experience_vectors = []
            experience_ids = []
            for exp_id, exp in brain.experience_storage._experiences.items():
                experience_vectors.append(exp.get_context_vector())
                experience_ids.append(exp_id)
            
            if experience_vectors:
                padded_query = query_sensory + [0.0, 0.0, 0.0, 0.0]  # Pad to 8D
                similar_experiences = brain.similarity_engine.find_similar_experiences(
                    padded_query, experience_vectors, experience_ids, max_results=20
                )
            similarity_time = (time.time() - similarity_start) * 1000
            similarity_times.append(similarity_time)
            
            # Test full brain cycle
            cycle_start = time.time()
            predicted_action, brain_state = brain.process_sensory_input(query_sensory)
            cycle_time = (time.time() - cycle_start) * 1000
            brain_cycle_times.append(cycle_time)
        
        print("done")
        
        # Calculate statistics
        avg_similarity_time = np.mean(similarity_times)
        avg_cycle_time = np.mean(brain_cycle_times)
        max_cycle_time = np.max(brain_cycle_times)
        std_cycle_time = np.std(brain_cycle_times)
        
        # Check hierarchical performance if available
        hierarchical_stats = brain.similarity_engine.get_performance_stats().get('hierarchical_indexing', {})
        regions = hierarchical_stats.get('total_regions', 0)
        search_efficiency = hierarchical_stats.get('search_efficiency', 1.0)
        
        result = {
            'experiences': experience_count,
            'avg_similarity_time': avg_similarity_time,
            'avg_cycle_time': avg_cycle_time,
            'max_cycle_time': max_cycle_time,
            'std_cycle_time': std_cycle_time,
            'hierarchical_regions': regions,
            'search_efficiency': search_efficiency,
            'within_400ms_limit': max_cycle_time <= target_think_time
        }
        results.append(result)
        
        # Print results
        status = "✅" if result['within_400ms_limit'] else "❌"
        print(f"   {status} Avg: {avg_cycle_time:.1f}ms, Max: {max_cycle_time:.1f}ms, "
              f"Similarity: {avg_similarity_time:.1f}ms")
        if regions > 0:
            print(f"      Hierarchical: {regions} regions, {search_efficiency:.1f}x efficiency")
        print()
        
        # Update max viable count
        if result['within_400ms_limit']:
            max_viable_experiences = experience_count
        else:
            break  # Stop testing once we exceed the limit
    
    brain.finalize_session()
    
    # Analysis and recommendations
    print("📈 CAPACITY ANALYSIS:")
    print("-" * 30)
    
    print(f"✅ Maximum viable experiences: {max_viable_experiences:,}")
    
    if max_viable_experiences > 0:
        # Find the result for max viable experiences
        max_result = next(r for r in results if r['experiences'] == max_viable_experiences)
        print(f"   Average cycle time: {max_result['avg_cycle_time']:.1f}ms")
        print(f"   Peak cycle time: {max_result['max_cycle_time']:.1f}ms")
        print(f"   Similarity search: {max_result['avg_similarity_time']:.1f}ms")
        
        if max_result['hierarchical_regions'] > 0:
            print(f"   Hierarchical regions: {max_result['hierarchical_regions']}")
            print(f"   Search efficiency: {max_result['search_efficiency']:.1f}x")
        
        # Performance breakdown
        similarity_ratio = max_result['avg_similarity_time'] / max_result['avg_cycle_time']
        print(f"   Similarity search: {similarity_ratio:.1%} of cycle time")
        
        # Scaling analysis
        if len(results) >= 2:
            print(f"\n📊 SCALING ANALYSIS:")
            for i, result in enumerate(results):
                if i == 0:
                    scaling_factor = 1.0
                else:
                    prev_result = results[i-1]
                    exp_ratio = result['experiences'] / prev_result['experiences']
                    time_ratio = result['avg_cycle_time'] / prev_result['avg_cycle_time']
                    scaling_factor = time_ratio / exp_ratio
                
                scaling_desc = "Sub-linear" if scaling_factor < 0.9 else \
                             "Linear" if scaling_factor < 1.1 else "Super-linear"
                
                print(f"   {result['experiences']:>6,} exp: {result['avg_cycle_time']:>6.1f}ms "
                      f"({scaling_desc} scaling: {scaling_factor:.2f}x)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   🎯 Optimal capacity: {max_viable_experiences:,} experiences")
        print(f"   🚀 Performance mode: {max_result['avg_cycle_time']:.0f}ms average think time")
        
        if max_viable_experiences >= 10000:
            print(f"   ✨ Excellent: System handles 10k+ experiences efficiently")
        elif max_viable_experiences >= 5000:
            print(f"   ✅ Good: System handles 5k+ experiences well")
        elif max_viable_experiences >= 1000:
            print(f"   ⚠️  Limited: Consider optimizations for larger datasets")
        else:
            print(f"   ⚠️  Constrained: May need performance optimization")
        
        # Memory management advice
        conservative_limit = int(max_viable_experiences * 0.8)
        print(f"   🛡️  Conservative limit: {conservative_limit:,} experiences (80% of max)")
        
    else:
        print("❌ No viable configuration found within 400ms limit")
        print("   Consider system optimization or increasing think time limit")
    
    print(f"\n🔧 SYSTEM CONFIGURATION:")
    print(f"   Hardware: GPU={brain.similarity_engine.use_gpu}")
    print(f"   Hierarchical indexing: {brain.similarity_engine.use_hierarchical_indexing}")
    print(f"   Utility-based activation: {brain.use_utility_based_activation}")
    print(f"   Think time limit: {target_think_time}ms")


if __name__ == "__main__":
    test_memory_capacity_limit()