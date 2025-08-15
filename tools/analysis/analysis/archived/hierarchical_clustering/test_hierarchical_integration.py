#!/usr/bin/env python3
"""
Test Hierarchical Integration

Tests the production hierarchical clustering integration with the brain,
targeting the 10k experiences in <100ms goal.
"""

import sys
import os
import time
import math

# Set up path to access brain modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


def test_hierarchical_integration():
    """Test hierarchical clustering integration with the production brain."""
    print("🚀 TESTING HIERARCHICAL CLUSTERING INTEGRATION")
    print("=" * 60)
    print("Target: 10k experiences searchable in <100ms")
    print()
    
    # Create brain with hierarchical indexing enabled
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        enable_phase2_adaptations=False,
        quiet_mode=True
    )
    
    # Verify hierarchical indexing is enabled
    if not brain.similarity_engine.use_hierarchical_indexing:
        print("❌ FATAL: Hierarchical indexing not enabled!")
        return
    
    print("✅ Hierarchical indexing enabled in similarity engine")
    print()
    
    # Test scaling at different experience counts
    test_sizes = [100, 500, 1000, 2000, 5000, 10000]
    results = []
    
    for target_size in test_sizes:
        print(f"📊 Testing {target_size} experiences...")
        
        # Add experiences up to target size
        start_time = time.time()
        current_count = len(brain.experience_storage._experiences)
        
        for i in range(current_count, target_size):
            # Create diverse experiences that should cluster naturally
            angle = i * 0.05
            type_offset = (i % 3) * 2.0  # Three types of experiences
            
            sensory = [
                0.5 + 0.3 * math.sin(angle + type_offset),
                0.5 + 0.3 * math.cos(angle * 1.1 + type_offset),
                0.5 + 0.2 * math.sin(angle * 0.7 + type_offset),
                0.5 + 0.2 * math.cos(angle * 1.3 + type_offset)
            ]
            
            predicted_action, _ = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        setup_time = time.time() - start_time
        
        # Test search performance with hierarchical clustering
        search_times = []
        for _ in range(10):
            query_sensory = [0.5, 0.4, 0.6, 0.3]
            
            search_start = time.time()
            predicted_action, brain_state = brain.process_sensory_input(query_sensory)
            search_time = (time.time() - search_start) * 1000
            search_times.append(search_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        
        # Get hierarchical indexing statistics
        similarity_stats = brain.similarity_engine.get_performance_stats()
        hierarchical_stats = similarity_stats.get('hierarchical_indexing', {})
        
        result = {
            'size': target_size,
            'setup_time': setup_time,
            'avg_search_time': avg_search_time,
            'regions': hierarchical_stats.get('total_regions', 0),
            'avg_region_size': hierarchical_stats.get('avg_region_size', 0),
            'search_efficiency': hierarchical_stats.get('search_efficiency', 1.0),
            'experiences_per_search': hierarchical_stats.get('avg_experiences_per_search', target_size)
        }
        results.append(result)
        
        print(f"   Setup: {setup_time:.2f}s")
        print(f"   Search: {avg_search_time:.1f}ms")
        print(f"   Regions: {result['regions']}")
        print(f"   Avg region size: {result['avg_region_size']:.1f}")
        print(f"   Search efficiency: {result['search_efficiency']:.1f}x")
        print(f"   Experiences searched: {result['experiences_per_search']:.1f}")
        
        # Check if we achieved the 10k target
        if target_size >= 10000:
            if avg_search_time < 100:
                print(f"   🎉 SUCCESS: {target_size} experiences in {avg_search_time:.1f}ms!")
            else:
                print(f"   ⚠️  CLOSE: {target_size} experiences in {avg_search_time:.1f}ms (target: <100ms)")
        
        print()
        
        # Stop if search gets too slow
        if avg_search_time > 500:
            print(f"   ❌ STOPPING: Search too slow at {target_size} experiences")
            break
    
    # Analyze scaling behavior
    print("📈 SCALING ANALYSIS:")
    print("-" * 40)
    
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        
        size_ratio = last['size'] / first['size']
        time_ratio = last['avg_search_time'] / first['avg_search_time']
        
        print(f"   Experience scaling: {first['size']} → {last['size']} ({size_ratio:.1f}x)")
        print(f"   Time scaling: {first['avg_search_time']:.1f}ms → {last['avg_search_time']:.1f}ms ({time_ratio:.1f}x)")
        
        if time_ratio < size_ratio:
            print("   ✅ Sub-linear scaling achieved!")
        else:
            print("   ⚠️  Linear or super-linear scaling")
        
        # Project to higher scales
        if last['size'] < 10000:
            projected_10k = first['avg_search_time'] * (10000 / first['size']) * (time_ratio / size_ratio)
            print(f"   📊 Projected 10k performance: {projected_10k:.1f}ms")
    
    print()
    print("🧠 HIERARCHICAL BRAIN REGIONALIZATION:")
    print("-" * 50)
    
    final_stats = brain.similarity_engine.get_performance_stats()
    hierarchical_final = final_stats.get('hierarchical_indexing', {})
    
    if hierarchical_final:
        print(f"   Total experiences: {hierarchical_final.get('total_experiences', 0)}")
        print(f"   Brain regions: {hierarchical_final.get('total_regions', 0)}")
        print(f"   Average region size: {hierarchical_final.get('avg_region_size', 0):.1f}")
        print(f"   Largest region: {hierarchical_final.get('max_region_size', 0)}")
        print(f"   Search efficiency: {hierarchical_final.get('search_efficiency', 1.0):.1f}x faster than linear")
        
        specialization_scores = hierarchical_final.get('specialization_scores', [])
        if specialization_scores:
            avg_specialization = sum(specialization_scores) / len(specialization_scores)
            print(f"   Average specialization: {avg_specialization:.3f}")
    
    brain.finalize_session()
    
    # Summary
    print()
    print("🎯 SUMMARY:")
    print("-" * 20)
    
    best_result = min(results, key=lambda r: r['avg_search_time'])
    print(f"   Best performance: {best_result['avg_search_time']:.1f}ms ({best_result['size']} experiences)")
    
    ten_k_results = [r for r in results if r['size'] >= 10000]
    if ten_k_results:
        ten_k_result = ten_k_results[0]
        if ten_k_result['avg_search_time'] < 100:
            print(f"   🏆 TARGET ACHIEVED: 10k+ experiences in {ten_k_result['avg_search_time']:.1f}ms")
        else:
            print(f"   🎯 CLOSE TO TARGET: 10k+ experiences in {ten_k_result['avg_search_time']:.1f}ms")
    else:
        print("   ❌ Did not reach 10k experiences")


if __name__ == "__main__":
    test_hierarchical_integration()