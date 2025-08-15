#!/usr/bin/env python3
"""
Test 10K Challenge

Direct test to see if we can achieve 10k experiences with <100ms search time
using hierarchical clustering.
"""

import sys
import os
import time
import math

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


def test_10k_challenge():
    """Test if we can achieve 10k experiences with <100ms search."""
    print("🎯 10K EXPERIENCE CHALLENGE")
    print("=" * 50)
    print("Goal: 10,000 experiences searchable in <100ms")
    print()
    
    # Create brain with hierarchical indexing but minimal other features
    print("🧠 Creating optimized brain for 10k challenge...")
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=False,  # Disable for speed
        enable_phase2_adaptations=False,
        quiet_mode=True
    )
    
    print("✅ Brain created with hierarchical indexing enabled")
    print()
    
    # Add experiences in batches to track progress
    batch_size = 1000
    target_total = 10000
    
    print(f"📝 Adding {target_total} experiences in batches of {batch_size}...")
    
    total_start_time = time.time()
    
    for batch_num in range(target_total // batch_size):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, target_total)
        
        batch_start_time = time.time()
        
        for i in range(batch_start, batch_end):
            # Create simple, diverse experiences
            sensory = [
                0.1 + 0.0001 * i,
                0.2 + 0.0001 * i,
                0.3 + 0.0001 * i,
                0.4 + 0.0001 * i
            ]
            
            # Simple prediction to avoid complex processing
            predicted_action = [s * 0.5 for s in sensory]
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        batch_time = time.time() - batch_start_time
        current_total = batch_end
        
        print(f"   Batch {batch_num + 1}: {current_total} experiences ({batch_time:.2f}s)")
        
        # Test search performance at this scale
        if current_total >= 1000:  # Only test after we have meaningful data
            query_sensory = [0.5, 0.4, 0.6, 0.3]
            
            search_start = time.time()
            predicted_action, brain_state = brain.process_sensory_input(query_sensory)
            search_time = (time.time() - search_start) * 1000
            
            print(f"       Search time: {search_time:.1f}ms")
            
            # Get hierarchical stats
            similarity_stats = brain.similarity_engine.get_performance_stats()
            hierarchical_stats = similarity_stats.get('hierarchical_indexing', {})
            regions = hierarchical_stats.get('total_regions', 0)
            efficiency = hierarchical_stats.get('search_efficiency', 1.0)
            
            print(f"       Regions: {regions}, Efficiency: {efficiency:.1f}x")
            
            # Check if we're meeting the target early
            if current_total >= 10000 and search_time < 100:
                print(f"   🎉 SUCCESS: {current_total} experiences in {search_time:.1f}ms!")
                break
            elif search_time > 200:
                print(f"   ⚠️  Performance degrading: {search_time:.1f}ms")
                break
    
    total_time = time.time() - total_start_time
    final_experience_count = len(brain.experience_storage._experiences)
    
    print(f"\n✅ Setup completed: {final_experience_count} experiences in {total_time:.2f}s")
    
    # Final performance test
    print("\n🔍 FINAL PERFORMANCE TEST:")
    print("-" * 40)
    
    search_times = []
    for i in range(10):
        query_sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
        
        search_start = time.time()
        predicted_action, brain_state = brain.process_sensory_input(query_sensory)
        search_time = (time.time() - search_start) * 1000
        search_times.append(search_time)
    
    avg_search_time = sum(search_times) / len(search_times)
    min_search_time = min(search_times)
    max_search_time = max(search_times)
    
    print(f"   Average search time: {avg_search_time:.1f}ms")
    print(f"   Range: {min_search_time:.1f}ms - {max_search_time:.1f}ms")
    
    # Get final hierarchical statistics
    print("\n📊 FINAL HIERARCHICAL STATISTICS:")
    print("-" * 45)
    
    final_stats = brain.similarity_engine.get_performance_stats()
    hierarchical_final = final_stats.get('hierarchical_indexing', {})
    
    if hierarchical_final:
        total_experiences = hierarchical_final.get('total_experiences', 0)
        total_regions = hierarchical_final.get('total_regions', 0)
        avg_region_size = hierarchical_final.get('avg_region_size', 0)
        search_efficiency = hierarchical_final.get('search_efficiency', 1.0)
        experiences_per_search = hierarchical_final.get('avg_experiences_per_search', 0)
        
        print(f"   Total experiences: {total_experiences}")
        print(f"   Brain regions: {total_regions}")
        print(f"   Average region size: {avg_region_size:.1f}")
        print(f"   Search efficiency: {search_efficiency:.1f}x faster than linear")
        print(f"   Experiences searched per query: {experiences_per_search:.1f}")
        
        # Calculate theoretical linear search time
        linear_search_time = avg_search_time * search_efficiency
        print(f"   Theoretical linear search time: {linear_search_time:.1f}ms")
    
    brain.finalize_session()
    
    # Final verdict
    print("\n🏆 FINAL VERDICT:")
    print("-" * 25)
    
    if final_experience_count >= 10000 and avg_search_time < 100:
        print(f"   🎉 SUCCESS: {final_experience_count} experiences in {avg_search_time:.1f}ms")
        print("   🎯 TARGET ACHIEVED: 10k experiences searchable in <100ms!")
    elif final_experience_count >= 10000:
        print(f"   🔄 CLOSE: {final_experience_count} experiences in {avg_search_time:.1f}ms")
        print(f"   📈 {100 - avg_search_time:.1f}ms away from 100ms target")
    else:
        print(f"   📊 PARTIAL: {final_experience_count} experiences, {avg_search_time:.1f}ms search")
        
        # Project to 10k
        if final_experience_count > 1000:
            scale_factor = 10000 / final_experience_count
            projected_time = avg_search_time * scale_factor
            print(f"   🔮 Projected 10k time: {projected_time:.1f}ms")


if __name__ == "__main__":
    test_10k_challenge()