#!/usr/bin/env python3
"""
Debug Hierarchical Regression

Investigates why hierarchical clustering is causing severe performance regression
instead of improvement.
"""

import sys
import os
import time

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


def debug_hierarchical_regression():
    """Debug the performance regression with hierarchical clustering."""
    print("🔍 DEBUGGING HIERARCHICAL CLUSTERING REGRESSION")
    print("=" * 60)
    print("Investigating why performance degrades instead of improves")
    print()
    
    # Test both configurations
    print("🧪 TESTING WITH AND WITHOUT HIERARCHICAL CLUSTERING")
    print("-" * 60)
    
    # Test 1: WITHOUT hierarchical clustering
    print("1. Testing WITHOUT hierarchical clustering...")
    
    brain_no_hierarchical = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=False,
        enable_phase2_adaptations=False,
        quiet_mode=True
    )
    
    # Disable hierarchical clustering
    brain_no_hierarchical.similarity_engine.use_hierarchical_indexing = False
    brain_no_hierarchical.similarity_engine.hierarchical_index = None
    
    # Add experiences and test performance
    print("   Adding 100 experiences...")
    start_time = time.time()
    
    for i in range(100):
        sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
        predicted_action, _ = brain_no_hierarchical.process_sensory_input(sensory)
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain_no_hierarchical.store_experience(sensory, predicted_action, outcome, predicted_action)
    
    setup_time_no_hier = time.time() - start_time
    
    # Test search performance
    search_times_no_hier = []
    for i in range(10):
        query_sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
        search_start = time.time()
        predicted_action, brain_state = brain_no_hierarchical.process_sensory_input(query_sensory)
        search_time = (time.time() - search_start) * 1000
        search_times_no_hier.append(search_time)
    
    avg_no_hier = sum(search_times_no_hier) / len(search_times_no_hier)
    
    print(f"   Setup: {setup_time_no_hier:.2f}s")
    print(f"   Average search: {avg_no_hier:.1f}ms")
    
    brain_no_hierarchical.finalize_session()
    
    # Test 2: WITH hierarchical clustering
    print("\n2. Testing WITH hierarchical clustering...")
    
    brain_with_hierarchical = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=False,
        enable_phase2_adaptations=False,
        quiet_mode=True
    )
    
    # Verify hierarchical clustering is enabled
    print(f"   Hierarchical indexing enabled: {brain_with_hierarchical.similarity_engine.use_hierarchical_indexing}")
    
    # Add same experiences and test performance
    print("   Adding 100 experiences...")
    start_time = time.time()
    
    for i in range(100):
        sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
        predicted_action, _ = brain_with_hierarchical.process_sensory_input(sensory)
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain_with_hierarchical.store_experience(sensory, predicted_action, outcome, predicted_action)
    
    setup_time_with_hier = time.time() - start_time
    
    # Get hierarchical statistics
    similarity_stats = brain_with_hierarchical.similarity_engine.get_performance_stats()
    hierarchical_stats = similarity_stats.get('hierarchical_indexing', {})
    
    print(f"   Hierarchical stats: {hierarchical_stats}")
    
    # Test search performance
    search_times_with_hier = []
    for i in range(10):
        query_sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
        search_start = time.time()
        predicted_action, brain_state = brain_with_hierarchical.process_sensory_input(query_sensory)
        search_time = (time.time() - search_start) * 1000
        search_times_with_hier.append(search_time)
    
    avg_with_hier = sum(search_times_with_hier) / len(search_times_with_hier)
    
    print(f"   Setup: {setup_time_with_hier:.2f}s")
    print(f"   Average search: {avg_with_hier:.1f}ms")
    
    brain_with_hierarchical.finalize_session()
    
    # Analysis
    print(f"\n📊 COMPARISON ANALYSIS:")
    print("-" * 40)
    print(f"   WITHOUT hierarchical: {avg_no_hier:.1f}ms")
    print(f"   WITH hierarchical: {avg_with_hier:.1f}ms")
    
    if avg_with_hier > avg_no_hier:
        regression = ((avg_with_hier - avg_no_hier) / avg_no_hier) * 100
        print(f"   ❌ REGRESSION: {regression:+.1f}% slower with hierarchical clustering")
    else:
        improvement = ((avg_no_hier - avg_with_hier) / avg_no_hier) * 100
        print(f"   ✅ IMPROVEMENT: {improvement:.1f}% faster with hierarchical clustering")
    
    # Test scaling with more experiences
    print(f"\n🔬 SCALING TEST - LARGER DATASET:")
    print("-" * 50)
    
    # Test with 500 experiences
    brain_scaling = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=False,
        enable_phase2_adaptations=False,
        quiet_mode=True
    )
    
    print("   Adding 500 experiences...")
    start_time = time.time()
    
    for i in range(500):
        sensory = [0.1 + 0.001 * i, 0.2 + 0.001 * i, 0.3 + 0.001 * i, 0.4 + 0.001 * i]
        predicted_action, _ = brain_scaling.process_sensory_input(sensory)
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain_scaling.store_experience(sensory, predicted_action, outcome, predicted_action)
    
    setup_time_500 = time.time() - start_time
    
    # Test search performance at 500 experiences
    search_times_500 = []
    for i in range(5):
        query_sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
        search_start = time.time()
        predicted_action, brain_state = brain_scaling.process_sensory_input(query_sensory)
        search_time = (time.time() - search_start) * 1000
        search_times_500.append(search_time)
    
    avg_500 = sum(search_times_500) / len(search_times_500)
    
    # Get final hierarchical statistics
    final_stats = brain_scaling.similarity_engine.get_performance_stats()
    final_hierarchical = final_stats.get('hierarchical_indexing', {})
    
    print(f"   Setup: {setup_time_500:.2f}s")
    print(f"   Average search: {avg_500:.1f}ms")
    print(f"   Regions: {final_hierarchical.get('total_regions', 0)}")
    print(f"   Search efficiency: {final_hierarchical.get('search_efficiency', 1.0):.1f}x")
    
    brain_scaling.finalize_session()
    
    # Identify the problem
    print(f"\n🎯 PROBLEM IDENTIFICATION:")
    print("-" * 35)
    
    if avg_500 > 500:  # If very slow
        print("   ❌ Hierarchical clustering is causing severe slowdown")
        print("   🔍 Possible causes:")
        print("      1. Index update overhead during experience storage")
        print("      2. Inefficient region search algorithm")
        print("      3. Memory allocation issues in hierarchical index")
        print("      4. Interference with other brain systems")
    elif final_hierarchical.get('search_efficiency', 1.0) < 2.0:
        print("   ⚠️  Hierarchical search not being used effectively")
        print("   🔍 Possible causes:")
        print("      1. Too few regions created")
        print("      2. Threshold settings preventing hierarchical search")
        print("      3. Integration issues with similarity engine")
    else:
        print("   🤔 Performance issue may be elsewhere")


if __name__ == "__main__":
    debug_hierarchical_regression()