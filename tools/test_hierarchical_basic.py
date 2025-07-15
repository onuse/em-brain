#!/usr/bin/env python3
"""
Test Hierarchical Basic

Quick test to verify hierarchical clustering integration works correctly.
"""

import sys
import os
import time

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


def test_hierarchical_basic():
    """Basic test of hierarchical clustering integration."""
    print("🧪 BASIC HIERARCHICAL CLUSTERING TEST")
    print("=" * 50)
    
    # Create brain with hierarchical indexing
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        enable_phase2_adaptations=False,
        quiet_mode=True
    )
    
    # Verify hierarchical indexing is enabled
    print(f"✅ Hierarchical indexing enabled: {brain.similarity_engine.use_hierarchical_indexing}")
    print(f"✅ Hierarchical index created: {brain.similarity_engine.hierarchical_index is not None}")
    print()
    
    # Add a small number of experiences
    print("📝 Adding 50 experiences...")
    start_time = time.time()
    
    for i in range(50):
        sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
        predicted_action, _ = brain.process_sensory_input(sensory)
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain.store_experience(sensory, predicted_action, outcome, predicted_action)
    
    setup_time = time.time() - start_time
    print(f"✅ Setup completed in {setup_time:.2f}s")
    
    # Test search performance
    print("\n🔍 Testing search performance...")
    search_times = []
    
    for i in range(5):
        query_sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
        
        search_start = time.time()
        predicted_action, brain_state = brain.process_sensory_input(query_sensory)
        search_time = (time.time() - search_start) * 1000
        search_times.append(search_time)
        
        print(f"   Search {i+1}: {search_time:.1f}ms")
    
    avg_search_time = sum(search_times) / len(search_times)
    print(f"✅ Average search time: {avg_search_time:.1f}ms")
    
    # Get hierarchical statistics
    print("\n📊 Hierarchical Index Statistics:")
    similarity_stats = brain.similarity_engine.get_performance_stats()
    hierarchical_stats = similarity_stats.get('hierarchical_indexing', {})
    
    if hierarchical_stats:
        print(f"   Total experiences: {hierarchical_stats.get('total_experiences', 0)}")
        print(f"   Brain regions: {hierarchical_stats.get('total_regions', 0)}")
        print(f"   Average region size: {hierarchical_stats.get('avg_region_size', 0):.1f}")
        print(f"   Search efficiency: {hierarchical_stats.get('search_efficiency', 1.0):.1f}x")
        print(f"   Experiences searched per query: {hierarchical_stats.get('avg_experiences_per_search', 0):.1f}")
    else:
        print("   No hierarchical statistics available")
    
    # Check if hierarchical search was actually used
    if hierarchical_stats and hierarchical_stats.get('search_efficiency', 1.0) > 1.0:
        print("✅ Hierarchical search is working!")
    else:
        print("⚠️  Hierarchical search may not be activated (too few experiences)")
    
    brain.finalize_session()
    print("\n🎉 Basic test completed successfully!")


if __name__ == "__main__":
    test_hierarchical_basic()