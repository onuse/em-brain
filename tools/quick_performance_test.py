#!/usr/bin/env python3
"""
Quick performance test of the accelerated similarity engine
"""

import sys
sys.path.append('.')

import time
import numpy as np
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode

def quick_test():
    """Quick test of similarity performance with realistic brain size"""
    print("Quick Performance Test - Accelerated Similarity Engine")
    print("=" * 55)
    
    # Create realistic brain size
    graph = WorldGraph()
    n_nodes = 2140  # Actual brain size from metadata
    
    print(f"Creating brain with {n_nodes} nodes...")
    
    np.random.seed(42)
    start_time = time.time()
    
    for i in range(n_nodes):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (n_nodes - i) / rate
            print(f"  {i}/{n_nodes} nodes ({i/n_nodes*100:.1f}%) - ETA: {eta:.1f}s")
        
        # Realistic 8D context
        context = [np.random.uniform(0, 1) for _ in range(8)]
        
        node = ExperienceNode(
            mental_context=context,
            action_taken={'forward': 0.5, 'turn': 0.2},
            predicted_sensory=[1.0, 2.0, 3.0],
            actual_sensory=[1.1, 2.1, 3.1],
            prediction_error=0.1
        )
        
        graph.add_node(node)
    
    creation_time = time.time() - start_time
    print(f"Brain creation completed: {creation_time:.2f}s")
    
    # Test similarity search
    print(f"\\nTesting similarity search performance...")
    target_context = [0.5, 0.3, 0.8, 0.2, 0.1, 0.9, 0.6, 0.4]
    
    # Single search test
    start_time = time.time()
    similar_nodes = graph.find_similar_nodes(target_context, similarity_threshold=0.6, max_results=10)
    single_search_time = time.time() - start_time
    
    print(f"Single similarity search: {single_search_time:.6f}s")
    print(f"Found {len(similar_nodes)} similar nodes")
    
    # Multiple searches (simulating brain traversals)
    print(f"\\nTesting multiple searches (brain traversal simulation)...")
    n_searches = 10
    
    start_time = time.time()
    for i in range(n_searches):
        # Vary context slightly
        varied_context = [x + (i * 0.1) for x in target_context]
        similar_nodes = graph.find_similar_nodes(varied_context, 0.6, 10)
    
    multi_search_time = time.time() - start_time
    avg_search_time = multi_search_time / n_searches
    
    print(f"{n_searches} searches: {multi_search_time:.4f}s")
    print(f"Average per search: {avg_search_time:.6f}s")
    print(f"Searches per second: {1/avg_search_time:.1f}")
    
    # Get engine stats
    stats = graph.get_graph_statistics()
    sim_stats = stats['similarity_engine']
    
    print(f"\\nEngine Statistics:")
    print(f"  Method: {sim_stats['acceleration_method']}")
    print(f"  GPU Available: {sim_stats['gpu_available']}")
    print(f"  Total searches: {sim_stats['total_searches']}")
    print(f"  Cache hit rate: {sim_stats['cache_hit_rate']:.1%}")
    
    # Estimate impact on demo
    print(f"\\nDemo Impact Estimation:")
    
    # Typical brain prediction involves 3-10 similarity searches
    searches_per_prediction = 5
    prediction_time = searches_per_prediction * avg_search_time
    
    print(f"  Similarity searches per prediction: {searches_per_prediction}")
    print(f"  Time for similarity in one prediction: {prediction_time:.6f}s")
    print(f"  Potential demo FPS (similarity only): {1/prediction_time:.1f}")
    
    # Compare with original bottleneck
    original_fps = 1.4
    print(f"  Original demo FPS: {original_fps}")
    
    if prediction_time < (1/original_fps):
        improvement = (1/original_fps) / prediction_time
        print(f"  Potential overall improvement: {improvement:.1f}x faster")
        print(f"  🚀 Similarity search is no longer the bottleneck!")
    else:
        print(f"  ⚠️  Similarity search still a bottleneck")
    
    return avg_search_time

if __name__ == "__main__":
    quick_test()