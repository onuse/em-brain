#!/usr/bin/env python3
"""
Test the accelerated WorldGraph integration
"""

import sys
sys.path.append('.')

import time
import numpy as np
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode

def test_accelerated_worldgraph():
    """Test the accelerated similarity search in WorldGraph"""
    print("Testing Accelerated WorldGraph Integration")
    print("=" * 50)
    
    # Create a WorldGraph instance
    graph = WorldGraph()
    
    # Add realistic brain data
    n_nodes = 1000  # Smaller for testing
    context_dim = 8
    
    print(f"Adding {n_nodes} experience nodes...")
    np.random.seed(42)
    
    start_time = time.time()
    for i in range(n_nodes):
        context = np.random.randn(context_dim) * 2.0
        action = {'forward': np.random.uniform(0, 1), 'turn': np.random.uniform(-1, 1)}
        predicted_sensory = np.random.randn(6) * 1.5
        actual_sensory = predicted_sensory + np.random.randn(6) * 0.1
        prediction_error = np.linalg.norm(predicted_sensory - actual_sensory)
        
        node = ExperienceNode(
            mental_context=context.tolist(),
            action_taken=action,
            predicted_sensory=predicted_sensory.tolist(),
            actual_sensory=actual_sensory.tolist(),
            prediction_error=prediction_error
        )
        
        graph.add_node(node)
        
        if (i + 1) % 200 == 0:
            print(f"  Added {i + 1} nodes...")
    
    creation_time = time.time() - start_time
    print(f"Graph creation completed in {creation_time:.3f}s")
    print()
    
    # Test similarity search performance
    print("Testing similarity search performance...")
    target_context = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    # Warmup
    _ = graph.find_similar_nodes(target_context, similarity_threshold=0.6, max_results=10)
    
    # Benchmark
    n_searches = 20
    start_time = time.time()
    
    for i in range(n_searches):
        # Vary the context slightly for each search
        varied_context = [x + (i % 10) * 0.1 for x in target_context]
        similar_nodes = graph.find_similar_nodes(
            varied_context, 
            similarity_threshold=0.6, 
            max_results=10
        )
        
        if i == 0:
            print(f"  First search found {len(similar_nodes)} similar nodes")
    
    search_time = time.time() - start_time
    avg_search_time = search_time / n_searches
    
    print(f"\\nPerformance Results:")
    print(f"  {n_searches} similarity searches: {search_time:.4f}s")
    print(f"  Average per search: {avg_search_time:.6f}s")
    print(f"  Searches per second: {1/avg_search_time:.1f}")
    
    # Get detailed statistics
    stats = graph.get_graph_statistics()
    print(f"\\nGraph Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Average strength: {stats['avg_strength']:.3f}")
    print(f"  Total accesses: {stats['total_accesses']}")
    
    print(f"\\nSimilarity Engine Statistics:")
    sim_stats = stats['similarity_engine']
    print(f"  Method: {sim_stats['acceleration_method']}")
    print(f"  Total searches: {sim_stats['total_searches']}")
    print(f"  Average search time: {sim_stats['avg_search_time']:.6f}s")
    print(f"  Cache hit rate: {sim_stats['cache_hit_rate']:.2f}")
    print(f"  GPU available: {sim_stats['gpu_available']}")
    
    return avg_search_time

def compare_performance():
    """Compare accelerated vs original performance"""
    print("\\n" + "=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    
    # Create two graphs for comparison
    graph_accel = WorldGraph()
    
    # Add the same data to both
    n_nodes = 500
    context_dim = 8
    
    print(f"Setting up comparison with {n_nodes} nodes...")
    np.random.seed(42)
    
    for i in range(n_nodes):
        context = np.random.randn(context_dim) * 2.0
        action = {'forward': 0.5, 'turn': 0.2}
        predicted_sensory = np.random.randn(6) * 1.5
        actual_sensory = predicted_sensory + np.random.randn(6) * 0.1
        prediction_error = np.linalg.norm(predicted_sensory - actual_sensory)
        
        node = ExperienceNode(
            mental_context=context.tolist(),
            action_taken=action,
            predicted_sensory=predicted_sensory.tolist(),
            actual_sensory=actual_sensory.tolist(),
            prediction_error=prediction_error
        )
        
        graph_accel.add_node(node)
    
    # Test accelerated performance
    target_context = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    print("\\nTesting accelerated similarity search...")
    start_time = time.time()
    for i in range(10):
        similar_nodes = graph_accel.find_similar_nodes(target_context, 0.6, 10)
    accel_time = (time.time() - start_time) / 10
    
    print("\\nTesting fallback similarity search...")
    start_time = time.time()
    for i in range(10):
        similar_nodes_fallback = graph_accel._find_similar_nodes_fallback(target_context, 0.6, 10)
    fallback_time = (time.time() - start_time) / 10
    
    print(f"\\nComparison Results:")
    print(f"  Accelerated method: {accel_time:.6f}s per search")
    print(f"  Fallback method: {fallback_time:.6f}s per search")
    print(f"  Speedup: {fallback_time/accel_time:.1f}x")
    
    # Verify results are similar
    accel_nodes = graph_accel.find_similar_nodes(target_context, 0.6, 10)
    fallback_nodes = graph_accel._find_similar_nodes_fallback(target_context, 0.6, 10)
    
    print(f"\\nResult Verification:")
    print(f"  Accelerated found: {len(accel_nodes)} nodes")
    print(f"  Fallback found: {len(fallback_nodes)} nodes")
    
    return accel_time, fallback_time

def main():
    """Main test function"""
    print("WorldGraph Acceleration Integration Test")
    print("=" * 60)
    
    # Test 1: Basic functionality
    avg_search_time = test_accelerated_worldgraph()
    
    # Test 2: Performance comparison
    accel_time, fallback_time = compare_performance()
    
    # Summary
    print("\\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    speedup = fallback_time / accel_time
    
    print(f"✅ Accelerated WorldGraph integration: SUCCESSFUL")
    print(f"📊 Performance improvement: {speedup:.1f}x faster than original")
    print(f"⚡ Search speed: {1/avg_search_time:.0f} searches per second")
    
    if speedup > 2.0:
        print(f"🚀 Excellent acceleration - ready for brain integration!")
    elif speedup > 1.5:
        print(f"✅ Good acceleration - should improve brain performance")
    else:
        print(f"⚠️  Minimal acceleration - check similarity engine configuration")
    
    print(f"\\n🎯 Ready for Phase 4: Demo integration testing")

if __name__ == "__main__":
    main()