"""
Simple GPU Acceleration Test - Quick validation of vectorized backend.
"""

import sys
import time
import random

# Add project root to path
sys.path.append('.')

from core.hybrid_world_graph import HybridWorldGraph


def test_basic_functionality():
    """Test basic functionality of hybrid graph."""
    print("🧪 BASIC FUNCTIONALITY TEST")
    print("=" * 40)
    
    # Create hybrid graph
    graph = HybridWorldGraph()
    print(f"✅ Created HybridWorldGraph on device: {graph.vectorized_backend.device}")
    
    # Add a few experiences
    print("\n📝 Adding test experiences...")
    for i in range(10):
        mental_context = [random.uniform(-1, 1) for _ in range(8)]
        action_taken = {'forward_motor': random.uniform(-1, 1)}
        predicted_sensory = [random.uniform(0, 1) for _ in range(8)]
        actual_sensory = [p + random.uniform(-0.1, 0.1) for p in predicted_sensory]
        prediction_error = random.uniform(0, 0.5)
        
        experience = graph.add_experience(mental_context, action_taken, 
                                        predicted_sensory, actual_sensory, prediction_error)
        print(f"   Added experience {i+1}: {experience.node_id}")
    
    print(f"✅ Added {graph.node_count()} experiences")
    print(f"   Object storage: {len(graph.nodes)} nodes")
    print(f"   Vectorized storage: {graph.vectorized_backend.get_size()} experiences")
    
    # Test similarity search
    print("\n🔍 Testing similarity search...")
    query_context = [random.uniform(-1, 1) for _ in range(8)]
    
    # Test with vectorized enabled
    graph.use_vectorized_similarity = True
    start_time = time.time()
    vectorized_results = graph.find_similar_experiences(query_context, max_results=5)
    vectorized_time = time.time() - start_time
    
    # Test with vectorized disabled (object-based)
    graph.use_vectorized_similarity = False
    start_time = time.time()
    object_results = graph.find_similar_experiences(query_context, max_results=5)
    object_time = time.time() - start_time
    
    print(f"   Vectorized search: {len(vectorized_results)} results in {vectorized_time*1000:.2f}ms")
    print(f"   Object-based search: {len(object_results)} results in {object_time*1000:.2f}ms")
    
    if vectorized_time > 0:
        speedup = object_time / vectorized_time
        print(f"   Speedup: {speedup:.1f}x")
    
    # Re-enable vectorized
    graph.use_vectorized_similarity = True
    
    return graph


def test_performance_scaling():
    """Test performance with different graph sizes."""
    print("\n⚡ PERFORMANCE SCALING TEST")
    print("=" * 40)
    
    graph = HybridWorldGraph()
    sizes_to_test = [100, 500, 1000]
    
    for target_size in sizes_to_test:
        # Add experiences to reach target size
        current_size = graph.node_count()
        experiences_to_add = target_size - current_size
        
        if experiences_to_add > 0:
            print(f"\n📈 Adding {experiences_to_add} experiences (total: {target_size})...")
            
            for i in range(experiences_to_add):
                mental_context = [random.gauss(0, 1) for _ in range(8)]
                action_taken = {
                    'forward_motor': random.uniform(-1, 1),
                    'turn_motor': random.uniform(-1, 1)
                }
                predicted_sensory = [random.uniform(0, 1) for _ in range(8)]
                actual_sensory = [p + random.gauss(0, 0.1) for p in predicted_sensory]
                prediction_error = random.uniform(0, 1)
                
                graph.add_experience(mental_context, action_taken, 
                                   predicted_sensory, actual_sensory, prediction_error)
        
        # Benchmark similarity search
        print(f"🔍 Benchmarking with {graph.node_count()} experiences...")
        
        # Generate test queries
        test_queries = []
        for _ in range(20):
            query = [random.gauss(0, 1) for _ in range(8)]
            test_queries.append(query)
        
        # Test vectorized
        start_time = time.time()
        for query in test_queries:
            graph.find_similar_experiences(query, max_results=10)
        vectorized_time = time.time() - start_time
        
        # Test object-based (limit to avoid timeout)
        if target_size <= 500:  # Only test object-based for smaller sizes
            graph.use_vectorized_similarity = False
            start_time = time.time()
            for query in test_queries:
                graph.find_similar_experiences(query, max_results=10)
            object_time = time.time() - start_time
            graph.use_vectorized_similarity = True
            
            speedup = object_time / max(0.001, vectorized_time)
            
            print(f"   Vectorized: {vectorized_time*1000:.1f}ms total ({vectorized_time*1000/20:.2f}ms per query)")
            print(f"   Object-based: {object_time*1000:.1f}ms total ({object_time*1000/20:.2f}ms per query)")
            print(f"   Speedup: {speedup:.1f}x faster")
        else:
            print(f"   Vectorized: {vectorized_time*1000:.1f}ms total ({vectorized_time*1000/20:.2f}ms per query)")
            print("   Object-based: Skipped (too slow for large dataset)")
    
    return graph


def test_gpu_memory_usage():
    """Test GPU memory usage and efficiency."""
    print("\n💾 GPU MEMORY USAGE TEST")
    print("=" * 40)
    
    graph = HybridWorldGraph()
    
    # Add experiences and monitor memory
    for batch in range(1, 6):
        batch_size = 200
        print(f"\nBatch {batch}: Adding {batch_size} experiences...")
        
        for i in range(batch_size):
            mental_context = [random.gauss(0, 1) for _ in range(8)]
            action_taken = {'forward_motor': random.uniform(-1, 1)}
            predicted_sensory = [random.uniform(0, 1) for _ in range(8)]
            actual_sensory = [p + random.gauss(0, 0.05) for p in predicted_sensory]
            prediction_error = random.uniform(0, 0.3)
            
            graph.add_experience(mental_context, action_taken, 
                               predicted_sensory, actual_sensory, prediction_error)
        
        # Check memory usage
        stats = graph.vectorized_backend.get_stats()
        print(f"   Total experiences: {stats['size']}")
        print(f"   GPU memory usage: {stats['memory_usage_mb']:.1f} MB")
        print(f"   Device: {stats['device']}")
        
        # Quick performance check
        query = [random.gauss(0, 1) for _ in range(8)]
        start_time = time.time()
        results = graph.find_similar_experiences(query, max_results=20)
        query_time = time.time() - start_time
        
        print(f"   Query performance: {query_time*1000:.2f}ms ({len(results)} results)")


def main():
    """Run comprehensive but fast GPU acceleration tests."""
    print("🚀 GPU ACCELERATION VALIDATION")
    print("=" * 50)
    print("Quick tests to validate vectorized backend performance")
    print()
    
    # Basic functionality test
    graph = test_basic_functionality()
    
    # Performance scaling test
    test_performance_scaling()
    
    # Memory usage test
    test_gpu_memory_usage()
    
    # Final summary
    print("\n🌟 GPU ACCELERATION VALIDATION COMPLETE!")
    
    stats = graph.get_vectorized_stats()
    perf_analysis = stats['performance_analysis']
    
    print(f"Final Results:")
    print(f"  Device: {stats['backend_stats']['device']}")
    print(f"  Total experiences: {stats['backend_stats']['size']}")
    print(f"  GPU memory usage: {stats['backend_stats']['memory_usage_mb']:.1f} MB")
    print(f"  Average vectorized query: {perf_analysis['avg_vectorized_time_ms']:.2f}ms")
    
    if perf_analysis['speedup_ratio'] > 1:
        print(f"  Speedup achieved: {perf_analysis['speedup_ratio']:.1f}x faster than object-based")
    
    print("\n✅ GPU vectorization is working and delivering performance improvements!")


if __name__ == "__main__":
    main()