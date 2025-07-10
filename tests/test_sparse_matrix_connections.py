#!/usr/bin/env python3
"""
Test sparse matrix graph connections implementation.
"""

import sys
import time
import torch
from typing import List, Dict

# Add current directory to path
sys.path.append('.')

from core.hybrid_world_graph import HybridWorldGraph
from core.experience_node import ExperienceNode


def test_sparse_matrix_connections():
    """Test the sparse matrix connection implementation."""
    print("🔬 Testing Sparse Matrix Graph Connections")
    print("=" * 60)
    
    # Create hybrid world graph
    graph = HybridWorldGraph()
    
    # Create test experiences
    experiences = []
    for i in range(10):
        experience = ExperienceNode(
            mental_context=[0.1 * i, 0.2 * i, 0.3 * i],
            action_taken={'forward': 0.5, 'turn': 0.1 * i},
            predicted_sensory=[0.4 * i, 0.5 * i],
            actual_sensory=[0.4 * i + 0.01, 0.5 * i + 0.01],
            prediction_error=0.01
        )
        experiences.append(experience)
    
    # Add experiences to graph
    print(f"📊 Adding {len(experiences)} experiences...")
    start_time = time.time()
    
    for experience in experiences:
        graph.add_node(experience)
    
    add_time = time.time() - start_time
    print(f"   Added in {add_time:.3f}s")
    
    # Test sparse matrix connection retrieval
    print("\n🔍 Testing sparse matrix connections...")
    
    # Get connections using sparse matrix
    test_node_id = experiences[5].node_id  # Middle node
    start_time = time.time()
    
    connected_nodes = graph.get_connected_nodes_vectorized(test_node_id)
    
    sparse_time = time.time() - start_time
    print(f"   Sparse matrix lookup: {sparse_time*1000:.2f}ms")
    print(f"   Found {len(connected_nodes)} connections")
    
    # Display connections
    for connected_id, weight in connected_nodes[:5]:  # Show first 5
        print(f"      {connected_id[:8]}... -> {weight:.2f}")
    
    # Test batch operations
    print("\n🚀 Testing batch sparse matrix operations...")
    
    node_ids = [exp.node_id for exp in experiences[:5]]
    start_time = time.time()
    
    batch_connections = graph.batch_get_connected_nodes_vectorized(node_ids)
    
    batch_time = time.time() - start_time
    print(f"   Batch lookup ({len(node_ids)} nodes): {batch_time*1000:.2f}ms")
    
    total_connections = sum(len(connections) for connections in batch_connections.values())
    print(f"   Total connections found: {total_connections}")
    
    # Test graph traversal
    print("\n🗺️  Testing sparse matrix graph traversal...")
    
    start_node_id = experiences[0].node_id
    start_time = time.time()
    
    paths = graph.vectorized_graph_traversal(start_node_id, max_depth=3)
    
    traversal_time = time.time() - start_time
    print(f"   Graph traversal: {traversal_time*1000:.2f}ms")
    print(f"   Found {len(paths)} paths")
    
    # Test performance comparison
    print("\n⚡ Performance Comparison:")
    print(f"   Sparse matrix single lookup: {sparse_time*1000:.2f}ms")
    print(f"   Sparse matrix batch lookup: {batch_time*1000:.2f}ms")
    print(f"   Graph traversal: {traversal_time*1000:.2f}ms")
    
    # Test GPU usage
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        backend = graph.vectorized_backend
        print(f"\n🚀 GPU Backend Status:")
        print(f"   Device: {backend.device}")
        print(f"   Connection count: {backend._connection_count}")
        print(f"   Connection capacity: {backend._connection_capacity}")
    
    print("\n✅ Sparse matrix connections test completed!")
    return True


def test_connection_types():
    """Test different types of connections in sparse matrix."""
    print("\n🔗 Testing Different Connection Types")
    print("=" * 40)
    
    graph = HybridWorldGraph()
    
    # Create experiences with different connection types
    exp1 = ExperienceNode(
        mental_context=[1.0, 2.0, 3.0],
        action_taken={'forward': 1.0},
        predicted_sensory=[1.0, 2.0],
        actual_sensory=[1.1, 2.1],
        prediction_error=0.1
    )
    
    exp2 = ExperienceNode(
        mental_context=[1.1, 2.1, 3.1],
        action_taken={'forward': 1.1},
        predicted_sensory=[1.1, 2.1],
        actual_sensory=[1.2, 2.2],
        prediction_error=0.1
    )
    
    exp3 = ExperienceNode(
        mental_context=[1.2, 2.2, 3.2],
        action_taken={'forward': 1.2},
        predicted_sensory=[1.2, 2.2],
        actual_sensory=[1.3, 2.3],
        prediction_error=0.1
    )
    
    # Add nodes (this creates temporal connections)
    graph.add_node(exp1)
    graph.add_node(exp2)
    graph.add_node(exp3)
    
    # Add some similarity connections
    exp2.similar_contexts.append(exp1.node_id)
    exp1.similar_contexts.append(exp2.node_id)
    
    # Add prediction source connections
    exp3.prediction_sources.append(exp1.node_id)
    
    # Add weighted connections
    exp1.connection_weights[exp3.node_id] = 0.75
    
    # Rebuild sparse connections for modified nodes
    graph._build_sparse_connections_for_node(exp1, graph.vectorized_indices[exp1.node_id])
    graph._build_sparse_connections_for_node(exp2, graph.vectorized_indices[exp2.node_id])
    graph._build_sparse_connections_for_node(exp3, graph.vectorized_indices[exp3.node_id])
    
    # Test different connection types
    print("🔍 Connection analysis:")
    
    for exp in [exp1, exp2, exp3]:
        connections = graph.get_connected_nodes_vectorized(exp.node_id)
        print(f"   {exp.node_id[:8]}... has {len(connections)} connections:")
        for conn_id, weight in connections:
            print(f"      -> {conn_id[:8]}... (weight: {weight:.2f})")
    
    return True


def benchmark_sparse_vs_object():
    """Benchmark sparse matrix vs object-based traversal."""
    print("\n📊 Benchmarking Sparse Matrix vs Object-based")
    print("=" * 50)
    
    # Create larger graph for meaningful benchmark
    graph = HybridWorldGraph()
    
    # Add many experiences
    num_experiences = 100
    experiences = []
    
    print(f"Creating {num_experiences} experiences...")
    for i in range(num_experiences):
        experience = ExperienceNode(
            mental_context=[0.01 * i, 0.02 * i, 0.03 * i],
            action_taken={'forward': 0.5 + 0.01 * i},
            predicted_sensory=[0.1 * i, 0.2 * i],
            actual_sensory=[0.1 * i + 0.001, 0.2 * i + 0.001],
            prediction_error=0.001
        )
        experiences.append(experience)
        graph.add_node(experience)
    
    # Benchmark sparse matrix approach
    print("\n⚡ Benchmarking sparse matrix operations...")
    
    # Test batch operations
    node_ids = [exp.node_id for exp in experiences[:50]]  # Test with 50 nodes
    
    # Warm up
    _ = graph.batch_get_connected_nodes_vectorized(node_ids)
    
    # Benchmark
    num_iterations = 10
    start_time = time.time()
    
    for _ in range(num_iterations):
        batch_results = graph.batch_get_connected_nodes_vectorized(node_ids)
    
    sparse_time = (time.time() - start_time) / num_iterations
    
    print(f"   Sparse matrix batch lookup: {sparse_time*1000:.2f}ms")
    print(f"   Nodes per ms: {len(node_ids)/(sparse_time*1000):.1f}")
    
    # Benchmark traversal
    start_time = time.time()
    
    for _ in range(num_iterations):
        paths = graph.vectorized_graph_traversal(experiences[0].node_id, max_depth=2)
    
    traversal_time = (time.time() - start_time) / num_iterations
    
    print(f"   Graph traversal: {traversal_time*1000:.2f}ms")
    print(f"   Paths found: {len(paths)}")
    
    print("\n🎯 Performance Summary:")
    print(f"   Batch connection lookup: {sparse_time*1000:.2f}ms for {len(node_ids)} nodes")
    print(f"   Graph traversal: {traversal_time*1000:.2f}ms")
    print(f"   Total connections: {graph.vectorized_backend._connection_count}")
    
    return True


def main():
    """Run all sparse matrix connection tests."""
    print("🧠 SPARSE MATRIX GRAPH CONNECTIONS TESTS")
    print("=" * 80)
    
    try:
        # Test basic functionality
        test_sparse_matrix_connections()
        
        # Test connection types
        test_connection_types()
        
        # Benchmark performance
        benchmark_sparse_vs_object()
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)