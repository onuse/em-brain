"""
Performance tests for the core data structures.
Tests scaling behavior with 1000+ nodes.
"""

import pytest
import time
import random
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode


class TestPerformance:
    """Performance test cases for large-scale operations."""
    
    def test_1000_node_creation_and_storage(self):
        """Test creating and storing 1000+ experience nodes."""
        graph = WorldGraph()
        start_time = time.time()
        
        # Create 1000 nodes
        for i in range(1000):
            node = ExperienceNode(
                mental_context=[random.random() for _ in range(10)],
                action_taken={"forward_motor": random.uniform(-1, 1), "turn_motor": random.uniform(-1, 1)},
                predicted_sensory=[random.random() for _ in range(22)],
                actual_sensory=[random.random() for _ in range(22)],
                prediction_error=random.random()
            )
            graph.add_node(node)
        
        creation_time = time.time() - start_time
        
        assert graph.node_count() == 1000
        assert creation_time < 5.0  # Should complete in under 5 seconds
        print(f"Created 1000 nodes in {creation_time:.3f} seconds")
    
    def test_similarity_search_performance(self):
        """Test similarity search performance with 1000+ nodes."""
        graph = WorldGraph()
        
        # Create 1000 nodes with varied contexts
        for i in range(1000):
            # Create clusters of similar contexts
            base_context = [float(i % 10), float(i % 5), float(i % 3)]
            noise = [random.uniform(-0.1, 0.1) for _ in range(7)]
            context = base_context + noise
            
            node = ExperienceNode(
                mental_context=context,
                action_taken={"motor": random.uniform(-1, 1)},
                predicted_sensory=[random.random() for _ in range(22)],
                actual_sensory=[random.random() for _ in range(22)],
                prediction_error=random.random()
            )
            graph.add_node(node)
        
        # Test similarity search performance
        search_context = [1.0, 2.0, 1.0] + [0.0] * 7
        
        start_time = time.time()
        similar_nodes = graph.find_similar_nodes(search_context, similarity_threshold=0.7, max_results=10)
        search_time = time.time() - start_time
        
        assert len(similar_nodes) <= 10
        assert search_time < 1.0  # Should complete in under 1 second
        print(f"Similarity search through 1000 nodes took {search_time:.3f} seconds")
    
    def test_strength_updates_performance(self):
        """Test strength update performance with many nodes."""
        graph = WorldGraph()
        node_ids = []
        
        # Create nodes
        for i in range(1000):
            node = ExperienceNode(
                mental_context=[random.random() for _ in range(5)],
                action_taken={"motor": random.uniform(-1, 1)},
                predicted_sensory=[random.random() for _ in range(22)],
                actual_sensory=[random.random() for _ in range(22)],
                prediction_error=random.random()
            )
            node_id = graph.add_node(node)
            node_ids.append(node_id)
        
        # Test batch strength updates
        start_time = time.time()
        for i in range(100):  # Update 100 random nodes
            node_id = random.choice(node_ids)
            graph.strengthen_node(node_id, 0.1)
        update_time = time.time() - start_time
        
        assert update_time < 1.0  # Should complete quickly
        print(f"100 strength updates on 1000-node graph took {update_time:.3f} seconds")
    
    def test_memory_usage_scaling(self):
        """Test that memory usage scales reasonably with node count."""
        import sys
        
        graph = WorldGraph()
        
        # Measure initial memory
        initial_nodes = 100
        for i in range(initial_nodes):
            node = ExperienceNode(
                mental_context=[random.random() for _ in range(10)],
                action_taken={"motor": random.uniform(-1, 1)},
                predicted_sensory=[random.random() for _ in range(22)],
                actual_sensory=[random.random() for _ in range(22)],
                prediction_error=random.random()
            )
            graph.add_node(node)
        
        initial_size = sys.getsizeof(graph.nodes)
        
        # Add more nodes
        additional_nodes = 900
        for i in range(additional_nodes):
            node = ExperienceNode(
                mental_context=[random.random() for _ in range(10)],
                action_taken={"motor": random.uniform(-1, 1)},
                predicted_sensory=[random.random() for _ in range(22)],
                actual_sensory=[random.random() for _ in range(22)],
                prediction_error=random.random()
            )
            graph.add_node(node)
        
        final_size = sys.getsizeof(graph.nodes)
        
        # Memory should scale roughly linearly
        size_ratio = final_size / initial_size
        node_ratio = (initial_nodes + additional_nodes) / initial_nodes
        
        # Allow some overhead, but should be reasonably close to linear
        assert size_ratio <= node_ratio * 2.0
        print(f"Memory scaling: {node_ratio:.1f}x nodes, {size_ratio:.1f}x memory")
    
    def test_temporal_chain_performance(self):
        """Test temporal chain operations with many nodes."""
        graph = WorldGraph()
        
        # Create a long temporal chain
        for i in range(1000):
            node = ExperienceNode(
                mental_context=[float(i)] * 5,
                action_taken={"motor": random.uniform(-1, 1)},
                predicted_sensory=[float(i)] * 22,
                actual_sensory=[float(i)] * 22,
                prediction_error=random.random()
            )
            graph.add_node(node)
        
        # Test temporal sequence retrieval
        first_node = list(graph.nodes.values())[0]
        
        start_time = time.time()
        sequence = graph.get_temporal_sequence(first_node.node_id, length=50)
        sequence_time = time.time() - start_time
        
        assert len(sequence) == 50
        assert sequence_time < 0.1  # Should be very fast
        print(f"Retrieved 50-node temporal sequence in {sequence_time:.4f} seconds")
    
    def test_merge_performance_large_graph(self):
        """Test node merging performance with large graph."""
        graph = WorldGraph()
        
        # Create nodes with some similar contexts (to enable merging)
        for i in range(500):
            # Create pairs of similar nodes
            base_context = [float(i % 100), float(i % 50)]
            
            node1 = ExperienceNode(
                mental_context=base_context + [0.0] * 3,
                action_taken={"motor": 0.5},
                predicted_sensory=[1.0] * 22,
                actual_sensory=[1.0] * 22,
                prediction_error=0.1,
                strength=0.3  # Low strength to enable merging
            )
            
            node2 = ExperienceNode(
                mental_context=base_context + [0.1] * 3,  # Very similar
                action_taken={"motor": 0.51},
                predicted_sensory=[1.01] * 22,
                actual_sensory=[1.01] * 22,
                prediction_error=0.11,
                strength=0.3  # Low strength to enable merging
            )
            
            graph.add_node(node1)
            graph.add_node(node2)
        
        initial_count = graph.node_count()
        
        start_time = time.time()
        merges_performed = graph.merge_similar_nodes(similarity_threshold=0.95, strength_threshold=0.5)
        merge_time = time.time() - start_time
        
        final_count = graph.node_count()
        
        assert merges_performed > 0
        assert final_count < initial_count
        assert merge_time < 5.0  # Should complete in reasonable time
        print(f"Merged {merges_performed} nodes from {initial_count} to {final_count} in {merge_time:.3f} seconds")
    
    def test_graph_statistics_performance(self):
        """Test statistics calculation performance with large graph."""
        graph = WorldGraph()
        
        # Create 1000 nodes
        for i in range(1000):
            node = ExperienceNode(
                mental_context=[random.random() for _ in range(8)],
                action_taken={"motor": random.uniform(-1, 1)},
                predicted_sensory=[random.random() for _ in range(22)],
                actual_sensory=[random.random() for _ in range(22)],
                prediction_error=random.random(),
                strength=random.uniform(0.1, 2.0),
                times_accessed=random.randint(0, 50)
            )
            graph.add_node(node)
        
        # Test statistics calculation
        start_time = time.time()
        stats = graph.get_graph_statistics()
        stats_time = time.time() - start_time
        
        assert stats["total_nodes"] == 1000
        assert "avg_strength" in stats
        assert "max_strength" in stats
        assert "min_strength" in stats
        assert stats_time < 0.5  # Should be fast
        print(f"Calculated statistics for 1000 nodes in {stats_time:.4f} seconds")
    
    def test_decay_all_performance(self):
        """Test decay operation performance on large graph."""
        graph = WorldGraph()
        
        # Create 1000 nodes
        for i in range(1000):
            node = ExperienceNode(
                mental_context=[random.random() for _ in range(5)],
                action_taken={"motor": random.uniform(-1, 1)},
                predicted_sensory=[random.random() for _ in range(22)],
                actual_sensory=[random.random() for _ in range(22)],
                prediction_error=random.random(),
                strength=random.uniform(0.5, 2.0)
            )
            graph.add_node(node)
        
        # Test decay all operation
        start_time = time.time()
        graph.decay_all_nodes(decay_rate=0.001)
        decay_time = time.time() - start_time
        
        assert decay_time < 1.0  # Should complete quickly
        print(f"Applied decay to 1000 nodes in {decay_time:.4f} seconds")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print statements