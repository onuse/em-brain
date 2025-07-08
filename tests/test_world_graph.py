"""
Unit tests for WorldGraph class.
"""

import pytest
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode


class TestWorldGraph:
    """Test cases for WorldGraph functionality."""
    
    def test_empty_graph(self):
        """Test empty graph initialization."""
        graph = WorldGraph()
        
        assert not graph.has_nodes()
        assert graph.node_count() == 0
        assert graph.latest_node_id is None
        assert graph.get_latest_node() is None
        assert graph.total_nodes_created == 0
        assert graph.total_merges_performed == 0
    
    def test_add_single_node(self):
        """Test adding a single node to the graph."""
        graph = WorldGraph()
        
        node = ExperienceNode(
            mental_context=[0.1, 0.2],
            action_taken={"motor": 0.5},
            predicted_sensory=[1.0, 2.0],
            actual_sensory=[1.1, 2.1],
            prediction_error=0.1
        )
        
        node_id = graph.add_node(node)
        
        assert graph.has_nodes()
        assert graph.node_count() == 1
        assert graph.latest_node_id == node_id
        assert graph.get_latest_node() == node
        assert graph.total_nodes_created == 1
        assert graph.node_exists(node_id)
        assert graph.get_node(node_id) == node
    
    def test_add_multiple_nodes(self):
        """Test adding multiple nodes and temporal linking."""
        graph = WorldGraph()
        
        node1 = ExperienceNode(
            mental_context=[0.1, 0.2],
            action_taken={"motor": 0.5},
            predicted_sensory=[1.0, 2.0],
            actual_sensory=[1.1, 2.1],
            prediction_error=0.1
        )
        
        node2 = ExperienceNode(
            mental_context=[0.2, 0.3],
            action_taken={"motor": 0.6},
            predicted_sensory=[1.2, 2.2],
            actual_sensory=[1.3, 2.3],
            prediction_error=0.2
        )
        
        id1 = graph.add_node(node1)
        id2 = graph.add_node(node2)
        
        assert graph.node_count() == 2
        assert graph.latest_node_id == id2
        assert graph.total_nodes_created == 2
        
        # Check temporal linking
        assert node1.temporal_successor == id2
        assert node2.temporal_predecessor == id1
        assert node1.temporal_predecessor is None
        assert node2.temporal_successor is None
    
    def test_remove_node(self):
        """Test removing nodes from the graph."""
        graph = WorldGraph()
        
        node1 = ExperienceNode(
            mental_context=[0.1, 0.2],
            action_taken={"motor": 0.5},
            predicted_sensory=[1.0, 2.0],
            actual_sensory=[1.1, 2.1],
            prediction_error=0.1
        )
        
        node2 = ExperienceNode(
            mental_context=[0.2, 0.3],
            action_taken={"motor": 0.6},
            predicted_sensory=[1.2, 2.2],
            actual_sensory=[1.3, 2.3],
            prediction_error=0.2
        )
        
        id1 = graph.add_node(node1)
        id2 = graph.add_node(node2)
        
        # Remove first node
        assert graph.remove_node(id1)
        assert graph.node_count() == 1
        assert not graph.node_exists(id1)
        assert graph.get_node(id1) is None
        
        # Check temporal chain is fixed
        assert node2.temporal_predecessor is None
        
        # Try to remove non-existent node
        assert not graph.remove_node("non_existent")
    
    def test_find_similar_nodes(self):
        """Test finding nodes with similar contexts."""
        graph = WorldGraph()
        
        # Add nodes with similar contexts
        node1 = ExperienceNode(
            mental_context=[1.0, 2.0, 3.0],
            action_taken={"motor": 0.5},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        node2 = ExperienceNode(
            mental_context=[1.1, 2.1, 3.1],  # Similar to node1
            action_taken={"motor": 0.6},
            predicted_sensory=[1.1],
            actual_sensory=[1.1],
            prediction_error=0.0
        )
        
        node3 = ExperienceNode(
            mental_context=[5.0, 6.0, 7.0],  # Different from others
            action_taken={"motor": 0.7},
            predicted_sensory=[5.0],
            actual_sensory=[5.0],
            prediction_error=0.0
        )
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        # Find similar nodes to node1's context
        similar = graph.find_similar_nodes([1.0, 2.0, 3.0], similarity_threshold=0.7)
        
        assert len(similar) >= 1
        assert node1 in similar
        
        # With high threshold, should find fewer matches
        similar_strict = graph.find_similar_nodes([1.0, 2.0, 3.0], similarity_threshold=0.95)
        assert len(similar_strict) <= len(similar)
    
    def test_find_nodes_by_action(self):
        """Test finding nodes by similar actions."""
        graph = WorldGraph()
        
        node1 = ExperienceNode(
            mental_context=[1.0],
            action_taken={"forward_motor": 0.5, "turn_motor": 0.0},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        node2 = ExperienceNode(
            mental_context=[2.0],
            action_taken={"forward_motor": 0.55, "turn_motor": 0.05},  # Similar action
            predicted_sensory=[2.0],
            actual_sensory=[2.0],
            prediction_error=0.0
        )
        
        node3 = ExperienceNode(
            mental_context=[3.0],
            action_taken={"forward_motor": 1.0, "turn_motor": 0.5},  # Different action
            predicted_sensory=[3.0],
            actual_sensory=[3.0],
            prediction_error=0.0
        )
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        # Find nodes with similar actions
        similar = graph.find_nodes_by_action(
            {"forward_motor": 0.5, "turn_motor": 0.0}, 
            tolerance=0.1
        )
        
        assert node1 in similar
        assert node2 in similar
        assert node3 not in similar
    
    def test_strength_tracking(self):
        """Test node strength tracking and updates."""
        graph = WorldGraph()
        
        node = ExperienceNode(
            mental_context=[1.0],
            action_taken={"motor": 0.5},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        node_id = graph.add_node(node)
        
        # Test strength update
        graph.update_node_strength(node_id, 2.5)
        assert node.strength == 2.5
        assert node.times_accessed == 1
        
        # Test strengthen
        graph.strengthen_node(node_id, 0.5)
        assert node.strength == 3.0
        
        # Test weaken
        graph.weaken_node(node_id, 0.2)
        assert node.strength == 2.8
    
    def test_get_nodes_by_strength(self):
        """Test retrieving nodes by strength ranges."""
        graph = WorldGraph()
        
        # Add nodes with different strengths
        node1 = ExperienceNode(
            mental_context=[1.0],
            action_taken={"motor": 0.5},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        node2 = ExperienceNode(
            mental_context=[2.0],
            action_taken={"motor": 0.6},
            predicted_sensory=[2.0],
            actual_sensory=[2.0],
            prediction_error=0.0
        )
        
        id1 = graph.add_node(node1)
        id2 = graph.add_node(node2)
        
        # Set different strengths
        graph.update_node_strength(id1, 0.5)
        graph.update_node_strength(id2, 2.0)
        
        # Test range queries
        weak_nodes = graph.get_nodes_by_strength_range(0.0, 1.0)
        strong_nodes = graph.get_nodes_by_strength_range(1.5, 3.0)
        
        assert node1 in weak_nodes
        assert node2 in strong_nodes
        assert node1 not in strong_nodes
        assert node2 not in weak_nodes
    
    def test_get_weakest_strongest_nodes(self):
        """Test getting weakest and strongest nodes."""
        graph = WorldGraph()
        
        nodes = []
        for i in range(5):
            node = ExperienceNode(
                mental_context=[float(i)],
                action_taken={"motor": 0.5},
                predicted_sensory=[float(i)],
                actual_sensory=[float(i)],
                prediction_error=0.0
            )
            node_id = graph.add_node(node)
            graph.update_node_strength(node_id, float(i))
            nodes.append(node)
        
        weakest = graph.get_weakest_nodes(2)
        strongest = graph.get_strongest_nodes(2)
        
        assert len(weakest) == 2
        assert len(strongest) == 2
        assert weakest[0].strength <= weakest[1].strength
        assert strongest[0].strength >= strongest[1].strength
    
    def test_merge_similar_nodes(self):
        """Test merging of similar nodes."""
        graph = WorldGraph()
        
        # Add two similar nodes with low strength
        node1 = ExperienceNode(
            mental_context=[1.0, 2.0],
            action_taken={"motor": 0.5},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        node2 = ExperienceNode(
            mental_context=[1.01, 2.01],  # Very similar
            action_taken={"motor": 0.51},
            predicted_sensory=[1.01],
            actual_sensory=[1.01],
            prediction_error=0.0
        )
        
        id1 = graph.add_node(node1)
        id2 = graph.add_node(node2)
        
        # Set low strength to make them mergeable
        graph.update_node_strength(id1, 0.3)
        graph.update_node_strength(id2, 0.3)
        
        initial_count = graph.node_count()
        merges = graph.merge_similar_nodes(similarity_threshold=0.95, strength_threshold=0.5)
        
        assert merges > 0
        assert graph.node_count() < initial_count
        assert graph.total_merges_performed > 0
    
    def test_temporal_sequence(self):
        """Test getting temporal sequences of nodes."""
        graph = WorldGraph()
        
        nodes = []
        for i in range(3):
            node = ExperienceNode(
                mental_context=[float(i)],
                action_taken={"motor": 0.5},
                predicted_sensory=[float(i)],
                actual_sensory=[float(i)],
                prediction_error=0.0
            )
            graph.add_node(node)
            nodes.append(node)
        
        # Get sequence starting from first node
        sequence = graph.get_temporal_sequence(nodes[0].node_id, length=3)
        
        assert len(sequence) == 3
        assert sequence[0] == nodes[0]
        assert sequence[1] == nodes[1]
        assert sequence[2] == nodes[2]
    
    def test_decay_all_nodes(self):
        """Test applying decay to all nodes."""
        graph = WorldGraph()
        
        node = ExperienceNode(
            mental_context=[1.0],
            action_taken={"motor": 0.5},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        node_id = graph.add_node(node)
        initial_strength = node.strength
        
        graph.decay_all_nodes(decay_rate=0.1)
        
        assert node.strength < initial_strength
        assert node.strength >= 0.0
    
    def test_graph_statistics(self):
        """Test graph statistics calculation."""
        graph = WorldGraph()
        
        # Empty graph
        stats = graph.get_graph_statistics()
        assert stats["total_nodes"] == 0
        
        # Add some nodes
        for i in range(3):
            node = ExperienceNode(
                mental_context=[float(i)] * 2,
                action_taken={"motor": 0.5},
                predicted_sensory=[float(i)],
                actual_sensory=[float(i)],
                prediction_error=0.0
            )
            graph.add_node(node)
        
        stats = graph.get_graph_statistics()
        assert stats["total_nodes"] == 3
        assert stats["total_merges"] == 0
        assert "avg_strength" in stats
        assert "max_strength" in stats
        assert "min_strength" in stats
        assert "avg_context_length" in stats
        assert "temporal_chain_length" in stats
        assert "avg_access_count" in stats
        assert "total_accesses" in stats
    
    def test_large_graph_performance(self):
        """Test performance with a larger number of nodes."""
        graph = WorldGraph()
        
        # Add 100 nodes
        for i in range(100):
            node = ExperienceNode(
                mental_context=[float(i % 10), float(i % 5)],
                action_taken={"motor": float(i % 3) / 2.0},
                predicted_sensory=[float(i % 7)],
                actual_sensory=[float(i % 7)],
                prediction_error=float(i % 4) / 10.0
            )
            graph.add_node(node)
        
        assert graph.node_count() == 100
        
        # Test similarity search performance
        similar = graph.find_similar_nodes([1.0, 2.0], similarity_threshold=0.5)
        assert len(similar) >= 0
        
        # Test strength operations
        graph.decay_all_nodes(decay_rate=0.001)
        assert graph.node_count() == 100


if __name__ == "__main__":
    pytest.main([__file__])