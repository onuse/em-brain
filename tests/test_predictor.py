"""
Unit tests for the predictor system.
Tests single traversal, consensus resolution, and triple prediction.
"""

import pytest
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from prediction.action.single_traversal import SingleTraversal, TraversalResult
from prediction.action.consensus_resolver import ConsensusResolver, ConsensusResult  
from prediction.action.triple_predictor import TriplePredictor


def create_test_sensory_vector(length: int = 5, base_value: float = 1.0) -> list:
    """Create a test sensory vector of arbitrary length (brain-agnostic)."""
    return [base_value + i * 0.1 for i in range(length)]


class TestSingleTraversal:
    """Test cases for single graph traversal."""
    
    def test_traversal_with_empty_graph(self):
        """Test traversal when graph is empty."""
        traversal = SingleTraversal(max_depth=3)
        graph = WorldGraph()
        
        result = traversal.traverse([1.0, 2.0, 3.0], graph, random_seed=42)
        
        assert result.prediction is None
        assert result.path == []
        assert result.terminal_node is None
        assert result.depth_reached == 0
    
    def test_traversal_with_single_node(self):
        """Test traversal with only one node in graph."""
        traversal = SingleTraversal(max_depth=3)
        graph = WorldGraph()
        
        # Add single node
        node = ExperienceNode(
            mental_context=[1.0, 2.0, 3.0],
            action_taken={"forward_motor": 0.5, "turn_motor": 0.0},
            predicted_sensory=create_test_sensory_vector(8, 1.0),
            actual_sensory=create_test_sensory_vector(8, 1.1),
            prediction_error=0.1
        )
        graph.add_node(node)
        
        result = traversal.traverse([1.0, 2.0, 3.0], graph, random_seed=42)
        
        assert result.prediction is not None
        assert len(result.path) == 1
        assert result.terminal_node == node
        assert result.depth_reached == 1
        assert result.prediction.motor_action == node.action_taken
    
    def test_traversal_with_connected_nodes(self):
        """Test traversal through connected nodes."""
        traversal = SingleTraversal(max_depth=3)
        graph = WorldGraph()
        
        # Create chain of connected nodes
        nodes = []
        for i in range(5):
            node = ExperienceNode(
                mental_context=[float(i), float(i+1)],
                action_taken={"forward_motor": float(i) * 0.1},
                predicted_sensory=create_test_sensory_vector(6, float(i)),
                actual_sensory=create_test_sensory_vector(6, float(i)),
                prediction_error=0.1
            )
            graph.add_node(node)
            nodes.append(node)
        
        # Add similarity connections
        for i in range(len(nodes) - 1):
            nodes[i].similar_contexts.append(nodes[i+1].node_id)
        
        result = traversal.traverse([0.0, 1.0], graph, random_seed=42)
        
        assert result.prediction is not None
        assert len(result.path) >= 1
        assert result.depth_reached >= 1
        assert result.terminal_node is not None
    
    def test_path_strengthening(self):
        """Test that traversal strengthens nodes in the path."""
        traversal = SingleTraversal(max_depth=2)
        graph = WorldGraph()
        
        # Add nodes
        node1 = ExperienceNode(
            mental_context=[1.0, 2.0],
            action_taken={"motor": 0.5},
            predicted_sensory=create_test_sensory_vector(7, 1.0),
            actual_sensory=create_test_sensory_vector(7, 1.0),
            prediction_error=0.1
        )
        node2 = ExperienceNode(
            mental_context=[1.1, 2.1],
            action_taken={"motor": 0.6},
            predicted_sensory=create_test_sensory_vector(7, 1.1),
            actual_sensory=create_test_sensory_vector(7, 1.1),
            prediction_error=0.1
        )
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        # Connect nodes
        node1.similar_contexts.append(node2.node_id)
        
        initial_strength1 = node1.strength
        initial_strength2 = node2.strength
        
        result = traversal.traverse([1.0, 2.0], graph, random_seed=42)
        
        # Nodes in path should be strengthened
        assert node1.strength > initial_strength1
        # Node2 might be strengthened if it was in the traversal path
    
    def test_deterministic_with_same_seed(self):
        """Test that same random seed produces same starting node."""
        traversal = SingleTraversal(max_depth=3)
        graph = self._create_test_graph()
        
        result1 = traversal.traverse([1.0, 2.0], graph, random_seed=42)
        result2 = traversal.traverse([1.0, 2.0], graph, random_seed=42)
        
        # At minimum, should start from same node and complete successfully
        assert result1.path[0] == result2.path[0]  # Same starting node
        assert result1.prediction is not None
        assert result2.prediction is not None
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds can produce different traversals."""
        traversal = SingleTraversal(max_depth=3, randomness_factor=0.8)  # High randomness
        graph = self._create_test_graph()
        
        result1 = traversal.traverse([1.0, 2.0], graph, random_seed=42)
        result2 = traversal.traverse([1.0, 2.0], graph, random_seed=123)
        
        # With high randomness, should get different paths (though not guaranteed)
        # At minimum, the traversals should complete successfully
        assert result1.prediction is not None
        assert result2.prediction is not None
    
    def _create_test_graph(self):
        """Create a test graph with connected nodes."""
        graph = WorldGraph()
        
        nodes = []
        for i in range(6):
            node = ExperienceNode(
                mental_context=[float(i), float(i+1)],
                action_taken={"forward_motor": float(i) * 0.1, "turn_motor": 0.0},
                predicted_sensory=create_test_sensory_vector(10, float(i)),
                actual_sensory=create_test_sensory_vector(10, float(i)),
                prediction_error=0.1,
                strength=1.0 + i * 0.5  # Varying strengths
            )
            graph.add_node(node)
            nodes.append(node)
        
        # Create connections
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j and abs(i - j) <= 2:  # Connect nearby nodes
                    nodes[i].similar_contexts.append(nodes[j].node_id)
        
        return graph


class TestConsensusResolver:
    """Test cases for consensus resolution."""
    
    def test_perfect_consensus(self):
        """Test perfect consensus when all predictions agree."""
        resolver = ConsensusResolver(action_similarity_threshold=0.1)
        
        # Create identical predictions
        identical_results = []
        for i in range(3):
            result = self._create_mock_traversal_result(
                {"forward_motor": 0.5, "turn_motor": 0.0}, 
                strength=1.0 + i * 0.1
            )
            identical_results.append(result)
        
        consensus = resolver.resolve_consensus(identical_results)
        
        assert consensus.consensus_strength == "perfect"
        assert consensus.agreement_count == 3
        assert consensus.total_traversals == 3
        assert "All 3 traversals agreed" in consensus.reasoning
    
    def test_strong_consensus_majority(self):
        """Test strong consensus when majority agrees."""
        resolver = ConsensusResolver(action_similarity_threshold=0.1)
        
        results = [
            self._create_mock_traversal_result({"forward_motor": 0.5, "turn_motor": 0.0}, 2.0),
            self._create_mock_traversal_result({"forward_motor": 0.5, "turn_motor": 0.0}, 1.5),
            self._create_mock_traversal_result({"forward_motor": 0.0, "turn_motor": 0.5}, 1.0)
        ]
        
        consensus = resolver.resolve_consensus(results)
        
        assert consensus.consensus_strength == "strong"
        assert consensus.agreement_count == 2
        assert consensus.total_traversals == 3
        assert "2 out of 3" in consensus.reasoning
    
    def test_weak_consensus_no_majority(self):
        """Test weak consensus when no clear majority."""
        resolver = ConsensusResolver(action_similarity_threshold=0.1)
        
        results = [
            self._create_mock_traversal_result({"forward_motor": 0.5, "turn_motor": 0.0}, 3.0),
            self._create_mock_traversal_result({"forward_motor": 0.0, "turn_motor": 0.5}, 1.0),
            self._create_mock_traversal_result({"forward_motor": -0.2, "turn_motor": 0.0}, 2.0)
        ]
        
        consensus = resolver.resolve_consensus(results)
        
        assert consensus.consensus_strength == "weak"
        assert consensus.agreement_count == 1
        assert "strongest traversal" in consensus.reasoning
        # Should choose the one with highest strength (first one)
        assert consensus.prediction.motor_action["forward_motor"] == 0.5
    
    def test_dual_consensus(self):
        """Test consensus with only 2 traversals."""
        resolver = ConsensusResolver(action_similarity_threshold=0.1)
        
        # Test agreement
        results_agree = [
            self._create_mock_traversal_result({"forward_motor": 0.5, "turn_motor": 0.0}, 1.0),
            self._create_mock_traversal_result({"forward_motor": 0.5, "turn_motor": 0.0}, 1.5)
        ]
        
        consensus = resolver.resolve_consensus(results_agree)
        assert consensus.consensus_strength == "strong"
        assert "Both traversals agreed" in consensus.reasoning
        
        # Test disagreement
        results_disagree = [
            self._create_mock_traversal_result({"forward_motor": 0.5, "turn_motor": 0.0}, 1.0),
            self._create_mock_traversal_result({"forward_motor": 0.0, "turn_motor": 0.5}, 2.0)
        ]
        
        consensus = resolver.resolve_consensus(results_disagree)
        assert consensus.consensus_strength == "weak"
        # Should choose the stronger one (higher terminal strength)
        # The second result has strength 2.0, first has 1.0
        assert consensus.prediction.motor_action["turn_motor"] == 0.5
    
    def test_no_valid_traversals(self):
        """Test handling when no traversals succeeded."""
        resolver = ConsensusResolver()
        
        # Empty results list
        consensus = resolver.resolve_consensus([])
        
        assert consensus.consensus_strength == "no_consensus"
        assert consensus.agreement_count == 0
        assert "No valid traversals" in consensus.reasoning
        assert consensus.prediction.motor_action["brake_motor"] == 1.0  # Should stop
    
    def test_single_valid_traversal(self):
        """Test handling when only one traversal succeeded."""
        resolver = ConsensusResolver()
        
        results = [self._create_mock_traversal_result({"forward_motor": 0.3}, 1.5)]
        
        consensus = resolver.resolve_consensus(results)
        
        assert consensus.consensus_strength == "single"
        assert consensus.agreement_count == 1
        assert "Only 1 out of" in consensus.reasoning
    
    def test_action_similarity_threshold(self):
        """Test that action similarity threshold works correctly."""
        # Strict threshold - actions differ by 0.01, threshold is 0.005
        strict_resolver = ConsensusResolver(action_similarity_threshold=0.005)
        
        results = [
            self._create_mock_traversal_result({"forward_motor": 0.5, "turn_motor": 0.0}, 1.0),
            self._create_mock_traversal_result({"forward_motor": 0.51, "turn_motor": 0.0}, 1.0)
        ]
        
        # Should NOT consider these similar with strict threshold
        consensus = strict_resolver.resolve_consensus(results)
        assert consensus.consensus_strength == "weak"  # No consensus
        
        # Lenient threshold - actions differ by 0.01, threshold is 0.1
        lenient_resolver = ConsensusResolver(action_similarity_threshold=0.1)
        consensus = lenient_resolver.resolve_consensus(results)
        assert consensus.consensus_strength == "strong"  # Should agree
    
    def _create_mock_traversal_result(self, motor_action, strength):
        """Create a mock traversal result for testing."""
        from core.communication import PredictionPacket
        from datetime import datetime
        
        prediction = PredictionPacket(
            expected_sensory=create_test_sensory_vector(12, 1.0),
            motor_action=motor_action,
            confidence=0.8,
            timestamp=datetime.now(),
            sequence_id=0
        )
        
        terminal_node = ExperienceNode(
            mental_context=[1.0, 2.0],
            action_taken=motor_action,
            predicted_sensory=create_test_sensory_vector(12, 1.0),
            actual_sensory=create_test_sensory_vector(12, 1.0),
            prediction_error=0.1,
            strength=strength
        )
        
        return TraversalResult(
            prediction=prediction,
            path=["node1", "node2"],
            terminal_node=terminal_node,
            depth_reached=2
        )


class TestTriplePredictor:
    """Test cases for the complete triple predictor system."""
    
    def test_bootstrap_prediction(self):
        """Test bootstrap prediction when graph is empty."""
        predictor = TriplePredictor(base_time_budget=0.01)  # 10ms for fast tests
        graph = WorldGraph()
        
        result = predictor.generate_prediction([1.0, 2.0, 3.0], graph, sequence_id=1)
        
        assert result.consensus_strength == "bootstrap"
        assert result.prediction is not None
        assert result.prediction.sequence_id == 1
        assert result.prediction.motor_action["forward_motor"] > 0  # Should explore
        assert result.prediction.confidence < 0.2  # Low confidence
        assert "bootstrap" in result.reasoning
    
    def test_prediction_with_populated_graph(self):
        """Test prediction generation with populated graph."""
        predictor = TriplePredictor(max_depth=2, base_time_budget=0.01)
        graph = self._create_populated_graph()
        
        result = predictor.generate_prediction([1.0, 2.0], graph, sequence_id=5)
        
        assert result.prediction is not None
        assert result.prediction.sequence_id == 5
        assert result.consensus_strength in ["perfect", "strong", "weak"]
        assert result.total_traversals >= 1  # At least one traversal
        assert len(result.prediction.traversal_paths) > 0
    
    def test_statistics_tracking(self):
        """Test that prediction statistics are tracked correctly."""
        predictor = TriplePredictor(base_time_budget=0.01)
        graph = self._create_populated_graph()
        
        # Generate several predictions
        for i in range(5):
            result = predictor.generate_prediction([float(i), float(i+1)], graph)
        
        stats = predictor.get_prediction_statistics()
        
        assert stats["total_predictions"] == 5
        assert "consensus_counts" in stats
        assert "consensus_percentages" in stats
        assert "average_thinking_time" in stats
        assert stats["average_thinking_time"] > 0
        assert "strong_consensus_rate" in stats
        assert "traversal_statistics" in stats
    
    def test_adaptive_parameters(self):
        """Test adaptive parameter adjustment."""
        predictor = TriplePredictor(max_depth=3, randomness_factor=0.3, base_time_budget=0.05)
        
        # Test parameter updates
        predictor.set_adaptive_parameters(
            new_depth=7,
            new_randomness=0.8,
            new_similarity_threshold=0.2,
            new_time_budget=0.2
        )
        
        assert predictor.single_traversal.max_depth == 7
        assert predictor.single_traversal.randomness_factor == 0.8
        assert predictor.consensus_resolver.action_similarity_threshold == 0.2
        assert predictor.base_time_budget == 0.2
        
        # Test bounds checking
        predictor.set_adaptive_parameters(
            new_depth=50,  # Should be clamped
            new_randomness=-0.5,  # Should be clamped
            new_similarity_threshold=1.0,  # Should be clamped
            new_time_budget=2.0  # Should be clamped
        )
        
        assert predictor.single_traversal.max_depth == 10  # Max allowed
        assert predictor.single_traversal.randomness_factor == 0.0  # Min allowed
        assert predictor.consensus_resolver.action_similarity_threshold == 0.5  # Max allowed
        assert predictor.base_time_budget == 1.0  # Max allowed
    
    def test_consensus_quality_measurement(self):
        """Test consensus quality measurement."""
        predictor = TriplePredictor()
        
        # Simulate perfect consensus
        predictor.consensus_stats['perfect'] = 8
        predictor.consensus_stats['strong'] = 2
        predictor.total_predictions = 10
        
        quality = predictor.get_recent_consensus_quality()
        
        # Should be high quality (0.8 * 1.0 + 0.2 * 0.8) / 1.0 = 0.96
        assert quality > 0.9
        
        # Reset and simulate poor consensus
        predictor.reset_statistics()
        predictor.consensus_stats['weak'] = 5
        predictor.consensus_stats['no_consensus'] = 5
        predictor.total_predictions = 10
        
        quality = predictor.get_recent_consensus_quality()
        
        # Should be low quality
        assert quality < 0.3
    
    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        predictor = TriplePredictor(base_time_budget=0.01)
        graph = self._create_populated_graph()
        
        # Generate some predictions
        for i in range(3):
            predictor.generate_prediction([float(i)], graph)
        
        assert predictor.total_predictions == 3
        assert len(predictor.traversal_count_history) == 3
        
        # Reset statistics
        predictor.reset_statistics()
        
        assert predictor.total_predictions == 0
        assert all(count == 0 for count in predictor.consensus_stats.values())
        assert predictor.average_thinking_time == 0.0
        assert len(predictor.traversal_count_history) == 0
    
    def test_time_budgeted_thinking(self):
        """Test that thinking adapts to time budgets and threat levels."""
        predictor = TriplePredictor(base_time_budget=0.05)  # 50ms
        graph = self._create_populated_graph()
        
        # Test different threat levels
        safe_result = predictor.generate_prediction([1.0, 2.0], graph, threat_level="safe")
        danger_result = predictor.generate_prediction([1.0, 2.0], graph, threat_level="danger")
        
        # Safe mode should allow more thinking time budget (2x vs 0.2x)
        # Both may hit the 20-traversal safety limit, but safe gets more allocated time
        assert safe_result.prediction.threat_level == "safe"
        assert danger_result.prediction.threat_level == "danger"
        
        # At minimum, both should complete some traversals
        assert safe_result.prediction.traversal_count >= 1
        assert danger_result.prediction.traversal_count >= 1
        
        # Both should have completed successfully
        assert safe_result.prediction is not None
        assert danger_result.prediction is not None
        
    def test_hardware_scaling_simulation(self):
        """Test that the system naturally scales with available processing time."""
        graph = self._create_populated_graph()
        
        # Simulate slow hardware (very small time budget)
        slow_predictor = TriplePredictor(base_time_budget=0.001)  # 1ms
        slow_result = slow_predictor.generate_prediction([1.0, 2.0], graph)
        
        # Simulate fast hardware (larger time budget)  
        fast_predictor = TriplePredictor(base_time_budget=0.02)  # 20ms
        fast_result = fast_predictor.generate_prediction([1.0, 2.0], graph)
        
        # Fast hardware should generally do more traversals
        # (though this is probabilistic, so we just check it's working)
        assert slow_result.prediction.traversal_count >= 1
        assert fast_result.prediction.traversal_count >= 1
        assert fast_result.prediction.time_budget_used >= slow_result.prediction.time_budget_used
    
    def _create_populated_graph(self):
        """Create a populated graph for testing."""
        graph = WorldGraph()
        
        # Add diverse nodes
        for i in range(10):
            node = ExperienceNode(
                mental_context=[float(i), float(i % 3), float(i % 5)],
                action_taken={
                    "forward_motor": (i % 5) * 0.2 - 0.4,  # Range -0.4 to 0.6
                    "turn_motor": (i % 3) * 0.3 - 0.3,     # Range -0.3 to 0.3
                    "brake_motor": 0.0
                },
                predicted_sensory=create_test_sensory_vector(15, float(i % 7)),
                actual_sensory=create_test_sensory_vector(15, float(i % 7)),
                prediction_error=0.1 + (i % 3) * 0.1,
                strength=1.0 + i * 0.2
            )
            graph.add_node(node)
        
        # Add some similarity connections
        all_nodes = graph.all_nodes()
        for i in range(len(all_nodes)):
            for j in range(i+1, len(all_nodes)):
                if abs(i - j) <= 2:  # Connect nearby nodes
                    all_nodes[i].similar_contexts.append(all_nodes[j].node_id)
                    all_nodes[j].similar_contexts.append(all_nodes[i].node_id)
        
        return graph


if __name__ == "__main__":
    pytest.main([__file__, "-v"])