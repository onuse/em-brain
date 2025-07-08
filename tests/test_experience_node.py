"""
Unit tests for ExperienceNode class.
"""

import pytest
from datetime import datetime
from core.experience_node import ExperienceNode


class TestExperienceNode:
    """Test cases for ExperienceNode functionality."""
    
    def test_creation_with_required_fields(self):
        """Test creating an ExperienceNode with all required fields."""
        mental_context = [0.1, 0.2, 0.3, 0.4]
        action = {"forward_motor": 0.5, "turn_motor": 0.0}
        predicted = [1.0, 2.0, 3.0]
        actual = [1.1, 2.1, 2.9]
        error = 0.15
        
        node = ExperienceNode(
            mental_context=mental_context,
            action_taken=action,
            predicted_sensory=predicted,
            actual_sensory=actual,
            prediction_error=error
        )
        
        assert node.mental_context == mental_context
        assert node.action_taken == action
        assert node.predicted_sensory == predicted
        assert node.actual_sensory == actual
        assert node.prediction_error == error
        assert node.strength == 1.0
        assert node.times_accessed == 0
        assert isinstance(node.node_id, str)
        assert isinstance(node.timestamp, datetime)
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        node = ExperienceNode(
            mental_context=[0.1, 0.2],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        assert node.strength == 1.0
        assert node.times_accessed == 0
        assert node.merge_count == 0
        assert node.temporal_predecessor is None
        assert node.temporal_successor is None
        assert node.similar_contexts == []
        assert node.prediction_sources == []
    
    def test_unique_node_ids(self):
        """Test that each node gets a unique ID."""
        node1 = ExperienceNode(
            mental_context=[0.1],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        node2 = ExperienceNode(
            mental_context=[0.1],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        assert node1.node_id != node2.node_id
    
    def test_access_node(self):
        """Test the access_node method updates counters correctly."""
        node = ExperienceNode(
            mental_context=[0.1],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        initial_time = node.last_accessed
        initial_count = node.times_accessed
        
        node.access_node()
        
        assert node.times_accessed == initial_count + 1
        assert node.last_accessed > initial_time
    
    def test_calculate_prediction_accuracy(self):
        """Test prediction accuracy calculation."""
        # Perfect prediction
        node1 = ExperienceNode(
            mental_context=[0.1],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0, 2.0, 3.0],
            actual_sensory=[1.0, 2.0, 3.0],
            prediction_error=0.0
        )
        
        assert node1.calculate_prediction_accuracy() == 1.0
        
        # Poor prediction
        node2 = ExperienceNode(
            mental_context=[0.1],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0, 2.0, 3.0],
            actual_sensory=[3.0, 4.0, 5.0],
            prediction_error=3.46  # sqrt((2^2 + 2^2 + 2^2)) = sqrt(12) ≈ 3.46
        )
        
        accuracy = node2.calculate_prediction_accuracy()
        assert 0.0 <= accuracy <= 1.0
        assert accuracy < 1.0
    
    def test_calculate_prediction_accuracy_mismatched_lengths(self):
        """Test prediction accuracy with mismatched array lengths."""
        node = ExperienceNode(
            mental_context=[0.1],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0, 2.0, 3.0],
            actual_sensory=[1.0, 2.0],  # Different length
            prediction_error=0.0
        )
        
        assert node.calculate_prediction_accuracy() == 0.0
    
    def test_is_similar_context(self):
        """Test context similarity detection."""
        node = ExperienceNode(
            mental_context=[1.0, 2.0, 3.0],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        # Identical context
        assert node.is_similar_context([1.0, 2.0, 3.0], threshold=0.7)
        
        # Similar context
        assert node.is_similar_context([1.1, 2.1, 3.1], threshold=0.7)
        
        # Different context
        assert not node.is_similar_context([5.0, 6.0, 7.0], threshold=0.7)
        
        # Different length
        assert not node.is_similar_context([1.0, 2.0], threshold=0.7)
    
    def test_merge_with(self):
        """Test node merging functionality."""
        node1 = ExperienceNode(
            mental_context=[1.0, 2.0],
            action_taken={"forward_motor": 0.5, "turn_motor": 0.0},
            predicted_sensory=[3.0, 4.0],
            actual_sensory=[3.1, 4.1],
            prediction_error=0.1,
            strength=2.0,
            times_accessed=5
        )
        
        node2 = ExperienceNode(
            mental_context=[1.2, 2.2],
            action_taken={"forward_motor": 0.7, "turn_motor": 0.1},
            predicted_sensory=[3.2, 4.2],
            actual_sensory=[3.3, 4.3],
            prediction_error=0.2,
            strength=1.5,
            times_accessed=3
        )
        
        merged = node1.merge_with(node2, weight=0.5)
        
        # Check merged values are averages (with floating point tolerance)
        assert abs(merged.mental_context[0] - 1.1) < 0.01
        assert abs(merged.mental_context[1] - 2.1) < 0.01
        assert abs(merged.predicted_sensory[0] - 3.1) < 0.01
        assert abs(merged.predicted_sensory[1] - 4.1) < 0.01
        assert abs(merged.actual_sensory[0] - 3.2) < 0.01
        assert abs(merged.actual_sensory[1] - 4.2) < 0.01
        assert abs(merged.prediction_error - 0.15) < 0.01
        
        # Check aggregated values
        assert merged.times_accessed == 8
        assert merged.strength == 2.1  # max(2.0, 1.5) + 0.1
        assert merged.merge_count == 1
    
    def test_equality_and_hashing(self):
        """Test node equality and hashing based on node_id."""
        node1 = ExperienceNode(
            mental_context=[0.1],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        node2 = ExperienceNode(
            mental_context=[0.1],
            action_taken={"motor": 0.0},
            predicted_sensory=[1.0],
            actual_sensory=[1.0],
            prediction_error=0.0
        )
        
        # Different nodes should not be equal
        assert node1 != node2
        assert hash(node1) != hash(node2)
        
        # Same node should be equal to itself
        assert node1 == node1
        assert hash(node1) == hash(node1)
    
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        node = ExperienceNode(
            mental_context=[],
            action_taken={},
            predicted_sensory=[],
            actual_sensory=[],
            prediction_error=0.0
        )
        
        assert node.calculate_prediction_accuracy() == 0.0
        assert node.is_similar_context([], threshold=0.7)
        assert not node.is_similar_context([1.0], threshold=0.7)


if __name__ == "__main__":
    pytest.main([__file__])