"""
Unit tests for the Brain Interface - tests world-agnostic brain architecture.
Demonstrates that brain adapts to arbitrary sensory vector lengths.
"""

import pytest
from datetime import datetime
from core.brain_interface import BrainInterface
from core.communication import SensoryPacket
from predictor.triple_predictor import TriplePredictor


class TestBrainInterface:
    """Test cases for the world-agnostic brain interface."""
    
    def test_brain_learns_sensory_dimensions(self):
        """Test that brain automatically learns sensory vector length from brainstem."""
        predictor = TriplePredictor(base_time_budget=0.01)
        brain = BrainInterface(predictor)
        
        # First sensory input teaches brain the vector length
        sensory_packet_5 = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0, 4.0, 5.0],  # 5 sensors
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        # Brain should learn from first input
        assert brain.sensory_vector_length is None
        
        prediction = brain.process_sensory_input(
            sensory_packet_5, 
            mental_context=[0.0, 0.0, 1.0],  # position, energy
            threat_level="normal"
        )
        
        # Brain should have learned the vector length
        assert brain.sensory_vector_length == 5
        assert len(prediction.expected_sensory) == 5
        
    def test_brain_adapts_to_different_sensor_counts(self):
        """Test that different brainstems can use different sensor counts."""
        # Test with 3-sensor robot
        predictor_3 = TriplePredictor(base_time_budget=0.01)
        brain_3 = BrainInterface(predictor_3)
        
        sensory_3 = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0],  # 3 sensors
            actuator_positions=[0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        prediction_3 = brain_3.process_sensory_input(
            sensory_3, [0.0, 1.0], "normal"
        )
        
        # Test with 47-sensor robot (like Mars rover)
        predictor_47 = TriplePredictor(base_time_budget=0.01)
        brain_47 = BrainInterface(predictor_47)
        
        sensory_47 = SensoryPacket(
            sensor_values=list(range(47)),  # 47 sensors
            actuator_positions=[0.0] * 10,
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        prediction_47 = brain_47.process_sensory_input(
            sensory_47, [0.0, 1.0], "normal"
        )
        
        # Each brain adapts to its brainstem
        assert brain_3.sensory_vector_length == 3
        assert brain_47.sensory_vector_length == 47
        assert len(prediction_3.expected_sensory) == 3
        assert len(prediction_47.expected_sensory) == 47
        
    def test_brain_validates_consistent_sensory_length(self):
        """Test that brain rejects inconsistent sensory vector lengths."""
        predictor = TriplePredictor(base_time_budget=0.01)
        brain = BrainInterface(predictor)
        
        # First input with 4 sensors
        sensory_4 = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0, 4.0],
            actuator_positions=[0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        brain.process_sensory_input(sensory_4, [0.0], "normal")
        assert brain.sensory_vector_length == 4
        
        # Second input with different length should fail
        sensory_6 = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            actuator_positions=[0.0],
            timestamp=datetime.now(),
            sequence_id=2
        )
        
        with pytest.raises(ValueError, match="Inconsistent sensory length"):
            brain.process_sensory_input(sensory_6, [0.0], "normal")
            
    def test_experience_creation_with_arbitrary_sensors(self):
        """Test that experiences are created correctly with arbitrary sensor counts."""
        predictor = TriplePredictor(base_time_budget=0.01)
        brain = BrainInterface(predictor)
        
        # Process two sensory inputs to create one experience
        sensory_1 = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # 6 sensors
            actuator_positions=[0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        sensory_2 = SensoryPacket(
            sensor_values=[1.1, 2.1, 3.1, 4.1, 5.1, 6.1],  # Slightly different
            actuator_positions=[0.0, 0.0],
            timestamp=datetime.now(),
            sequence_id=2
        )
        
        # First prediction
        brain.process_sensory_input(sensory_1, [0.0, 1.0], "normal")
        assert brain.world_graph.get_graph_statistics()["total_nodes"] == 0  # No experience yet
        
        # Second prediction creates experience from first
        brain.process_sensory_input(sensory_2, [0.1, 1.0], "normal")
        assert brain.world_graph.get_graph_statistics()["total_nodes"] == 1  # Experience created
        
        # Check that experience has correct sensory lengths
        nodes = brain.world_graph.all_nodes()
        experience = nodes[0]
        assert len(experience.predicted_sensory) == 6
        assert len(experience.actual_sensory) == 6
        
    def test_threat_level_affects_thinking_time(self):
        """Test that different threat levels affect prediction generation."""
        predictor = TriplePredictor(base_time_budget=0.02)
        brain = BrainInterface(predictor)
        
        sensory = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0],
            actuator_positions=[0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        # Safe thinking (more time)
        safe_pred = brain.process_sensory_input(sensory, [0.0], "safe")
        
        # Danger thinking (less time)
        danger_pred = brain.process_sensory_input(sensory, [0.0], "danger")
        
        # Both should work, but with different characteristics
        assert safe_pred is not None
        assert danger_pred is not None
        assert safe_pred.threat_level == "safe"
        assert danger_pred.threat_level == "danger"
        
    def test_brain_statistics_are_comprehensive(self):
        """Test that brain provides comprehensive statistics."""
        predictor = TriplePredictor(base_time_budget=0.01)
        brain = BrainInterface(predictor)
        
        sensory = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0, 4.0],
            actuator_positions=[0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        brain.process_sensory_input(sensory, [0.0, 1.0], "normal")
        
        stats = brain.get_brain_statistics()
        
        assert "graph_stats" in stats
        assert "predictor_stats" in stats
        assert "interface_stats" in stats
        assert stats["interface_stats"]["sensory_vector_length"] == 4
        assert stats["interface_stats"]["is_learning"] is True
        
    def test_brain_reset_preserves_learned_dimensions(self):
        """Test that brain reset keeps learned sensory dimensions."""
        predictor = TriplePredictor(base_time_budget=0.01)
        brain = BrainInterface(predictor)
        
        sensory = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0, 4.0, 5.0],
            actuator_positions=[0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        brain.process_sensory_input(sensory, [0.0], "normal")
        assert brain.sensory_vector_length == 5
        
        # Reset brain
        brain.reset_brain()
        
        # Should remember interface dimensions
        assert brain.sensory_vector_length == 5
        assert brain.world_graph.get_graph_statistics()["total_nodes"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])