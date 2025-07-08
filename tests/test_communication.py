"""
Unit tests for communication data structures.
"""

import pytest
from datetime import datetime
from core.communication import PredictionPacket, SensoryPacket


class TestPredictionPacket:
    """Test cases for PredictionPacket functionality."""
    
    def test_creation_with_required_fields(self):
        """Test creating a PredictionPacket with all required fields."""
        expected_sensory = [1.0, 2.0, 3.0]
        motor_action = {"forward_motor": 0.5, "turn_motor": 0.0}
        confidence = 0.8
        timestamp = datetime.now()
        sequence_id = 123
        
        packet = PredictionPacket(
            expected_sensory=expected_sensory,
            motor_action=motor_action,
            confidence=confidence,
            timestamp=timestamp,
            sequence_id=sequence_id
        )
        
        assert packet.expected_sensory == expected_sensory
        assert packet.motor_action == motor_action
        assert packet.confidence == confidence
        assert packet.timestamp == timestamp
        assert packet.sequence_id == sequence_id
        assert packet.traversal_paths == []
        assert packet.consensus_strength == "unknown"
        assert packet.thinking_depth == 0
    
    def test_is_high_confidence(self):
        """Test high confidence detection."""
        packet = PredictionPacket(
            expected_sensory=[1.0],
            motor_action={"motor": 0.5},
            confidence=0.8,
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        assert packet.is_high_confidence(threshold=0.7)
        assert not packet.is_high_confidence(threshold=0.9)
    
    def test_add_traversal_path(self):
        """Test adding traversal paths."""
        packet = PredictionPacket(
            expected_sensory=[1.0],
            motor_action={"motor": 0.5},
            confidence=0.8,
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        path1 = ["node1", "node2", "node3"]
        path2 = ["node4", "node5"]
        
        packet.add_traversal_path(path1)
        packet.add_traversal_path(path2)
        
        assert len(packet.traversal_paths) == 2
        assert packet.traversal_paths[0] == path1
        assert packet.traversal_paths[1] == path2
    
    def test_get_consensus_info(self):
        """Test getting consensus information."""
        packet = PredictionPacket(
            expected_sensory=[1.0],
            motor_action={"motor": 0.5},
            confidence=0.8,
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        packet.add_traversal_path(["node1", "node2"])
        packet.add_traversal_path(["node3", "node4"])
        packet.consensus_strength = "strong"
        packet.thinking_depth = 5
        
        info = packet.get_consensus_info()
        
        assert info["paths_count"] == 2
        assert info["consensus_strength"] == "strong"
        assert info["thinking_depth"] == 5
        assert info["confidence"] == 0.8


class TestSensoryPacket:
    """Test cases for SensoryPacket functionality."""
    
    def test_creation_with_required_fields(self):
        """Test creating a SensoryPacket with all required fields."""
        sensor_values = [1.0, 2.0, 3.0, 4.0]
        actuator_positions = [0.5, 0.0, 0.8]
        timestamp = datetime.now()
        sequence_id = 456
        
        packet = SensoryPacket(
            sensor_values=sensor_values,
            actuator_positions=actuator_positions,
            timestamp=timestamp,
            sequence_id=sequence_id
        )
        
        assert packet.sensor_values == sensor_values
        assert packet.actuator_positions == actuator_positions
        assert packet.timestamp == timestamp
        assert packet.sequence_id == sequence_id
        assert packet.brainstem_prediction is None
        assert packet.network_latency == 0.0
    
    def test_get_sensor_count(self):
        """Test getting sensor count."""
        packet = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0],
            actuator_positions=[0.5],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        assert packet.get_sensor_count() == 3
    
    def test_get_actuator_count(self):
        """Test getting actuator count."""
        packet = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0],
            actuator_positions=[0.5, 0.0],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        assert packet.get_actuator_count() == 2
    
    def test_is_sensor_in_range(self):
        """Test sensor range checking."""
        packet = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0],
            actuator_positions=[0.5],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        assert packet.is_sensor_in_range(0, 0.5, 1.5)
        assert not packet.is_sensor_in_range(0, 2.0, 3.0)
        assert not packet.is_sensor_in_range(10, 0.0, 5.0)  # Invalid index
    
    def test_get_sensor_value(self):
        """Test getting sensor values with defaults."""
        packet = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0],
            actuator_positions=[0.5],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        assert packet.get_sensor_value(0) == 1.0
        assert packet.get_sensor_value(1) == 2.0
        assert packet.get_sensor_value(10, default=99.0) == 99.0
    
    def test_calculate_prediction_error(self):
        """Test prediction error calculation."""
        packet = SensoryPacket(
            sensor_values=[1.0, 2.0, 3.0],
            actuator_positions=[0.5],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        # Perfect prediction
        error = packet.calculate_prediction_error([1.0, 2.0, 3.0])
        assert error == 0.0
        
        # Prediction with error
        error = packet.calculate_prediction_error([1.5, 2.5, 3.5])
        assert error > 0.0
        
        # Mismatched lengths
        error = packet.calculate_prediction_error([1.0, 2.0])
        assert error == float('inf')
    
    def test_with_brainstem_prediction(self):
        """Test with brainstem prediction metadata."""
        brainstem_pred = {"motor1": 0.5, "motor2": 0.0}
        
        packet = SensoryPacket(
            sensor_values=[1.0, 2.0],
            actuator_positions=[0.5, 0.0],
            timestamp=datetime.now(),
            sequence_id=1,
            brainstem_prediction=brainstem_pred,
            network_latency=0.05
        )
        
        assert packet.brainstem_prediction == brainstem_pred
        assert packet.network_latency == 0.05
    
    def test_empty_sensor_values(self):
        """Test handling of empty sensor values."""
        packet = SensoryPacket(
            sensor_values=[],
            actuator_positions=[],
            timestamp=datetime.now(),
            sequence_id=1
        )
        
        assert packet.get_sensor_count() == 0
        assert packet.get_actuator_count() == 0
        assert packet.get_sensor_value(0, default=5.0) == 5.0
        assert not packet.is_sensor_in_range(0, 0.0, 1.0)
        
        # Error calculation with empty arrays
        error = packet.calculate_prediction_error([])
        assert error == 0.0


if __name__ == "__main__":
    pytest.main([__file__])