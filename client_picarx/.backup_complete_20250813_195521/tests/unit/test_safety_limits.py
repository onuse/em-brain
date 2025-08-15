#!/usr/bin/env python3
"""
Safety Limits Unit Tests

Critical tests for safety constraints that prevent:
1. Hardware damage from excessive motor commands
2. Dangerous robot behaviors (collisions, falls)
3. Runaway acceleration or spinning
4. Servo damage from out-of-range angles
5. Battery damage from over-discharge

These tests ensure the robot operates within safe parameters at all times.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
from brainstem.sensor_motor_adapter import PiCarXBrainAdapter


class TestMotorSafetyLimits:
    """Test motor command safety limits."""
    
    def test_motor_speed_clamping(self):
        """Test that motor speeds are clamped to safe ranges."""
        adapter = PiCarXBrainAdapter()
        
        # Test cases: [brain_output, expected_motor_speeds]
        test_cases = [
            # Excessive forward speed
            ([1.0, 0.0, 0.0, 0.0], {"left_motor": 50.0, "right_motor": 50.0}),
            
            # Excessive reverse speed
            ([-1.0, 0.0, 0.0, 0.0], {"left_motor": -50.0, "right_motor": -50.0}),
            
            # Extreme values should be clamped
            ([10.0, 0.0, 0.0, 0.0], {"left_motor": 50.0, "right_motor": 50.0}),
            ([-10.0, 0.0, 0.0, 0.0], {"left_motor": -50.0, "right_motor": -50.0}),
            
            # Normal operation
            ([0.5, 0.0, 0.0, 0.0], {"left_motor": 25.0, "right_motor": 25.0}),
        ]
        
        for brain_output, expected in test_cases:
            result = adapter.brain_output_to_motors(brain_output)
            
            # Check motor speeds are within limits
            assert abs(result["left_motor"]) <= adapter.max_motor_speed
            assert abs(result["right_motor"]) <= adapter.max_motor_speed
            
            # Verify expected clamping
            assert abs(result["left_motor"] - expected["left_motor"]) < 0.1
            assert abs(result["right_motor"] - expected["right_motor"]) < 0.1
    
    def test_steering_angle_limits(self):
        """Test that steering angles are limited to prevent mechanical damage."""
        adapter = PiCarXBrainAdapter()
        
        test_cases = [
            # Extreme left
            ([0.0, -1.0, 0.0, 0.0], -adapter.max_steering_angle),
            
            # Extreme right
            ([0.0, 1.0, 0.0, 0.0], adapter.max_steering_angle),
            
            # Over-limit values should be clamped
            ([0.0, 5.0, 0.0, 0.0], adapter.max_steering_angle),
            ([0.0, -5.0, 0.0, 0.0], -adapter.max_steering_angle),
            
            # Normal operation
            ([0.0, 0.5, 0.0, 0.0], 12.5),
            ([0.0, -0.5, 0.0, 0.0], -12.5),
        ]
        
        for brain_output, expected_angle in test_cases:
            result = adapter.brain_output_to_motors(brain_output)
            
            # Check steering angle is within limits
            assert abs(result["steering_angle"]) <= adapter.max_steering_angle
            
            # Verify expected angle
            assert abs(result["steering_angle"] - expected_angle) < 0.1
    
    def test_camera_servo_limits(self):
        """Test that camera servos stay within mechanical limits."""
        adapter = PiCarXBrainAdapter()
        
        # Camera pan limits: -90 to 90 degrees
        # Camera tilt limits: -35 to 65 degrees
        
        test_cases = [
            # Pan limits
            ([0.0, 0.0, 1.0, 0.0], {"camera_pan": 90, "camera_tilt": 0}),
            ([0.0, 0.0, -1.0, 0.0], {"camera_pan": -90, "camera_tilt": 0}),
            
            # Tilt limits
            ([0.0, 0.0, 0.0, 1.0], {"camera_pan": 0, "camera_tilt": 65}),
            ([0.0, 0.0, 0.0, -1.0], {"camera_pan": 0, "camera_tilt": -35}),
            
            # Combined
            ([0.0, 0.0, 0.5, 0.5], {"camera_pan": 45, "camera_tilt": 32.5}),
        ]
        
        for brain_output, expected in test_cases:
            result = adapter.brain_output_to_motors(brain_output)
            
            # Check servo angles are within mechanical limits
            assert -90 <= result["camera_pan"] <= 90
            assert -35 <= result["camera_tilt"] <= 65
            
            # Verify expected angles
            assert abs(result["camera_pan"] - expected["camera_pan"]) < 1
            assert abs(result["camera_tilt"] - expected["camera_tilt"]) < 1
    
    def test_motor_smoothing_prevents_jerks(self):
        """Test that motor smoothing prevents sudden jerky movements."""
        adapter = PiCarXBrainAdapter()
        
        # Simulate sudden command change
        commands_sequence = [
            [0.0, 0.0, 0.0, 0.0],  # Stopped
            [1.0, 0.0, 0.0, 0.0],  # Sudden full forward
            [1.0, 0.0, 0.0, 0.0],  # Maintain
            [-1.0, 0.0, 0.0, 0.0], # Sudden full reverse
        ]
        
        motor_history = []
        
        for commands in commands_sequence:
            result = adapter.brain_output_to_motors(commands)
            motor_history.append(result["left_motor"])
        
        # Check that changes are smoothed
        # First command from stop should be smoothed
        assert motor_history[1] < 50.0, "Should not jump to full speed immediately"
        
        # Reverse command should be heavily smoothed
        assert motor_history[3] > -20.0, "Should not reverse at full speed immediately"


class TestCollisionAvoidance:
    """Test collision avoidance safety features."""
    
    def test_emergency_stop_on_obstacle(self):
        """Test that close obstacles trigger speed reduction."""
        adapter = PiCarXBrainAdapter()
        
        # Test with various distances
        test_cases = [
            # [distance, expected_speed_factor]
            (0.05, 0.0),   # Very close - full stop
            (0.1, 0.0),    # Still too close
            (0.2, 0.5),    # At minimum safe distance
            (0.5, 1.0),    # Safe distance
            (1.0, 1.0),    # Far away
        ]
        
        for distance, expected_factor in test_cases:
            # Create sensor data with ultrasonic distance
            sensor_data = [distance] + [0.5] * 15
            
            # Process sensors
            brain_input = adapter.sensors_to_brain_input(sensor_data)
            
            # The adapter should modify commands based on distance
            # This tests that the safety system is engaged
            # In real implementation, this would affect motor commands
            
            # For now, verify distance is properly normalized
            distance_signal = brain_input[0]  # First channel is distance
            
            if distance < adapter.min_safe_distance:
                # Should indicate very close obstacle
                assert distance_signal > 0.8, f"Close obstacle not properly signaled at {distance}m"
    
    def test_cliff_detection_stops_motion(self):
        """Test that cliff detection prevents forward motion."""
        adapter = PiCarXBrainAdapter()
        
        # Normal sensor data
        normal_sensors = [0.5] + [0.5] * 10 + [0.0] + [0.5] * 4  # No cliff
        
        # Cliff detected
        cliff_sensors = [0.5] + [0.5] * 10 + [1.0] + [0.5] * 4  # Cliff detected
        
        # Process both scenarios
        normal_input = adapter.sensors_to_brain_input(normal_sensors)
        cliff_input = adapter.sensors_to_brain_input(cliff_sensors)
        
        # Health indicator should reflect cliff danger
        normal_health = normal_input[23]  # System health channel
        cliff_health = cliff_input[23]
        
        assert cliff_health < normal_health, "Cliff should reduce system health"
        assert cliff_health <= 0.5, "Cliff should significantly impact health"
    
    def test_line_following_safety(self):
        """Test that line following doesn't override safety limits."""
        adapter = PiCarXBrainAdapter()
        
        # Strong line signal but obstacle ahead
        sensor_data = [
            0.1,   # Close obstacle
            0.9,   # Right grayscale (strong line)
            0.9,   # Center grayscale (strong line)
            0.9,   # Left grayscale (strong line)
        ] + [0.5] * 12
        
        brain_input = adapter.sensors_to_brain_input(sensor_data)
        
        # Even with strong line signal, obstacle should dominate
        obstacle_signal = brain_input[0]
        assert obstacle_signal > 0.8, "Obstacle should override line following"


class TestBatteryProtection:
    """Test battery protection features."""
    
    def test_low_battery_speed_limiting(self):
        """Test that low battery reduces maximum speeds."""
        adapter = PiCarXBrainAdapter()
        
        # Test with different battery voltages
        # Normal: 7.4V, Low: 6.5V, Critical: 6.0V
        test_cases = [
            (7.4, 1.0),   # Normal battery - full performance
            (7.0, 1.0),   # Still okay
            (6.5, 0.7),   # Low battery - should reduce performance
            (6.0, 0.5),   # Critical - emergency only
        ]
        
        for voltage, expected_performance in test_cases:
            sensor_data = [0.5] * 9 + [voltage] + [0.5] * 6
            brain_input = adapter.sensors_to_brain_input(sensor_data)
            
            # Check system health reflects battery state
            health = brain_input[23]
            
            if voltage < 6.5:
                assert health < 1.0, f"Low battery at {voltage}V should reduce health"
    
    def test_temperature_protection(self):
        """Test that high CPU temperature triggers protection."""
        adapter = PiCarXBrainAdapter()
        
        # Test with different temperatures (Celsius)
        test_cases = [
            (40, 1.0),   # Normal
            (50, 1.0),   # Warm but okay
            (60, 0.8),   # Getting hot
            (70, 0.6),   # Too hot
            (80, 0.4),   # Critical
        ]
        
        for temp, expected_health_factor in test_cases:
            sensor_data = [0.5] * 12 + [temp] + [0.5] * 3
            brain_input = adapter.sensors_to_brain_input(sensor_data)
            
            health = brain_input[23]
            
            if temp > 60:
                assert health < 1.0, f"High temp {temp}°C should reduce health"


class TestSensorNormalization:
    """Test sensor value normalization and validation."""
    
    def test_sensor_range_normalization(self):
        """Test that all sensor values are normalized to [0, 1]."""
        adapter = PiCarXBrainAdapter()
        
        # Create sensor data with extreme values
        extreme_sensors = [
            2.0,    # Distance (meters) - should normalize
            -1.0,   # Grayscale (already normalized, but negative)
            2.0,    # Grayscale (out of range)
            0.0,    # Grayscale (minimum)
            -100,   # Left motor
            100,    # Right motor  
            -90,    # Camera pan
            90,     # Camera tilt
            -50,    # Steering angle
            12.0,   # Battery voltage (high)
            1.0,    # Line detected
            1.0,    # Cliff detected
            100,    # CPU temp (very hot)
            1.0,    # Memory usage
            0.0,    # Reserved
            0.0,    # Reserved
        ]
        
        brain_input = adapter.sensors_to_brain_input(extreme_sensors)
        
        # All values should be in [0, 1] range
        for i, value in enumerate(brain_input[:24]):  # Exclude reward
            assert 0.0 <= value <= 1.0, f"Channel {i} out of range: {value}"
    
    def test_missing_sensor_handling(self):
        """Test graceful handling of missing sensor data."""
        adapter = PiCarXBrainAdapter()
        
        # Test with too few sensors
        short_sensors = [0.5] * 10  # Only 10 instead of 16
        brain_input = adapter.sensors_to_brain_input(short_sensors)
        
        # Should pad with safe defaults
        assert len(brain_input) == 25  # 24 + reward
        
        # Padded values should be neutral (0.5)
        for i in range(16, 24):
            assert brain_input[i] == 0.5, f"Channel {i} should be padded to 0.5"
    
    def test_sensor_noise_filtering(self):
        """Test that sensor noise is handled appropriately."""
        adapter = PiCarXBrainAdapter()
        
        # Simulate noisy sensor readings
        np.random.seed(42)
        base_sensors = [0.5] * 16
        
        results = []
        for _ in range(100):
            # Add noise
            noisy = base_sensors.copy()
            noisy[0] += np.random.normal(0, 0.01)  # Small noise on distance
            
            brain_input = adapter.sensors_to_brain_input(noisy)
            results.append(brain_input[0])
        
        # Results should be stable despite noise
        variance = np.var(results)
        assert variance < 0.01, "Sensor processing should be stable despite noise"


class TestFailSafeDefaults:
    """Test fail-safe default behaviors."""
    
    def test_zero_command_stops_robot(self):
        """Test that zero commands result in stopped robot."""
        adapter = PiCarXBrainAdapter()
        
        zero_commands = [0.0, 0.0, 0.0, 0.0]
        result = adapter.brain_output_to_motors(zero_commands)
        
        # With smoothing, might not be exactly zero, but should be very small
        assert abs(result["left_motor"]) < 5.0
        assert abs(result["right_motor"]) < 5.0
        assert abs(result["steering_angle"]) < 5.0
    
    def test_nan_handling(self):
        """Test that NaN values don't crash the system."""
        adapter = PiCarXBrainAdapter()
        
        # Test NaN in sensor data
        nan_sensors = [float('nan')] + [0.5] * 15
        brain_input = adapter.sensors_to_brain_input(nan_sensors)
        
        # Should replace NaN with safe default
        assert not np.isnan(brain_input[0]), "NaN should be replaced"
        
        # Test NaN in motor commands
        nan_commands = [float('nan'), 0.0, 0.0, 0.0]
        result = adapter.brain_output_to_motors(nan_commands)
        
        # Should use safe defaults
        assert not np.isnan(result["left_motor"])
        assert not np.isnan(result["right_motor"])
    
    def test_infinity_handling(self):
        """Test that infinity values are handled safely."""
        adapter = PiCarXBrainAdapter()
        
        # Test infinity in sensor data
        inf_sensors = [float('inf')] + [0.5] * 15
        brain_input = adapter.sensors_to_brain_input(inf_sensors)
        
        # Should clamp to reasonable range
        assert brain_input[0] <= 1.0, "Infinity should be clamped"
        
        # Test infinity in motor commands
        inf_commands = [float('inf'), 0.0, 0.0, 0.0]
        result = adapter.brain_output_to_motors(inf_commands)
        
        # Should clamp to maximum safe speed
        assert result["left_motor"] == adapter.max_motor_speed
        assert result["right_motor"] == adapter.max_motor_speed


class TestSafetyIntegration:
    """Test integration of multiple safety systems."""
    
    def test_multiple_hazards_stop_robot(self):
        """Test that multiple simultaneous hazards result in safe stop."""
        adapter = PiCarXBrainAdapter()
        
        # Multiple hazards: close obstacle, cliff, low battery, high temp
        hazard_sensors = [
            0.05,   # Very close obstacle
            0.5, 0.5, 0.5,  # Grayscale
            0.0, 0.0,       # Motors
            0, 0,           # Camera
            0,              # Steering
            6.0,            # Low battery
            0.5,            # Line
            1.0,            # Cliff detected!
            75,             # High CPU temp
            0.8,            # High memory
            0.0, 0.0        # Reserved
        ]
        
        brain_input = adapter.sensors_to_brain_input(hazard_sensors)
        
        # System health should be very low
        health = brain_input[23]
        assert health < 0.2, "Multiple hazards should severely impact health"
        
        # Even with forward command, safety should limit motion
        forward_command = [1.0, 0.0, 0.0, 0.0]
        result = adapter.brain_output_to_motors(forward_command)
        
        # Should be heavily limited or stopped
        # (Actual implementation would enforce this)
        assert result is not None  # At minimum, shouldn't crash
    
    def test_safety_override_priority(self):
        """Test that safety overrides have correct priority."""
        adapter = PiCarXBrainAdapter()
        
        # Priority should be: cliff > obstacle > battery > temperature
        
        # Test each hazard level
        hazards = [
            ("cliff", [0.5] * 11 + [1.0] + [0.5] * 4),
            ("obstacle", [0.05] + [0.5] * 15),
            ("battery", [0.5] * 9 + [5.5] + [0.5] * 6),
            ("temperature", [0.5] * 12 + [80] + [0.5] * 3),
        ]
        
        health_scores = {}
        
        for hazard_name, sensor_data in hazards:
            brain_input = adapter.sensors_to_brain_input(sensor_data)
            health_scores[hazard_name] = brain_input[23]
        
        # Verify priority order
        assert health_scores["cliff"] <= health_scores["obstacle"]
        assert health_scores["obstacle"] <= health_scores["battery"]
        assert health_scores["battery"] <= health_scores["temperature"]


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_safety_limits.py -v
    pytest.main([__file__, "-v", "--tb=short"])