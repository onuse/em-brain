#!/usr/bin/env python3
"""
Fixed Sensor-Motor Adapter for PiCar-X to Brain Interface

This module handles the conversion between:
- PiCar-X sensors (16 channels) → Brain inputs (16 channels) - NO PADDING!
- Brain outputs (4 channels) → PiCar-X motors (5 channels)

Key changes:
- Removed reward signal generation (brain discovers rewards through experience)
- Fixed 16→16 sensor mapping (no artificial padding to 24)
- All normalization and safety limits preserved
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import time

try:
    from ..config.brainstem_config import BrainstemConfig, get_config
except ImportError:
    # Handle relative import when running as main script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.brainstem_config import BrainstemConfig, get_config


class PiCarXBrainAdapter:
    """
    Adapts between PiCar-X hardware interface and brain expectations.
    
    Key responsibilities:
    1. Normalize 16 sensor channels for brain input
    2. Convert 4 brain outputs to 5 motor commands
    3. Apply safety limits and smoothing
    
    NO REWARD SIGNALS - Brain learns through experience!
    """
    
    def __init__(self, config: BrainstemConfig = None):
        """Initialize the adapter."""
        self.config = config or get_config()
        
        # Motor smoothing
        self.prev_motor_commands = [0.0] * self.config.motors.picarx_motor_count
        
        # State tracking for derived sensors
        self.prev_distance = None
        self.prev_steering = 0.0
        
        print("🔧 PiCar-X Brain Adapter initialized (No rewards, pure 16-channel)")
    
    def sensors_to_brain_input(self, sensor_data: List[float]) -> List[float]:
        """
        Convert PiCar-X 16 sensors to brain's 16 normalized inputs.
        
        Args:
            sensor_data: List of 16 sensor values from PiCar-X
            
        Returns:
            List of 16 normalized values for brain input
        """
        if len(sensor_data) != self.config.sensors.picarx_sensor_count:
            print(f"⚠️  Expected {self.config.sensors.picarx_sensor_count} sensors, got {len(sensor_data)}")
            # Pad or truncate to exactly 16
            if len(sensor_data) < 16:
                sensor_data = sensor_data + [0.5] * (16 - len(sensor_data))
            else:
                sensor_data = sensor_data[:16]
        
        # Initialize with normalized sensor values
        brain_input = [0.5] * 16  # Start with neutral values
        
        # === Direct sensor mappings (0-11) ===
        # 0: Ultrasonic distance (normalized, inverted - closer = higher)
        distance = sensor_data[0]  # meters
        brain_input[0] = 1.0 - min(distance / self.config.sensors.ultrasonic_max_distance, 1.0)
        
        # 1-3: Grayscale sensors (already 0-1)
        brain_input[1] = sensor_data[1]  # Right grayscale
        brain_input[2] = sensor_data[2]  # Center grayscale
        brain_input[3] = sensor_data[3]  # Left grayscale
        
        # 4-5: Motor speeds (normalized to 0-1)
        brain_input[4] = (sensor_data[4] + self.config.sensors.motor_range_half) / (2 * self.config.sensors.motor_range_half)
        brain_input[5] = (sensor_data[5] + self.config.sensors.motor_range_half) / (2 * self.config.sensors.motor_range_half)
        
        # 6-7: Camera servos (normalized to 0-1)
        brain_input[6] = (sensor_data[6] + self.config.sensors.camera_pan_offset) / self.config.sensors.camera_pan_range
        brain_input[7] = (sensor_data[7] + self.config.sensors.camera_tilt_offset) / self.config.sensors.camera_tilt_range
        
        # 8: Steering angle (normalized to 0-1)
        brain_input[8] = (sensor_data[8] + self.config.sensors.steering_offset) / self.config.sensors.steering_range
        
        # 9: Battery voltage (normalized)
        brain_input[9] = (sensor_data[9] - self.config.sensors.battery_min) / (self.config.sensors.battery_max - self.config.sensors.battery_min)
        brain_input[9] = max(0.0, min(1.0, brain_input[9]))  # Clamp to 0-1
        
        # 10: Line detection (binary, already 0 or 1)
        brain_input[10] = sensor_data[10]
        
        # 11: Cliff detection (binary, already 0 or 1)
        brain_input[11] = sensor_data[11]
        
        # === Derived/computed sensors (12-15) ===
        # 12: CPU temperature (normalized)
        brain_input[12] = min(sensor_data[12] / self.config.sensors.cpu_temp_critical, 1.0)
        
        # 13: Distance gradient (how fast distance is changing)
        if self.prev_distance is not None:
            distance_change = (distance - self.prev_distance) * 10  # Scale up small changes
            brain_input[13] = 0.5 + np.clip(distance_change, -0.5, 0.5)  # Center at 0.5
        else:
            brain_input[13] = 0.5
        self.prev_distance = distance
        
        # 14: Angular velocity estimate (from motor differential)
        angular_vel = (sensor_data[5] - sensor_data[4]) / (2 * self.config.sensors.motor_range_half)
        brain_input[14] = 0.5 + np.clip(angular_vel * 0.5, -0.5, 0.5)  # Center at 0.5
        
        # 15: System health indicator (composite)
        health = 1.0
        if sensor_data[9] < self.config.sensors.low_battery_threshold:
            health -= 0.3
        if sensor_data[12] > self.config.sensors.high_temp_threshold:
            health -= 0.2
        if sensor_data[11] > 0:  # Cliff detected
            health -= 0.5
        brain_input[15] = max(0.0, health)
        
        # Ensure all values are in [0, 1] range
        brain_input = [max(0.0, min(1.0, v)) for v in brain_input]
        
        return brain_input
    
    def brain_output_to_motors(self, brain_output: List[float]) -> Dict[str, float]:
        """
        Convert brain's 4 outputs to PiCar-X's 5 motor commands.
        
        Args:
            brain_output: List of 4 values from brain (range varies)
            
        Returns:
            Dictionary with 5 motor commands
        
        Brain output mapping:
        - Channel 0: Forward/backward (primary locomotion)
        - Channel 1: Left/right (steering)
        - Channel 2: Camera pan
        - Channel 3: Camera tilt / auxiliary
        """
        if len(brain_output) < 4:
            print(f"⚠️  Expected 4 brain outputs, got {len(brain_output)}, padding with zeros")
            brain_output = brain_output + [0.0] * (4 - len(brain_output))
        
        # Initialize motor commands
        commands = {}
        
        # === Locomotion (channels 0-1) ===
        # Channel 0: Forward/backward movement
        forward_command = brain_output[0]
        forward_speed = np.clip(forward_command * self.config.motors.max_motor_speed, 
                                -self.config.motors.max_motor_speed, 
                                self.config.motors.max_motor_speed)
        
        # Channel 1: Steering
        steering_command = brain_output[1]
        steering_angle = np.clip(steering_command * self.config.motors.max_steering_angle,
                                 -self.config.motors.max_steering_angle,
                                 self.config.motors.max_steering_angle)
        
        # Apply differential drive for turning
        if abs(steering_angle) > self.config.sensors.straight_steering_threshold:
            # Reduce speed when turning
            turn_factor = 1.0 - (abs(steering_angle) / self.config.motors.max_steering_angle) * 0.5
            forward_speed *= turn_factor
            
            # Differential speed for turning
            turn_differential = (steering_angle / self.config.motors.max_steering_angle) * forward_speed * 0.3
            commands['left_motor'] = forward_speed + turn_differential
            commands['right_motor'] = forward_speed - turn_differential
        else:
            # Straight movement
            commands['left_motor'] = forward_speed
            commands['right_motor'] = forward_speed
        
        commands['steering_servo'] = steering_angle
        
        # === Camera control (channels 2-3) ===
        # Channel 2: Camera pan
        pan_command = brain_output[2] if len(brain_output) > 2 else 0.0
        commands['camera_pan_servo'] = np.clip(
            pan_command * self.config.motors.camera_pan_range,
            self.config.motors.camera_pan_min,
            self.config.motors.camera_pan_max
        )
        
        # Channel 3: Camera tilt
        tilt_command = brain_output[3] if len(brain_output) > 3 else 0.0
        commands['camera_tilt_servo'] = np.clip(
            tilt_command * self.config.motors.camera_tilt_range,
            self.config.motors.camera_tilt_min,
            self.config.motors.camera_tilt_max
        )
        
        # === Apply smoothing ===
        for i, key in enumerate(['left_motor', 'right_motor', 'steering_servo', 
                                 'camera_pan_servo', 'camera_tilt_servo']):
            if key in ['left_motor', 'right_motor']:
                alpha = self.config.motors.motor_smoothing_alpha
            elif key == 'steering_servo':
                alpha = self.config.motors.steering_smoothing_alpha
            else:
                alpha = self.config.motors.servo_smoothing_alpha
            
            # Exponential smoothing
            smoothed = alpha * commands[key] + (1 - alpha) * self.prev_motor_commands[i]
            commands[key] = smoothed
            self.prev_motor_commands[i] = smoothed
        
        return commands
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information about the adapter state."""
        return {
            'prev_distance': self.prev_distance,
            'prev_motor_commands': self.prev_motor_commands,
            'sensor_dimensions': self.config.sensors.picarx_sensor_count,
            'motor_dimensions': self.config.motors.picarx_motor_count,
            'brain_input_dim': 16,  # Fixed!
            'brain_output_dim': self.config.motors.brain_output_dimensions
        }


def test_adapter():
    """Test the sensor-motor adapter."""
    print("🧪 Testing Fixed PiCar-X Brain Adapter")
    print("=" * 50)
    
    adapter = PiCarXBrainAdapter()
    
    # Test sensor data (16 channels)
    test_sensors = [
        0.5,   # Distance (meters)
        0.3, 0.8, 0.3,  # Grayscale sensors
        20, 20,  # Motor speeds
        0, 0,    # Camera servos
        0,       # Steering angle
        7.4,     # Battery voltage
        1,       # Line detected
        0,       # No cliff
        45,      # CPU temp
        0.3,     # Memory usage (unused)
        1000,    # Free disk (unused)
        0        # Reserved
    ]
    
    print(f"\nInput sensors (16 values): {test_sensors}")
    
    # Convert to brain input
    brain_input = adapter.sensors_to_brain_input(test_sensors)
    print(f"\nBrain input (16 normalized): {[f'{v:.2f}' for v in brain_input]}")
    
    # Test brain output
    test_brain_output = [0.5, 0.2, 0.0, 0.1]  # Forward with slight right turn
    print(f"\nBrain output (4 values): {test_brain_output}")
    
    # Convert to motors
    motor_commands = adapter.brain_output_to_motors(test_brain_output)
    print(f"\nMotor commands:")
    for key, value in motor_commands.items():
        print(f"  {key}: {value:.2f}")
    
    # Show debug info
    debug = adapter.get_debug_info()
    print(f"\nDebug info:")
    for key, value in debug.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Adapter test complete - NO REWARDS, pure 16-channel mapping!")


if __name__ == "__main__":
    test_adapter()