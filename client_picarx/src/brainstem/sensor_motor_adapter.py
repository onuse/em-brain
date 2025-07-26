#!/usr/bin/env python3
"""
Sensor-Motor Adapter for PiCar-X to Brain Interface

This module handles the conversion between:
- PiCar-X sensors (16 channels) → Brain inputs (24 channels)
- Brain outputs (4 channels) → PiCar-X motors (5 channels)

It also handles normalization, safety limits, and reward signal generation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import time


class PiCarXBrainAdapter:
    """
    Adapts between PiCar-X hardware interface and brain expectations.
    
    Key responsibilities:
    1. Expand 16 sensor channels to 24 brain inputs
    2. Normalize sensor ranges to [0, 1]
    3. Generate reward signal (channel 24)
    4. Convert 4 brain outputs to 5 motor commands
    5. Apply safety limits and smoothing
    """
    
    def __init__(self):
        """Initialize the adapter."""
        # Motor smoothing
        self.motor_smoothing_factor = 0.3
        self.prev_motor_commands = [0.0] * 5
        
        # Reward calculation state
        self.prev_distance = None
        self.collision_cooldown = 0
        self.exploration_bonus_timer = 0
        self.line_following_score = 0.0
        
        # Safety parameters
        self.min_safe_distance = 0.2  # meters
        self.max_motor_speed = 50.0   # percent
        self.max_steering_angle = 25.0  # degrees
        
        print("🔧 PiCar-X Brain Adapter initialized")
    
    def sensors_to_brain_input(self, sensor_data: List[float]) -> List[float]:
        """
        Convert PiCar-X 16 sensors to brain's expected 24 inputs.
        
        Args:
            sensor_data: List of 16 sensor values from PiCar-X
            
        Returns:
            List of 24 normalized values for brain input
        """
        if len(sensor_data) != 16:
            print(f"⚠️  Expected 16 sensors, got {len(sensor_data)}")
            sensor_data = sensor_data[:16] + [0.5] * (16 - len(sensor_data))
        
        brain_input = [0.5] * 25  # 24 sensors + 1 reward
        
        # === Spatial sensors (0-5) ===
        # Use ultrasonic for rough position estimation
        distance = sensor_data[0]  # meters
        brain_input[0] = 1.0 - min(distance / 2.0, 1.0)  # X: closer = higher
        brain_input[1] = 0.5  # Y: neutral (no lateral distance sensor)
        brain_input[2] = 0.5  # Z: neutral (no vertical sensor)
        
        # Grayscale sensors for additional spatial info
        brain_input[3] = sensor_data[1]  # Right grayscale (already normalized)
        brain_input[4] = sensor_data[2]  # Center grayscale
        brain_input[5] = sensor_data[3]  # Left grayscale
        
        # === Motion sensors (6-11) ===
        brain_input[6] = (sensor_data[4] + 1.0) / 2.0  # Left motor (-1,1) → (0,1)
        brain_input[7] = (sensor_data[5] + 1.0) / 2.0  # Right motor
        brain_input[8] = (sensor_data[8] + 30) / 60    # Steering angle normalized
        brain_input[9] = (sensor_data[6] + 90) / 180   # Camera pan normalized
        brain_input[10] = (sensor_data[7] + 35) / 100  # Camera tilt normalized
        brain_input[11] = sensor_data[9] / 8.4         # Battery voltage normalized
        
        # === Status sensors (12-17) ===
        brain_input[12] = sensor_data[10]  # Line detected
        brain_input[13] = sensor_data[11]  # Cliff detected
        brain_input[14] = sensor_data[12] / 100.0  # CPU temperature
        brain_input[15] = sensor_data[13]  # Memory usage
        
        # Angular velocity estimate from motor differential
        angular_vel = (sensor_data[5] - sensor_data[4]) / 2.0
        brain_input[16] = (angular_vel + 1.0) / 2.0
        
        # Forward velocity estimate
        forward_vel = (sensor_data[4] + sensor_data[5]) / 2.0
        brain_input[17] = (forward_vel + 1.0) / 2.0
        
        # === Derived sensors (18-23) ===
        # Create richer sensory experience with derived values
        
        # Obstacle gradient (how fast distance is changing)
        if self.prev_distance is not None:
            distance_change = distance - self.prev_distance
            brain_input[18] = (distance_change + 0.5) / 1.0  # Normalized change
        self.prev_distance = distance
        
        # Line following quality
        center_strength = sensor_data[2]
        side_strength = (sensor_data[1] + sensor_data[3]) / 2.0
        line_quality = center_strength - side_strength if center_strength > 0.3 else 0
        brain_input[19] = min(max(line_quality + 0.5, 0), 1)
        
        # Steering effort (how hard we're turning)
        brain_input[20] = abs(sensor_data[8]) / 30.0
        
        # Speed magnitude
        speed_mag = abs(forward_vel)
        brain_input[21] = min(speed_mag, 1.0)
        
        # Exploration indicator (variety in sensor readings)
        sensor_variance = np.var(sensor_data[:4])
        brain_input[22] = min(sensor_variance * 10, 1.0)
        
        # System health indicator
        health = 1.0
        if sensor_data[9] < 6.5:  # Low battery
            health -= 0.3
        if sensor_data[12] > 60:  # High temperature
            health -= 0.2
        if sensor_data[11] > 0:   # Cliff detected
            health -= 0.5
        brain_input[23] = max(health, 0.0)
        
        # === Reward signal (24) ===
        brain_input[24] = self._calculate_reward(sensor_data)
        
        return brain_input
    
    def brain_output_to_motors(self, brain_output: List[float]) -> Dict[str, float]:
        """
        Convert brain's 4 outputs to PiCar-X's 5 motor commands.
        
        Args:
            brain_output: List of 4 values from brain
            
        Returns:
            Dictionary with motor commands
        """
        if len(brain_output) < 4:
            brain_output = brain_output + [0.0] * (4 - len(brain_output))
        
        # Brain outputs:
        # 0: Forward/backward
        # 1: Left/right steering
        # 2: Camera control (pan)
        # 3: Additional control (we'll use for camera tilt)
        
        # Convert to differential drive
        forward = brain_output[0] * self.max_motor_speed
        steering = brain_output[1] * self.max_steering_angle
        
        # Safety: reduce speed when turning sharply
        turn_factor = 1.0 - abs(steering) / (self.max_steering_angle * 2)
        forward *= (0.5 + 0.5 * turn_factor)
        
        # Differential drive calculation
        if abs(steering) < 5:  # Going straight
            left_motor = forward
            right_motor = forward
        else:
            # Differential steering
            turn_radius_factor = 1.0 - abs(steering) / self.max_steering_angle
            if steering > 0:  # Turning right
                left_motor = forward
                right_motor = forward * turn_radius_factor
            else:  # Turning left
                left_motor = forward * turn_radius_factor
                right_motor = forward
        
        # Camera controls
        camera_pan = brain_output[2] * 45  # ±45 degrees
        camera_tilt = brain_output[3] * 30  # Reduced range for tilt
        
        # Apply smoothing
        motor_commands = [
            left_motor,
            right_motor,
            steering,
            camera_pan,
            camera_tilt
        ]
        
        for i in range(5):
            motor_commands[i] = (self.motor_smoothing_factor * self.prev_motor_commands[i] + 
                               (1 - self.motor_smoothing_factor) * motor_commands[i])
        
        self.prev_motor_commands = motor_commands.copy()
        
        # Safety limits
        if self.prev_distance and self.prev_distance < self.min_safe_distance:
            # Emergency brake
            if motor_commands[0] > 0 and motor_commands[1] > 0:
                safety_factor = self.prev_distance / self.min_safe_distance
                motor_commands[0] *= safety_factor
                motor_commands[1] *= safety_factor
        
        return {
            'left_motor': np.clip(motor_commands[0], -self.max_motor_speed, self.max_motor_speed),
            'right_motor': np.clip(motor_commands[1], -self.max_motor_speed, self.max_motor_speed),
            'steering_servo': np.clip(motor_commands[2], -self.max_steering_angle, self.max_steering_angle),
            'camera_pan_servo': np.clip(motor_commands[3], -90, 90),
            'camera_tilt_servo': np.clip(motor_commands[4], -35, 65)
        }
    
    def _calculate_reward(self, sensor_data: List[float]) -> float:
        """
        Calculate reward signal based on robot behavior.
        
        Rewards:
        - Safe forward movement
        - Successful line following
        - Exploration
        
        Penalties:
        - Collisions or near-misses
        - Getting stuck
        - System issues (low battery, high temp)
        """
        reward = 0.5  # Neutral baseline
        
        # Distance-based rewards
        distance = sensor_data[0]
        if distance < 0.1:
            reward -= 0.4  # Too close!
            self.collision_cooldown = 20
        elif distance < self.min_safe_distance:
            reward -= 0.2  # Getting close
        elif self.collision_cooldown > 0:
            self.collision_cooldown -= 1
            reward -= 0.1
        else:
            # Reward forward movement at safe distance
            forward_speed = (sensor_data[4] + sensor_data[5]) / 2.0
            if forward_speed > 0.1:
                reward += 0.2 * min(forward_speed, 1.0)
        
        # Line following reward
        center_line = sensor_data[2]
        if center_line > 0.6:  # Strong line signal
            reward += 0.15
            self.line_following_score = min(self.line_following_score + 0.1, 1.0)
        else:
            self.line_following_score *= 0.9
        
        reward += 0.1 * self.line_following_score
        
        # Exploration bonus
        if self.exploration_bonus_timer <= 0:
            # Reward changes in position/orientation
            steering_change = abs(sensor_data[8])
            if steering_change > 10:
                reward += 0.05
                self.exploration_bonus_timer = 10
        else:
            self.exploration_bonus_timer -= 1
        
        # System health penalties
        if sensor_data[9] < 6.5:  # Low battery
            reward -= 0.1
        if sensor_data[12] > 60:  # CPU too hot
            reward -= 0.05
        if sensor_data[11] > 0:  # Cliff detected
            reward -= 0.3
        
        # Ensure reward is in [0, 1]
        return np.clip(reward, 0.0, 1.0)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the adapter state."""
        return {
            'prev_distance': self.prev_distance,
            'collision_cooldown': self.collision_cooldown,
            'exploration_timer': self.exploration_bonus_timer,
            'line_score': self.line_following_score,
            'prev_motors': self.prev_motor_commands
        }


def test_adapter():
    """Test the sensor-motor adapter."""
    print("🧪 Testing PiCar-X Brain Adapter")
    print("=" * 50)
    
    adapter = PiCarXBrainAdapter()
    
    # Test sensor conversion
    mock_sensors = [
        0.5,    # ultrasonic (meters)
        0.3,    # grayscale right
        0.8,    # grayscale center (on line)
        0.3,    # grayscale left
        0.2,    # left motor speed
        0.2,    # right motor speed
        0,      # camera pan
        0,      # camera tilt
        0,      # steering
        7.4,    # battery
        1,      # line detected
        0,      # no cliff
        45,     # CPU temp
        0.3,    # memory
        1000,   # timestamp
        0       # reserved
    ]
    
    brain_input = adapter.sensors_to_brain_input(mock_sensors)
    print(f"\nSensor data → Brain input:")
    print(f"  Original sensors: {len(mock_sensors)} channels")
    print(f"  Brain input: {len(brain_input)} channels")
    print(f"  Reward signal: {brain_input[24]:.3f}")
    
    # Test motor conversion
    mock_brain_output = [0.5, 0.2, 0.0, 0.0]  # Forward with slight right turn
    motor_commands = adapter.brain_output_to_motors(mock_brain_output)
    
    print(f"\nBrain output → Motor commands:")
    print(f"  Brain output: {mock_brain_output}")
    print(f"  Motor commands:")
    for name, value in motor_commands.items():
        print(f"    {name}: {value:.1f}")
    
    print("\n✅ Adapter test complete!")


if __name__ == "__main__":
    test_adapter()