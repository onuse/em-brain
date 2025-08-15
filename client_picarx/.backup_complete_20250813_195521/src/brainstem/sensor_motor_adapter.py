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
    1. Expand 16 sensor channels to 24 brain inputs
    2. Normalize sensor ranges to [0, 1]
    3. Generate reward signal (channel 24)
    4. Convert 4 brain outputs to 5 motor commands
    5. Apply safety limits and smoothing
    """
    
    def __init__(self, config: BrainstemConfig = None):
        """Initialize the adapter."""
        self.config = config or get_config()
        
        # Motor smoothing
        self.prev_motor_commands = [0.0] * self.config.motors.picarx_motor_count
        
        # Reward calculation state
        self.prev_distance = None
        self.collision_cooldown = 0
        self.exploration_bonus_timer = 0
        self.line_following_score = 0.0
        
        print("🔧 PiCar-X Brain Adapter initialized with configuration")
    
    def sensors_to_brain_input(self, sensor_data: List[float]) -> List[float]:
        """
        Convert PiCar-X 16 sensors to brain's expected 24 inputs.
        
        Args:
            sensor_data: List of 16 sensor values from PiCar-X
            
        Returns:
            List of 24 normalized values for brain input
        """
        if len(sensor_data) != self.config.sensors.picarx_sensor_count:
            print(f"⚠️  Expected {self.config.sensors.picarx_sensor_count} sensors, got {len(sensor_data)}")
            sensor_data = (sensor_data[:self.config.sensors.picarx_sensor_count] + 
                          [self.config.sensors.neutral_value] * 
                          (self.config.sensors.picarx_sensor_count - len(sensor_data)))
        
        brain_input = [self.config.sensors.neutral_value] * (self.config.sensors.brain_input_dimensions + 1)  # sensors + reward
        
        # === Spatial sensors (0-5) ===
        # Use ultrasonic for rough position estimation
        distance = sensor_data[0]  # meters
        brain_input[0] = 1.0 - min(distance / self.config.sensors.spatial_distance_scale, 1.0)  # X: closer = higher
        brain_input[1] = self.config.sensors.neutral_value  # Y: neutral (no lateral distance sensor)
        brain_input[2] = self.config.sensors.neutral_value  # Z: neutral (no vertical sensor)
        
        # Grayscale sensors for additional spatial info
        brain_input[3] = sensor_data[1]  # Right grayscale (already normalized)
        brain_input[4] = sensor_data[2]  # Center grayscale
        brain_input[5] = sensor_data[3]  # Left grayscale
        
        # === Motion sensors (6-11) ===
        brain_input[6] = (sensor_data[4] + self.config.sensors.motor_range_half) / (2 * self.config.sensors.motor_range_half)  # Left motor
        brain_input[7] = (sensor_data[5] + self.config.sensors.motor_range_half) / (2 * self.config.sensors.motor_range_half)  # Right motor
        brain_input[8] = (sensor_data[8] + self.config.sensors.steering_offset) / self.config.sensors.steering_range    # Steering angle normalized
        brain_input[9] = (sensor_data[6] + self.config.sensors.camera_pan_offset) / self.config.sensors.camera_pan_range   # Camera pan normalized
        brain_input[10] = (sensor_data[7] + self.config.sensors.camera_tilt_offset) / self.config.sensors.camera_tilt_range  # Camera tilt normalized
        brain_input[11] = sensor_data[9] / self.config.sensors.battery_max         # Battery voltage normalized
        
        # === Status sensors (12-17) ===
        brain_input[12] = sensor_data[10]  # Line detected
        brain_input[13] = sensor_data[11]  # Cliff detected
        brain_input[14] = sensor_data[12] / self.config.sensors.cpu_temp_critical  # CPU temperature normalized
        brain_input[15] = sensor_data[13]  # Memory usage
        
        # Angular velocity estimate from motor differential
        angular_vel = (sensor_data[5] - sensor_data[4]) / (2 * self.config.sensors.motor_range_half)
        brain_input[16] = (angular_vel + self.config.sensors.motor_range_half) / (2 * self.config.sensors.motor_range_half)
        
        # Forward velocity estimate
        forward_vel = (sensor_data[4] + sensor_data[5]) / (2 * self.config.sensors.motor_range_half)
        brain_input[17] = (forward_vel + self.config.sensors.motor_range_half) / (2 * self.config.sensors.motor_range_half)
        
        # === Derived sensors (18-23) ===
        # Create richer sensory experience with derived values
        
        # Obstacle gradient (how fast distance is changing)
        if self.prev_distance is not None:
            distance_change = distance - self.prev_distance
            brain_input[18] = (distance_change + self.config.sensors.distance_change_offset) / self.config.sensors.distance_change_scale
        self.prev_distance = distance
        
        # Line following quality
        center_strength = sensor_data[2]
        side_strength = (sensor_data[1] + sensor_data[3]) / 2.0
        line_quality = center_strength - side_strength if center_strength > self.config.sensors.line_quality_threshold else 0
        brain_input[19] = min(max(line_quality + self.config.sensors.neutral_value, 0), 1)
        
        # Steering effort (how hard we're turning)
        brain_input[20] = abs(sensor_data[8]) / self.config.sensors.steering_effort_scale
        
        # Speed magnitude
        speed_mag = abs(forward_vel)
        brain_input[21] = min(speed_mag, self.config.sensors.motor_range_half)
        
        # Exploration indicator (variety in sensor readings)
        sensor_variance = np.var(sensor_data[:4])
        brain_input[22] = min(sensor_variance * self.config.sensors.exploration_variance_scale, self.config.sensors.motor_range_half)
        
        # System health indicator
        health = self.config.sensors.motor_range_half
        if sensor_data[9] < self.config.sensors.low_battery_threshold:  # Low battery
            health -= self.config.sensors.health_battery_penalty
        if sensor_data[12] > self.config.sensors.high_temp_threshold:  # High temperature
            health -= self.config.sensors.health_temp_penalty
        if sensor_data[11] > 0:   # Cliff detected
            health -= self.config.sensors.health_cliff_penalty
        brain_input[23] = max(health, 0.0)
        
        # === Reward signal (final index) ===
        brain_input[self.config.sensors.brain_input_dimensions] = self._calculate_reward(sensor_data)
        
        return brain_input
    
    def brain_output_to_motors(self, brain_output: List[float]) -> Dict[str, float]:
        """
        Convert brain's 4 outputs to PiCar-X's 5 motor commands.
        
        Args:
            brain_output: List of 4 values from brain
            
        Returns:
            Dictionary with motor commands
        """
        if len(brain_output) < self.config.motors.brain_output_dimensions:
            brain_output = brain_output + [0.0] * (self.config.motors.brain_output_dimensions - len(brain_output))
        
        # Brain outputs:
        # 0: Forward/backward
        # 1: Left/right steering
        # 2: Camera control (pan)
        # 3: Additional control (we'll use for camera tilt)
        
        # Convert to differential drive
        forward = brain_output[0] * self.config.motors.max_motor_speed
        steering = brain_output[1] * self.config.motors.max_steering_angle
        
        # Safety: reduce speed when turning sharply  
        turn_factor = self.config.sensors.motor_range_half - abs(steering) / (self.config.motors.max_steering_angle * 2)
        forward *= (self.config.motors.turn_speed_reduction_base + self.config.motors.turn_speed_reduction_factor * turn_factor)
        
        # Differential drive calculation
        if abs(steering) < self.config.sensors.straight_steering_threshold:  # Going straight
            left_motor = forward
            right_motor = forward
        else:
            # Differential steering
            turn_radius_factor = self.config.sensors.motor_range_half - abs(steering) / self.config.motors.max_steering_angle
            if steering > 0:  # Turning right
                left_motor = forward
                right_motor = forward * turn_radius_factor
            else:  # Turning left
                left_motor = forward * turn_radius_factor
                right_motor = forward
        
        # Camera controls
        camera_pan = brain_output[2] * self.config.motors.camera_pan_range
        camera_tilt = brain_output[3] * self.config.motors.camera_tilt_range
        
        # Apply smoothing
        motor_commands = [
            left_motor,
            right_motor,
            steering,
            camera_pan,
            camera_tilt
        ]
        
        for i in range(self.config.motors.picarx_motor_count):
            motor_commands[i] = (self.config.motors.motor_smoothing_alpha * self.prev_motor_commands[i] + 
                               (1 - self.config.motors.motor_smoothing_alpha) * motor_commands[i])
        
        self.prev_motor_commands = motor_commands.copy()
        
        # Safety limits
        if self.prev_distance and self.prev_distance < self.config.safety.min_safe_distance:
            # Emergency brake
            if motor_commands[0] > 0.0 and motor_commands[1] > 0.0:
                safety_factor = self.prev_distance / self.config.safety.min_safe_distance
                motor_commands[0] *= safety_factor
                motor_commands[1] *= safety_factor
        
        return {
            'left_motor': np.clip(motor_commands[0], -self.config.motors.max_motor_speed, self.config.motors.max_motor_speed),
            'right_motor': np.clip(motor_commands[1], -self.config.motors.max_motor_speed, self.config.motors.max_motor_speed),
            'steering_servo': np.clip(motor_commands[2], -self.config.motors.max_steering_angle, self.config.motors.max_steering_angle),
            'camera_pan_servo': np.clip(motor_commands[3], self.config.motors.camera_pan_min, self.config.motors.camera_pan_max),
            'camera_tilt_servo': np.clip(motor_commands[4], self.config.motors.camera_tilt_min, self.config.motors.camera_tilt_max)
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
        reward = self.config.rewards.neutral_baseline  # Neutral baseline
        
        # Distance-based rewards
        distance = sensor_data[0]
        if distance < self.config.safety.emergency_stop_distance:
            reward -= self.config.rewards.collision_penalty
            self.collision_cooldown = self.config.rewards.collision_cooldown_cycles
        elif distance < self.config.safety.min_safe_distance:
            reward -= self.config.rewards.near_obstacle_penalty
        elif self.collision_cooldown > 0:
            self.collision_cooldown -= 1
            reward -= self.config.rewards.collision_cooldown_penalty
        else:
            # Reward forward movement at safe distance
            forward_speed = (sensor_data[4] + sensor_data[5]) / (2 * self.config.sensors.motor_range_half)
            if forward_speed > self.config.sensors.forward_speed_threshold:
                reward += self.config.rewards.forward_movement_reward * min(forward_speed, self.config.sensors.motor_range_half)
        
        # Line following reward
        center_line = sensor_data[2]
        if center_line > self.config.sensors.line_detection_strength:  # Strong line signal
            reward += self.config.rewards.line_detection_reward
            self.line_following_score = min(self.line_following_score + self.config.rewards.line_score_accumulation, self.config.sensors.motor_range_half)
        else:
            self.line_following_score *= self.config.rewards.line_score_decay
        
        reward += self.config.rewards.line_score_weight * self.line_following_score
        
        # Exploration bonus
        if self.exploration_bonus_timer <= 0:
            # Reward changes in position/orientation
            steering_change = abs(sensor_data[8])
            if steering_change > self.config.rewards.exploration_steering_threshold:
                reward += self.config.rewards.exploration_reward
                self.exploration_bonus_timer = self.config.rewards.exploration_cooldown
        else:
            self.exploration_bonus_timer -= 1
        
        # System health penalties
        if sensor_data[9] < self.config.sensors.low_battery_threshold:  # Low battery
            reward -= self.config.rewards.low_battery_penalty
        if sensor_data[12] > self.config.sensors.high_temp_threshold:  # CPU too hot
            reward -= self.config.rewards.high_temperature_penalty
        if sensor_data[11] > 0.0:  # Cliff detected
            reward -= self.config.rewards.cliff_detection_penalty
        
        # Ensure reward is in [0, 1]
        return np.clip(reward, 0.0, self.config.sensors.motor_range_half)
    
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
    
    # Test with default configuration
    config = get_config("testing")  # Use testing profile for safety
    adapter = PiCarXBrainAdapter(config)
    
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
    print(f"  Reward signal: {brain_input[config.sensors.brain_input_dimensions]:.3f}")
    print(f"  Configuration: {config.motors.max_motor_speed}% max speed, {config.safety.min_safe_distance}m safe distance")
    
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