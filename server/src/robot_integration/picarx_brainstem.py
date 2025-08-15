#!/usr/bin/env python3
"""
PiCar-X Brainstem Integration Layer

This layer handles:
1. Sensor normalization from PiCar-X to brain format
2. Motor command scaling from brain to PiCar-X
3. Safety limits and smooth motor transitions
4. Reward signal generation based on robot behavior
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class PiCarXSensorData:
    """Raw sensor data from PiCar-X"""
    ultrasonic_distance: float  # cm
    line_following: List[int]  # 3 sensors, 0-1023
    battery_voltage: float  # volts
    motor_speeds: List[float]  # current speeds
    servo_angles: List[float]  # steering, camera pan/tilt
    

class PiCarXBrainstem:
    """
    Brainstem layer for PiCar-X robot integration.
    
    Converts between robot hardware and brain interface.
    """
    
    def __init__(self, 
                 max_speed: float = 50.0,  # Max motor speed %
                 max_turn_rate: float = 30.0,  # Max steering angle
                 safety_distance: float = 20.0):  # cm
        
        self.max_speed = max_speed
        self.max_turn_rate = max_turn_rate
        self.safety_distance = safety_distance
        
        # Smooth transition parameters
        self.motor_smoothing = 0.3
        self.prev_motor_commands = [0.0, 0.0, 0.0, 0.0]
        
        # Reward calculation state
        self.prev_distance = None
        self.collision_penalty_cooldown = 0
        self.exploration_bonus_timer = 0
        
    def sensors_to_brain_input(self, sensors: PiCarXSensorData) -> List[float]:
        """
        Convert PiCar-X sensors to normalized brain input [0, 1].
        
        Brain expects 25 inputs:
        - 0-2: Spatial position/orientation estimates
        - 3-5: Line following sensors
        - 6-8: Obstacle/distance sensors
        - 9-15: Motion/dynamics sensors
        - 16-23: Additional sensory channels
        - 24: Reward signal
        """
        brain_input = [0.5] * 25  # Default neutral values
        
        # Spatial estimation (crude odometry)
        # These would be better with actual odometry
        brain_input[0] = 0.5  # X position estimate
        brain_input[1] = 0.5  # Y position estimate
        brain_input[2] = (sensors.servo_angles[0] + 30) / 60  # Steering angle normalized
        
        # Line following sensors (normalized from 0-1023)
        for i, val in enumerate(sensors.line_following[:3]):
            brain_input[3 + i] = val / 1023.0
        
        # Distance sensors
        # Normalize ultrasonic (0-200cm to 0-1, inverted so closer = higher)
        distance_normalized = 1.0 - min(sensors.ultrasonic_distance / 200.0, 1.0)
        brain_input[6] = distance_normalized  # Front distance
        brain_input[7] = distance_normalized * 0.8  # Estimated left
        brain_input[8] = distance_normalized * 0.8  # Estimated right
        
        # Motion dynamics
        if sensors.motor_speeds:
            brain_input[9] = (sensors.motor_speeds[0] + 100) / 200  # Left motor
            brain_input[10] = (sensors.motor_speeds[1] + 100) / 200  # Right motor
        
        # Battery level (important for behavior)
        brain_input[11] = min(sensors.battery_voltage / 8.4, 1.0)  # 8.4V = full
        
        # Angular velocity estimate (from motor differential)
        if len(sensors.motor_speeds) >= 2:
            angular_vel = (sensors.motor_speeds[1] - sensors.motor_speeds[0]) / 100
            brain_input[12] = (angular_vel + 1) / 2  # Normalize to [0,1]
        
        # Camera servo positions
        brain_input[13] = (sensors.servo_angles[1] + 90) / 180  # Pan
        brain_input[14] = (sensors.servo_angles[2] + 90) / 180  # Tilt
        
        # Fill remaining sensors with variations for richer input
        for i in range(15, 24):
            # Create some variation based on other sensors
            brain_input[i] = 0.5 + 0.2 * np.sin(i + brain_input[i % 6])
        
        # Calculate reward signal
        brain_input[24] = self._calculate_reward(sensors)
        
        return brain_input
    
    def brain_output_to_motors(self, brain_output: List[float]) -> Tuple[float, float, float]:
        """
        Convert brain motor commands to PiCar-X control.
        
        Brain outputs 4 values, we map to:
        - Forward/backward speed
        - Steering angle
        - Camera control (optional)
        
        Returns: (speed, steering_angle, camera_pan)
        """
        # Smooth motor commands
        for i in range(len(brain_output)):
            brain_output[i] = (self.motor_smoothing * self.prev_motor_commands[i] + 
                              (1 - self.motor_smoothing) * brain_output[i])
        self.prev_motor_commands = brain_output.copy()
        
        # Map brain outputs to motor controls
        # Brain output 0: Forward/backward (-1 to 1)
        # Brain output 1: Left/right (-1 to 1)
        
        # Convert to differential drive
        forward = brain_output[0] * self.max_speed
        turn = brain_output[1] * self.max_turn_rate
        
        # Safety limits based on distance
        if hasattr(self, '_last_distance') and self._last_distance < self.safety_distance:
            # Reduce forward speed when close to obstacles
            safety_factor = self._last_distance / self.safety_distance
            if forward > 0:
                forward *= safety_factor
        
        # Camera control from brain output 2 (if desired)
        camera_pan = brain_output[2] * 45 if len(brain_output) > 2 else 0
        
        # Apply final limits
        forward = np.clip(forward, -self.max_speed, self.max_speed)
        turn = np.clip(turn, -self.max_turn_rate, self.max_turn_rate)
        camera_pan = np.clip(camera_pan, -45, 45)
        
        return forward, turn, camera_pan
    
    def _calculate_reward(self, sensors: PiCarXSensorData) -> float:
        """
        Calculate reward signal for the brain.
        
        Rewards:
        - Forward movement without obstacles
        - Following lines (if detected)
        - Exploration
        
        Penalties:
        - Collisions or near-misses
        - Stuck behavior
        - Low battery
        """
        reward = 0.0
        
        # Store distance for safety checks
        self._last_distance = sensors.ultrasonic_distance
        
        # Distance-based rewards/penalties
        if sensors.ultrasonic_distance < 10:
            # Too close! Big penalty
            reward -= 0.8
            self.collision_penalty_cooldown = 10
        elif sensors.ultrasonic_distance < self.safety_distance:
            # Getting close, mild penalty
            reward -= 0.2
        elif self.collision_penalty_cooldown > 0:
            self.collision_penalty_cooldown -= 1
        else:
            # Safe distance, reward forward movement
            if sensors.motor_speeds:
                avg_speed = np.mean(np.abs(sensors.motor_speeds))
                reward += min(avg_speed / 50, 0.3)  # Max 0.3 reward for speed
        
        # Line following bonus
        line_center = sensors.line_following[1]
        line_sides = (sensors.line_following[0] + sensors.line_following[2]) / 2
        if line_center > line_sides * 1.5:
            # Centered on line
            reward += 0.2
        
        # Exploration bonus (changes in sensor readings)
        if self.exploration_bonus_timer <= 0:
            if self.prev_distance and abs(sensors.ultrasonic_distance - self.prev_distance) > 5:
                reward += 0.1
                self.exploration_bonus_timer = 10
        else:
            self.exploration_bonus_timer -= 1
        
        self.prev_distance = sensors.ultrasonic_distance
        
        # Battery penalty
        if sensors.battery_voltage < 6.5:
            reward -= 0.1
        
        # Normalize reward to [0, 1] range expected by brain
        return (reward + 1.0) / 2.0
    
    def create_demo_sensors(self, t: float) -> PiCarXSensorData:
        """Create demo sensor data for testing without real robot."""
        return PiCarXSensorData(
            ultrasonic_distance=50 + 30 * np.sin(t * 0.5),
            line_following=[
                int(512 + 300 * np.sin(t * 0.3)),
                int(512 + 300 * np.sin(t * 0.3 + 1)),
                int(512 + 300 * np.sin(t * 0.3 + 2))
            ],
            battery_voltage=7.4,
            motor_speeds=[20 * np.sin(t), 20 * np.sin(t + 0.5)],
            servo_angles=[15 * np.sin(t * 0.2), 0, 0]
        )


def test_brainstem():
    """Test the brainstem integration."""
    print("🧠 Testing PiCar-X Brainstem Integration")
    print("=" * 50)
    
    brainstem = PiCarXBrainstem()
    
    # Simulate some sensor readings
    for i in range(10):
        t = i * 0.1
        sensors = brainstem.create_demo_sensors(t)
        
        # Convert to brain input
        brain_input = brainstem.sensors_to_brain_input(sensors)
        
        # Simulate brain output (normally from brain.process_robot_cycle)
        brain_output = [
            0.3 * np.sin(t),  # Forward/back
            0.2 * np.sin(t * 2),  # Left/right
            0.1 * np.sin(t * 3),  # Camera
            0.0  # Unused
        ]
        
        # Convert to motor commands
        speed, steering, camera = brainstem.brain_output_to_motors(brain_output)
        
        if i % 3 == 0:
            print(f"\nCycle {i}:")
            print(f"  Distance: {sensors.ultrasonic_distance:.1f}cm")
            print(f"  Brain reward: {brain_input[24]:.3f}")
            print(f"  Motor commands: speed={speed:.1f}%, steer={steering:.1f}°")
    
    print("\n✅ Brainstem integration ready for deployment!")


if __name__ == "__main__":
    test_brainstem()