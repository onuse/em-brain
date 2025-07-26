#!/usr/bin/env python3
"""
Direct Brain Adapter for PiCar-X

This adapter works with the brain's actual dynamic dimension system,
rather than forcing a fixed 24-sensor format.

The brain will adapt its internal dimensions based on the robot's
actual sensor count (16 for PiCar-X).
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class DirectBrainAdapter:
    """
    Direct adapter that preserves PiCar-X's natural sensor dimensionality.
    
    Key differences from sensor_motor_adapter.py:
    1. Keeps 16 sensors (no expansion to 24)
    2. Adds reward as 17th sensor (brain convention)
    3. Maps brain's 4 outputs to PiCar-X's 5 motors
    """
    
    def __init__(self):
        """Initialize the direct adapter."""
        # Motor control
        self.motor_smoothing_factor = 0.3
        self.prev_motor_commands = [0.0] * 5
        
        # Reward calculation state
        self.prev_distance = None
        self.collision_cooldown = 0
        self.exploration_timer = 0
        
        # Safety parameters
        self.min_safe_distance = 0.2  # meters
        self.max_motor_speed = 50.0   # percent
        self.max_steering_angle = 25.0  # degrees
        
        print("🔧 Direct Brain Adapter initialized (16→17 sensors)")
    
    def prepare_brain_input(self, picarx_sensors: List[float]) -> List[float]:
        """
        Prepare PiCar-X sensors for brain input.
        
        Args:
            picarx_sensors: 16 sensor values from PiCar-X
            
        Returns:
            17 values: 16 sensors + 1 reward signal
        """
        if len(picarx_sensors) != 16:
            print(f"⚠️  Expected 16 sensors, got {len(picarx_sensors)}")
            picarx_sensors = picarx_sensors[:16] + [0.5] * (16 - len(picarx_sensors))
        
        # Calculate reward based on robot state
        reward = self._calculate_reward(picarx_sensors)
        
        # Return original sensors + reward
        return picarx_sensors + [reward]
    
    def brain_output_to_motors(self, brain_output: List[float]) -> Dict[str, float]:
        """
        Convert brain's 4 outputs to PiCar-X's 5 motor commands.
        
        Same as before - brain outputs are abstract, we interpret them.
        """
        if len(brain_output) < 4:
            brain_output = brain_output + [0.0] * (4 - len(brain_output))
        
        # Brain outputs (abstract motor space):
        # 0: Primary motion axis (forward/back)
        # 1: Secondary motion axis (turn/strafe)
        # 2: Tertiary control (camera pan)
        # 3: Quaternary control (camera tilt)
        
        forward = brain_output[0] * self.max_motor_speed
        steering = brain_output[1] * self.max_steering_angle
        
        # Reduce speed when turning
        turn_factor = 1.0 - abs(steering) / (self.max_steering_angle * 2)
        forward *= (0.5 + 0.5 * turn_factor)
        
        # Differential drive
        if abs(steering) < 5:
            left_motor = forward
            right_motor = forward
        else:
            radius_factor = 1.0 - abs(steering) / self.max_steering_angle
            if steering > 0:
                left_motor = forward
                right_motor = forward * radius_factor
            else:
                left_motor = forward * radius_factor
                right_motor = forward
        
        # Camera controls from abstract axes
        camera_pan = brain_output[2] * 45
        camera_tilt = brain_output[3] * 30
        
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
        
        # Safety limits based on distance
        distance = self.prev_distance if self.prev_distance else 1.0
        if distance < self.min_safe_distance and motor_commands[0] > 0:
            safety_factor = distance / self.min_safe_distance
            motor_commands[0] *= safety_factor
            motor_commands[1] *= safety_factor
        
        return {
            'left_motor': np.clip(motor_commands[0], -self.max_motor_speed, self.max_motor_speed),
            'right_motor': np.clip(motor_commands[1], -self.max_motor_speed, self.max_motor_speed),
            'steering_servo': np.clip(motor_commands[2], -self.max_steering_angle, self.max_steering_angle),
            'camera_pan_servo': np.clip(motor_commands[3], -90, 90),
            'camera_tilt_servo': np.clip(motor_commands[4], -35, 65)
        }
    
    def _calculate_reward(self, sensors: List[float]) -> float:
        """
        Calculate reward signal based on PiCar-X sensor data.
        
        Using sensor indices from picarx_profile.json:
        0: ultrasonic_distance (meters)
        1-3: grayscale sensors
        4-5: motor speeds
        9: battery voltage
        10: line_detected
        11: cliff_detected
        """
        reward = 0.5  # Neutral baseline
        
        # Distance-based rewards
        distance = sensors[0]
        self.prev_distance = distance
        
        if distance < 0.1:
            reward -= 0.4
            self.collision_cooldown = 20
        elif distance < self.min_safe_distance:
            reward -= 0.2
        elif self.collision_cooldown > 0:
            self.collision_cooldown -= 1
            reward -= 0.1
        else:
            # Reward forward movement
            forward_speed = (sensors[4] + sensors[5]) / 2.0
            if forward_speed > 0.1:
                reward += 0.2 * min(forward_speed, 1.0)
        
        # Line following reward
        if sensors[10] > 0.5:  # Line detected
            center_line = sensors[2]
            if center_line > 0.6:
                reward += 0.15
        
        # Exploration bonus
        if self.exploration_timer <= 0:
            steering = sensors[8] if len(sensors) > 8 else 0
            if abs(steering) > 10:
                reward += 0.05
                self.exploration_timer = 10
        else:
            self.exploration_timer -= 1
        
        # System health
        if sensors[9] < 6.5:  # Low battery
            reward -= 0.1
        if sensors[11] > 0:  # Cliff
            reward -= 0.3
        
        return np.clip(reward, 0.0, 1.0)
    
    def get_expected_dimensions(self) -> Tuple[int, int]:
        """Get expected sensor and motor dimensions."""
        return 16, 4  # PiCar-X has 16 sensors, brain provides 4 motor outputs


def test_direct_adapter():
    """Test the direct adapter."""
    print("🧪 Testing Direct Brain Adapter")
    print("=" * 50)
    
    adapter = DirectBrainAdapter()
    
    # Mock PiCar-X sensors
    mock_sensors = [
        0.5,   # ultrasonic
        0.3,   # grayscale right
        0.8,   # grayscale center
        0.3,   # grayscale left
        0.2,   # left motor
        0.2,   # right motor
        0,     # camera pan
        0,     # camera tilt
        0,     # steering
        7.4,   # battery
        1,     # line detected
        0,     # no cliff
        45,    # CPU temp
        0.3,   # memory
        1000,  # timestamp
        0      # reserved
    ]
    
    # Prepare for brain
    brain_input = adapter.prepare_brain_input(mock_sensors)
    print(f"\nPiCar-X sensors: {len(mock_sensors)}")
    print(f"Brain input: {len(brain_input)} (includes reward)")
    print(f"Reward signal: {brain_input[-1]:.3f}")
    
    # Test motor conversion
    mock_brain_output = [0.5, 0.2, 0.0, 0.0]
    motor_commands = adapter.brain_output_to_motors(mock_brain_output)
    
    print(f"\nBrain output → Motors:")
    for name, value in motor_commands.items():
        print(f"  {name}: {value:.1f}")
    
    # Show expected dimensions
    sensor_dim, motor_dim = adapter.get_expected_dimensions()
    print(f"\nExpected dimensions for brain creation:")
    print(f"  sensory_dim: {sensor_dim}")
    print(f"  motor_dim: {motor_dim}")
    
    print("\n✅ Direct adapter ready!")
    print("\nTo create brain: factory.create(sensory_dim=16, motor_dim=4)")


if __name__ == "__main__":
    test_direct_adapter()