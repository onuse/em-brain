#!/usr/bin/env python3
"""
PiCar-X Brainstem

A custom brainstem that interfaces the minimal brain with a simulated PiCar-X robot.
This demonstrates how the 4-system minimal brain can control a realistic robot platform.

PiCar-X Hardware:
- 4WD chassis with servo steering
- Ultrasonic sensor (front)
- Camera (RGB, can pan/tilt)
- 2 line tracking sensors
- Buzzer, LEDs
- Servo motors for camera and ultrasonic sensor

This brainstem converts between:
- Robot sensors → Brain sensory input vectors
- Brain action vectors → Robot motor commands
"""

import sys
import os

# Add the brain/ directory (now the project root) to sys.path
current_dir = os.path.dirname(__file__)  # picar_x/
demos_dir = os.path.dirname(current_dir)  # demos/
brain_dir = os.path.dirname(demos_dir)   # brain/ (project root)
sys.path.insert(0, brain_dir)

try:
    from src.brain import MinimalBrain
except ImportError:
    # Try server path when running from different directory
    server_path = os.path.join(brain_dir, 'server')
    sys.path.insert(0, server_path)
    from src.brain import MinimalBrain
import numpy as np
import time
import math
import random
from typing import List, Tuple, Dict, Any


class PiCarXBrainstem:
    """
    Brainstem for PiCar-X robot - handles all sensory-motor translation.
    
    The minimal brain knows nothing about robots - it just processes vectors.
    This brainstem is the interface that translates between robot hardware
    and the abstract vector world that the brain operates in.
    """
    
    def __init__(self, enable_camera=True, enable_ultrasonics=True, enable_line_tracking=True):
        """Initialize the PiCar-X brainstem."""
        
        # Hardware capabilities
        self.enable_camera = enable_camera
        self.enable_ultrasonics = enable_ultrasonics  
        self.enable_line_tracking = enable_line_tracking
        
        # Robot state
        self.position = [0.0, 0.0, 0.0]  # x, y, heading
        self.velocity = [0.0, 0.0]       # forward_speed, angular_speed
        self.last_update = time.time()
        
        # Sensor state
        self.ultrasonic_distance = 100.0  # cm
        self.camera_pan_angle = 0.0       # degrees
        self.camera_tilt_angle = 0.0      # degrees
        self.line_sensors = [0.0, 0.0]   # left, right line detection
        self.camera_rgb = [0.5, 0.5, 0.5]  # simplified RGB values
        
        # Motor state
        self.motor_speed = 0.0      # -100 to 100
        self.steering_angle = 0.0   # -30 to 30 degrees
        
        # Initialize the minimal brain with vector stream architecture
        self.brain = MinimalBrain(quiet_mode=True, sensory_dim=16, motor_dim=8)
        
        # Performance tracking
        self.total_control_cycles = 0
        self.obstacle_encounters = 0
        self.successful_navigations = 0
        
        print("🤖 PiCar-X Brainstem initialized")
        print(f"   Camera: {'✅' if enable_camera else '❌'}")
        print(f"   Ultrasonics: {'✅' if enable_ultrasonics else '❌'}")
        print(f"   Line tracking: {'✅' if enable_line_tracking else '❌'}")
        print(f"   Brain: 4-system minimal brain ready")
    
    def get_sensory_vector(self) -> List[float]:
        """
        Convert all robot sensors into a unified sensory vector for the brain.
        
        The brain doesn't know about cameras or ultrasonics - it just sees
        a vector of numbers that represent the current sensory state.
        """
        sensors = []
        
        if self.enable_ultrasonics:
            # Ultrasonic distance sensor (normalized to 0-1, closer = higher value)
            normalized_distance = max(0.0, min(1.0, (200.0 - self.ultrasonic_distance) / 200.0))
            sensors.append(normalized_distance)
            
            # Add directional ultrasonic readings (simulated by panning)
            # Simulate multiple ultrasonic readings in different directions
            for angle_offset in [-30, -15, 0, 15, 30]:
                simulated_distance = self._simulate_ultrasonic_reading(self.camera_pan_angle + angle_offset)
                normalized = max(0.0, min(1.0, (200.0 - simulated_distance) / 200.0))
                sensors.append(normalized)
        
        if self.enable_camera:
            # Camera RGB values (3 channels, already normalized)
            sensors.extend(self.camera_rgb)
            
            # Camera pose information
            sensors.append(self.camera_pan_angle / 180.0)   # -1 to 1
            sensors.append(self.camera_tilt_angle / 90.0)   # -1 to 1
        
        if self.enable_line_tracking:
            # Line following sensors (already 0-1)
            sensors.extend(self.line_sensors)
        
        # Robot motion state
        sensors.append(self.motor_speed / 100.0)      # -1 to 1
        sensors.append(self.steering_angle / 30.0)    # -1 to 1
        sensors.append(math.sin(math.radians(self.position[2])))  # heading as sin/cos
        sensors.append(math.cos(math.radians(self.position[2])))
        
        # Pad to fixed size (16 sensors for consistency)
        while len(sensors) < 16:
            sensors.append(0.0)
        
        return sensors[:16]  # Ensure exactly 16 dimensions
    
    def execute_action_vector(self, action: List[float]) -> Dict[str, Any]:
        """
        Convert brain action vector into robot motor commands.
        
        The brain outputs abstract actions - this translates them into
        specific motor movements for the PiCar-X platform.
        """
        # Brain outputs 4D action vector: [motor_speed, steering, camera_pan, camera_tilt]
        if len(action) >= 4:
            # Extract and scale actions
            target_motor_speed = np.clip(action[0] * 100.0, -100.0, 100.0)
            target_steering = np.clip(action[1] * 30.0, -30.0, 30.0) 
            target_camera_pan = np.clip(action[2] * 180.0, -180.0, 180.0)
            target_camera_tilt = np.clip(action[3] * 90.0, -90.0, 90.0)
        else:
            # Fallback for smaller action vectors
            target_motor_speed = 0.0
            target_steering = 0.0
            target_camera_pan = 0.0
            target_camera_tilt = 0.0
        
        # Smooth motor control (simulate acceleration limits)
        motor_change_limit = 20.0  # max change per cycle
        steering_change_limit = 10.0
        
        self.motor_speed += np.clip(target_motor_speed - self.motor_speed, 
                                   -motor_change_limit, motor_change_limit)
        self.steering_angle += np.clip(target_steering - self.steering_angle,
                                      -steering_change_limit, steering_change_limit)
        
        # Smooth camera control
        camera_change_limit = 15.0
        self.camera_pan_angle += np.clip(target_camera_pan - self.camera_pan_angle,
                                        -camera_change_limit, camera_change_limit)
        self.camera_tilt_angle += np.clip(target_camera_tilt - self.camera_tilt_angle,
                                         -camera_change_limit, camera_change_limit)
        
        # Update robot physics
        self._update_robot_physics()
        
        # Update sensors based on new position
        self._update_sensors()
        
        return {
            'motor_speed': self.motor_speed,
            'steering_angle': self.steering_angle,
            'camera_pan': self.camera_pan_angle,
            'camera_tilt': self.camera_tilt_angle,
            'position': self.position.copy(),
            'ultrasonic_distance': self.ultrasonic_distance
        }
    
    def control_cycle(self) -> Dict[str, Any]:
        """
        Complete brainstem control cycle: sensors → brain → motors.
        
        This is the main control loop that runs continuously on the robot.
        """
        cycle_start = time.time()
        
        # 1. Get sensory input from robot hardware
        sensory_vector = self.get_sensory_vector()
        
        # 2. Brain processes sensory input and predicts action
        action_vector, brain_state = self.brain.process_sensory_input(
            sensory_vector, action_dimensions=4
        )
        
        # 3. Execute brain's action on robot hardware
        motor_state = self.execute_action_vector(action_vector)
        
        # 4. After action execution, get outcome sensors
        outcome_vector = self.get_sensory_vector()
        
        # 5. Store experience in brain for learning
        experience_id = self.brain.store_experience(
            sensory_vector, action_vector, outcome_vector, action_vector
        )
        
        # 6. Performance tracking
        self.total_control_cycles += 1
        cycle_time = time.time() - cycle_start
        
        # Detect obstacles and successful navigation
        if self.ultrasonic_distance < 20.0:
            self.obstacle_encounters += 1
        elif self.motor_speed > 20.0 and self.ultrasonic_distance > 50.0:
            self.successful_navigations += 1
        
        return {
            'cycle_time': cycle_time,
            'brain_state': brain_state,
            'motor_state': motor_state,
            'sensory_vector': sensory_vector,
            'action_vector': action_vector,
            'experience_id': experience_id,
            'performance': {
                'total_cycles': self.total_control_cycles,
                'obstacle_encounters': self.obstacle_encounters,
                'successful_navigations': self.successful_navigations
            }
        }
    
    def _update_robot_physics(self):
        """Update robot position based on current motor commands."""
        dt = time.time() - self.last_update
        self.last_update = time.time()
        
        if dt > 0.1:  # Skip large time jumps
            dt = 0.1
        
        # Convert motor speed and steering to velocity
        forward_speed = self.motor_speed * 0.01  # cm/s to m/s roughly
        
        # Ackermann steering approximation
        if abs(self.steering_angle) > 1.0:
            turning_radius = 1.0 / math.tan(math.radians(abs(self.steering_angle)))
            angular_speed = forward_speed / turning_radius
            if self.steering_angle < 0:
                angular_speed = -angular_speed
        else:
            angular_speed = 0.0
        
        # Update position
        self.position[2] += math.degrees(angular_speed * dt)  # heading
        self.position[2] = self.position[2] % 360.0  # wrap around
        
        # Move forward in current heading direction
        heading_rad = math.radians(self.position[2])
        self.position[0] += forward_speed * math.cos(heading_rad) * dt
        self.position[1] += forward_speed * math.sin(heading_rad) * dt
    
    def _update_sensors(self):
        """Update all sensor readings based on current robot state."""
        
        # Update ultrasonic sensor
        self.ultrasonic_distance = self._simulate_ultrasonic_reading(0)
        
        # Update camera (simulate environment-based RGB)
        # Simple simulation: RGB varies based on position and what robot "sees"
        self.camera_rgb = [
            0.3 + 0.4 * math.sin(self.position[0] * 0.1),
            0.3 + 0.4 * math.cos(self.position[1] * 0.1), 
            0.3 + 0.4 * math.sin((self.position[0] + self.position[1]) * 0.05)
        ]
        
        # Update line sensors (simulate line following scenario)
        # Lines appear at certain positions
        line_positions = [(10, 10), (20, 5), (30, 15), (15, 25)]
        min_distance_to_line = min(
            math.sqrt((self.position[0] - lx)**2 + (self.position[1] - ly)**2)
            for lx, ly in line_positions
        )
        
        line_strength = max(0.0, 1.0 - min_distance_to_line / 5.0)
        self.line_sensors = [
            line_strength + random.uniform(-0.1, 0.1),
            line_strength + random.uniform(-0.1, 0.1)
        ]
        self.line_sensors = [max(0.0, min(1.0, s)) for s in self.line_sensors]
    
    def _simulate_ultrasonic_reading(self, pan_angle_offset: float) -> float:
        """Simulate ultrasonic sensor reading in a given direction."""
        
        # Simulate obstacles in the environment
        obstacles = [
            (5, 5, 2),    # x, y, radius
            (15, 8, 1.5),
            (25, 12, 3),
            (8, 20, 2),
            (30, 30, 4)
        ]
        
        # Calculate sensor direction
        sensor_angle = math.radians(self.position[2] + pan_angle_offset)
        sensor_direction = [math.cos(sensor_angle), math.sin(sensor_angle)]
        
        # Cast ray from robot position in sensor direction
        min_distance = 200.0  # Max sensor range
        
        for obs_x, obs_y, obs_radius in obstacles:
            # Vector from robot to obstacle center
            to_obstacle = [obs_x - self.position[0], obs_y - self.position[1]]
            
            # Project obstacle center onto sensor ray
            projection = (to_obstacle[0] * sensor_direction[0] + 
                         to_obstacle[1] * sensor_direction[1])
            
            if projection > 0:  # Obstacle is in front of sensor
                # Distance from obstacle center to ray (safe calculation)
                distance_squared = to_obstacle[0]**2 + to_obstacle[1]**2
                projection_squared = projection**2
                perpendicular_squared = max(0.0, distance_squared - projection_squared)
                perpendicular_distance = math.sqrt(perpendicular_squared)
                
                if perpendicular_distance <= obs_radius:
                    # Ray hits obstacle
                    hit_distance = projection - math.sqrt(
                        obs_radius**2 - perpendicular_distance**2
                    )
                    if hit_distance > 0:
                        min_distance = min(min_distance, hit_distance * 100)  # Convert to cm
        
        # Add some sensor noise
        min_distance += random.uniform(-2.0, 2.0)
        return max(5.0, min(200.0, min_distance))
    
    def get_robot_status(self) -> Dict[str, Any]:
        """Get comprehensive robot status for monitoring."""
        brain_stats = self.brain.get_brain_stats()
        
        return {
            'position': {
                'x': self.position[0],
                'y': self.position[1], 
                'heading': self.position[2]
            },
            'sensors': {
                'ultrasonic_distance': self.ultrasonic_distance,
                'camera_rgb': self.camera_rgb,
                'line_sensors': self.line_sensors,
                'camera_pan': self.camera_pan_angle,
                'camera_tilt': self.camera_tilt_angle
            },
            'motors': {
                'speed': self.motor_speed,
                'steering': self.steering_angle
            },
            'brain': {
                'total_cycles': brain_stats['brain_summary']['total_cycles'],
                'uptime_seconds': brain_stats['brain_summary']['uptime_seconds'],
                'prediction_confidence': brain_stats['brain_summary']['prediction_confidence'],
                'architecture': brain_stats['brain_summary']['architecture'],
                'sensory_patterns': brain_stats['brain_summary']['streams']['sensory_patterns'],
                'motor_patterns': brain_stats['brain_summary']['streams']['motor_patterns']
            },
            'performance': {
                'control_cycles': self.total_control_cycles,
                'obstacle_encounters': self.obstacle_encounters,
                'successful_navigations': self.successful_navigations,
                'navigation_success_rate': self.successful_navigations / max(1, self.total_control_cycles)
            }
        }
    
    def reset_robot(self):
        """Reset robot to initial state (for testing)."""
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0]
        self.motor_speed = 0.0
        self.steering_angle = 0.0
        self.camera_pan_angle = 0.0
        self.camera_tilt_angle = 0.0
        
        self.total_control_cycles = 0
        self.obstacle_encounters = 0
        self.successful_navigations = 0
        
        self.brain.reset_brain()
        print("🔄 PiCar-X robot reset to initial state")


if __name__ == "__main__":
    # This module is designed to be imported, but can demonstrate basic functionality
    print("🤖 PiCar-X Brainstem - Robot Brain Interface")
    print("   This module connects the minimal brain to PiCar-X robot hardware")
    print("   Import this module to control a real or simulated PiCar-X robot")
    
    # Quick test
    brainstem = PiCarXBrainstem()
    
    print("\n🧪 Quick functionality test:")
    for i in range(3):
        result = brainstem.control_cycle()
        print(f"   Cycle {i+1}: {result['cycle_time']*1000:.1f}ms, "
              f"brain method: {result['brain_state']['prediction_method']}")
    
    status = brainstem.get_robot_status()
    print(f"\n📊 Robot status: Position ({status['position']['x']:.2f}, {status['position']['y']:.2f}), "
          f"Brain experiences: {status['brain']['total_experiences']}")