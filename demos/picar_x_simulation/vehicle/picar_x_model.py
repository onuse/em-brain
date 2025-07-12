"""
PiCar-X Vehicle Model

Accurate simulation of the SunFounder PiCar-X robot including:
- Physical dimensions and dynamics
- Sensor mounting and simulation
- Actuator behavior and limits
- Real-time physics integration
"""

import numpy as np
import time
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass
class VehicleState:
    """Current state of the PiCar-X vehicle."""
    position: List[float]        # [x, y] in meters
    orientation: float           # Heading in degrees
    velocity: List[float]        # [vx, vy] in m/s
    angular_velocity: float      # rad/s
    acceleration: List[float]    # [ax, ay] in m/s²


class PiCarXVehicle:
    """
    SunFounder PiCar-X Robot Simulation Model
    
    Accurately models the physical robot including:
    - Differential drive kinematics
    - Ackermann steering geometry
    - Sensor positions and characteristics
    - Motor dynamics and limitations
    """
    
    def __init__(self, position: List[float] = [0.0, 0.0], orientation: float = 0.0):
        """
        Initialize PiCar-X vehicle model.
        
        Args:
            position: Initial position [x, y] in meters
            orientation: Initial heading in degrees
        """
        # Vehicle dimensions (SunFounder PiCar-X specifications)
        self.length = 0.165         # 165mm
        self.width = 0.095          # 95mm
        self.height = 0.085         # 85mm
        self.wheelbase = 0.090      # 90mm (distance between front and rear axles)
        self.track_width = 0.065    # 65mm (distance between left and right wheels)
        self.mass = 0.200          # 200g
        
        # Vehicle state
        self.state = VehicleState(
            position=list(position),
            orientation=orientation,
            velocity=[0.0, 0.0],
            angular_velocity=0.0,
            acceleration=[0.0, 0.0]
        )
        
        # Motor system
        self.left_motor_speed = 0.0   # -1.0 to 1.0
        self.right_motor_speed = 0.0  # -1.0 to 1.0
        self.max_motor_speed = 0.5    # 0.5 m/s max speed
        self.motor_acceleration = 2.0  # 2.0 m/s² max acceleration
        
        # Steering system
        self.steering_angle = 0.0     # Current steering angle (degrees)
        self.target_steering_angle = 0.0
        self.max_steering_angle = 30.0  # ±30° steering range
        self.steering_rate = 90.0     # 90°/s steering rate
        
        # Camera servo system
        self.camera_pan_angle = 0.0   # Current pan angle
        self.camera_tilt_angle = 0.0  # Current tilt angle
        self.target_pan_angle = 0.0
        self.target_tilt_angle = 0.0
        self.max_pan_angle = 90.0     # ±90° pan range
        self.max_tilt_angle = 30.0    # ±30° tilt range
        self.servo_rate = 180.0       # 180°/s servo rate
        
        # Physics parameters
        self.tire_friction = 0.8      # Tire-ground friction coefficient
        self.air_resistance = 0.02    # Air resistance coefficient
        self.rotational_inertia = 0.001  # kg⋅m² (estimated)
        
        # Sensor mounting positions (relative to vehicle center)
        self.camera_offset = [0.060, 0.0, 0.040]     # 6cm forward, 4cm up
        self.ultrasonic_offset = [0.080, 0.0, 0.030] # 8cm forward, 3cm up
        self.imu_offset = [0.0, 0.0, 0.020]          # Center, 2cm up
        
        # Collision state
        self.in_collision = False
        self.collision_object = None
        
        # Performance tracking
        self.total_distance = 0.0
        self.max_speed_reached = 0.0
        self.simulation_time = 0.0
        
        # Reference to world (set by world when vehicle is added)
        self.world = None
        
        print(f"🚗 PiCar-X vehicle initialized at ({position[0]:.2f}, {position[1]:.2f})")
    
    def set_motor_speeds(self, left_speed: float, right_speed: float):
        """
        Set motor speeds for differential drive.
        
        Args:
            left_speed: Left motor speed (-1.0 to 1.0)
            right_speed: Right motor speed (-1.0 to 1.0)
        """
        self.left_motor_speed = np.clip(left_speed, -1.0, 1.0)
        self.right_motor_speed = np.clip(right_speed, -1.0, 1.0)
    
    def set_steering_angle(self, angle: float):
        """
        Set steering angle (Ackermann steering).
        
        Args:
            angle: Steering angle in degrees (±30°)
        """
        self.target_steering_angle = np.clip(angle, -self.max_steering_angle, self.max_steering_angle)
    
    def set_camera_angles(self, pan: float, tilt: float):
        """
        Set camera servo angles.
        
        Args:
            pan: Pan angle in degrees (±90°)
            tilt: Tilt angle in degrees (±30°)
        """
        self.target_pan_angle = np.clip(pan, -self.max_pan_angle, self.max_pan_angle)
        self.target_tilt_angle = np.clip(tilt, -self.max_tilt_angle, self.max_tilt_angle)
    
    def update_physics(self, dt: float):
        """
        Update vehicle physics simulation.
        
        Args:
            dt: Time step in seconds
        """
        self.simulation_time += dt
        
        # Update servo positions
        self._update_servos(dt)
        
        # Update vehicle dynamics
        if not self.in_collision:
            self._update_dynamics(dt)
        else:
            # In collision - reduce speeds
            self._handle_collision_dynamics(dt)
        
        # Update tracking statistics
        self._update_statistics(dt)
    
    def _update_servos(self, dt: float):
        """Update camera servo positions."""
        # Pan servo
        pan_error = self.target_pan_angle - self.camera_pan_angle
        max_pan_change = self.servo_rate * dt
        pan_change = np.clip(pan_error, -max_pan_change, max_pan_change)
        self.camera_pan_angle += pan_change
        
        # Tilt servo
        tilt_error = self.target_tilt_angle - self.camera_tilt_angle
        max_tilt_change = self.servo_rate * dt
        tilt_change = np.clip(tilt_error, -max_tilt_change, max_tilt_change)
        self.camera_tilt_angle += tilt_change
        
        # Steering servo
        steering_error = self.target_steering_angle - self.steering_angle
        max_steering_change = self.steering_rate * dt
        steering_change = np.clip(steering_error, -max_steering_change, max_steering_change)
        self.steering_angle += steering_change
    
    def _update_dynamics(self, dt: float):
        """Update vehicle kinematics and dynamics."""
        # Convert motor speeds to wheel velocities
        left_wheel_velocity = self.left_motor_speed * self.max_motor_speed
        right_wheel_velocity = self.right_motor_speed * self.max_motor_speed
        
        # Differential drive kinematics
        linear_velocity = (left_wheel_velocity + right_wheel_velocity) / 2.0
        angular_velocity = (right_wheel_velocity - left_wheel_velocity) / self.track_width
        
        # Add steering effect (simplified Ackermann steering)
        if abs(self.steering_angle) > 0.1:
            # Steering modifies the angular velocity
            steering_factor = np.tan(np.radians(self.steering_angle)) / self.wheelbase
            angular_velocity += linear_velocity * steering_factor
        
        # Apply physics constraints
        max_angular_velocity = 2 * np.pi  # 1 rev/sec max
        angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)
        
        # Store previous state for acceleration calculation
        prev_velocity = self.state.velocity.copy()
        
        # Update orientation
        self.state.angular_velocity = angular_velocity
        self.state.orientation += np.degrees(angular_velocity * dt)
        self.state.orientation = self.state.orientation % 360  # Keep in 0-360 range
        
        # Update velocity in world coordinates
        heading_rad = np.radians(self.state.orientation)
        self.state.velocity[0] = linear_velocity * np.cos(heading_rad)
        self.state.velocity[1] = linear_velocity * np.sin(heading_rad)
        
        # Apply air resistance
        speed = np.linalg.norm(self.state.velocity)
        if speed > 0:
            resistance_force = self.air_resistance * speed * speed
            resistance_factor = max(0, 1 - resistance_force * dt / max(0.001, speed))
            self.state.velocity[0] *= resistance_factor
            self.state.velocity[1] *= resistance_factor
        
        # Update position
        self.state.position[0] += self.state.velocity[0] * dt
        self.state.position[1] += self.state.velocity[1] * dt
        
        # Calculate acceleration
        self.state.acceleration[0] = (self.state.velocity[0] - prev_velocity[0]) / dt
        self.state.acceleration[1] = (self.state.velocity[1] - prev_velocity[1]) / dt
    
    def _handle_collision_dynamics(self, dt: float):
        """Handle physics when in collision."""
        # Reduce velocities in collision
        damping_factor = 0.8  # 20% velocity loss per second in collision
        self.state.velocity[0] *= (1 - damping_factor * dt)
        self.state.velocity[1] *= (1 - damping_factor * dt)
        self.state.angular_velocity *= (1 - damping_factor * dt)
        
        # Stop if velocity is very low
        if np.linalg.norm(self.state.velocity) < 0.01:
            self.state.velocity = [0.0, 0.0]
            self.state.angular_velocity = 0.0
    
    def _update_statistics(self, dt: float):
        """Update performance statistics."""
        # Total distance
        distance_increment = np.linalg.norm(self.state.velocity) * dt
        self.total_distance += distance_increment
        
        # Max speed
        current_speed = np.linalg.norm(self.state.velocity)
        self.max_speed_reached = max(self.max_speed_reached, current_speed)
    
    def handle_collision(self, obstacle):
        """
        Handle collision with an obstacle.
        
        Args:
            obstacle: The obstacle that was hit
        """
        self.in_collision = True
        self.collision_object = obstacle
        
        # Simple collision response - bounce back slightly
        bounce_velocity = 0.05  # 5cm/s bounce
        
        # Calculate bounce direction (opposite to movement)
        if np.linalg.norm(self.state.velocity) > 0:
            bounce_direction = -np.array(self.state.velocity) / np.linalg.norm(self.state.velocity)
            self.state.velocity[0] = bounce_direction[0] * bounce_velocity
            self.state.velocity[1] = bounce_direction[1] * bounce_velocity
        
        # Stop motors temporarily
        self.left_motor_speed *= 0.1
        self.right_motor_speed *= 0.1
    
    def clear_collision(self):
        """Clear collision state."""
        self.in_collision = False
        self.collision_object = None
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position."""
        return tuple(self.state.position)
    
    def get_bounding_box(self) -> Tuple[float, float]:
        """Get vehicle bounding box size."""
        return (self.length, self.width)
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """
        Get sensor data for brainstem processing.
        
        Returns:
            Dictionary with all sensor information
        """
        return {
            "position": self.state.position.copy(),
            "orientation": self.state.orientation,
            "velocity": self.state.velocity.copy(),
            "angular_velocity": self.state.angular_velocity,
            "acceleration": self.state.acceleration.copy(),
            "motor_speeds": {
                "left": self.left_motor_speed,
                "right": self.right_motor_speed
            },
            "steering_angle": self.steering_angle,
            "camera_angles": {
                "pan": self.camera_pan_angle,
                "tilt": self.camera_tilt_angle
            },
            "in_collision": self.in_collision
        }
    
    def get_camera_pose(self) -> Dict[str, Any]:
        """
        Get camera pose for rendering.
        
        Returns:
            Camera position and orientation for world rendering
        """
        # Calculate camera position in world coordinates
        heading_rad = np.radians(self.state.orientation)
        
        # Camera offset rotated by vehicle orientation
        camera_x = (self.state.position[0] + 
                   self.camera_offset[0] * np.cos(heading_rad) - 
                   self.camera_offset[1] * np.sin(heading_rad))
        camera_y = (self.state.position[1] + 
                   self.camera_offset[0] * np.sin(heading_rad) + 
                   self.camera_offset[1] * np.cos(heading_rad))
        
        # Camera orientation includes vehicle heading and pan angle
        camera_orientation = self.state.orientation + self.camera_pan_angle
        
        return {
            "position": [camera_x, camera_y, self.camera_offset[2]],
            "orientation": camera_orientation,
            "tilt": self.camera_tilt_angle,
            "field_of_view": 60.0
        }
    
    def get_ultrasonic_pose(self) -> Dict[str, Any]:
        """Get ultrasonic sensor pose for distance measurement."""
        heading_rad = np.radians(self.state.orientation)
        
        # Ultrasonic sensor position
        sensor_x = (self.state.position[0] + 
                   self.ultrasonic_offset[0] * np.cos(heading_rad))
        sensor_y = (self.state.position[1] + 
                   self.ultrasonic_offset[0] * np.sin(heading_rad))
        
        return {
            "position": [sensor_x, sensor_y, self.ultrasonic_offset[2]],
            "orientation": self.state.orientation,
            "max_range": 3.0,
            "beam_angle": 30.0
        }
    
    def apply_world_forces(self, gravity: float, air_resistance: float, dt: float):
        """
        Apply world-level forces (called by world simulation).
        
        Args:
            gravity: Gravitational acceleration
            air_resistance: Air resistance coefficient
            dt: Time step
        """
        # Gravity doesn't affect horizontal motion for ground vehicle
        # Air resistance already handled in _update_dynamics
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get vehicle status for display."""
        speed = np.linalg.norm(self.state.velocity)
        
        return {
            "position": self.state.position,
            "orientation": self.state.orientation,
            "speed": speed,
            "motor_speeds": [self.left_motor_speed, self.right_motor_speed],
            "steering_angle": self.steering_angle,
            "camera_angles": [self.camera_pan_angle, self.camera_tilt_angle],
            "collision": self.in_collision,
            "total_distance": self.total_distance,
            "simulation_time": self.simulation_time
        }
    
    def reset_vehicle(self, position: List[float] = None, orientation: float = None):
        """Reset vehicle to initial state."""
        if position is not None:
            self.state.position = list(position)
        if orientation is not None:
            self.state.orientation = orientation
        
        # Reset dynamics
        self.state.velocity = [0.0, 0.0]
        self.state.angular_velocity = 0.0
        self.state.acceleration = [0.0, 0.0]
        
        # Reset actuators
        self.left_motor_speed = 0.0
        self.right_motor_speed = 0.0
        self.steering_angle = 0.0
        self.target_steering_angle = 0.0
        self.camera_pan_angle = 0.0
        self.camera_tilt_angle = 0.0
        self.target_pan_angle = 0.0
        self.target_tilt_angle = 0.0
        
        # Reset collision state
        self.clear_collision()
        
        # Reset statistics
        self.total_distance = 0.0
        self.max_speed_reached = 0.0
        self.simulation_time = 0.0
        
        print(f"🔄 Vehicle reset to ({self.state.position[0]:.2f}, {self.state.position[1]:.2f})")
    
    def __str__(self) -> str:
        speed = np.linalg.norm(self.state.velocity)
        return (f"PiCarXVehicle(pos=({self.state.position[0]:.2f}, {self.state.position[1]:.2f}), "
                f"heading={self.state.orientation:.1f}°, speed={speed:.2f}m/s)")