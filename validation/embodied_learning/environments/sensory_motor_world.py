#!/usr/bin/env python3
"""
Sensory-Motor Coordination Environment

A scientifically rigorous simulated world for testing embodied intelligence.
This environment tests the brain's ability to learn sensory-motor coordination
in a realistic setting with:

- Multi-modal sensory input (distance, light, proprioception)
- Embodied constraints (energy, motor delays, sensor noise)
- Adaptive complexity (environment becomes more challenging)
- Temporal dynamics (state changes over time)

Scientific Questions:
1. Can the brain learn to coordinate multiple sensory modalities?
2. How does embodied constraint affect learning efficiency?
3. Does the brain discover emergent navigation strategies?
4. How does temporal dynamics affect prediction accuracy?
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import math

class ActionType(Enum):
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3

@dataclass
class RobotState:
    """Complete robot state for scientific analysis."""
    position: np.ndarray
    orientation: float
    battery: float
    motor_temperature: float
    last_action: Optional[ActionType]
    action_history: List[ActionType]
    
    def to_dict(self):
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation,
            'battery': self.battery,
            'motor_temperature': self.motor_temperature,
            'last_action': self.last_action.value if self.last_action else None,
            'action_history': [a.value for a in self.action_history[-10:]]  # Last 10 actions
        }

@dataclass
class EnvironmentState:
    """Environment state for reproducible experiments."""
    time_step: int
    light_positions: List[Tuple[float, float]]
    obstacle_positions: List[Tuple[float, float, float]]  # x, y, radius
    light_intensity: float
    noise_level: float
    
    def to_dict(self):
        return asdict(self)

class SensoryMotorWorld:
    """
    Scientifically rigorous sensory-motor learning environment.
    
    This environment simulates a robot learning to navigate toward light sources
    while avoiding obstacles, with realistic embodied constraints.
    """
    
    def __init__(self, 
                 world_size: float = 10.0,
                 num_light_sources: int = 2,
                 num_obstacles: int = 5,
                 battery_capacity: float = 1000.0,
                 random_seed: int = 42):
        """
        Initialize the sensory-motor world.
        
        Args:
            world_size: Size of the square world (meters)
            num_light_sources: Number of light sources
            num_obstacles: Number of obstacles
            battery_capacity: Battery capacity (arbitrary units)
            random_seed: Random seed for reproducibility
        """
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # World parameters
        self.world_size = world_size
        self.battery_capacity = battery_capacity
        self.time_step = 0
        self.dt = 0.1  # Time step (seconds)
        
        # Robot state
        self.robot_state = RobotState(
            position=np.array([world_size/2, world_size/2]),  # Start in center
            orientation=0.0,  # Facing right
            battery=battery_capacity,
            motor_temperature=20.0,  # Celsius
            last_action=None,
            action_history=[]
        )
        
        # Environment state
        self.environment_state = EnvironmentState(
            time_step=0,
            light_positions=self._generate_light_sources(num_light_sources),
            obstacle_positions=self._generate_obstacles(num_obstacles),
            light_intensity=1.0,
            noise_level=0.05
        )
        
        # Physics constants
        self.max_speed = 2.0  # m/s
        self.max_angular_speed = np.pi  # rad/s
        self.motor_delay = 0.1  # seconds
        self.sensor_noise_std = 0.05
        
        # Energy consumption (per action)
        self.energy_costs = {
            ActionType.MOVE_FORWARD: 2.0,
            ActionType.TURN_LEFT: 1.0,
            ActionType.TURN_RIGHT: 1.0,
            ActionType.STOP: 0.1
        }
        
        # Sensory system
        self.num_distance_sensors = 8
        self.max_sensor_range = 3.0  # meters
        self.num_light_sensors = 4
        
        # Learning adaptation
        self.difficulty_level = 1
        self.performance_history = []
        
        print(f"🌍 SensoryMotorWorld initialized:")
        print(f"   World size: {world_size}m × {world_size}m")
        print(f"   Light sources: {num_light_sources}")
        print(f"   Obstacles: {num_obstacles}")
        print(f"   Battery capacity: {battery_capacity}")
        print(f"   Random seed: {random_seed}")
    
    def _generate_light_sources(self, num_sources: int) -> List[Tuple[float, float]]:
        """Generate light source positions."""
        sources = []
        for i in range(num_sources):
            # Place light sources away from center and walls
            angle = (i / num_sources) * 2 * np.pi
            radius = self.world_size * 0.3
            x = self.world_size/2 + radius * np.cos(angle)
            y = self.world_size/2 + radius * np.sin(angle)
            sources.append((x, y))
        return sources
    
    def _generate_obstacles(self, num_obstacles: int) -> List[Tuple[float, float, float]]:
        """Generate obstacle positions (x, y, radius)."""
        obstacles = []
        for _ in range(num_obstacles):
            # Random position, avoiding center and light sources
            x = np.random.uniform(1.0, self.world_size - 1.0)
            y = np.random.uniform(1.0, self.world_size - 1.0)
            radius = np.random.uniform(0.2, 0.5)
            
            # Ensure not too close to center
            center_dist = np.linalg.norm([x - self.world_size/2, y - self.world_size/2])
            if center_dist > 1.0:
                obstacles.append((x, y, radius))
        
        return obstacles
    
    def get_sensory_input(self) -> List[float]:
        """
        Get 16-dimensional sensory input vector.
        
        Returns:
            16D vector: [position(2), orientation(1), battery(1), 
                        distance_sensors(8), light_sensors(4)]
        """
        
        # Position (normalized to [0, 1])
        norm_position = self.robot_state.position / self.world_size
        
        # Orientation (normalized to [0, 1])
        norm_orientation = (self.robot_state.orientation % (2 * np.pi)) / (2 * np.pi)
        
        # Battery (normalized to [0, 1])
        norm_battery = self.robot_state.battery / self.battery_capacity
        
        # Distance sensors
        distance_readings = self._get_distance_sensor_readings()
        
        # Light sensors
        light_readings = self._get_light_sensor_readings()
        
        # Combine into 16D vector
        sensory_vector = [
            norm_position[0], norm_position[1],  # Position (2D)
            norm_orientation,                     # Orientation (1D)
            norm_battery,                        # Battery (1D)
            *distance_readings,                  # Distance sensors (8D)
            *light_readings                      # Light sensors (4D)
        ]
        
        # Add sensor noise
        noise = np.random.normal(0, self.sensor_noise_std, len(sensory_vector))
        sensory_vector = np.array(sensory_vector) + noise
        
        # Clamp to valid range
        sensory_vector = np.clip(sensory_vector, 0.0, 1.0)
        
        return sensory_vector.tolist()
    
    def _get_distance_sensor_readings(self) -> List[float]:
        """Get distance sensor readings in 8 directions."""
        readings = []
        
        for i in range(self.num_distance_sensors):
            # Sensor angle relative to robot orientation
            sensor_angle = (i / self.num_distance_sensors) * 2 * np.pi
            absolute_angle = self.robot_state.orientation + sensor_angle
            
            # Cast ray in sensor direction
            distance = self._raycast_distance(absolute_angle)
            
            # Normalize to [0, 1] (0 = max range, 1 = very close)
            normalized_distance = max(0.0, 1.0 - distance / self.max_sensor_range)
            readings.append(normalized_distance)
        
        return readings
    
    def _raycast_distance(self, angle: float) -> float:
        """Cast a ray and return distance to nearest obstacle/wall."""
        ray_x = self.robot_state.position[0]
        ray_y = self.robot_state.position[1]
        
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Check distance to walls
        if dx > 0:
            wall_dist_x = (self.world_size - ray_x) / dx
        elif dx < 0:
            wall_dist_x = ray_x / (-dx)
        else:
            wall_dist_x = float('inf')
        
        if dy > 0:
            wall_dist_y = (self.world_size - ray_y) / dy
        elif dy < 0:
            wall_dist_y = ray_y / (-dy)
        else:
            wall_dist_y = float('inf')
        
        min_distance = min(wall_dist_x, wall_dist_y)
        
        # Check distance to obstacles
        for obs_x, obs_y, obs_radius in self.environment_state.obstacle_positions:
            # Ray-circle intersection
            # Vector from ray start to circle center
            to_center_x = obs_x - ray_x
            to_center_y = obs_y - ray_y
            
            # Project onto ray direction
            proj_length = to_center_x * dx + to_center_y * dy
            
            if proj_length > 0:  # Only consider obstacles ahead
                # Closest point on ray to circle center
                closest_x = ray_x + proj_length * dx
                closest_y = ray_y + proj_length * dy
                
                # Distance from circle center to closest point
                dist_to_ray = np.sqrt((closest_x - obs_x)**2 + (closest_y - obs_y)**2)
                
                if dist_to_ray <= obs_radius:
                    # Ray intersects circle
                    # Calculate intersection distance
                    intersection_offset = np.sqrt(obs_radius**2 - dist_to_ray**2)
                    intersection_dist = proj_length - intersection_offset
                    
                    if intersection_dist > 0:
                        min_distance = min(min_distance, intersection_dist)
        
        return min(min_distance, self.max_sensor_range)
    
    def _get_light_sensor_readings(self) -> List[float]:
        """Get light sensor readings in 4 directions."""
        readings = []
        
        for i in range(self.num_light_sensors):
            # Light sensor angle (90 degrees apart)
            sensor_angle = (i / self.num_light_sensors) * 2 * np.pi
            absolute_angle = self.robot_state.orientation + sensor_angle
            
            # Calculate light intensity in this direction
            light_intensity = self._calculate_light_intensity(absolute_angle)
            readings.append(light_intensity)
        
        return readings
    
    def _calculate_light_intensity(self, angle: float) -> float:
        """Calculate light intensity in a given direction."""
        total_intensity = 0.0
        
        for light_x, light_y in self.environment_state.light_positions:
            # Vector from robot to light
            to_light_x = light_x - self.robot_state.position[0]
            to_light_y = light_y - self.robot_state.position[1]
            
            # Distance to light
            light_distance = np.sqrt(to_light_x**2 + to_light_y**2)
            
            if light_distance > 0:
                # Angle to light
                light_angle = np.arctan2(to_light_y, to_light_x)
                
                # Angular difference from sensor direction
                angle_diff = abs(((light_angle - angle + np.pi) % (2 * np.pi)) - np.pi)
                
                # Light intensity falls off with distance and angle
                if angle_diff < np.pi/4:  # 45 degree field of view
                    intensity = (1.0 / (1.0 + light_distance)) * np.cos(angle_diff)
                    total_intensity += intensity
        
        return min(1.0, total_intensity * self.environment_state.light_intensity)
    
    def execute_action(self, action_vector: List[float]) -> Dict:
        """
        Execute action and return environment response.
        
        Args:
            action_vector: 4D action vector [forward, left, right, stop]
            
        Returns:
            Dict with execution results and metrics
        """
        
        # Interpret action vector
        action_type = self._interpret_action(action_vector)
        
        # Apply motor delay
        time.sleep(self.motor_delay)
        
        # Check energy constraints
        energy_cost = self.energy_costs[action_type]
        if self.robot_state.battery < energy_cost:
            # Insufficient energy - force stop
            action_type = ActionType.STOP
            energy_cost = 0.1
        
        # Execute action
        success = self._execute_motor_action(action_type)
        
        # Update energy
        self.robot_state.battery -= energy_cost
        
        # Update motor temperature (realistic heating)
        if action_type != ActionType.STOP:
            self.robot_state.motor_temperature += 0.5
        else:
            self.robot_state.motor_temperature = max(20.0, self.robot_state.motor_temperature - 0.1)
        
        # Update action history
        self.robot_state.last_action = action_type
        self.robot_state.action_history.append(action_type)
        
        # Update time
        self.time_step += 1
        self.environment_state.time_step = self.time_step
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        return {
            'success': success,
            'action_executed': action_type.value,
            'energy_cost': energy_cost,
            'robot_state': self.robot_state.to_dict(),
            'environment_state': self.environment_state.to_dict(),
            'metrics': metrics,
            'time_step': self.time_step
        }
    
    def _interpret_action(self, action_vector: List[float]) -> ActionType:
        """Interpret 4D action vector as discrete action."""
        # Find the action with highest activation
        max_idx = np.argmax(action_vector)
        
        action_map = {
            0: ActionType.MOVE_FORWARD,
            1: ActionType.TURN_LEFT,
            2: ActionType.TURN_RIGHT,
            3: ActionType.STOP
        }
        
        return action_map[max_idx]
    
    def _execute_motor_action(self, action: ActionType) -> bool:
        """Execute motor action with realistic physics."""
        
        if action == ActionType.MOVE_FORWARD:
            # Move forward in current orientation
            dx = self.max_speed * self.dt * np.cos(self.robot_state.orientation)
            dy = self.max_speed * self.dt * np.sin(self.robot_state.orientation)
            
            new_position = self.robot_state.position + np.array([dx, dy])
            
            # Check collision with walls
            if (0 <= new_position[0] <= self.world_size and 
                0 <= new_position[1] <= self.world_size):
                
                # Check collision with obstacles
                collision = False
                for obs_x, obs_y, obs_radius in self.environment_state.obstacle_positions:
                    distance = np.linalg.norm(new_position - np.array([obs_x, obs_y]))
                    if distance < obs_radius + 0.1:  # Robot has 0.1m radius
                        collision = True
                        break
                
                if not collision:
                    self.robot_state.position = new_position
                    return True
            
            return False  # Collision or out of bounds
        
        elif action == ActionType.TURN_LEFT:
            self.robot_state.orientation += self.max_angular_speed * self.dt
            self.robot_state.orientation = self.robot_state.orientation % (2 * np.pi)
            return True
        
        elif action == ActionType.TURN_RIGHT:
            self.robot_state.orientation -= self.max_angular_speed * self.dt
            self.robot_state.orientation = self.robot_state.orientation % (2 * np.pi)
            return True
        
        elif action == ActionType.STOP:
            return True
        
        return False
    
    def _calculate_metrics(self) -> Dict:
        """Calculate scientific metrics for analysis."""
        
        # Distance to nearest light source
        light_distances = []
        for light_x, light_y in self.environment_state.light_positions:
            distance = np.linalg.norm(self.robot_state.position - np.array([light_x, light_y]))
            light_distances.append(distance)
        
        min_light_distance = min(light_distances)
        
        # Exploration coverage (how much of the world has been explored)
        exploration_score = self._calculate_exploration_score()
        
        # Efficiency (distance traveled vs. progress toward goal)
        efficiency = self._calculate_efficiency()
        
        # Collision rate
        collision_rate = self._calculate_collision_rate()
        
        return {
            'min_light_distance': min_light_distance,
            'exploration_score': exploration_score,
            'efficiency': efficiency,
            'collision_rate': collision_rate,
            'battery_remaining': self.robot_state.battery / self.battery_capacity,
            'motor_temperature': self.robot_state.motor_temperature,
            'time_step': self.time_step
        }
    
    def _calculate_exploration_score(self) -> float:
        """Calculate how much of the world has been explored."""
        # Simple implementation: based on distance from starting position
        start_pos = np.array([self.world_size/2, self.world_size/2])
        current_distance = np.linalg.norm(self.robot_state.position - start_pos)
        max_distance = self.world_size * 0.7  # Maximum reasonable exploration
        
        return min(1.0, current_distance / max_distance)
    
    def _calculate_efficiency(self) -> float:
        """Calculate movement efficiency."""
        if len(self.robot_state.action_history) < 2:
            return 0.0
        
        # Count useful actions (forward movement)
        useful_actions = sum(1 for action in self.robot_state.action_history 
                           if action == ActionType.MOVE_FORWARD)
        
        total_actions = len(self.robot_state.action_history)
        
        return useful_actions / total_actions if total_actions > 0 else 0.0
    
    def _calculate_collision_rate(self) -> float:
        """Calculate collision rate based on recent actions."""
        if len(self.robot_state.action_history) < 10:
            return 0.0
        
        # Count failed forward movements (collisions)
        recent_actions = self.robot_state.action_history[-10:]
        failed_movements = 0
        
        # This is a simplified collision detection
        # In a real implementation, we'd track actual collision events
        return failed_movements / len(recent_actions)
    
    def get_state_summary(self) -> Dict:
        """Get complete state summary for logging."""
        return {
            'robot_state': self.robot_state.to_dict(),
            'environment_state': self.environment_state.to_dict(),
            'metrics': self._calculate_metrics(),
            'time_step': self.time_step
        }
    
    def reset(self, random_seed: Optional[int] = None):
        """Reset environment to initial state."""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Reset robot to center
        self.robot_state.position = np.array([self.world_size/2, self.world_size/2])
        self.robot_state.orientation = 0.0
        self.robot_state.battery = self.battery_capacity
        self.robot_state.motor_temperature = 20.0
        self.robot_state.last_action = None
        self.robot_state.action_history = []
        
        # Reset environment
        self.time_step = 0
        self.environment_state.time_step = 0
        
        print("🔄 Environment reset to initial state")
    
    def adapt_difficulty(self, performance_score: float):
        """Adapt environment difficulty based on performance."""
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-10:])
            
            if recent_performance > 0.8 and self.difficulty_level < 5:
                self.difficulty_level += 1
                self._increase_difficulty()
                print(f"📈 Difficulty increased to level {self.difficulty_level}")
            
            elif recent_performance < 0.4 and self.difficulty_level > 1:
                self.difficulty_level -= 1
                self._decrease_difficulty()
                print(f"📉 Difficulty decreased to level {self.difficulty_level}")
    
    def _increase_difficulty(self):
        """Increase environment difficulty."""
        # Add more obstacles
        new_obstacles = self._generate_obstacles(2)
        self.environment_state.obstacle_positions.extend(new_obstacles)
        
        # Increase sensor noise
        self.sensor_noise_std = min(0.2, self.sensor_noise_std + 0.02)
        
        # Reduce light intensity
        self.environment_state.light_intensity = max(0.3, self.environment_state.light_intensity - 0.1)
    
    def _decrease_difficulty(self):
        """Decrease environment difficulty."""
        # Remove some obstacles
        if len(self.environment_state.obstacle_positions) > 2:
            self.environment_state.obstacle_positions.pop()
        
        # Decrease sensor noise
        self.sensor_noise_std = max(0.01, self.sensor_noise_std - 0.02)
        
        # Increase light intensity
        self.environment_state.light_intensity = min(1.0, self.environment_state.light_intensity + 0.1)


if __name__ == "__main__":
    # Quick test of the environment
    env = SensoryMotorWorld(random_seed=42)
    
    print("\n🧪 Testing sensory-motor environment:")
    
    # Test sensory input
    sensors = env.get_sensory_input()
    print(f"📡 Sensory input: {len(sensors)}D vector")
    print(f"   Position: {sensors[0]:.3f}, {sensors[1]:.3f}")
    print(f"   Orientation: {sensors[2]:.3f}")
    print(f"   Battery: {sensors[3]:.3f}")
    print(f"   Distance sensors: {[f'{s:.3f}' for s in sensors[4:12]]}")
    print(f"   Light sensors: {[f'{s:.3f}' for s in sensors[12:16]]}")
    
    # Test action execution
    print("\n🚀 Testing action execution:")
    for i, action_name in enumerate(['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']):
        action_vector = [0.0] * 4
        action_vector[i] = 1.0
        
        result = env.execute_action(action_vector)
        print(f"   {action_name}: {'✅' if result['success'] else '❌'} "
              f"(energy: {result['energy_cost']:.1f})")
    
    print("\n📊 Final metrics:")
    metrics = env._calculate_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value:.3f}")
    
    print("\n🎯 Environment ready for scientific validation!")