"""
3D Simulation World Environment

Defines the virtual testing environment for the PiCar-X robot.
Includes room geometry, obstacles, physics simulation, and sensor interactions.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Obstacle:
    """Represents an obstacle in the simulation world."""
    position: Tuple[float, float]  # (x, y) position
    size: Tuple[float, float]      # (width, height)
    obstacle_type: str             # "box", "cylinder", "wall"
    height: float = 0.3            # Height in meters


class SimulationWorld:
    """
    3D Simulation Environment for PiCar-X Testing
    
    Provides:
    - Room boundaries and geometry
    - Static and dynamic obstacles
    - Ray casting for sensor simulation
    - Collision detection
    - Basic physics simulation
    """
    
    def __init__(self, room_width: float = 4.0, room_height: float = 3.0):
        """
        Initialize simulation world.
        
        Args:
            room_width: Room width in meters
            room_height: Room height in meters
        """
        self.room_width = room_width
        self.room_height = room_height
        self.wall_thickness = 0.1
        
        # World objects
        self.obstacles: List[Obstacle] = []
        self.vehicles = []  # Vehicles in the world
        
        # Physics state
        self.gravity = 9.81  # m/s²
        self.air_resistance = 0.02
        self.ground_friction = 0.8
        
        # Simulation time
        self.simulation_time = 0.0
        self.last_update = time.time()
        
        # Performance tracking
        self.total_updates = 0
        self.collision_checks = 0
        
        # Initialize world
        self._create_room_boundaries()
        self._create_default_obstacles()
        
        print(f"🌍 SimulationWorld created ({room_width}m x {room_height}m)")
    
    def _create_room_boundaries(self):
        """Create walls around the room."""
        # North wall
        self.obstacles.append(Obstacle(
            position=(self.room_width / 2, self.room_height - self.wall_thickness / 2),
            size=(self.room_width, self.wall_thickness),
            obstacle_type="wall",
            height=0.5
        ))
        
        # South wall
        self.obstacles.append(Obstacle(
            position=(self.room_width / 2, self.wall_thickness / 2),
            size=(self.room_width, self.wall_thickness),
            obstacle_type="wall",
            height=0.5
        ))
        
        # East wall
        self.obstacles.append(Obstacle(
            position=(self.room_width - self.wall_thickness / 2, self.room_height / 2),
            size=(self.wall_thickness, self.room_height),
            obstacle_type="wall",
            height=0.5
        ))
        
        # West wall
        self.obstacles.append(Obstacle(
            position=(self.wall_thickness / 2, self.room_height / 2),
            size=(self.wall_thickness, self.room_height),
            obstacle_type="wall",
            height=0.5
        ))
    
    def _create_default_obstacles(self):
        """Create some default obstacles for testing."""
        # Central obstacle
        self.obstacles.append(Obstacle(
            position=(2.0, 1.5),
            size=(0.3, 0.3),
            obstacle_type="box",
            height=0.2
        ))
        
        # Corner obstacles
        self.obstacles.append(Obstacle(
            position=(0.8, 0.8),
            size=(0.2, 0.2),
            obstacle_type="box",
            height=0.15
        ))
        
        self.obstacles.append(Obstacle(
            position=(3.2, 2.2),
            size=(0.25, 0.25),
            obstacle_type="box",
            height=0.18
        ))
        
        # Cylindrical obstacle
        self.obstacles.append(Obstacle(
            position=(1.5, 2.5),
            size=(0.15, 0.15),  # Radius represented as width/height
            obstacle_type="cylinder",
            height=0.25
        ))
    
    def add_vehicle(self, vehicle):
        """Add a vehicle to the world."""
        self.vehicles.append(vehicle)
        vehicle.world = self  # Give vehicle reference to world
        print(f"🚗 Vehicle added to world: {vehicle}")
    
    def update(self, dt: float):
        """
        Update world physics and state.
        
        Args:
            dt: Time step in seconds
        """
        current_time = time.time()
        self.simulation_time += dt
        self.total_updates += 1
        
        # Update vehicles (if they need world-level physics)
        for vehicle in self.vehicles:
            if hasattr(vehicle, 'apply_world_forces'):
                vehicle.apply_world_forces(self.gravity, self.air_resistance, dt)
        
        self.last_update = current_time
    
    def check_collisions(self):
        """Check for collisions between vehicles and obstacles."""
        self.collision_checks += 1
        
        for vehicle in self.vehicles:
            vehicle_pos = vehicle.get_position()
            vehicle_size = vehicle.get_bounding_box()
            
            # Check collision with each obstacle
            for obstacle in self.obstacles:
                if self._check_box_collision(
                    vehicle_pos, vehicle_size,
                    obstacle.position, obstacle.size
                ):
                    # Handle collision
                    vehicle.handle_collision(obstacle)
    
    def _check_box_collision(self, pos1: Tuple[float, float], size1: Tuple[float, float],
                           pos2: Tuple[float, float], size2: Tuple[float, float]) -> bool:
        """
        Check collision between two rectangular objects.
        
        Args:
            pos1, size1: Position and size of first object
            pos2, size2: Position and size of second object
            
        Returns:
            True if collision detected
        """
        # AABB (Axis-Aligned Bounding Box) collision detection
        x1_min, x1_max = pos1[0] - size1[0]/2, pos1[0] + size1[0]/2
        y1_min, y1_max = pos1[1] - size1[1]/2, pos1[1] + size1[1]/2
        
        x2_min, x2_max = pos2[0] - size2[0]/2, pos2[0] + size2[0]/2
        y2_min, y2_max = pos2[1] - size2[1]/2, pos2[1] + size2[1]/2
        
        # Check overlap
        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)
    
    def cast_ray(self, start_pos: List[float], direction: float, 
                 max_range: float = 3.0, beam_angle: float = 30.0) -> float:
        """
        Cast a ray for ultrasonic sensor simulation.
        
        Args:
            start_pos: Starting position [x, y]
            direction: Direction in degrees
            max_range: Maximum detection range
            beam_angle: Sensor beam width
            
        Returns:
            Distance to closest obstacle
        """
        min_distance = max_range
        
        # Convert direction to radians
        dir_rad = np.radians(direction)
        
        # Cast multiple rays within beam angle for realistic sensor simulation
        num_rays = 5
        half_beam = np.radians(beam_angle / 2)
        
        for i in range(num_rays):
            # Spread rays across beam angle
            ray_angle = dir_rad + np.linspace(-half_beam, half_beam, num_rays)[i]
            
            # Ray direction vector
            ray_dx = np.cos(ray_angle)
            ray_dy = np.sin(ray_angle)
            
            # Check intersection with each obstacle
            for obstacle in self.obstacles:
                distance = self._ray_box_intersection(
                    start_pos, [ray_dx, ray_dy], 
                    obstacle.position, obstacle.size
                )
                
                if distance is not None and distance < min_distance:
                    min_distance = distance
        
        return min_distance
    
    def _ray_box_intersection(self, ray_start: List[float], ray_dir: List[float],
                            box_center: Tuple[float, float], box_size: Tuple[float, float]) -> Optional[float]:
        """
        Calculate ray-box intersection distance.
        
        Args:
            ray_start: Ray starting position
            ray_dir: Ray direction (normalized)
            box_center: Box center position
            box_size: Box size (width, height)
            
        Returns:
            Distance to intersection or None if no intersection
        """
        # Box boundaries
        box_min_x = box_center[0] - box_size[0] / 2
        box_max_x = box_center[0] + box_size[0] / 2
        box_min_y = box_center[1] - box_size[1] / 2
        box_max_y = box_center[1] + box_size[1] / 2
        
        # Ray intersection with box (AABB method)
        if abs(ray_dir[0]) < 1e-6:  # Ray parallel to Y axis
            if ray_start[0] < box_min_x or ray_start[0] > box_max_x:
                return None
            t_min_x, t_max_x = float('-inf'), float('inf')
        else:
            t1_x = (box_min_x - ray_start[0]) / ray_dir[0]
            t2_x = (box_max_x - ray_start[0]) / ray_dir[0]
            t_min_x, t_max_x = min(t1_x, t2_x), max(t1_x, t2_x)
        
        if abs(ray_dir[1]) < 1e-6:  # Ray parallel to X axis
            if ray_start[1] < box_min_y or ray_start[1] > box_max_y:
                return None
            t_min_y, t_max_y = float('-inf'), float('inf')
        else:
            t1_y = (box_min_y - ray_start[1]) / ray_dir[1]
            t2_y = (box_max_y - ray_start[1]) / ray_dir[1]
            t_min_y, t_max_y = min(t1_y, t2_y), max(t1_y, t2_y)
        
        # Find intersection
        t_min = max(t_min_x, t_min_y)
        t_max = min(t_max_x, t_max_y)
        
        if t_min <= t_max and t_min > 0:
            return t_min
        
        return None
    
    def render_camera_view(self, camera_pose: Dict[str, Any], 
                          resolution: Tuple[int, int] = (640, 480),
                          fov: float = 60.0) -> np.ndarray:
        """
        Render camera view from simulation (placeholder).
        
        Args:
            camera_pose: Camera position and orientation
            resolution: Image resolution
            fov: Field of view in degrees
            
        Returns:
            RGB image array
        """
        # Placeholder implementation - returns simulated camera data
        # In a full implementation, this would use OpenGL or similar
        
        width, height = resolution
        
        # Create a simple synthetic image based on world state
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background color (floor)
        image[:, :] = [50, 50, 50]  # Dark gray floor
        
        # Add some basic environment visualization
        # This is a simplified 2D projection for now
        
        camera_x = camera_pose.get('position', [0, 0])[0]
        camera_y = camera_pose.get('position', [0, 0])[1]
        camera_angle = camera_pose.get('orientation', 0)
        
        # Draw obstacles as colored rectangles (very basic)
        for i, obstacle in enumerate(self.obstacles):
            if obstacle.obstacle_type != "wall":  # Skip walls for now
                # Project obstacle position to image coordinates
                rel_x = obstacle.position[0] - camera_x
                rel_y = obstacle.position[1] - camera_y
                
                # Simple projection (placeholder)
                if abs(rel_x) < 2.0 and abs(rel_y) < 2.0:
                    img_x = int(width/2 + rel_x * 100)
                    img_y = int(height/2 + rel_y * 100)
                    
                    if 0 <= img_x < width-20 and 0 <= img_y < height-20:
                        # Draw obstacle as colored square
                        color = [100 + i * 30, 150, 100]
                        image[img_y:img_y+20, img_x:img_x+20] = color
        
        return image
    
    def get_obstacles_in_range(self, position: Tuple[float, float], 
                              max_range: float) -> List[Obstacle]:
        """Get obstacles within range of a position."""
        nearby_obstacles = []
        
        for obstacle in self.obstacles:
            distance = np.sqrt(
                (obstacle.position[0] - position[0])**2 +
                (obstacle.position[1] - position[1])**2
            )
            
            if distance <= max_range:
                nearby_obstacles.append(obstacle)
        
        return nearby_obstacles
    
    def is_position_valid(self, position: Tuple[float, float], 
                         size: Tuple[float, float] = (0.165, 0.095)) -> bool:
        """
        Check if a position is valid (no collisions).
        
        Args:
            position: Position to check
            size: Object size (default: PiCar-X dimensions)
            
        Returns:
            True if position is valid
        """
        for obstacle in self.obstacles:
            if self._check_box_collision(position, size, obstacle.position, obstacle.size):
                return False
        return True
    
    def get_world_bounds(self) -> Tuple[float, float, float, float]:
        """Get world boundaries (min_x, min_y, max_x, max_y)."""
        return (0.0, 0.0, self.room_width, self.room_height)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get world simulation statistics."""
        return {
            "room_dimensions": [self.room_width, self.room_height],
            "total_obstacles": len(self.obstacles),
            "vehicles": len(self.vehicles),
            "simulation_time": self.simulation_time,
            "total_updates": self.total_updates,
            "collision_checks": self.collision_checks,
            "physics": {
                "gravity": self.gravity,
                "air_resistance": self.air_resistance,
                "ground_friction": self.ground_friction
            }
        }
    
    def __str__(self) -> str:
        return (f"SimulationWorld({self.room_width}m x {self.room_height}m, "
                f"{len(self.obstacles)} obstacles, {len(self.vehicles)} vehicles)")