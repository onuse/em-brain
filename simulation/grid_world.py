"""
Grid world simulation environment for testing emergent intelligence.
Provides a 2D grid world with robot, obstacles, food, and dangers.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class Robot:
    """Robot entity in the grid world."""
    position: Tuple[int, int]      # (x, y) grid coordinates
    orientation: int = 0           # 0=North, 1=East, 2=South, 3=West
    health: float = 1.0           # 0.0 to 1.0
    energy: float = 1.0           # 0.0 to 1.0
    time_since_food: int = 0      # Steps since last food consumption
    time_since_damage: int = 0    # Steps since last damage taken
    
    def __post_init__(self):
        """Ensure valid initial state."""
        self.health = max(0.0, min(1.0, self.health))
        self.energy = max(0.0, min(1.0, self.energy))
        self.orientation = self.orientation % 4


class GridWorldSimulation:
    """
    A PyGame-based 2D grid world that simulates a robot environment.
    Acts as a virtual brainstem providing sensory input and responding to motor commands.
    """
    
    # Cell type constants
    EMPTY = 0.0
    WALL = 1.0
    FOOD = 0.5
    DANGER = -1.0
    PLANT = 0.3
    ROBOT = 0.8
    
    # Movement directions: [dx, dy] for each orientation
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
    
    def __init__(self, width: int = 40, height: int = 40, seed: Optional[int] = None):
        """Initialize the grid world simulation."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.width = width
        self.height = height
        self.world = self._generate_world()
        self.robot = Robot(position=self._find_safe_starting_position())
        self.step_count = 0
        self.total_food_consumed = 0
        self.total_collisions = 0
        
    def _generate_world(self) -> np.ndarray:
        """Generate a random grid world with walls, food, and dangers."""
        world = np.zeros((self.width, self.height))
        
        # Add walls around perimeter
        world[0, :] = self.WALL    # Top wall
        world[-1, :] = self.WALL   # Bottom wall
        world[:, 0] = self.WALL    # Left wall
        world[:, -1] = self.WALL   # Right wall
        
        # Add random internal obstacles with connectivity checks
        # Scale obstacle count with world size (aim for 8-12% coverage)
        total_cells = (self.width - 2) * (self.height - 2)
        obstacle_count = int(total_cells * 0.10)  # 10% coverage
        
        placed_obstacles = 0
        attempts = 0
        max_attempts = obstacle_count * 3
        
        while placed_obstacles < obstacle_count and attempts < max_attempts:
            x, y = random.randint(1, self.width-2), random.randint(1, self.height-2)
            
            # Don't place obstacles that would create isolated 1x1 cells
            if self._would_create_isolated_cell(world, x, y):
                attempts += 1
                continue
                
            world[x, y] = self.WALL
            placed_obstacles += 1
            attempts += 1
        
        # Add food sources (scale with world size - 3-5% coverage)
        food_count = int(total_cells * 0.04)  # 4% coverage
        placed_food = 0
        while placed_food < food_count:
            x, y = random.randint(1, self.width-2), random.randint(1, self.height-2)
            if world[x, y] == self.EMPTY:  # Only place on empty cells
                world[x, y] = self.FOOD
                placed_food += 1
        
        # Add danger zones (1-2% coverage)
        danger_count = int(total_cells * 0.015)  # 1.5% coverage
        placed_danger = 0
        while placed_danger < danger_count:
            x, y = random.randint(1, self.width-2), random.randint(1, self.height-2)
            if world[x, y] == self.EMPTY:  # Only place on empty cells
                world[x, y] = self.DANGER
                placed_danger += 1
        
        # Add plants for smell sensations (3-4% coverage)
        plant_count = int(total_cells * 0.035)  # 3.5% coverage
        placed_plants = 0
        while placed_plants < plant_count:
            x, y = random.randint(1, self.width-2), random.randint(1, self.height-2)
            if world[x, y] == self.EMPTY:  # Only place on empty cells
                world[x, y] = self.PLANT
                placed_plants += 1
        
        return world
    
    def _would_create_isolated_cell(self, world: np.ndarray, x: int, y: int) -> bool:
        """Check if placing a wall at (x,y) would create isolated 1x1 cells."""
        if world[x, y] != self.EMPTY:
            return False  # Can't place on non-empty cell anyway
        
        # Check all 8 neighbors to see if placing a wall here would isolate any empty cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the center cell
                
                check_x, check_y = x + dx, y + dy
                
                # Check bounds
                if not (1 <= check_x < self.width-1 and 1 <= check_y < self.height-1):
                    continue
                
                # If neighbor is empty, check if it would become isolated
                if world[check_x, check_y] == self.EMPTY:
                    # Count how many of its neighbors would be walls if we place wall at (x,y)
                    wall_neighbors = 0
                    for ddx in [-1, 0, 1]:
                        for ddy in [-1, 0, 1]:
                            if ddx == 0 and ddy == 0:
                                continue
                            
                            neighbor_x, neighbor_y = check_x + ddx, check_y + ddy
                            
                            # Check if this neighbor would be a wall
                            if (neighbor_x == x and neighbor_y == y):
                                wall_neighbors += 1  # Our proposed wall
                            elif (0 <= neighbor_x < self.width and 0 <= neighbor_y < self.height):
                                if world[neighbor_x, neighbor_y] == self.WALL:
                                    wall_neighbors += 1
                            else:
                                wall_neighbors += 1  # Out of bounds = wall
                    
                    # If cell would have 7 or 8 wall neighbors, it would be isolated/trapped
                    if wall_neighbors >= 7:
                        return True
        
        return False
    
    def _find_safe_starting_position(self) -> Tuple[int, int]:
        """Find a safe empty cell to start the robot."""
        for _ in range(100):  # Try up to 100 times
            x = random.randint(1, self.width-2)
            y = random.randint(1, self.height-2)
            if self.world[x, y] == self.EMPTY:
                return (x, y)
        
        # Fallback: find first empty cell
        for x in range(1, self.width-1):
            for y in range(1, self.height-1):
                if self.world[x, y] == self.EMPTY:
                    return (x, y)
        
        # Last resort: clear a cell
        x, y = self.width // 2, self.height // 2
        self.world[x, y] = self.EMPTY
        return (x, y)
    
    def get_sensor_readings(self) -> List[float]:
        """
        Get comprehensive sensor readings simulating robot sensors.
        Returns 24 floating-point values: 4 distance + 13 vision + 2 smell + 5 internal.
        """
        # Distance sensors (ultrasonic simulation)
        distance_sensors = self._calculate_distance_sensors()
        
        # Vision features (camera simulation)
        vision_features = self._calculate_vision_features()
        
        # Smell sensors (chemical/scent detection)
        smell_sensors = self._calculate_smell_sensors()
        
        # Internal state sensors
        internal_state = self._get_robot_internal_state()
        
        return distance_sensors + vision_features + smell_sensors + internal_state
    
    def _calculate_distance_sensors(self) -> List[float]:
        """Calculate distance sensor readings in 4 directions."""
        distances = []
        x, y = self.robot.position
        max_range = 5  # Maximum sensor range in cells
        
        # Check each direction: front, left, right, back relative to robot orientation
        sensor_directions = [
            self.robot.orientation,                    # front
            (self.robot.orientation - 1) % 4,         # left
            (self.robot.orientation + 1) % 4,         # right
            (self.robot.orientation + 2) % 4          # back
        ]
        
        for direction in sensor_directions:
            dx, dy = self.DIRECTIONS[direction]
            distance = 0
            
            for step in range(1, max_range + 1):
                check_x, check_y = x + dx * step, y + dy * step
                
                # Check bounds
                if not (0 <= check_x < self.width and 0 <= check_y < self.height):
                    distance = step - 1
                    break
                
                # Check for obstacle
                if self.world[check_x, check_y] == self.WALL:
                    distance = step - 1
                    break
                
                distance = step
            
            # Normalize distance: 0.0 = adjacent obstacle, 1.0 = max range
            normalized_distance = min(distance / max_range, 1.0)
            distances.append(normalized_distance)
        
        return distances
    
    def _calculate_vision_features(self) -> List[float]:
        """Calculate vision features: 3x3 grid around robot + environmental features."""
        x, y = self.robot.position
        vision_data = []
        
        # 3x3 grid around robot (9 values)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                check_x, check_y = x + dx, y + dy
                
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    cell_value = self.world[check_x, check_y]
                else:
                    cell_value = self.WALL  # Out of bounds treated as wall
                
                vision_data.append(cell_value)
        
        # Environmental features (4 values)
        visible_range = 5
        visible_cells = []
        food_cells = []
        danger_cells = []
        
        # Gather visible cells in range
        for dx in range(-visible_range, visible_range + 1):
            for dy in range(-visible_range, visible_range + 1):
                check_x, check_y = x + dx, y + dy
                
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    cell_value = self.world[check_x, check_y]
                    visible_cells.append(cell_value)
                    
                    if cell_value == self.FOOD:
                        food_cells.append((dx, dy))
                    elif cell_value == self.DANGER:
                        danger_cells.append((dx, dy))
        
        # Nearest food direction (-1 if none visible, 0-3 for N,E,S,W)
        nearest_food_direction = -1.0
        if food_cells:
            # Find closest food
            closest_food = min(food_cells, key=lambda pos: abs(pos[0]) + abs(pos[1]))
            dx, dy = closest_food
            
            # Convert to direction
            if abs(dx) > abs(dy):
                nearest_food_direction = 1.0 if dx > 0 else 3.0  # East or West
            else:
                nearest_food_direction = 2.0 if dy > 0 else 0.0  # South or North
        
        # Nearest danger direction (-1 if none visible, 0-3 for N,E,S,W)
        nearest_danger_direction = -1.0
        if danger_cells:
            # Find closest danger
            closest_danger = min(danger_cells, key=lambda pos: abs(pos[0]) + abs(pos[1]))
            dx, dy = closest_danger
            
            # Convert to direction
            if abs(dx) > abs(dy):
                nearest_danger_direction = 1.0 if dx > 0 else 3.0  # East or West
            else:
                nearest_danger_direction = 2.0 if dy > 0 else 0.0  # South or North
        
        # Food and danger density
        total_visible = len(visible_cells)
        food_density = len(food_cells) / max(total_visible, 1)
        danger_density = len(danger_cells) / max(total_visible, 1)
        
        # Normalize direction values to 0-1 range
        nearest_food_direction = (nearest_food_direction + 1) / 4.0    # -1 to 3 -> 0 to 1
        nearest_danger_direction = (nearest_danger_direction + 1) / 4.0  # -1 to 3 -> 0 to 1
        
        vision_data.extend([
            nearest_food_direction,
            nearest_danger_direction,
            food_density,
            danger_density
        ])
        
        return vision_data
    
    def _calculate_smell_sensors(self) -> List[float]:
        """Calculate smell sensor readings from nearby plants."""
        x, y = self.robot.position
        smell_range = 3  # Plants can be smelled within 3 cells
        
        # Find all plants within smell range
        plant_scents = []
        
        for dx in range(-smell_range, smell_range + 1):
            for dy in range(-smell_range, smell_range + 1):
                check_x, check_y = x + dx, y + dy
                
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    if self.world[check_x, check_y] == self.PLANT:
                        # Calculate distance to plant
                        distance = (dx*dx + dy*dy) ** 0.5
                        if distance <= smell_range:
                            # Smell intensity decreases with distance
                            intensity = max(0.0, 1.0 - (distance / smell_range))
                            plant_scents.append((dx, dy, intensity))
        
        # Calculate smell direction and intensity
        if not plant_scents:
            return [0.0, 0.0]  # No plants nearby: no direction, no intensity
        
        # Find strongest scent
        strongest_scent = max(plant_scents, key=lambda s: s[2])
        dx, dy, max_intensity = strongest_scent
        
        # Convert direction to angle (0-1 range representing 0-360 degrees)
        import math
        if dx == 0 and dy == 0:
            direction_angle = 0.0  # Plant is right here
        else:
            # Calculate angle in radians, then normalize to 0-1
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            direction_angle = angle_deg / 360.0
        
        # Average intensity of all nearby plants
        total_intensity = sum(s[2] for s in plant_scents)
        average_intensity = total_intensity / len(plant_scents)
        
        return [direction_angle, average_intensity]
    
    def _get_robot_internal_state(self) -> List[float]:
        """Get robot's internal state sensors."""
        # Normalize time counters (max expected time between events)
        max_time = 100
        normalized_food_time = min(self.robot.time_since_food / max_time, 1.0)
        normalized_damage_time = min(self.robot.time_since_damage / max_time, 1.0)
        
        # Normalize orientation to 0-1 range
        normalized_orientation = self.robot.orientation / 3.0
        
        return [
            self.robot.health,
            self.robot.energy,
            normalized_orientation,
            normalized_food_time,
            normalized_damage_time
        ]
    
    def execute_motor_commands(self, commands: Dict[str, float]) -> bool:
        """
        Process motor commands and update robot state.
        Returns True if robot is still alive, False if terminated.
        """
        # Extract motor commands with defaults
        forward_strength = commands.get("forward_motor", 0.0)
        turn_strength = commands.get("turn_motor", 0.0)
        brake_strength = commands.get("brake_motor", 0.0)
        
        # Clamp values to valid ranges
        forward_strength = max(-1.0, min(1.0, forward_strength))
        turn_strength = max(-1.0, min(1.0, turn_strength))
        brake_strength = max(0.0, min(1.0, brake_strength))
        
        # Apply braking (reduces other motor effectiveness)
        movement_factor = 1.0 - brake_strength
        
        # Process turning (changes orientation)
        if abs(turn_strength) > 0.3:  # Threshold for turning
            if turn_strength > 0:
                self.robot.orientation = (self.robot.orientation + 1) % 4  # Turn right
            else:
                self.robot.orientation = (self.robot.orientation - 1) % 4  # Turn left
        
        # Process forward/backward movement
        elif abs(forward_strength) > 0.2:  # Threshold for movement
            if forward_strength > 0:
                self._attempt_move_forward(movement_factor)
            else:
                self._attempt_move_backward(movement_factor)
        
        # Apply energy cost for any action
        energy_cost = (abs(forward_strength) + abs(turn_strength)) * 0.01
        self.robot.energy = max(0.0, self.robot.energy - energy_cost)
        
        # Process environment effects
        self._process_environment_effects()
        
        # Update step counter
        self.step_count += 1
        
        # Check for termination
        if self.robot.health <= 0.0:
            return False  # Robot died
        
        return True  # Robot still alive
    
    def _attempt_move_forward(self, movement_factor: float):
        """Attempt to move robot forward in current orientation."""
        x, y = self.robot.position
        dx, dy = self.DIRECTIONS[self.robot.orientation]
        
        new_x, new_y = x + dx, y + dy
        
        # Check bounds
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            self._handle_collision()
            return
        
        # Check for wall collision
        if self.world[new_x, new_y] == self.WALL:
            self._handle_collision()
            return
        
        # Movement successful
        self.robot.position = (new_x, new_y)
    
    def _attempt_move_backward(self, movement_factor: float):
        """Attempt to move robot backward (opposite to current orientation)."""
        x, y = self.robot.position
        dx, dy = self.DIRECTIONS[self.robot.orientation]
        
        # Move in opposite direction
        new_x, new_y = x - dx, y - dy
        
        # Check bounds
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            self._handle_collision()
            return
        
        # Check for wall collision
        if self.world[new_x, new_y] == self.WALL:
            self._handle_collision()
            return
        
        # Movement successful
        self.robot.position = (new_x, new_y)
    
    def _handle_collision(self):
        """Handle collision with wall or boundary."""
        self.robot.health = max(0.0, self.robot.health - 0.005)  # Very light collision damage: 0.5%
        self.total_collisions += 1
    
    def _process_environment_effects(self):
        """Process effects from the current cell and general world effects."""
        x, y = self.robot.position
        cell_value = self.world[x, y]
        
        if cell_value == self.FOOD:  # Food
            self.robot.energy = min(1.0, self.robot.energy + 0.2)  # Restore energy
            self.robot.health = min(1.0, self.robot.health + 0.1)  # Minor health boost
            self.world[x, y] = self.EMPTY  # Consume the food
            self.robot.time_since_food = 0
            self.total_food_consumed += 1
            
        elif cell_value == self.DANGER:  # Danger
            self.robot.health = max(0.0, self.robot.health - 0.002)  # Very light damage: 0.2% per step
            self.robot.time_since_damage = 0
            
        elif cell_value == self.PLANT:  # Plant
            # Plants provide no direct effect - they're just for smell sensations
            # Robot can walk through plants without consuming or damaging them
            pass
        
        # Natural energy depletion
        self.robot.energy = max(0.0, self.robot.energy - 0.00002)  # Very slow energy drain (50,000 steps)
        
        # Health deterioration if no energy
        if self.robot.energy <= 0.0:
            self.robot.health = max(0.0, self.robot.health - 0.0001)  # Very slow starvation
        
        # Update time counters
        self.robot.time_since_food += 1
        self.robot.time_since_damage += 1
    
    def reset_robot(self):
        """Reset robot to safe starting position with full health/energy."""
        self.robot = Robot(position=self._find_safe_starting_position())
        self.step_count = 0
    
    def get_simulation_stats(self) -> Dict[str, any]:
        """Get current simulation statistics."""
        return {
            "step_count": self.step_count,
            "robot_health": self.robot.health,
            "robot_energy": self.robot.energy,
            "robot_position": self.robot.position,
            "robot_orientation": self.robot.orientation,
            "total_food_consumed": self.total_food_consumed,
            "total_collisions": self.total_collisions,
            "time_since_food": self.robot.time_since_food,
            "time_since_damage": self.robot.time_since_damage
        }
    
    def get_world_copy(self) -> np.ndarray:
        """Get a copy of the current world state for visualization."""
        world_copy = self.world.copy()
        # Mark robot position
        x, y = self.robot.position
        world_copy[x, y] = self.ROBOT
        return world_copy