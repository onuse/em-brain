# Grid World Simulation Requirements

## Overview
A PyGame-based 2D grid world that simulates a robot environment for testing the emergent intelligence brain. The simulation acts as a "virtual brainstem" providing sensory input and responding to motor commands.

## Core Simulation

### Grid Environment
- **Size**: 20x20 grid (scalable)
- **Cell Types**:
  - `Empty` (0.0) - Safe to traverse
  - `Wall` (1.0) - Blocks movement, causes collision
  - `Food` (0.5) - Beneficial, disappears when consumed
  - `Danger` (-1.0) - Harmful, causes damage
  - `Robot` (0.8) - Current robot position

### Robot Entity
- **Position**: Integer grid coordinates (x, y)
- **Orientation**: North, East, South, West (0, 1, 2, 3)
- **Health**: 0.0 to 1.0 (starts at 1.0)
- **Energy**: 0.0 to 1.0 (starts at 1.0, depletes over time)

## Sensor System (Simulating PiCar-X sensors)

### Distance Sensors (Ultrasonic simulation)
```python
sensor_readings = [
    front_distance,    # Distance to obstacle in front (0.0 to 1.0)
    left_distance,     # Distance to obstacle on left
    right_distance,    # Distance to obstacle on right
    back_distance      # Distance to obstacle behind
]
```
- Distance normalized: 0.0 = adjacent obstacle, 1.0 = max sensor range (5 cells)

### Vision Sensor (Camera simulation)
```python
vision_features = [
    # 3x3 grid around robot, flattened (9 values)
    cell_nw, cell_n, cell_ne,
    cell_w,  cell_c, cell_e,
    cell_sw, cell_s, cell_se,
    
    # Environmental features (4 values)
    nearest_food_direction,    # 0-3 for N,E,S,W or -1 if none visible
    nearest_danger_direction,  # 0-3 for N,E,S,W or -1 if none visible
    food_density,             # Fraction of visible cells containing food
    danger_density            # Fraction of visible cells containing danger
]
```

### Internal Sensors (Robot state)
```python
internal_state = [
    current_health,    # 0.0 to 1.0
    current_energy,    # 0.0 to 1.0
    orientation,       # 0-3 normalized to 0.0-1.0
    time_since_food,   # Normalized time since last food consumption
    time_since_damage  # Normalized time since last damage taken
]
```

**Total Sensor Vector**: 4 + 13 + 5 = 22 floating-point values

## Motor System (Simulating PiCar-X actuators)

### Movement Motors
```python
motor_commands = {
    "forward_motor": value,    # -1.0 to 1.0 (negative = backward)
    "turn_motor": value,       # -1.0 to 1.0 (negative = left, positive = right)
    "brake_motor": value       # 0.0 to 1.0 (strength of braking)
}
```

### Motor Response Logic
```python
def process_motor_commands(robot, commands):
    # Movement processing
    forward_strength = commands.get("forward_motor", 0.0)
    turn_strength = commands.get("turn_motor", 0.0)
    brake_strength = commands.get("brake_motor", 0.0)
    
    # Apply braking (reduces other motor effectiveness)
    movement_factor = 1.0 - brake_strength
    
    # Turning (changes orientation)
    if abs(turn_strength) > 0.3:  # Threshold for turning
        if turn_strength > 0:
            robot.orientation = (robot.orientation + 1) % 4  # Turn right
        else:
            robot.orientation = (robot.orientation - 1) % 4  # Turn left
    
    # Forward/backward movement
    elif abs(forward_strength) > 0.2:  # Threshold for movement
        if forward_strength > 0:
            attempt_move_forward(robot, movement_factor)
        else:
            attempt_move_backward(robot, movement_factor)
    
    # Energy cost for any action
    energy_cost = (abs(forward_strength) + abs(turn_strength)) * 0.01
    robot.energy = max(0.0, robot.energy - energy_cost)
```

## Environment Dynamics

### World Generation
```python
def generate_world(width=20, height=20):
    world = np.zeros((width, height))
    
    # Add walls around perimeter
    world[0, :] = 1.0    # Top wall
    world[-1, :] = 1.0   # Bottom wall
    world[:, 0] = 1.0    # Left wall
    world[:, -1] = 1.0   # Right wall
    
    # Add random internal obstacles (10-15% coverage)
    for _ in range(random.randint(20, 40)):
        x, y = random.randint(1, width-2), random.randint(1, height-2)
        world[x, y] = 1.0
    
    # Add food sources (5-8% coverage)
    for _ in range(random.randint(15, 25)):
        x, y = random.randint(1, width-2), random.randint(1, height-2)
        if world[x, y] == 0.0:  # Only place on empty cells
            world[x, y] = 0.5
    
    # Add danger zones (2-3% coverage)
    for _ in range(random.randint(5, 10)):
        x, y = random.randint(1, width-2), random.randint(1, height-2)
        if world[x, y] == 0.0:  # Only place on empty cells
            world[x, y] = -1.0
    
    return world
```

### Environmental Effects
```python
def process_environment_effects(robot, world):
    x, y = robot.position
    cell_value = world[x, y]
    
    if cell_value == 0.5:  # Food
        robot.energy = min(1.0, robot.energy + 0.2)  # Restore energy
        robot.health = min(1.0, robot.health + 0.1)  # Minor health boost
        world[x, y] = 0.0  # Consume the food
        robot.time_since_food = 0
        
    elif cell_value == -1.0:  # Danger
        robot.health = max(0.0, robot.health - 0.1)  # Take damage
        robot.time_since_damage = 0
        
    elif cell_value == 1.0:  # Wall collision
        robot.health = max(0.0, robot.health - 0.05)  # Minor collision damage
        # Don't actually move into wall - previous position maintained
    
    # Natural energy depletion
    robot.energy = max(0.0, robot.energy - 0.001)  # Slow energy drain
    
    # Health deterioration if no energy
    if robot.energy <= 0.0:
        robot.health = max(0.0, robot.health - 0.005)
    
    # Update time counters
    robot.time_since_food += 1
    robot.time_since_damage += 1
```

## PyGame Visualization

### Visual Elements
- **Grid cells**: Color-coded rectangles
  - Gray: Empty space
  - Black: Walls
  - Green: Food
  - Red: Danger zones
- **Robot**: Blue circle with orientation indicator
- **UI Elements**:
  - Health bar (top-left)
  - Energy bar (top-left)
  - Step counter
  - Current sensor readings (text overlay)

### Real-time Display
```python
def render_world(screen, world, robot, sensor_data):
    # Clear screen
    screen.fill((255, 255, 255))
    
    # Draw grid
    cell_size = 30
    for x in range(world.shape[0]):
        for y in range(world.shape[1]):
            cell_value = world[x, y]
            color = get_cell_color(cell_value)
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # Grid lines
    
    # Draw robot
    robot_x, robot_y = robot.position
    center = (robot_x * cell_size + cell_size//2, robot_y * cell_size + cell_size//2)
    pygame.draw.circle(screen, (0, 0, 255), center, cell_size//3)
    
    # Draw orientation indicator
    orientation_offset = [(0, -10), (10, 0), (0, 10), (-10, 0)][robot.orientation]
    end_point = (center[0] + orientation_offset[0], center[1] + orientation_offset[1])
    pygame.draw.line(screen, (255, 255, 255), center, end_point, 3)
    
    # Draw UI
    draw_health_bar(screen, robot.health)
    draw_energy_bar(screen, robot.energy)
    draw_sensor_data(screen, sensor_data)
```

## Communication Interface (Brainstem Simulation)

### Brain-Simulation Protocol
```python
class GridWorldBrainstem:
    def __init__(self):
        self.world = generate_world()
        self.robot = Robot(position=find_safe_starting_position(self.world))
        self.step_count = 0
    
    def get_sensor_readings(self):
        """Return sensor data in same format as real brainstem"""
        distance_sensors = calculate_distance_sensors(self.robot, self.world)
        vision_features = calculate_vision_features(self.robot, self.world)
        internal_state = get_robot_internal_state(self.robot)
        
        return {
            'sensor_values': distance_sensors + vision_features + internal_state,
            'actuator_positions': [0.0, 0.0, 0.0],  # Not meaningful in grid world
            'timestamp': time.time(),
            'sequence_id': self.step_count
        }
    
    def execute_motor_commands(self, commands):
        """Process motor commands and update robot state"""
        process_motor_commands(self.robot, commands)
        process_environment_effects(self.robot, self.world)
        self.step_count += 1
        
        # Check for termination conditions
        if self.robot.health <= 0.0:
            print("Robot died!")
            self.reset_robot()
    
    def reset_robot(self):
        """Reset robot to safe starting position with full health/energy"""
        self.robot = Robot(position=find_safe_starting_position(self.world))
        
    def get_hardware_capabilities(self):
        """Mimic real brainstem hardware discovery"""
        return {
            'sensors': [
                {'id': 'distance_sensors', 'type': 'distance', 'data_size': 4},
                {'id': 'vision_features', 'type': 'camera_features', 'data_size': 13},
                {'id': 'internal_state', 'type': 'internal', 'data_size': 5}
            ],
            'actuators': [
                {'id': 'forward_motor', 'type': 'motor', 'range': [-1.0, 1.0]},
                {'id': 'turn_motor', 'type': 'motor', 'range': [-1.0, 1.0]},
                {'id': 'brake_motor', 'type': 'motor', 'range': [0.0, 1.0]}
            ]
        }
```

## Learning Objectives

The robot should naturally discover:
1. **Basic navigation** - Moving without hitting walls
2. **Food seeking** - Approaching green food sources
3. **Danger avoidance** - Staying away from red danger zones
4. **Energy management** - Balancing exploration with conservation
5. **Spatial memory** - Remembering where food/danger locations are

## Success Metrics

- **Survival time**: How long robot stays alive
- **Food collection**: Number of food sources consumed
- **Collision rate**: Frequency of wall collisions (should decrease)
- **Exploration coverage**: Percentage of world explored
- **Behavioral complexity**: Diversity of motor command patterns

## Implementation Notes

- **Simulation speed**: Should run faster than real-time for rapid learning
- **Deterministic**: Same random seed should produce identical results
- **Configurable**: Easy to adjust world size, food/danger density, etc.
- **Logging**: Track all robot actions and outcomes for analysis
- **Multiple worlds**: Ability to test on different generated environments

This simulation provides a rich but manageable environment for testing whether the brain develops intelligent behaviors through pure experience-based learning.