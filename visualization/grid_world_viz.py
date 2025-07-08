"""
Grid world visualization using PyGame.
Displays the robot, environment, sensors, and navigation state in real-time.
"""

import pygame
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from simulation.grid_world import GridWorldSimulation
from simulation.brainstem_sim import GridWorldBrainstem


class GridWorldVisualizer:
    """Real-time PyGame visualization of the grid world simulation."""
    
    # Color definitions
    COLORS = {
        'background': (255, 255, 255),  # White
        'empty': (240, 240, 240),       # Light gray
        'wall': (50, 50, 50),           # Dark gray
        'food': (0, 200, 0),            # Green
        'danger': (200, 0, 0),          # Red
        'plant': (0, 150, 50),          # Dark green
        'robot': (0, 100, 255),         # Blue
        'robot_orientation': (255, 255, 255),  # White
        'sensor_ray': (255, 255, 0),    # Yellow
        'grid_lines': (200, 200, 200),  # Light gray
        'ui_background': (30, 30, 30),  # Dark gray
        'ui_text': (255, 255, 255),     # White
        'ui_good': (0, 255, 0),         # Green
        'ui_warning': (255, 255, 0),    # Yellow
        'ui_danger': (255, 0, 0),       # Red
    }
    
    def __init__(self, brainstem: GridWorldBrainstem, cell_size: int = 25, ui_width: int = 300):
        """
        Initialize the grid world visualizer.
        
        Args:
            brainstem: The brainstem simulation to visualize
            cell_size: Size of each grid cell in pixels
            ui_width: Width of the UI panel in pixels
        """
        self.brainstem = brainstem
        self.simulation = brainstem.simulation
        self.cell_size = cell_size
        self.ui_width = ui_width
        
        # Calculate window dimensions
        self.grid_width = self.simulation.width * cell_size
        self.grid_height = self.simulation.height * cell_size
        self.window_width = self.grid_width + ui_width
        self.window_height = max(self.grid_height, 600)  # Minimum height for UI
        
        # Initialize pygame
        pygame.init()
        
        # Only create display if not being used as a component (ui_width > 0)
        if ui_width > 0:
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Emergent Intelligence Robot - Grid World")
        else:
            # Component mode - screen will be set by parent
            self.screen = None
        
        # Fonts for UI
        self.font_large = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        
        # Visualization settings
        self.show_sensor_rays = True
        self.show_grid_lines = True
        self.show_vision_overlay = False
        
        # History tracking for graphs
        self.health_history = []
        self.energy_history = []
        self.step_history = []
        self.max_history_length = 200
    
    def get_cell_color(self, cell_value: float) -> Tuple[int, int, int]:
        """Get the color for a grid cell based on its value."""
        if cell_value == self.simulation.EMPTY:
            return self.COLORS['empty']
        elif cell_value == self.simulation.WALL:
            return self.COLORS['wall']
        elif cell_value == self.simulation.FOOD:
            return self.COLORS['food']
        elif cell_value == self.simulation.DANGER:
            return self.COLORS['danger']
        elif cell_value == self.simulation.PLANT:
            return self.COLORS['plant']
        elif cell_value == self.simulation.ROBOT:
            return self.COLORS['robot']
        else:
            return self.COLORS['empty']
    
    def draw_grid(self):
        """Draw the grid world environment."""
        world_state = self.simulation.get_world_copy()
        
        # Draw cells
        for x in range(self.simulation.width):
            for y in range(self.simulation.height):
                cell_value = world_state[x, y]
                color = self.get_cell_color(cell_value)
                
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw grid lines
                if self.show_grid_lines:
                    pygame.draw.rect(self.screen, self.COLORS['grid_lines'], rect, 1)
    
    def draw_robot(self):
        """Draw the robot with orientation indicator."""
        robot = self.simulation.robot
        x, y = robot.position
        
        # Robot center
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        center = (center_x, center_y)
        
        # Draw robot body
        radius = self.cell_size // 3
        pygame.draw.circle(self.screen, self.COLORS['robot'], center, radius)
        
        # Draw orientation indicator
        direction_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        dx, dy = direction_vectors[robot.orientation]
        
        indicator_length = radius - 2
        end_x = center_x + dx * indicator_length
        end_y = center_y + dy * indicator_length
        
        pygame.draw.line(
            self.screen,
            self.COLORS['robot_orientation'],
            center,
            (end_x, end_y),
            3
        )
        
        # Draw health/energy indicator around robot
        if robot.health < 1.0 or robot.energy < 1.0:
            # Health arc (red)
            if robot.health < 1.0:
                health_angle = robot.health * 180  # Half circle for health
                self._draw_arc(center, radius + 3, 0, health_angle, self.COLORS['ui_danger'], 2)
            
            # Energy arc (yellow)
            if robot.energy < 1.0:
                energy_angle = robot.energy * 180  # Half circle for energy
                self._draw_arc(center, radius + 3, 180, 180 + energy_angle, self.COLORS['ui_warning'], 2)
    
    def draw_sensor_rays(self):
        """Draw sensor rays showing distance sensor readings."""
        if not self.show_sensor_rays:
            return
        
        robot = self.simulation.robot
        x, y = robot.position
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        
        # Get sensor readings
        sensor_packet = self.brainstem.get_sensor_readings()
        distance_sensors = sensor_packet.sensor_values[:4]  # First 4 are distance sensors
        
        # Direction mappings: front, left, right, back relative to robot orientation
        sensor_directions = [
            robot.orientation,                    # front
            (robot.orientation - 1) % 4,         # left
            (robot.orientation + 1) % 4,         # right
            (robot.orientation + 2) % 4          # back
        ]
        
        direction_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        
        for i, (distance, direction) in enumerate(zip(distance_sensors, sensor_directions)):
            dx, dy = direction_vectors[direction]
            
            # Calculate ray length based on distance reading
            max_sensor_range = 5 * self.cell_size  # 5 cells max range
            ray_length = distance * max_sensor_range
            
            end_x = center_x + dx * ray_length
            end_y = center_y + dy * ray_length
            
            # Draw sensor ray
            alpha = 128 if distance > 0.5 else 255  # More opaque for close obstacles
            color = (*self.COLORS['sensor_ray'], alpha)
            
            # Create surface for alpha blending
            ray_surface = pygame.Surface((2, int(ray_length) + 1))
            ray_surface.set_alpha(alpha)
            ray_surface.fill(self.COLORS['sensor_ray'])
            
            pygame.draw.line(
                self.screen,
                self.COLORS['sensor_ray'],
                (center_x, center_y),
                (end_x, end_y),
                1
            )
    
    def draw_vision_overlay(self):
        """Draw 3x3 vision grid overlay around robot."""
        if not self.show_vision_overlay:
            return
        
        robot = self.simulation.robot
        x, y = robot.position
        
        # Draw 3x3 grid around robot
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                grid_x, grid_y = x + dx, y + dy
                
                if 0 <= grid_x < self.simulation.width and 0 <= grid_y < self.simulation.height:
                    rect = pygame.Rect(
                        grid_x * self.cell_size + 2,
                        grid_y * self.cell_size + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )
                    pygame.draw.rect(self.screen, self.COLORS['sensor_ray'], rect, 2)
    
    def draw_ui_panel(self):
        """Draw the UI panel with robot stats and controls."""
        # UI panel background
        ui_rect = pygame.Rect(self.grid_width, 0, self.ui_width, self.window_height)
        pygame.draw.rect(self.screen, self.COLORS['ui_background'], ui_rect)
        
        # Get current stats
        stats = self.brainstem.get_simulation_stats()
        sensor_packet = self.brainstem.get_sensor_readings()
        
        y_offset = 10
        
        # Title
        title_text = self.font_large.render("Robot Status", True, self.COLORS['ui_text'])
        self.screen.blit(title_text, (self.grid_width + 10, y_offset))
        y_offset += 40
        
        # Robot stats
        self._draw_ui_stat("Position", f"{stats['robot_position']}", y_offset)
        y_offset += 25
        
        self._draw_ui_stat("Orientation", self._get_direction_name(stats['robot_orientation']), y_offset)
        y_offset += 25
        
        # Health bar
        self._draw_progress_bar("Health", stats['robot_health'], y_offset, 
                               self.COLORS['ui_good'], self.COLORS['ui_danger'])
        y_offset += 40
        
        # Energy bar
        self._draw_progress_bar("Energy", stats['robot_energy'], y_offset,
                               self.COLORS['ui_warning'], self.COLORS['ui_danger'])
        y_offset += 40
        
        # Simulation stats
        self._draw_ui_stat("Step", str(stats['step_count']), y_offset)
        y_offset += 25
        
        self._draw_ui_stat("Food Consumed", str(stats['total_food_consumed']), y_offset)
        y_offset += 25
        
        self._draw_ui_stat("Collisions", str(stats['total_collisions']), y_offset)
        y_offset += 25
        
        self._draw_ui_stat("Since Food", str(stats['time_since_food']), y_offset)
        y_offset += 25
        
        self._draw_ui_stat("Since Damage", str(stats['time_since_damage']), y_offset)
        y_offset += 40
        
        # Sensor readings
        sensor_title = self.font_medium.render("Sensors", True, self.COLORS['ui_text'])
        self.screen.blit(sensor_title, (self.grid_width + 10, y_offset))
        y_offset += 30
        
        # Distance sensors
        distance_sensors = sensor_packet.sensor_values[:4]
        directions = ["Front", "Left", "Right", "Back"]
        for direction, distance in zip(directions, distance_sensors):
            self._draw_ui_stat(direction, f"{distance:.2f}", y_offset, small=True)
            y_offset += 20
        
        # Internal sensors
        y_offset += 10
        internal_sensors = sensor_packet.sensor_values[17:22]
        internal_names = ["Health", "Energy", "Orient", "Food Timer", "Damage Timer"]
        for name, value in zip(internal_names, internal_sensors):
            self._draw_ui_stat(name, f"{value:.2f}", y_offset, small=True)
            y_offset += 20
        
        # Controls info
        y_offset += 20
        controls_title = self.font_medium.render("Controls", True, self.COLORS['ui_text'])
        self.screen.blit(controls_title, (self.grid_width + 10, y_offset))
        y_offset += 25
        
        controls = [
            "SPACE: Pause/Resume",
            "S: Toggle Sensors", 
            "V: Toggle Vision",
            "G: Toggle Grid",
            "R: Reset Robot",
            "ESC: Exit"
        ]
        
        for control in controls:
            control_text = self.font_small.render(control, True, self.COLORS['ui_text'])
            self.screen.blit(control_text, (self.grid_width + 10, y_offset))
            y_offset += 18
        
        # Update history tracking
        self.health_history.append(stats['robot_health'])
        self.energy_history.append(stats['robot_energy'])
        self.step_history.append(stats['step_count'])
        
        # Trim history if too long
        if len(self.health_history) > self.max_history_length:
            self.health_history.pop(0)
            self.energy_history.pop(0)
            self.step_history.pop(0)
    
    def _draw_ui_stat(self, label: str, value: str, y: int, small: bool = False):
        """Draw a labeled statistic in the UI panel."""
        font = self.font_small if small else self.font_medium
        
        label_text = font.render(f"{label}:", True, self.COLORS['ui_text'])
        value_text = font.render(value, True, self.COLORS['ui_text'])
        
        self.screen.blit(label_text, (self.grid_width + 10, y))
        self.screen.blit(value_text, (self.grid_width + 120, y))
    
    def _draw_progress_bar(self, label: str, value: float, y: int, good_color: Tuple[int, int, int], 
                          bad_color: Tuple[int, int, int]):
        """Draw a progress bar for a value between 0 and 1."""
        bar_width = self.ui_width - 40
        bar_height = 15
        
        # Label
        label_text = self.font_medium.render(f"{label}:", True, self.COLORS['ui_text'])
        self.screen.blit(label_text, (self.grid_width + 10, y))
        
        # Background bar
        bar_rect = pygame.Rect(self.grid_width + 10, y + 20, bar_width, bar_height)
        pygame.draw.rect(self.screen, (100, 100, 100), bar_rect)
        
        # Filled portion
        fill_width = int(bar_width * value)
        fill_rect = pygame.Rect(self.grid_width + 10, y + 20, fill_width, bar_height)
        
        # Color based on value
        if value > 0.7:
            color = good_color
        elif value > 0.3:
            color = self.COLORS['ui_warning']
        else:
            color = bad_color
        
        pygame.draw.rect(self.screen, color, fill_rect)
        
        # Border
        pygame.draw.rect(self.screen, self.COLORS['ui_text'], bar_rect, 1)
        
        # Value text
        value_text = self.font_small.render(f"{value:.1%}", True, self.COLORS['ui_text'])
        self.screen.blit(value_text, (self.grid_width + bar_width - 30, y + 22))
    
    def _draw_arc(self, center: Tuple[int, int], radius: int, start_angle: float, 
                  end_angle: float, color: Tuple[int, int, int], width: int):
        """Draw an arc (pygame doesn't have a simple arc function)."""
        # This is a simplified arc drawing - for a full implementation you'd want to use pygame.gfxdraw
        import math
        
        points = []
        angle_step = 5  # degrees
        
        for angle in range(int(start_angle), int(end_angle), angle_step):
            rad = math.radians(angle)
            x = center[0] + radius * math.cos(rad)
            y = center[1] + radius * math.sin(rad)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, width)
    
    def _get_direction_name(self, orientation: int) -> str:
        """Get human-readable direction name."""
        directions = ["North", "East", "South", "West"]
        return directions[orientation]
    
    def render(self):
        """Render one frame of the visualization."""
        # Clear screen
        self.screen.fill(self.COLORS['background'])
        
        # Draw components
        self.draw_grid()
        self.draw_robot()
        self.draw_sensor_rays()
        self.draw_vision_overlay()
        self.draw_ui_panel()
        
        # Update display
        pygame.display.flip()
    
    def handle_events(self) -> Dict[str, bool]:
        """
        Handle pygame events and return control flags.
        
        Returns:
            Dictionary with control flags like 'quit', 'pause', etc.
        """
        events = {
            'quit': False,
            'pause': False,
            'reset': False,
            'step': False
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events['quit'] = True
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    events['quit'] = True
                elif event.key == pygame.K_SPACE:
                    events['pause'] = True
                elif event.key == pygame.K_r:
                    events['reset'] = True
                elif event.key == pygame.K_RETURN:
                    events['step'] = True
                elif event.key == pygame.K_s:
                    self.show_sensor_rays = not self.show_sensor_rays
                elif event.key == pygame.K_v:
                    self.show_vision_overlay = not self.show_vision_overlay
                elif event.key == pygame.K_g:
                    self.show_grid_lines = not self.show_grid_lines
        
        return events
    
    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()