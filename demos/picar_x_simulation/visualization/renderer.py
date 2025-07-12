"""
Simulation Renderer

3D visualization system for the PiCar-X simulation environment.
Provides real-time rendering of the world, vehicle, and sensor data.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class SimulationRenderer:
    """
    3D Simulation Renderer
    
    Provides real-time visualization of:
    - 3D world environment
    - Vehicle position and orientation
    - Sensor data visualization
    - Debug information overlay
    """
    
    def __init__(self, world, vehicle, window_size: Tuple[int, int] = (800, 600)):
        """
        Initialize simulation renderer.
        
        Args:
            world: Simulation world object
            vehicle: Vehicle model object
            window_size: Window dimensions (width, height)
        """
        self.world = world
        self.vehicle = vehicle
        self.window_size = window_size
        
        # Rendering state
        self.initialized = False
        self.running = True
        
        # Camera settings
        self.camera_position = [2.0, 1.5, 2.0]  # Above and behind vehicle
        self.camera_target = [2.0, 1.5, 0.0]    # Looking at center of room
        self.camera_up = [0.0, 0.0, 1.0]        # Z-up coordinate system
        
        # Rendering options
        self.show_debug_info = True
        self.show_sensor_rays = True
        self.show_vehicle_trail = True
        self.vehicle_trail = []
        self.max_trail_length = 100
        
        # Try to initialize pygame
        try:
            import pygame
            self.pygame = pygame
            self._initialize_pygame()
            print("🎮 3D renderer initialized with pygame")
        except ImportError:
            print("⚠️  pygame not available - renderer disabled")
            raise ImportError("pygame required for visualization")
    
    def _initialize_pygame(self):
        """Initialize pygame for 2D visualization (placeholder for 3D)."""
        self.pygame.init()
        self.screen = self.pygame.display.set_mode(self.window_size)
        self.pygame.display.set_caption("PiCar-X Simulation")
        self.clock = self.pygame.time.Clock()
        self.font = self.pygame.font.Font(None, 24)
        
        # Colors
        self.colors = {
            "background": (30, 30, 30),
            "floor": (60, 60, 60),
            "walls": (100, 100, 100),
            "obstacles": (150, 100, 50),
            "vehicle": (100, 150, 255),
            "sensor_ray": (255, 100, 100),
            "trail": (100, 255, 100),
            "text": (255, 255, 255)
        }
        
        # Rendering scale (pixels per meter)
        self.scale = 150  # 150 pixels = 1 meter
        
        self.initialized = True
    
    def update(self) -> bool:
        """
        Update renderer and handle events.
        
        Returns:
            True to continue simulation, False to exit
        """
        if not self.initialized:
            return True
        
        # Handle pygame events
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE:
                    return False
                elif event.key == self.pygame.K_r:
                    # Reset vehicle
                    self.vehicle.reset_vehicle([2.0, 1.5], 0.0)
                    self.vehicle_trail.clear()
                elif event.key == self.pygame.K_d:
                    # Toggle debug info
                    self.show_debug_info = not self.show_debug_info
                elif event.key == self.pygame.K_s:
                    # Toggle sensor rays
                    self.show_sensor_rays = not self.show_sensor_rays
                elif event.key == self.pygame.K_t:
                    # Toggle vehicle trail
                    self.show_vehicle_trail = not self.show_vehicle_trail
        
        # Render the scene
        self._render_scene()
        
        # Update display
        self.pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
        
        return True
    
    def _render_scene(self):
        """Render the complete simulation scene."""
        # Clear screen
        self.screen.fill(self.colors["background"])
        
        # Render world elements
        self._render_world()
        self._render_obstacles()
        self._render_vehicle()
        
        # Render sensor visualization
        if self.show_sensor_rays:
            self._render_sensor_rays()
        
        # Render vehicle trail
        if self.show_vehicle_trail:
            self._render_vehicle_trail()
        
        # Render debug information
        if self.show_debug_info:
            self._render_debug_info()
        
        # Render controls help
        self._render_controls_help()
    
    def _render_world(self):
        """Render world boundaries and floor."""
        # Convert world coordinates to screen coordinates
        world_bounds = self.world.get_world_bounds()
        
        # Floor (room area)
        floor_rect = self.pygame.Rect(
            world_bounds[0] * self.scale,
            world_bounds[1] * self.scale,
            (world_bounds[2] - world_bounds[0]) * self.scale,
            (world_bounds[3] - world_bounds[1]) * self.scale
        )
        self.pygame.draw.rect(self.screen, self.colors["floor"], floor_rect)
        
        # Room outline
        self.pygame.draw.rect(self.screen, self.colors["walls"], floor_rect, 3)
    
    def _render_obstacles(self):
        """Render obstacles in the world."""
        for obstacle in self.world.obstacles:
            if obstacle.obstacle_type != "wall":  # Walls already drawn as room outline
                # Convert position and size to screen coordinates
                screen_x = obstacle.position[0] * self.scale - (obstacle.size[0] * self.scale) // 2
                screen_y = obstacle.position[1] * self.scale - (obstacle.size[1] * self.scale) // 2
                screen_w = obstacle.size[0] * self.scale
                screen_h = obstacle.size[1] * self.scale
                
                obstacle_rect = self.pygame.Rect(screen_x, screen_y, screen_w, screen_h)
                
                if obstacle.obstacle_type == "cylinder":
                    # Draw cylinder as circle
                    radius = int(screen_w // 2)
                    center = (int(screen_x + radius), int(screen_y + radius))
                    self.pygame.draw.circle(self.screen, self.colors["obstacles"], center, radius)
                else:
                    # Draw box
                    self.pygame.draw.rect(self.screen, self.colors["obstacles"], obstacle_rect)
                    self.pygame.draw.rect(self.screen, self.colors["walls"], obstacle_rect, 2)
    
    def _render_vehicle(self):
        """Render the PiCar-X vehicle."""
        vehicle_pos = self.vehicle.get_position()
        vehicle_size = self.vehicle.get_bounding_box()
        vehicle_heading = self.vehicle.state.orientation
        
        # Add to trail
        if len(self.vehicle_trail) == 0 or \
           np.linalg.norm(np.array(vehicle_pos) - np.array(self.vehicle_trail[-1])) > 0.05:
            self.vehicle_trail.append(list(vehicle_pos))
            if len(self.vehicle_trail) > self.max_trail_length:
                self.vehicle_trail.pop(0)
        
        # Convert to screen coordinates
        screen_x = vehicle_pos[0] * self.scale
        screen_y = vehicle_pos[1] * self.scale
        screen_w = vehicle_size[0] * self.scale
        screen_h = vehicle_size[1] * self.scale
        
        # Create vehicle rectangle
        vehicle_rect = self.pygame.Rect(
            screen_x - screen_w // 2,
            screen_y - screen_h // 2,
            screen_w,
            screen_h
        )
        
        # Draw vehicle body
        self.pygame.draw.rect(self.screen, self.colors["vehicle"], vehicle_rect)
        self.pygame.draw.rect(self.screen, self.colors["walls"], vehicle_rect, 2)
        
        # Draw heading indicator
        heading_length = 25
        heading_end_x = screen_x + heading_length * np.cos(np.radians(vehicle_heading))
        heading_end_y = screen_y + heading_length * np.sin(np.radians(vehicle_heading))
        
        self.pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (int(screen_x), int(screen_y)),
            (int(heading_end_x), int(heading_end_y)),
            3
        )
        
        # Draw camera direction
        camera_pose = self.vehicle.get_camera_pose()
        camera_angle = camera_pose["orientation"]
        camera_length = 20
        camera_end_x = screen_x + camera_length * np.cos(np.radians(camera_angle))
        camera_end_y = screen_y + camera_length * np.sin(np.radians(camera_angle))
        
        self.pygame.draw.line(
            self.screen,
            (255, 255, 0),
            (int(screen_x), int(screen_y)),
            (int(camera_end_x), int(camera_end_y)),
            2
        )
    
    def _render_sensor_rays(self):
        """Render ultrasonic sensor rays."""
        ultrasonic_pose = self.vehicle.get_ultrasonic_pose()
        
        # Simulate sensor measurement
        if hasattr(self.world, 'cast_ray'):
            sensor_pos = ultrasonic_pose["position"][:2]
            sensor_angle = ultrasonic_pose["orientation"]
            max_range = ultrasonic_pose["max_range"]
            
            distance = self.world.cast_ray(sensor_pos, sensor_angle, max_range, 30.0)
            
            # Draw sensor ray
            screen_start_x = sensor_pos[0] * self.scale
            screen_start_y = sensor_pos[1] * self.scale
            
            ray_end_x = sensor_pos[0] + distance * np.cos(np.radians(sensor_angle))
            ray_end_y = sensor_pos[1] + distance * np.sin(np.radians(sensor_angle))
            screen_end_x = ray_end_x * self.scale
            screen_end_y = ray_end_y * self.scale
            
            self.pygame.draw.line(
                self.screen,
                self.colors["sensor_ray"],
                (int(screen_start_x), int(screen_start_y)),
                (int(screen_end_x), int(screen_end_y)),
                2
            )
            
            # Draw detection point
            if distance < max_range:
                self.pygame.draw.circle(
                    self.screen,
                    self.colors["sensor_ray"],
                    (int(screen_end_x), int(screen_end_y)),
                    5
                )
    
    def _render_vehicle_trail(self):
        """Render vehicle movement trail."""
        if len(self.vehicle_trail) < 2:
            return
        
        # Draw trail as connected line segments
        screen_points = []
        for pos in self.vehicle_trail:
            screen_x = int(pos[0] * self.scale)
            screen_y = int(pos[1] * self.scale)
            screen_points.append((screen_x, screen_y))
        
        if len(screen_points) >= 2:
            self.pygame.draw.lines(
                self.screen,
                self.colors["trail"],
                False,
                screen_points,
                2
            )
    
    def _render_debug_info(self):
        """Render debug information overlay."""
        debug_lines = []
        
        # Vehicle status
        vehicle_status = self.vehicle.get_status()
        debug_lines.append(f"Position: ({vehicle_status['position'][0]:.2f}, {vehicle_status['position'][1]:.2f})")
        debug_lines.append(f"Heading: {vehicle_status['orientation']:.1f}°")
        debug_lines.append(f"Speed: {vehicle_status['speed']:.2f} m/s")
        debug_lines.append(f"Motors: L={vehicle_status['motor_speeds'][0]:.2f} R={vehicle_status['motor_speeds'][1]:.2f}")
        debug_lines.append(f"Steering: {vehicle_status['steering_angle']:.1f}°")
        debug_lines.append(f"Camera: Pan={vehicle_status['camera_angles'][0]:.1f}° Tilt={vehicle_status['camera_angles'][1]:.1f}°")
        debug_lines.append(f"Distance: {vehicle_status['total_distance']:.2f}m")
        debug_lines.append(f"Time: {vehicle_status['simulation_time']:.1f}s")
        
        # Collision status
        if vehicle_status['collision']:
            debug_lines.append("⚠️ COLLISION")
        
        # World statistics
        world_stats = self.world.get_statistics()
        debug_lines.append(f"World: {world_stats['total_obstacles']} obstacles")
        debug_lines.append(f"Updates: {world_stats['total_updates']}")
        
        # Render debug text
        y_offset = 10
        for line in debug_lines:
            text_surface = self.font.render(line, True, self.colors["text"])
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25
    
    def _render_controls_help(self):
        """Render controls help at bottom of screen."""
        help_lines = [
            "Controls: R=Reset, D=Debug, S=Sensors, T=Trail, ESC=Exit"
        ]
        
        y_offset = self.window_size[1] - 30
        for line in help_lines:
            text_surface = self.font.render(line, True, self.colors["text"])
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 20
    
    def cleanup(self):
        """Clean up renderer resources."""
        if self.initialized and hasattr(self, 'pygame'):
            self.pygame.quit()
            print("🧹 Renderer cleanup complete")
    
    def __str__(self) -> str:
        return f"SimulationRenderer({self.window_size[0]}x{self.window_size[1]}, initialized={self.initialized})"