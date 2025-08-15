"""
3D Wireframe Renderer for PiCar-X Simulation

Provides immersive 3D visualization using PyOpenGL with wireframe graphics.
Features movable camera, sensor visualization, and 2D overlay for data.
"""

import numpy as np
import pygame
from pygame.locals import *
from typing import Dict, Any, Tuple, List, Optional

# Try to import OpenGL
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. Install with: pip install PyOpenGL PyOpenGL_accelerate")


class Camera3D:
    """
    3D Camera with multiple control modes.
    
    Supports free camera, follow camera, and fixed views.
    """
    
    def __init__(self):
        # Camera position and orientation
        self.position = np.array([5.0, 5.0, 3.0])
        self.target = np.array([2.0, 1.5, 0.0])
        self.up = np.array([0.0, 0.0, 1.0])
        
        # Spherical coordinates for orbit camera
        self.distance = 5.0
        self.yaw = -45.0      # Horizontal angle
        self.pitch = -30.0    # Vertical angle
        
        # Camera movement
        self.move_speed = 3.0
        self.rotate_speed = 90.0
        self.zoom_speed = 2.0
        
        # Camera modes
        self.mode = "free"  # "free", "follow", "top", "side"
        self.smooth_factor = 0.1
        
        # Mouse control
        self.mouse_sensitivity = 0.3
        self.last_mouse_pos = None
    
    def update_orbit_position(self):
        """Update camera position based on spherical coordinates."""
        # Convert spherical to Cartesian
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        
        # Calculate position relative to target
        x = self.distance * np.cos(pitch_rad) * np.cos(yaw_rad)
        y = self.distance * np.cos(pitch_rad) * np.sin(yaw_rad)
        z = self.distance * np.sin(pitch_rad)
        
        self.position = self.target + np.array([x, y, z])
    
    def handle_mouse(self, dx: float, dy: float, buttons: Tuple[bool, bool, bool]):
        """Handle mouse input for camera control."""
        if buttons[0]:  # Left mouse button
            # Rotate camera
            self.yaw += dx * self.mouse_sensitivity
            self.pitch = np.clip(self.pitch - dy * self.mouse_sensitivity, -89, 89)
            self.update_orbit_position()
    
    def handle_keys(self, keys, dt: float):
        """Handle keyboard input for camera movement."""
        if self.mode == "free":
            # Calculate forward and right vectors
            forward = self.target - self.position
            forward[2] = 0  # Keep movement horizontal
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, self.up)
            
            # WASD movement
            if keys[K_w]:
                self.position += forward * self.move_speed * dt
                self.target += forward * self.move_speed * dt
            if keys[K_s]:
                self.position -= forward * self.move_speed * dt
                self.target -= forward * self.move_speed * dt
            if keys[K_a]:
                self.position -= right * self.move_speed * dt
                self.target -= right * self.move_speed * dt
            if keys[K_d]:
                self.position += right * self.move_speed * dt
                self.target += right * self.move_speed * dt
            
            # QE for up/down
            if keys[K_q]:
                self.position[2] -= self.move_speed * dt
                self.target[2] -= self.move_speed * dt
            if keys[K_e]:
                self.position[2] += self.move_speed * dt
                self.target[2] += self.move_speed * dt
    
    def handle_scroll(self, direction: int):
        """Handle mouse wheel for zoom."""
        self.distance = np.clip(self.distance - direction * 0.5, 1.0, 20.0)
        self.update_orbit_position()
    
    def set_mode(self, mode: str, vehicle_pos: Optional[np.ndarray] = None):
        """Switch camera mode."""
        self.mode = mode
        
        if mode == "top":
            self.position = np.array([2.0, 1.5, 10.0])
            self.target = np.array([2.0, 1.5, 0.0])
            self.up = np.array([0.0, 1.0, 0.0])
        elif mode == "side":
            self.position = np.array([10.0, 1.5, 1.0])
            self.target = np.array([2.0, 1.5, 0.0])
            self.up = np.array([0.0, 0.0, 1.0])
        elif mode == "follow" and vehicle_pos is not None:
            self.target = np.array([vehicle_pos[0], vehicle_pos[1], 0.3])
            self.update_orbit_position()
            self.up = np.array([0.0, 0.0, 1.0])
        elif mode == "free":
            self.up = np.array([0.0, 0.0, 1.0])
    
    def follow_vehicle(self, vehicle_pos: np.ndarray, vehicle_heading: float):
        """Update camera to follow vehicle (smooth following)."""
        if self.mode == "follow":
            # Smooth target following
            target_pos = np.array([vehicle_pos[0], vehicle_pos[1], 0.3])
            self.target = self.target * (1 - self.smooth_factor) + target_pos * self.smooth_factor
            
            # Update camera position
            self.update_orbit_position()


class WireframeRenderer:
    """
    Wireframe rendering utilities for 3D visualization.
    
    Provides functions to draw wireframe primitives like boxes,
    cylinders, grids, etc.
    """
    
    @staticmethod
    def draw_line(p1: np.ndarray, p2: np.ndarray, color: Tuple[float, float, float] = (1, 1, 1)):
        """Draw a line between two points."""
        glColor3f(*color)
        glBegin(GL_LINES)
        glVertex3f(*p1)
        glVertex3f(*p2)
        glEnd()
    
    @staticmethod
    def draw_wireframe_box(center: np.ndarray, size: np.ndarray, 
                          color: Tuple[float, float, float] = (1, 1, 1)):
        """Draw a wireframe box."""
        x, y, z = center
        w, h, d = size / 2
        
        # Define box vertices
        vertices = [
            [x - w, y - h, z - d], [x + w, y - h, z - d],
            [x + w, y + h, z - d], [x - w, y + h, z - d],
            [x - w, y - h, z + d], [x + w, y - h, z + d],
            [x + w, y + h, z + d], [x - w, y + h, z + d]
        ]
        
        # Define edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        glColor3f(*color)
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3f(*vertices[vertex])
        glEnd()
    
    @staticmethod
    def draw_wireframe_cylinder(center: np.ndarray, radius: float, height: float,
                               segments: int = 12, color: Tuple[float, float, float] = (1, 1, 1)):
        """Draw a wireframe cylinder."""
        x, y, z = center
        
        # Generate circle points
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        
        glColor3f(*color)
        glBegin(GL_LINES)
        
        # Draw top and bottom circles
        for i in range(segments):
            # Current and next point
            angle1 = angles[i]
            angle2 = angles[(i + 1) % segments]
            
            # Top circle
            x1, y1 = x + radius * np.cos(angle1), y + radius * np.sin(angle1)
            x2, y2 = x + radius * np.cos(angle2), y + radius * np.sin(angle2)
            glVertex3f(x1, y1, z + height/2)
            glVertex3f(x2, y2, z + height/2)
            
            # Bottom circle
            glVertex3f(x1, y1, z - height/2)
            glVertex3f(x2, y2, z - height/2)
            
            # Vertical lines
            glVertex3f(x1, y1, z - height/2)
            glVertex3f(x1, y1, z + height/2)
        
        glEnd()
    
    @staticmethod
    def draw_wireframe_cone(center: np.ndarray, radius: float, height: float,
                           segments: int = 12, color: Tuple[float, float, float] = (1, 1, 1)):
        """Draw a wireframe cone (for sensor visualization)."""
        x, y, z = center
        
        # Generate base circle points
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        
        glColor3f(*color)
        glBegin(GL_LINES)
        
        # Draw base circle and lines to apex
        for i in range(segments):
            angle1 = angles[i]
            angle2 = angles[(i + 1) % segments]
            
            # Base circle
            x1, y1 = x + radius * np.cos(angle1), y + radius * np.sin(angle1)
            x2, y2 = x + radius * np.cos(angle2), y + radius * np.sin(angle2)
            glVertex3f(x1, y1, z)
            glVertex3f(x2, y2, z)
            
            # Lines to apex
            glVertex3f(x1, y1, z)
            glVertex3f(x, y, z + height)
        
        glEnd()
    
    @staticmethod
    def draw_grid(size: float, spacing: float, color: Tuple[float, float, float] = (0.3, 0.3, 0.3)):
        """Draw a grid on the XY plane at z=0."""
        glColor3f(*color)
        glBegin(GL_LINES)
        
        # Calculate grid range
        half_size = size / 2
        num_lines = int(size / spacing) + 1
        
        # Draw lines parallel to X axis
        for i in range(num_lines):
            y = -half_size + i * spacing
            glVertex3f(-half_size, y, 0)
            glVertex3f(half_size, y, 0)
        
        # Draw lines parallel to Y axis
        for i in range(num_lines):
            x = -half_size + i * spacing
            glVertex3f(x, -half_size, 0)
            glVertex3f(x, half_size, 0)
        
        glEnd()
    
    @staticmethod
    def draw_axes(origin: np.ndarray, length: float = 1.0):
        """Draw coordinate axes."""
        x, y, z = origin
        
        # X axis - Red
        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        glVertex3f(x, y, z)
        glVertex3f(x + length, y, z)
        glEnd()
        
        # Y axis - Green
        glColor3f(0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(x, y, z)
        glVertex3f(x, y + length, z)
        glEnd()
        
        # Z axis - Blue
        glColor3f(0, 0, 1)
        glBegin(GL_LINES)
        glVertex3f(x, y, z)
        glVertex3f(x, y, z + length)
        glEnd()


class Renderer3D:
    """
    3D Wireframe Renderer for PiCar-X Simulation
    
    Provides full 3D visualization with movable camera,
    wireframe graphics, and 2D overlay information.
    """
    
    def __init__(self, world, vehicle, window_size: Tuple[int, int] = (1024, 768)):
        """
        Initialize 3D renderer.
        
        Args:
            world: Simulation world object
            vehicle: Vehicle model object
            window_size: Window dimensions
        """
        if not OPENGL_AVAILABLE:
            raise ImportError("PyOpenGL is required for 3D visualization")
        
        self.world = world
        self.vehicle = vehicle
        self.window_size = window_size
        
        # Initialize pygame with OpenGL
        pygame.init()
        pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("PiCar-X 3D Simulation")
        
        # Initialize OpenGL
        self._init_opengl()
        
        # Camera system
        self.camera = Camera3D()
        
        # Rendering components
        self.wireframe = WireframeRenderer()
        
        # UI state
        self.show_grid = True
        self.show_sensors = True
        self.show_trail = True
        self.show_overlay = True
        self.vehicle_trail = []
        self.max_trail_points = 200
        
        # Performance tracking
        self.clock = pygame.time.Clock()
        self.fps = 0
        self.frame_count = 0
        
        # Font for 2D overlay
        pygame.font.init()
        self.font = pygame.font.Font(None, 18)
        self.font_large = pygame.font.Font(None, 24)
        
        print("🎮 3D wireframe renderer initialized")
    
    def _init_opengl(self):
        """Initialize OpenGL settings."""
        # Set up viewport
        glViewport(0, 0, *self.window_size)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.window_size[0] / self.window_size[1], 0.1, 100.0)
        
        # Enable line smoothing
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Set line width
        glLineWidth(1.5)
        
        # Set clear color
        glClearColor(0.1, 0.1, 0.1, 1.0)
    
    def update(self) -> bool:
        """
        Update renderer and handle events.
        
        Returns:
            True to continue, False to exit
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_1:
                    self.camera.set_mode("free")
                elif event.key == K_2:
                    vehicle_pos = np.array([*self.vehicle.get_position(), 0])
                    self.camera.set_mode("follow", vehicle_pos)
                elif event.key == K_3:
                    self.camera.set_mode("top")
                elif event.key == K_4:
                    self.camera.set_mode("side")
                elif event.key == K_g:
                    self.show_grid = not self.show_grid
                elif event.key == K_r:
                    self.vehicle.reset_vehicle([2.0, 1.5], 0.0)
                    self.vehicle_trail.clear()
                elif event.key == K_t:
                    self.show_trail = not self.show_trail
                elif event.key == K_o:
                    self.show_overlay = not self.show_overlay
                elif event.key == K_SPACE:
                    # Reset camera
                    self.camera = Camera3D()
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.camera.handle_scroll(1)
                elif event.button == 5:  # Scroll down
                    self.camera.handle_scroll(-1)
        
        # Handle mouse movement
        mouse_buttons = pygame.mouse.get_pressed()
        if any(mouse_buttons):
            mouse_pos = pygame.mouse.get_pos()
            if self.camera.last_mouse_pos:
                dx = mouse_pos[0] - self.camera.last_mouse_pos[0]
                dy = mouse_pos[1] - self.camera.last_mouse_pos[1]
                self.camera.handle_mouse(dx, dy, mouse_buttons)
            self.camera.last_mouse_pos = mouse_pos
        else:
            self.camera.last_mouse_pos = None
        
        # Update camera
        dt = self.clock.tick(60) / 1000.0
        keys = pygame.key.get_pressed()
        self.camera.handle_keys(keys, dt)
        
        # Follow vehicle if in follow mode
        if self.camera.mode == "follow":
            vehicle_pos = np.array([*self.vehicle.get_position(), 0])
            self.camera.follow_vehicle(vehicle_pos, self.vehicle.state.orientation)
        
        # Update trail
        self._update_trail()
        
        # Render frame
        self._render_frame()
        
        # Update FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.fps = self.clock.get_fps()
        
        return True
    
    def _update_trail(self):
        """Update vehicle trail."""
        if self.show_trail:
            current_pos = self.vehicle.get_position()
            
            # Add point if moved enough
            if (len(self.vehicle_trail) == 0 or 
                np.linalg.norm(np.array(current_pos) - np.array(self.vehicle_trail[-1][:2])) > 0.05):
                self.vehicle_trail.append([current_pos[0], current_pos[1], 0.05])
                
                # Limit trail length
                if len(self.vehicle_trail) > self.max_trail_points:
                    self.vehicle_trail.pop(0)
    
    def _render_frame(self):
        """Render complete frame."""
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up camera view
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*self.camera.position, *self.camera.target, *self.camera.up)
        
        # Render 3D scene
        self._render_3d_scene()
        
        # Render 2D overlay
        if self.show_overlay:
            self._render_2d_overlay()
        
        # Swap buffers
        pygame.display.flip()
    
    def _render_3d_scene(self):
        """Render all 3D elements."""
        # Draw coordinate axes at origin
        self.wireframe.draw_axes(np.array([0, 0, 0]), 0.5)
        
        # Draw floor grid
        if self.show_grid:
            glPushMatrix()
            glTranslatef(2.0, 1.5, 0)  # Center on room
            self.wireframe.draw_grid(6.0, 0.5)
            glPopMatrix()
        
        # Draw room walls
        self._render_room()
        
        # Draw obstacles
        self._render_obstacles()
        
        # Draw vehicle
        self._render_vehicle()
        
        # Draw sensors
        if self.show_sensors:
            self._render_sensors()
        
        # Draw trail
        if self.show_trail and len(self.vehicle_trail) > 1:
            self._render_trail()
    
    def _render_room(self):
        """Render room boundaries."""
        room_w, room_h = self.world.room_width, self.world.room_height
        wall_height = 0.5
        
        # Draw room as wireframe box (no floor/ceiling)
        glColor3f(0.7, 0.7, 0.7)
        glBegin(GL_LINES)
        
        # Floor rectangle
        glVertex3f(0, 0, 0)
        glVertex3f(room_w, 0, 0)
        glVertex3f(room_w, 0, 0)
        glVertex3f(room_w, room_h, 0)
        glVertex3f(room_w, room_h, 0)
        glVertex3f(0, room_h, 0)
        glVertex3f(0, room_h, 0)
        glVertex3f(0, 0, 0)
        
        # Top rectangle
        glVertex3f(0, 0, wall_height)
        glVertex3f(room_w, 0, wall_height)
        glVertex3f(room_w, 0, wall_height)
        glVertex3f(room_w, room_h, wall_height)
        glVertex3f(room_w, room_h, wall_height)
        glVertex3f(0, room_h, wall_height)
        glVertex3f(0, room_h, wall_height)
        glVertex3f(0, 0, wall_height)
        
        # Vertical edges
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, wall_height)
        glVertex3f(room_w, 0, 0)
        glVertex3f(room_w, 0, wall_height)
        glVertex3f(room_w, room_h, 0)
        glVertex3f(room_w, room_h, wall_height)
        glVertex3f(0, room_h, 0)
        glVertex3f(0, room_h, wall_height)
        
        glEnd()
    
    def _render_obstacles(self):
        """Render obstacles in the world."""
        for obstacle in self.world.obstacles:
            if obstacle.obstacle_type == "wall":
                continue  # Already rendered as room
            
            center = np.array([obstacle.position[0], obstacle.position[1], obstacle.height / 2])
            
            if obstacle.obstacle_type == "cylinder":
                radius = obstacle.size[0] / 2
                self.wireframe.draw_wireframe_cylinder(
                    center, radius, obstacle.height,
                    segments=12, color=(1.0, 0.6, 0.0)
                )
            else:  # box
                size = np.array([obstacle.size[0], obstacle.size[1], obstacle.height])
                self.wireframe.draw_wireframe_box(
                    center, size, color=(1.0, 0.6, 0.0)
                )
    
    def _render_vehicle(self):
        """Render the PiCar-X vehicle."""
        vehicle_pos = self.vehicle.get_position()
        vehicle_size = self.vehicle.get_bounding_box()
        vehicle_heading = self.vehicle.state.orientation
        
        # Vehicle center
        center = np.array([vehicle_pos[0], vehicle_pos[1], self.vehicle.height / 2])
        size = np.array([vehicle_size[0], vehicle_size[1], self.vehicle.height])
        
        # Draw vehicle body (with rotation)
        glPushMatrix()
        glTranslatef(vehicle_pos[0], vehicle_pos[1], 0)
        glRotatef(vehicle_heading, 0, 0, 1)
        glTranslatef(-vehicle_pos[0], -vehicle_pos[1], 0)
        
        self.wireframe.draw_wireframe_box(center, size, color=(0.0, 0.8, 1.0))
        
        # Draw wheels (4 small cylinders)
        wheel_positions = [
            [-self.vehicle.wheelbase/2, -self.vehicle.track_width/2],
            [-self.vehicle.wheelbase/2, self.vehicle.track_width/2],
            [self.vehicle.wheelbase/2, -self.vehicle.track_width/2],
            [self.vehicle.wheelbase/2, self.vehicle.track_width/2]
        ]
        
        for i, (wx, wy) in enumerate(wheel_positions):
            wheel_center = np.array([
                vehicle_pos[0] + wx,
                vehicle_pos[1] + wy,
                0.015  # Wheel radius
            ])
            
            # Apply steering to front wheels
            if i >= 2:  # Front wheels
                glPushMatrix()
                glTranslatef(wheel_center[0], wheel_center[1], wheel_center[2])
                glRotatef(self.vehicle.steering_angle, 0, 0, 1)
                glTranslatef(-wheel_center[0], -wheel_center[1], -wheel_center[2])
            
            self.wireframe.draw_wireframe_cylinder(
                wheel_center, 0.015, 0.01,
                segments=8, color=(0.5, 0.5, 0.5)
            )
            
            if i >= 2:
                glPopMatrix()
        
        # Draw direction indicator
        glBegin(GL_LINES)
        glColor3f(1, 1, 1)
        glVertex3f(vehicle_pos[0], vehicle_pos[1], self.vehicle.height)
        glVertex3f(
            vehicle_pos[0] + 0.3,
            vehicle_pos[1],
            self.vehicle.height
        )
        glEnd()
        
        glPopMatrix()
    
    def _render_sensors(self):
        """Render sensor visualizations."""
        # Ultrasonic sensor cone
        ultrasonic_pose = self.vehicle.get_ultrasonic_pose()
        sensor_pos = ultrasonic_pose["position"]
        sensor_heading = ultrasonic_pose["orientation"]
        
        # Measure distance
        distance = self.world.cast_ray(sensor_pos[:2], sensor_heading, 3.0, 30.0)
        
        # Draw sensor cone
        glPushMatrix()
        glTranslatef(sensor_pos[0], sensor_pos[1], sensor_pos[2])
        glRotatef(sensor_heading, 0, 0, 1)
        
        # Draw cone representing sensor beam
        cone_length = min(distance, 3.0)
        cone_radius = cone_length * np.tan(np.radians(15))  # 30° beam angle
        
        glRotatef(-90, 0, 1, 0)  # Point cone forward
        self.wireframe.draw_wireframe_cone(
            np.array([0, 0, 0]),
            cone_radius, cone_length,
            segments=8, color=(1.0, 0.2, 0.2)
        )
        
        glPopMatrix()
        
        # Camera frustum
        camera_pose = self.vehicle.get_camera_pose()
        cam_pos = camera_pose["position"]
        cam_heading = camera_pose["orientation"]
        cam_tilt = camera_pose["tilt"]
        fov = camera_pose["field_of_view"]
        
        # Draw camera frustum (simplified pyramid)
        glPushMatrix()
        glTranslatef(cam_pos[0], cam_pos[1], cam_pos[2])
        glRotatef(cam_heading, 0, 0, 1)
        glRotatef(cam_tilt, 0, 1, 0)
        
        # Frustum parameters
        near = 0.1
        far = 2.0
        half_fov = np.radians(fov / 2)
        
        # Calculate frustum corners
        near_height = near * np.tan(half_fov)
        far_height = far * np.tan(half_fov)
        aspect = 640.0 / 480.0  # Camera aspect ratio
        near_width = near_height * aspect
        far_width = far_height * aspect
        
        # Draw frustum lines
        glColor3f(1.0, 1.0, 0.0)
        glBegin(GL_LINES)
        
        # From camera to far corners
        glVertex3f(0, 0, 0)
        glVertex3f(far, -far_width, -far_height)
        glVertex3f(0, 0, 0)
        glVertex3f(far, far_width, -far_height)
        glVertex3f(0, 0, 0)
        glVertex3f(far, far_width, far_height)
        glVertex3f(0, 0, 0)
        glVertex3f(far, -far_width, far_height)
        
        # Far rectangle
        glVertex3f(far, -far_width, -far_height)
        glVertex3f(far, far_width, -far_height)
        glVertex3f(far, far_width, -far_height)
        glVertex3f(far, far_width, far_height)
        glVertex3f(far, far_width, far_height)
        glVertex3f(far, -far_width, far_height)
        glVertex3f(far, -far_width, far_height)
        glVertex3f(far, -far_width, -far_height)
        
        glEnd()
        glPopMatrix()
    
    def _render_trail(self):
        """Render vehicle movement trail."""
        # Draw trail as connected lines with fading
        glBegin(GL_LINE_STRIP)
        
        for i, pos in enumerate(self.vehicle_trail):
            # Fade older points
            alpha = i / len(self.vehicle_trail)
            glColor3f(0.0, alpha, alpha * 0.5)
            glVertex3f(*pos)
        
        glEnd()
    
    def _render_2d_overlay(self):
        """Render 2D overlay information."""
        # Switch to 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window_size[0], self.window_size[1], 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test for 2D
        glDisable(GL_DEPTH_TEST)
        
        # Render overlay panels
        self._render_vehicle_status()
        self._render_sensor_data()
        self._render_brain_stats()
        self._render_controls_help()
        self._render_camera_info()
        
        # Re-enable depth test
        glEnable(GL_DEPTH_TEST)
        
        # Restore matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def _render_text(self, text: str, x: int, y: int, font=None, color=(255, 255, 255)):
        """Render text at specified position."""
        if font is None:
            font = self.font
        
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
        glRasterPos2f(x, y)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                    GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    
    def _render_vehicle_status(self):
        """Render vehicle status panel."""
        x, y = 10, 10
        line_height = 20
        
        self._render_text("Vehicle Status", x, y, self.font_large)
        y += 25
        self._render_text("─" * 20, x, y)
        y += line_height
        
        vehicle_status = self.vehicle.get_status()
        
        self._render_text(f"Position: ({vehicle_status['position'][0]:.2f}, {vehicle_status['position'][1]:.2f})", x, y)
        y += line_height
        self._render_text(f"Heading: {vehicle_status['orientation']:.1f}°", x, y)
        y += line_height
        self._render_text(f"Speed: {vehicle_status['speed']:.3f} m/s", x, y)
        y += line_height
        self._render_text(f"Motors: L={vehicle_status['motor_speeds'][0]:.2f} R={vehicle_status['motor_speeds'][1]:.2f}", x, y)
        y += line_height
        self._render_text(f"Steering: {vehicle_status['steering_angle']:.1f}°", x, y)
        y += line_height
        
        if vehicle_status['collision']:
            self._render_text("⚠️ COLLISION", x, y, color=(255, 100, 100))
    
    def _render_sensor_data(self):
        """Render sensor data panel."""
        x = self.window_size[0] - 250
        y = 10
        line_height = 20
        
        self._render_text("Sensor Readings", x, y, self.font_large)
        y += 25
        self._render_text("─" * 20, x, y)
        y += line_height
        
        # Ultrasonic distance
        ultrasonic_pose = self.vehicle.get_ultrasonic_pose()
        distance = self.world.cast_ray(
            ultrasonic_pose["position"][:2],
            ultrasonic_pose["orientation"],
            3.0, 30.0
        )
        self._render_text(f"Ultrasonic: {distance:.3f}m", x, y)
        y += line_height
        
        # Camera angles
        self._render_text(f"Camera: Pan={self.vehicle.camera_pan_angle:.1f}°", x, y)
        y += line_height
        self._render_text(f"        Tilt={self.vehicle.camera_tilt_angle:.1f}°", x, y)
        y += line_height
        
        # IMU data (simulated)
        self._render_text("IMU:", x, y)
        y += line_height
        self._render_text(f"  Heading: {self.vehicle.state.orientation:.1f}°", x, y)
        y += line_height
        self._render_text(f"  Angular: {self.vehicle.state.angular_velocity:.2f} rad/s", x, y)
    
    def _render_brain_stats(self):
        """Render brain connection stats."""
        x = 10
        y = self.window_size[1] - 150
        line_height = 20
        
        self._render_text("Brain Connection", x, y, self.font_large)
        y += 25
        self._render_text("─" * 20, x, y)
        y += line_height
        
        # Get brain stats from simulation (brainstem is part of simulation, not vehicle)
        if hasattr(self, '_simulation_ref') and hasattr(self._simulation_ref, 'brainstem'):
            brain_stats = self._simulation_ref.brainstem.get_brain_connection_stats()
            status_color = (100, 255, 100) if brain_stats['connected'] else (255, 200, 100)
            self._render_text(f"Status: {brain_stats['status']}", x, y, color=status_color)
            y += line_height
            self._render_text(f"Experiences: {brain_stats['experiences']}", x, y)
            y += line_height
            self._render_text(f"Predictions: {brain_stats['predictions']}", x, y)
        else:
            self._render_text("Status: No brainstem", x, y)
    
    def _render_controls_help(self):
        """Render controls help."""
        x = self.window_size[0] - 250
        y = self.window_size[1] - 150
        line_height = 20
        
        self._render_text("Camera Controls", x, y, self.font_large)
        y += 25
        self._render_text("─" * 20, x, y)
        y += line_height
        
        controls = [
            "WASD+Mouse: Move camera",
            "1-4: Switch views",
            "R: Reset vehicle",
            "G: Toggle grid",
            "T: Toggle trail",
            "O: Toggle overlay",
            "Space: Reset camera"
        ]
        
        for control in controls:
            self._render_text(control, x, y)
            y += line_height
    
    def _render_camera_info(self):
        """Render camera information."""
        x = self.window_size[0] // 2 - 100
        y = self.window_size[1] - 30
        
        self._render_text(f"Camera: {self.camera.mode.upper()} | FPS: {self.fps:.0f}", x, y)
    
    def cleanup(self):
        """Clean up renderer resources."""
        pygame.quit()
        print("🧹 3D renderer cleanup complete")
    
    def __str__(self) -> str:
        return f"Renderer3D({self.window_size[0]}x{self.window_size[1]}, mode={self.camera.mode})"