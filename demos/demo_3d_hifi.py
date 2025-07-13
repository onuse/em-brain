#!/usr/bin/env python3
"""
High-Fidelity 3D Demo - PiCar-X Spatial Intelligence Visualization

Interactive 3D visualization with full camera control for understanding
robot spatial navigation and brain activity. Uses OpenGL for real-time
performance with clean engineering visualization style.

Features:
- Full 3D scene with room environment and obstacles
- Mouse orbit camera controls (rotate, zoom, pan)
- Real-time robot movement with brain state visualization
- Sensor beam visualization (ultrasonic cones)
- Clean geometric shapes for maximum legibility
- Professional HUD with brain metrics

Controls:
- Mouse: Orbit camera around scene
- Scroll: Zoom in/out
- Right-click drag: Pan camera
- R: Reset camera to default position
- Space: Pause simulation
- ESC: Exit demo

Run with: python3 demo_runner.py 3d_new
"""

import sys
import os
import math
import time
from typing import Tuple, List

# Add the brain/ directory to import minimal as a package
current_dir = os.path.dirname(__file__)  # demos/
brain_dir = os.path.dirname(current_dir)   # brain/
sys.path.insert(0, brain_dir)

import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

# Handle both direct execution and module import
try:
    from .picar_x_simulation.picar_x_brainstem import PiCarXBrainstem
except ImportError:
    from picar_x_simulation.picar_x_brainstem import PiCarXBrainstem


class Camera3D:
    """Professional 3D camera with orbit controls."""
    
    def __init__(self, distance=20.0, elevation=30.0, azimuth=45.0):
        """Initialize camera with orbit parameters."""
        self.distance = distance  # Distance from target
        self.elevation = elevation  # Elevation angle (degrees)
        self.azimuth = azimuth  # Azimuth angle (degrees)
        self.target = [0.0, 0.0, 0.0]  # Look-at target
        self.min_distance = 5.0
        self.max_distance = 50.0
        
        # Mouse control state
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)
        self.pan_mode = False
        
    def handle_mouse_event(self, event):
        """Handle mouse input for camera control."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.mouse_down = True
                self.pan_mode = False
            elif event.button == 3:  # Right click
                self.mouse_down = True
                self.pan_mode = True
            self.last_mouse_pos = pygame.mouse.get_pos()
            
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_down = False
            self.pan_mode = False
            
        elif event.type == pygame.MOUSEMOTION and self.mouse_down:
            mouse_pos = pygame.mouse.get_pos()
            dx = mouse_pos[0] - self.last_mouse_pos[0]
            dy = mouse_pos[1] - self.last_mouse_pos[1]
            
            if self.pan_mode:
                # Pan camera target
                sensitivity = 0.02
                self.target[0] -= dx * sensitivity
                self.target[2] += dy * sensitivity
            else:
                # Orbit camera
                sensitivity = 0.5
                self.azimuth += dx * sensitivity
                self.elevation -= dy * sensitivity
                self.elevation = max(-89, min(89, self.elevation))
                
            self.last_mouse_pos = mouse_pos
            
        elif event.type == pygame.MOUSEWHEEL:
            # Zoom with mouse wheel
            zoom_speed = 1.0
            self.distance -= event.y * zoom_speed
            self.distance = max(self.min_distance, min(self.max_distance, self.distance))
    
    def reset_view(self):
        """Reset camera to default position."""
        self.distance = 20.0
        self.elevation = 30.0
        self.azimuth = 45.0
        self.target = [0.0, 0.0, 0.0]
    
    def apply_view_matrix(self):
        """Apply camera transformation to OpenGL."""
        # Calculate camera position from orbit parameters
        elevation_rad = math.radians(self.elevation)
        azimuth_rad = math.radians(self.azimuth)
        
        camera_x = self.target[0] + self.distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        camera_y = self.target[1] + self.distance * math.sin(elevation_rad)
        camera_z = self.target[2] + self.distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        
        # Set up view matrix
        gluLookAt(
            camera_x, camera_y, camera_z,  # Camera position
            self.target[0], self.target[1], self.target[2],  # Look at target
            0.0, 1.0, 0.0  # Up vector
        )


class Scene3D:
    """3D scene renderer with clean geometric shapes."""
    
    def __init__(self, world_size=30):
        """Initialize 3D scene."""
        self.world_size = world_size
        self.setup_lighting()
        
        # Colors for clean engineering visualization
        self.colors = {
            'room_walls': (0.9, 0.9, 0.9, 0.3),      # Light gray, transparent
            'room_floor': (0.8, 0.8, 0.8, 1.0),      # Gray floor
            'grid_lines': (0.6, 0.6, 0.6, 1.0),      # Grid
            'obstacle': (0.8, 0.2, 0.2, 1.0),        # Red obstacles
            'robot_body': (0.2, 0.6, 0.9, 1.0),      # Blue robot
            'robot_direction': (1.0, 1.0, 0.2, 1.0), # Yellow direction
            'sensor_beam': (0.2, 1.0, 0.2, 0.3),     # Green sensor beam
            'trail': (0.0, 0.4, 0.8, 0.6),           # Blue trail
        }
    
    def setup_lighting(self):
        """Set up OpenGL lighting for professional appearance."""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Ambient light
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 20.0, 10.0, 1.0])
        
        # Material properties
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 32.0)
    
    def draw_room(self):
        """Draw room environment with walls and floor."""
        size = self.world_size / 2
        
        # Draw floor
        glColor4f(*self.colors['room_floor'])
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-size, 0, -size)
        glVertex3f(size, 0, -size)
        glVertex3f(size, 0, size)
        glVertex3f(-size, 0, size)
        glEnd()
        
        # Draw grid lines on floor
        glDisable(GL_LIGHTING)
        glColor4f(*self.colors['grid_lines'])
        glBegin(GL_LINES)
        
        # Grid lines every 2 units
        for i in range(-int(size), int(size) + 1, 2):
            # X direction lines
            glVertex3f(i, 0.01, -size)
            glVertex3f(i, 0.01, size)
            # Z direction lines
            glVertex3f(-size, 0.01, i)
            glVertex3f(size, 0.01, i)
        
        glEnd()
        glEnable(GL_LIGHTING)
        
        # Draw transparent walls
        wall_height = 5.0
        glColor4f(*self.colors['room_walls'])
        
        # Front wall
        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glVertex3f(-size, 0, -size)
        glVertex3f(-size, wall_height, -size)
        glVertex3f(size, wall_height, -size)
        glVertex3f(size, 0, -size)
        glEnd()
        
        # Back wall
        glBegin(GL_QUADS)
        glNormal3f(0, 0, -1)
        glVertex3f(-size, 0, size)
        glVertex3f(size, 0, size)
        glVertex3f(size, wall_height, size)
        glVertex3f(-size, wall_height, size)
        glEnd()
        
        # Left wall
        glBegin(GL_QUADS)
        glNormal3f(1, 0, 0)
        glVertex3f(-size, 0, -size)
        glVertex3f(-size, 0, size)
        glVertex3f(-size, wall_height, size)
        glVertex3f(-size, wall_height, -size)
        glEnd()
        
        # Right wall
        glBegin(GL_QUADS)
        glNormal3f(-1, 0, 0)
        glVertex3f(size, 0, -size)
        glVertex3f(size, wall_height, -size)
        glVertex3f(size, wall_height, size)
        glVertex3f(size, 0, size)
        glEnd()
    
    def draw_cylinder(self, x, z, radius, height, segments=16):
        """Draw a cylinder (for obstacles)."""
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            # Normal for cylinder sides
            glNormal3f(cos_a, 0, sin_a)
            
            # Bottom vertex
            glVertex3f(x + radius * cos_a, 0, z + radius * sin_a)
            # Top vertex
            glVertex3f(x + radius * cos_a, height, z + radius * sin_a)
        glEnd()
        
        # Top cap
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 1, 0)
        glVertex3f(x, height, z)  # Center
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            glVertex3f(x + radius * math.cos(angle), height, z + radius * math.sin(angle))
        glEnd()
    
    def draw_obstacles(self, obstacles):
        """Draw cylindrical obstacles."""
        glColor4f(*self.colors['obstacle'])
        for obs_x, obs_z, obs_size in obstacles:
            self.draw_cylinder(obs_x, obs_z, obs_size / 2, 3.0)
    
    def draw_box(self, x, y, z, width, height, depth):
        """Draw a box (for robot)."""
        # Define box vertices
        w, h, d = width/2, height/2, depth/2
        
        vertices = [
            # Front face
            (-w, -h, d), (w, -h, d), (w, h, d), (-w, h, d),
            # Back face
            (-w, -h, -d), (-w, h, -d), (w, h, -d), (w, -h, -d),
        ]
        
        faces = [
            (0, 1, 2, 3),  # Front
            (4, 5, 6, 7),  # Back
            (0, 3, 5, 4),  # Left
            (1, 7, 6, 2),  # Right
            (3, 2, 6, 5),  # Top
            (0, 4, 7, 1),  # Bottom
        ]
        
        normals = [
            (0, 0, 1),   # Front
            (0, 0, -1),  # Back
            (-1, 0, 0),  # Left
            (1, 0, 0),   # Right
            (0, 1, 0),   # Top
            (0, -1, 0),  # Bottom
        ]
        
        glPushMatrix()
        glTranslatef(x, y, z)
        
        for i, face in enumerate(faces):
            glBegin(GL_QUADS)
            glNormal3f(*normals[i])
            for vertex_index in face:
                glVertex3f(*vertices[vertex_index])
            glEnd()
        
        glPopMatrix()
    
    def draw_robot(self, robot_x, robot_y, robot_z, robot_heading):
        """Draw robot with direction indicator."""
        # Robot body
        glColor4f(*self.colors['robot_body'])
        self.draw_box(robot_x, robot_y + 0.5, robot_z, 2.0, 1.0, 3.0)
        
        # Direction indicator (arrow)
        glDisable(GL_LIGHTING)
        glColor4f(*self.colors['robot_direction'])
        
        # Calculate arrow direction
        heading_rad = math.radians(robot_heading)
        arrow_length = 2.0
        arrow_x = robot_x + arrow_length * math.cos(heading_rad)
        arrow_z = robot_z + arrow_length * math.sin(heading_rad)
        
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glVertex3f(robot_x, robot_y + 1.5, robot_z)
        glVertex3f(arrow_x, robot_y + 1.5, arrow_z)
        glEnd()
        
        # Arrow head
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glVertex3f(arrow_x, robot_y + 1.5, arrow_z)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_sensor_beam(self, robot_x, robot_y, robot_z, robot_heading, distance, max_range=20):
        """Draw ultrasonic sensor beam as a cone."""
        if distance >= max_range:
            return
        
        glDisable(GL_LIGHTING)
        glColor4f(*self.colors['sensor_beam'])
        
        # Calculate beam direction
        heading_rad = math.radians(robot_heading)
        beam_end_x = robot_x + (distance / 5.0) * math.cos(heading_rad)
        beam_end_z = robot_z + (distance / 5.0) * math.sin(heading_rad)
        
        # Draw beam cone
        beam_radius = distance / 20.0  # Cone gets wider with distance
        segments = 8
        
        glBegin(GL_TRIANGLE_FAN)
        # Cone apex (robot position)
        glVertex3f(robot_x, robot_y + 1.0, robot_z)
        
        # Cone base
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            offset_x = beam_radius * math.cos(angle)
            offset_y = beam_radius * math.sin(angle)
            glVertex3f(beam_end_x + offset_x, robot_y + 1.0 + offset_y, beam_end_z)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_trail(self, trail_points):
        """Draw robot movement trail."""
        if len(trail_points) < 2:
            return
        
        glDisable(GL_LIGHTING)
        glColor4f(*self.colors['trail'])
        glLineWidth(2.0)
        
        glBegin(GL_LINE_STRIP)
        for point in trail_points[-50:]:  # Show last 50 points
            glVertex3f(point[0], 0.1, point[1])  # Slightly above floor
        glEnd()
        
        glEnable(GL_LIGHTING)


class HighFidelity3DDemo:
    """High-fidelity 3D demonstration with full camera control."""
    
    def __init__(self, world_size=30):
        """Initialize the 3D demo."""
        self.world_size = world_size
        
        # Initialize pygame and OpenGL
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        
        pygame.display.set_mode((self.screen_width, self.screen_height), 
                              pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("PiCar-X High-Fidelity 3D Demo - Spatial Intelligence Visualization")
        
        # Set up 3D viewport
        glViewport(0, 0, self.screen_width, self.screen_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.screen_width / self.screen_height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Initialize components
        self.camera = Camera3D()
        self.scene = Scene3D(world_size)
        
        # Create robot brainstem
        self.robot = PiCarXBrainstem(
            enable_camera=True,
            enable_ultrasonics=True,
            enable_line_tracking=True
        )
        
        # Scale robot to 3D world
        scale_factor = world_size / 50.0
        self.robot.position[0] *= scale_factor
        self.robot.position[1] *= scale_factor
        
        # Environment obstacles (scaled to 3D world)
        self.obstacles = [
            (8, 8, 3),     # x, z, size
            (15, 5, 2),
            (6, 20, 2.5),
            (22, 18, 3),
            (3, 15, 2),
            (25, 7, 2.5),
            (12, 25, 4),
            (20, 3, 2)
        ]
        
        # Demo state
        self.running = True
        self.paused = False
        self.step_count = 0
        self.trail_points = []
        
        print(f"🎯 High-Fidelity 3D Demo initialized")
        print(f"   Resolution: {self.screen_width}x{self.screen_height}")
        print(f"   World size: {world_size}x{world_size}")
        print(f"   OpenGL context ready")
    
    def handle_events(self):
        """Handle pygame and camera events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.camera.reset_view()
                    print("🔄 Camera view reset")
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"⏸️ Simulation {'paused' if self.paused else 'resumed'}")
            else:
                # Pass mouse events to camera
                self.camera.handle_mouse_event(event)
    
    def update_simulation(self):
        """Update robot simulation if not paused."""
        if not self.paused:
            # Robot control cycle
            cycle_result = self.robot.control_cycle()
            self.step_count += 1
            
            # Record trail
            pos = [self.robot.position[0], self.robot.position[1]]
            self.trail_points.append(pos)
            
            # Keep trail manageable
            if len(self.trail_points) > 200:
                self.trail_points.pop(0)
            
            # Keep robot in bounds
            self.robot.position[0] = max(1, min(self.world_size - 1, self.robot.position[0]))
            self.robot.position[1] = max(1, min(self.world_size - 1, self.robot.position[1]))
            
            return cycle_result
        
        return None
    
    def render_frame(self):
        """Render one frame of the 3D scene."""
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply camera transformation
        self.camera.apply_view_matrix()
        
        # Draw scene components
        self.scene.draw_room()
        self.scene.draw_obstacles(self.obstacles)
        
        # Draw robot trail
        self.scene.draw_trail(self.trail_points)
        
        # Draw robot
        robot_x, robot_z = self.robot.position[0], self.robot.position[1]
        robot_y = 0  # Robot sits on floor
        robot_heading = self.robot.position[2]
        
        self.scene.draw_robot(robot_x, robot_y, robot_z, robot_heading)
        
        # Draw sensor beam
        if hasattr(self.robot, 'ultrasonic_distance') and self.robot.ultrasonic_distance < 50:
            self.scene.draw_sensor_beam(
                robot_x, robot_y, robot_z, robot_heading,
                self.robot.ultrasonic_distance
            )
        
        # Swap buffers
        pygame.display.flip()
    
    def run_demo(self, duration_seconds=300):
        """Run the high-fidelity 3D demo."""
        print(f"\n🚀 Starting High-Fidelity 3D Demo ({duration_seconds}s)")
        print("   Controls:")
        print("     Mouse: Orbit camera")
        print("     Scroll: Zoom")
        print("     Right-click drag: Pan")
        print("     R: Reset camera")
        print("     Space: Pause/resume")
        print("     ESC: Exit")
        print("   Observing spatial intelligence emergence...")
        
        start_time = time.time()
        clock = pygame.time.Clock()
        
        while self.running and (time.time() - start_time) < duration_seconds:
            # Handle input
            self.handle_events()
            
            if not self.running:
                break
            
            # Update simulation
            cycle_result = self.update_simulation()
            
            # Render frame
            self.render_frame()
            
            # Control frame rate (60 FPS target)
            clock.tick(60)
            
            # Progress updates
            if self.step_count % 100 == 0 and cycle_result:
                brain_state = cycle_result['brain_state']
                print(f"📍 Step {self.step_count}: "
                      f"Pos({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f}) | "
                      f"Brain: {brain_state['total_experiences']} exp | "
                      f"Confidence: {brain_state['prediction_confidence']:.3f}")
        
        # Demo complete
        pygame.quit()
        
        print(f"\n✅ High-Fidelity 3D demo complete!")
        print(f"   Duration: {time.time() - start_time:.1f}s")
        print(f"   Steps: {self.step_count}")
        print(f"   Final experiences: {self.robot.brain.total_experiences}")


def main():
    """Run the PiCar-X high-fidelity 3D demonstration."""
    
    print("🎯 PiCar-X High-Fidelity 3D Demo")
    print("   Professional spatial intelligence visualization")
    print("   Full camera control for comprehensive observation")
    
    try:
        demo = HighFidelity3DDemo(world_size=30)
        demo.run_demo(duration_seconds=300)
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        pygame.quit()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()


if __name__ == "__main__":
    main()