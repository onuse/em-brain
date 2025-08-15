#!/usr/bin/env python3
"""
PiCar-X Wireframe Demo

A Battlezone-inspired wireframe visualization using pygame instead of OpenGL.
Creates the same authentic 1980s vector graphics aesthetic with simpler technology.

Features:
- First-person robot perspective
- Green wireframe graphics on black background  
- Classic geometric obstacles
- Real-time brain learning visualization
- No OpenGL dependencies - works everywhere
"""

import sys
import os

# Add the brain/ directory to import minimal as a package
current_dir = os.path.dirname(__file__)  # picar_x/
demos_dir = os.path.dirname(current_dir)  # demos/
minimal_dir = os.path.dirname(demos_dir)  # minimal/
brain_dir = os.path.dirname(minimal_dir)   # brain/
sys.path.insert(0, brain_dir)

import pygame
import numpy as np
import math
import time

# Handle both direct execution and module import
try:
    from .picar_x_brainstem import PiCarXBrainstem
except ImportError:
    from picar_x_brainstem import PiCarXBrainstem


class WireframeRenderer:
    """
    Battlezone-style wireframe renderer using pygame 2D graphics.
    Creates authentic 1980s vector graphics with 3D perspective projection.
    """
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PiCar-X Wireframe Demo - Battlezone Style")
        self.clock = pygame.time.Clock()
        
        # Battlezone colors (classic green on black)
        self.bg_color = (0, 0, 0)           # Black background
        self.line_color = (0, 255, 0)       # Bright green lines
        self.horizon_color = (0, 150, 0)    # Darker green horizon
        self.sensor_color = (255, 255, 0)   # Yellow for sensors
        self.hud_color = (0, 255, 0)        # Green HUD elements
        
        # 3D projection parameters
        self.fov = 60.0  # Field of view in degrees
        self.camera_height = 1.5  # Robot camera height
        self.view_distance = 50.0  # Maximum view distance
        
        # Perspective projection matrix
        self.setup_projection()
        
        print("🎮 Wireframe renderer initialized (pygame)")
        print(f"   Resolution: {width}x{height}")
        print(f"   Camera height: {self.camera_height}m")
        print(f"   Field of view: {self.fov}°")
    
    def setup_projection(self):
        """Setup 3D to 2D perspective projection."""
        self.focal_length = self.width / (2 * math.tan(math.radians(self.fov / 2)))
        self.center_x = self.width // 2
        self.center_y = self.height // 2
    
    def project_3d_to_2d(self, world_x, world_y, world_z, camera_x, camera_y, camera_z, camera_heading):
        """Project 3D world coordinates to 2D screen coordinates."""
        
        # Translate to camera space
        dx = world_x - camera_x
        dy = world_y - camera_y  
        dz = world_z - camera_z
        
        # Rotate around camera heading
        heading_rad = math.radians(camera_heading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        
        # Camera space coordinates (z forward, x right, y up)
        cam_x = dx * cos_h + dz * sin_h
        cam_y = dy
        cam_z = -dx * sin_h + dz * cos_h
        
        # Perspective projection
        if cam_z <= 0.1:  # Behind camera or too close
            return None
        
        screen_x = self.center_x + (cam_x * self.focal_length) / cam_z
        screen_y = self.center_y - (cam_y * self.focal_length) / cam_z
        
        # Check if point is within screen bounds (with margin)
        if -100 <= screen_x <= self.width + 100 and -100 <= screen_y <= self.height + 100:
            return (int(screen_x), int(screen_y))
        return None
    
    def clear_screen(self):
        """Clear screen with classic black background."""
        self.screen.fill(self.bg_color)
    
    def draw_horizon_grid(self, camera_pos, camera_heading):
        """Draw classic Battlezone horizon grid."""
        
        camera_x, camera_y, camera_z = camera_pos
        
        # Horizon line (at ground level)
        horizon_points = []
        for x in range(-50, 51, 10):
            point = self.project_3d_to_2d(x, 0, 50, camera_x, camera_y, camera_z, camera_heading)
            if point:
                horizon_points.append(point)
        
        if len(horizon_points) >= 2:
            pygame.draw.lines(self.screen, self.horizon_color, False, horizon_points, 1)
        
        # Ground grid lines
        for i in range(-5, 6):
            # Parallel lines (extending forward)
            line_points = []
            for z in range(0, 50, 5):
                point = self.project_3d_to_2d(i * 5, 0, z, camera_x, camera_y, camera_z, camera_heading)
                if point:
                    line_points.append(point)
            
            if len(line_points) >= 2:
                pygame.draw.lines(self.screen, self.horizon_color, False, line_points, 1)
            
            # Perpendicular lines (every 10 units)
            if i % 2 == 0 and i != 0:
                perp_points = []
                for x in range(-25, 26, 5):
                    point = self.project_3d_to_2d(x, 0, i * 5, camera_x, camera_y, camera_z, camera_heading)
                    if point:
                        perp_points.append(point)
                
                if len(perp_points) >= 2:
                    pygame.draw.lines(self.screen, self.horizon_color, False, perp_points, 1)
    
    def draw_wireframe_cube(self, center, size, camera_pos, camera_heading):
        """Draw a wireframe cube (classic obstacle)."""
        
        x, y, z = center
        s = size / 2
        camera_x, camera_y, camera_z = camera_pos
        
        # Define cube vertices
        vertices_3d = [
            (x-s, y, z-s), (x+s, y, z-s), (x+s, y+size, z-s), (x-s, y+size, z-s),  # Front face
            (x-s, y, z+s), (x+s, y, z+s), (x+s, y+size, z+s), (x-s, y+size, z+s)   # Back face
        ]
        
        # Project to 2D
        vertices_2d = []
        for vx, vy, vz in vertices_3d:
            point = self.project_3d_to_2d(vx, vy, vz, camera_x, camera_y, camera_z, camera_heading)
            vertices_2d.append(point)
        
        # Draw cube edges
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # Front face
            (4,5), (5,6), (6,7), (7,4),  # Back face
            (0,4), (1,5), (2,6), (3,7)   # Connecting edges
        ]
        
        for edge in edges:
            p1, p2 = vertices_2d[edge[0]], vertices_2d[edge[1]]
            if p1 and p2:
                pygame.draw.line(self.screen, self.line_color, p1, p2, 2)
    
    def draw_pyramid(self, center, base_size, height, camera_pos, camera_heading):
        """Draw a wireframe pyramid."""
        
        x, y, z = center
        s = base_size / 2
        camera_x, camera_y, camera_z = camera_pos
        
        # Define pyramid vertices
        vertices_3d = [
            (x-s, y, z-s), (x+s, y, z-s), (x+s, y, z+s), (x-s, y, z+s),  # Base
            (x, y + height, z)  # Apex
        ]
        
        # Project to 2D
        vertices_2d = []
        for vx, vy, vz in vertices_3d:
            point = self.project_3d_to_2d(vx, vy, vz, camera_x, camera_y, camera_z, camera_heading)
            vertices_2d.append(point)
        
        # Draw pyramid edges
        # Base edges
        for i in range(4):
            p1, p2 = vertices_2d[i], vertices_2d[(i + 1) % 4]
            if p1 and p2:
                pygame.draw.line(self.screen, self.line_color, p1, p2, 2)
        
        # Edges to apex
        apex = vertices_2d[4]
        if apex:
            for i in range(4):
                base_point = vertices_2d[i]
                if base_point:
                    pygame.draw.line(self.screen, self.line_color, base_point, apex, 2)
    
    def draw_sensor_beam(self, robot_pos, robot_heading, distance, max_distance=20):
        """Draw ultrasonic sensor beam."""
        
        if distance >= max_distance:
            return
        
        camera_x, camera_z = robot_pos[0], robot_pos[1]
        camera_y = self.camera_height
        
        # Calculate beam end point
        heading_rad = math.radians(robot_heading)
        beam_length = distance / 5.0  # Scale distance
        
        end_x = camera_x + beam_length * math.cos(heading_rad)
        end_z = camera_z + beam_length * math.sin(heading_rad)
        
        # Project beam to screen
        start_point = self.project_3d_to_2d(camera_x, camera_y, camera_z, camera_x, camera_y, camera_z, robot_heading)
        end_point = self.project_3d_to_2d(end_x, camera_y, end_z, camera_x, camera_y, camera_z, robot_heading)
        
        if start_point and end_point:
            pygame.draw.line(self.screen, self.sensor_color, start_point, end_point, 3)
            # Draw detection point
            pygame.draw.circle(self.screen, self.sensor_color, end_point, 5)
    
    def draw_crosshairs(self):
        """Draw classic Battlezone crosshairs in screen center."""
        
        cx, cy = self.center_x, self.center_y
        size = 20
        
        # Main crosshairs
        pygame.draw.line(self.screen, self.hud_color, (cx - size, cy), (cx + size, cy), 2)
        pygame.draw.line(self.screen, self.hud_color, (cx, cy - size), (cx, cy + size), 2)
        
        # Corner brackets (classic Battlezone style)
        bracket_size = 8
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            bx, by = cx + dx * (size + 8), cy + dy * (size + 8)
            
            # Horizontal bracket
            pygame.draw.line(self.screen, self.hud_color, 
                           (bx, by), (bx + dx * bracket_size, by), 2)
            # Vertical bracket
            pygame.draw.line(self.screen, self.hud_color, 
                           (bx, by), (bx, by + dy * bracket_size), 2)
    
    def draw_hud_text(self, robot_pos, brain_state, step_count):
        """Draw HUD text overlay."""
        
        # Setup font
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        # HUD information
        hud_lines = [
            f"POS: ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})",
            f"HDG: {robot_pos[2]:.0f}°",
            f"STEP: {step_count}",
            f"EXP: {brain_state.get('total_experiences', 0)}",
            f"CONF: {brain_state.get('prediction_confidence', 0):.2f}",
            f"MODE: {brain_state.get('prediction_method', 'UNKNOWN')[:8]}"
        ]
        
        # Draw HUD background
        hud_rect = pygame.Rect(10, 10, 180, len(hud_lines) * 25 + 10)
        pygame.draw.rect(self.screen, (0, 50, 0), hud_rect)
        pygame.draw.rect(self.screen, self.line_color, hud_rect, 2)
        
        # Draw HUD text
        for i, line in enumerate(hud_lines):
            text_surface = small_font.render(line, True, self.hud_color)
            self.screen.blit(text_surface, (15, 15 + i * 22))


class PiCarXWireframeDemo:
    """
    Complete wireframe demo showing robot's first-person Battlezone perspective.
    """
    
    def __init__(self, world_size=50):
        self.world_size = world_size
        
        # Create robot brainstem
        self.robot = PiCarXBrainstem(
            enable_camera=True,
            enable_ultrasonics=True,
            enable_line_tracking=True
        )
        
        # Create wireframe renderer
        self.renderer = WireframeRenderer(width=800, height=600)
        
        # Environment obstacles (x, z, type, size)
        self.obstacles = [
            (15, 15, 'cube', 4),
            (25, 8, 'pyramid', 3),
            (10, 35, 'cube', 3),
            (35, 30, 'pyramid', 4),
            (5, 25, 'cube', 2),
            (40, 12, 'pyramid', 3),
            (20, 40, 'cube', 5),
            (30, 5, 'pyramid', 2)
        ]
        
        # Demo state
        self.running = True
        self.step_count = 0
        
        print(f"🎮 Wireframe Demo initialized")
        print(f"   World: {world_size}x{world_size}m")
        print(f"   Obstacles: {len(self.obstacles)}")
        print(f"   Controls: ESC=quit, SPACE=pause")
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    print(f"⏸️  Paused at step {self.step_count}")
                    print(f"   Robot: ({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f})")
                    print(f"   Brain: {self.robot.brain.total_experiences} experiences")
    
    def render_frame(self, brain_state):
        """Render one frame of the wireframe view."""
        
        # Clear screen
        self.renderer.clear_screen()
        
        # Camera position (robot's perspective)
        camera_pos = [self.robot.position[0], self.renderer.camera_height, self.robot.position[1]]
        camera_heading = self.robot.position[2]
        
        # Draw horizon grid
        self.renderer.draw_horizon_grid(camera_pos, camera_heading)
        
        # Draw obstacles
        for obs_x, obs_z, obs_type, obs_size in self.obstacles:
            center = [obs_x, 0, obs_z]
            
            if obs_type == 'cube':
                self.renderer.draw_wireframe_cube(center, obs_size, camera_pos, camera_heading)
            elif obs_type == 'pyramid':
                self.renderer.draw_pyramid(center, obs_size, obs_size, camera_pos, camera_heading)
        
        # Draw ultrasonic sensor beam
        if self.robot.ultrasonic_distance < 50:
            self.renderer.draw_sensor_beam(
                [self.robot.position[0], self.robot.position[1]], 
                self.robot.position[2], 
                self.robot.ultrasonic_distance / 10.0
            )
        
        # Draw crosshairs and HUD
        self.renderer.draw_crosshairs()
        self.renderer.draw_hud_text(self.robot.position, brain_state, self.step_count)
        
        # Update display
        pygame.display.flip()
    
    def run_demo(self, duration_seconds=60):
        """Run the complete wireframe demo."""
        
        print(f"\n🎮 Starting Wireframe Demo ({duration_seconds}s)")
        print("   Experience authentic Battlezone-style robot vision!")
        print("   Green wireframes show what the minimal brain 'sees'")
        
        start_time = time.time()
        
        while self.running and (time.time() - start_time) < duration_seconds:
            
            # Handle events
            self.handle_events()
            
            if not self.running:
                break
            
            # Robot control cycle
            cycle_result = self.robot.control_cycle()
            brain_state = cycle_result['brain_state']
            self.step_count += 1
            
            # Keep robot in bounds
            self.robot.position[0] = max(2, min(self.world_size - 2, self.robot.position[0]))
            self.robot.position[1] = max(2, min(self.world_size - 2, self.robot.position[1]))
            
            # Render frame
            self.render_frame(brain_state)
            
            # Control frame rate
            self.renderer.clock.tick(30)  # 30 FPS
            
            # Progress updates
            if self.step_count % 100 == 0:
                print(f"📍 Step {self.step_count}: Pos({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f}) | "
                      f"Brain: {brain_state['total_experiences']} exp | "
                      f"Method: {brain_state['prediction_method']}")
        
        # Demo complete
        pygame.quit()
        
        print(f"\n✅ Wireframe demo complete!")
        status = self.robot.get_robot_status()
        print(f"   Final experiences: {status['brain']['total_experiences']}")
        print(f"   Consensus rate: {status['brain']['consensus_rate']:.2%}")
        print(f"   Steps completed: {self.step_count}")


def main():
    """Run the PiCar-X wireframe demonstration."""
    
    print("🎮 PiCar-X Wireframe Demo - Battlezone-Style Vector Graphics")
    print("   Authentic 1980s green wireframe aesthetics using pygame")
    print("   Shows robot's first-person perspective with classic HUD")
    
    try:
        demo = PiCarXWireframeDemo(world_size=50)
        demo.run_demo(duration_seconds=60)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        pygame.quit()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()


if __name__ == "__main__":
    main()