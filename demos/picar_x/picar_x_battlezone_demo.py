#!/usr/bin/env python3
"""
PiCar-X Battlezone Demo

A Battlezone (1980) inspired first-person view of the robot's perspective.
Classic green wireframe graphics show what the minimal brain "sees" as it learns.

Features:
- First-person robot camera view
- Green wireframe obstacles (classic Battlezone style)
- Real-time brain learning visualization
- Authentic retro vector graphics aesthetic
- Shows ultrasonic sensor rays and camera field of view
"""

import sys
import os

# Add the brain/ directory to import minimal as a package
current_dir = os.path.dirname(__file__)  # picar_x/
demos_dir = os.path.dirname(current_dir)  # demos/
minimal_dir = os.path.dirname(demos_dir)  # minimal/
brain_dir = os.path.dirname(minimal_dir)   # brain/
sys.path.insert(0, brain_dir)

import pyglet
import numpy as np
import math
import time

# Import OpenGL functions
try:
    from pyglet.gl import *
    OPENGL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OpenGL import failed: {e}")
    OPENGL_AVAILABLE = False

# Handle both direct execution and module import
try:
    from .picar_x_brainstem import PiCarXBrainstem
except ImportError:
    from picar_x_brainstem import PiCarXBrainstem


class BattlezoneRenderer:
    """
    Battlezone-style 3D wireframe renderer using pyglet and OpenGL.
    Recreates the classic 1980 vector graphics aesthetic.
    """
    
    def __init__(self, width=800, height=600):
        if not OPENGL_AVAILABLE:
            raise RuntimeError("OpenGL not available - Battlezone demo requires OpenGL context")
            
        self.width = width
        self.height = height
        
        # Create pyglet window
        self.window = pyglet.window.Window(width, height, caption="PiCar-X Battlezone - Minimal Brain")
        
        # Battlezone colors (classic green on black)
        self.bg_color = (0, 0, 0, 1)        # Black background
        self.line_color = (0, 1, 0, 1)      # Bright green lines
        self.horizon_color = (0, 0.6, 0, 1) # Darker green horizon
        self.sensor_color = (1, 1, 0, 1)    # Yellow for sensors
        self.target_color = (1, 0, 0, 1)    # Red for targets/goals
        
        # 3D projection parameters
        self.fov = 60.0  # Field of view in degrees
        self.near = 0.1
        self.far = 200.0
        
        # Camera parameters (robot's perspective)
        self.camera_height = 1.5  # Robot camera height
        
        # Setup OpenGL
        self.setup_opengl()
        
        # Vertex lists for efficient rendering
        self.vertex_lists = {}
        
        print("🎮 Battlezone renderer initialized")
        print(f"   Resolution: {width}x{height}")
        print(f"   Camera height: {self.camera_height}m")
    
    def setup_opengl(self):
        """Setup OpenGL for wireframe 3D rendering."""
        
        try:
            # Enable depth testing
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            
            # Enable line smoothing for crisp wireframes
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            
            # Set line width for visible wireframes (with fallback)
            try:
                glLineWidth(2.0)
            except:
                # Fallback if line width setting fails
                glLineWidth(1.0)
        except Exception as e:
            print(f"⚠️  OpenGL setup warning: {e}")
            print("   Demo will continue with basic OpenGL settings")
        
        # Setup projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Perspective projection (classic Battlezone field of view)
        aspect = self.width / self.height
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        
        projection_matrix = [
            f/aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (self.far + self.near)/(self.near - self.far), (2*self.far*self.near)/(self.near - self.far),
            0, 0, -1, 0
        ]
        
        glMultMatrixf((GLfloat * 16)(*projection_matrix))
        
        # Switch to modelview matrix
        glMatrixMode(GL_MODELVIEW)
    
    def clear_screen(self):
        """Clear screen with classic black background."""
        glClearColor(*self.bg_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    def set_camera(self, robot_pos, robot_heading):
        """Set camera to robot's first-person perspective."""
        
        glLoadIdentity()
        
        # Robot position and orientation
        x, y = robot_pos[0], robot_pos[1]
        heading_rad = math.radians(robot_heading)
        
        # Camera position (slightly above robot)
        cam_x = x
        cam_y = self.camera_height
        cam_z = y
        
        # Look direction (forward from robot)
        look_x = x + math.cos(heading_rad)
        look_y = self.camera_height
        look_z = y + math.sin(heading_rad)
        
        # Up vector
        up_x, up_y, up_z = 0, 1, 0
        
        # Create look-at matrix manually (pyglet doesn't have gluLookAt)
        forward = np.array([look_x - cam_x, look_y - cam_y, look_z - cam_z])
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, [up_x, up_y, up_z])
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Create view matrix
        view_matrix = [
            right[0], up[0], -forward[0], 0,
            right[1], up[1], -forward[1], 0,
            right[2], up[2], -forward[2], 0,
            -np.dot(right, [cam_x, cam_y, cam_z]),
            -np.dot(up, [cam_x, cam_y, cam_z]),
             np.dot(forward, [cam_x, cam_y, cam_z]), 1
        ]
        
        glMultMatrixf((GLfloat * 16)(*view_matrix))
    
    def draw_horizon_grid(self):
        """Draw classic Battlezone horizon grid."""
        
        glColor4f(*self.horizon_color)
        glBegin(GL_LINES)
        
        # Horizon line
        glVertex3f(-100, 0, 0)
        glVertex3f(100, 0, 0)
        
        # Ground grid lines (extending into distance)
        for i in range(-10, 11, 2):
            # Parallel lines
            glVertex3f(i * 5, 0, 0)
            glVertex3f(i * 5, 0, 100)
            
            # Perpendicular lines (every 10 units)
            if i % 2 == 0:
                glVertex3f(-50, 0, i * 5)
                glVertex3f(50, 0, i * 5)
        
        glEnd()
    
    def draw_wireframe_cube(self, center, size):
        """Draw a wireframe cube (classic obstacle)."""
        
        x, y, z = center
        s = size / 2
        
        # Define cube vertices
        vertices = [
            [x-s, y, z-s], [x+s, y, z-s], [x+s, y+size, z-s], [x-s, y+size, z-s],  # Front face
            [x-s, y, z+s], [x+s, y, z+s], [x+s, y+size, z+s], [x-s, y+size, z+s]   # Back face
        ]
        
        # Define cube edges
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # Front face
            (4,5), (5,6), (6,7), (7,4),  # Back face
            (0,4), (1,5), (2,6), (3,7)   # Connecting edges
        ]
        
        glColor4f(*self.line_color)
        glBegin(GL_LINES)
        
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            glVertex3f(*v1)
            glVertex3f(*v2)
        
        glEnd()
    
    def draw_pyramid(self, center, base_size, height):
        """Draw a wireframe pyramid."""
        
        x, y, z = center
        s = base_size / 2
        
        # Define pyramid vertices
        base = [
            [x-s, y, z-s], [x+s, y, z-s], [x+s, y, z+s], [x-s, y, z+s]
        ]
        apex = [x, y + height, z]
        
        glColor4f(*self.line_color)
        glBegin(GL_LINES)
        
        # Draw base
        for i in range(4):
            v1 = base[i]
            v2 = base[(i + 1) % 4]
            glVertex3f(*v1)
            glVertex3f(*v2)
        
        # Draw edges to apex
        for vertex in base:
            glVertex3f(*vertex)
            glVertex3f(*apex)
        
        glEnd()
    
    def draw_ultrasonic_beam(self, robot_pos, robot_heading, distance, max_distance=50):
        """Draw ultrasonic sensor beam (yellow cone)."""
        
        if distance >= max_distance:
            return  # No beam if no detection
        
        x, y = robot_pos[0], robot_pos[1]
        heading_rad = math.radians(robot_heading)
        
        # Beam parameters
        beam_width = 15  # degrees
        beam_segments = 8
        
        glColor4f(*self.sensor_color)
        glBegin(GL_LINES)
        
        # Draw beam cone
        for i in range(beam_segments):
            angle = heading_rad + math.radians(beam_width * (i / beam_segments - 0.5))
            
            end_x = x + distance * math.cos(angle)
            end_z = y + distance * math.sin(angle)
            
            # Beam line
            glVertex3f(x, self.camera_height, y)
            glVertex3f(end_x, self.camera_height, end_z)
        
        # Draw beam outline
        for i in [-1, 1]:
            angle = heading_rad + math.radians(beam_width * i * 0.5)
            end_x = x + distance * math.cos(angle)
            end_z = y + distance * math.sin(angle)
            
            glVertex3f(x, self.camera_height, y)
            glVertex3f(end_x, self.camera_height, end_z)
        
        glEnd()
    
    def draw_crosshairs(self):
        """Draw classic Battlezone crosshairs in screen center."""
        
        # Switch to 2D rendering for HUD elements
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for HUD
        glDisable(GL_DEPTH_TEST)
        
        cx, cy = self.width // 2, self.height // 2
        size = 20
        
        glColor4f(*self.line_color)
        glBegin(GL_LINES)
        
        # Horizontal line
        glVertex2f(cx - size, cy)
        glVertex2f(cx + size, cy)
        
        # Vertical line
        glVertex2f(cx, cy - size)
        glVertex2f(cx, cy + size)
        
        # Corner brackets (classic Battlezone style)
        bracket_size = 5
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            bx, by = cx + dx * (size + 5), cy + dy * (size + 5)
            
            glVertex2f(bx, by)
            glVertex2f(bx + dx * bracket_size, by)
            
            glVertex2f(bx, by)
            glVertex2f(bx, by + dy * bracket_size)
        
        glEnd()
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


class PiCarXBattlezoneDemo:
    """
    Complete Battlezone-style demo showing robot's first-person perspective.
    """
    
    def __init__(self, world_size=50):
        self.world_size = world_size
        
        # Create robot brainstem
        self.robot = PiCarXBrainstem(
            enable_camera=True,
            enable_ultrasonics=True,
            enable_line_tracking=True
        )
        
        # Create Battlezone renderer
        self.renderer = BattlezoneRenderer(width=800, height=600)
        
        # Environment obstacles (x, z, type)
        self.obstacles = [
            (15, 15, 'cube', 4),    # x, z, type, size
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
        
        # Setup pyglet event handlers
        self.setup_events()
        
        print(f"🎮 Battlezone Demo initialized")
        print(f"   World: {world_size}x{world_size}m")
        print(f"   Obstacles: {len(self.obstacles)}")
        print(f"   Controls: ESC=quit, SPACE=pause")
    
    def setup_events(self):
        """Setup pyglet event handlers."""
        
        @self.renderer.window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.ESCAPE:
                self.running = False
                self.renderer.window.close()
            elif symbol == pyglet.window.key.SPACE:
                print(f"⏸️  Paused at step {self.step_count}")
                print(f"   Robot position: ({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f})")
                print(f"   Brain experiences: {self.robot.brain.total_experiences}")
        
        @self.renderer.window.event
        def on_draw():
            self.render_frame()
    
    def render_frame(self):
        """Render one frame of the Battlezone view."""
        
        # Clear screen
        self.renderer.clear_screen()
        
        # Set camera to robot's perspective
        robot_pos = [self.robot.position[0], self.robot.position[1]]
        robot_heading = self.robot.position[2]
        self.renderer.set_camera(robot_pos, robot_heading)
        
        # Draw horizon grid
        self.renderer.draw_horizon_grid()
        
        # Draw obstacles
        for obs_x, obs_z, obs_type, obs_size in self.obstacles:
            center = [obs_x, 0, obs_z]
            
            if obs_type == 'cube':
                self.renderer.draw_wireframe_cube(center, obs_size)
            elif obs_type == 'pyramid':
                self.renderer.draw_pyramid(center, obs_size, obs_size)
        
        # Draw ultrasonic sensor beam
        if self.robot.ultrasonic_distance < 50:
            self.renderer.draw_ultrasonic_beam(
                robot_pos, robot_heading, 
                self.robot.ultrasonic_distance / 10.0  # Scale to world units
            )
        
        # Draw crosshairs (HUD overlay)
        self.renderer.draw_crosshairs()
    
    def run_demo(self, duration_seconds=60):
        """Run the complete Battlezone demo."""
        
        print(f"\n🚀 Starting Battlezone Demo ({duration_seconds}s)")
        print("   Experience the robot's first-person perspective!")
        print("   Watch how the minimal brain learns to navigate...")
        
        start_time = time.time()
        
        def update(dt):
            """Update function called by pyglet scheduler."""
            
            if not self.running:
                return
            
            # Check if demo should end
            if time.time() - start_time > duration_seconds:
                self.running = False
                self.renderer.window.close()
                return
            
            # Robot control cycle
            cycle_result = self.robot.control_cycle()
            self.step_count += 1
            
            # Keep robot in bounds
            self.robot.position[0] = max(2, min(self.world_size - 2, self.robot.position[0]))
            self.robot.position[1] = max(2, min(self.world_size - 2, self.robot.position[1]))
            
            # Progress updates
            if self.step_count % 50 == 0:
                brain_state = cycle_result['brain_state']
                print(f"📍 Step {self.step_count}: Pos({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f}) | "
                      f"Brain: {brain_state['total_experiences']} exp | "
                      f"Method: {brain_state['prediction_method']}")
        
        # Schedule update function
        pyglet.clock.schedule_interval(update, 1/30.0)  # 30 FPS
        
        # Run pyglet app
        pyglet.app.run()
        
        # Demo complete
        print(f"\n✅ Battlezone demo complete!")
        status = self.robot.get_robot_status()
        print(f"   Final experiences: {status['brain']['total_experiences']}")
        print(f"   Consensus rate: {status['brain']['consensus_rate']:.2%}")
        print(f"   Steps completed: {self.step_count}")


def main():
    """Run the PiCar-X Battlezone demonstration."""
    
    print("🎮 PiCar-X Battlezone Demo - Retro First-Person Robot Vision")
    print("   Inspired by the classic 1980 wireframe tank game")
    print("   Shows what the minimal brain 'sees' as it learns to navigate")
    
    if not OPENGL_AVAILABLE:
        print("\n❌ OpenGL not available in this environment")
        print("💡 This demo requires a GUI environment with OpenGL support")
        print("💡 Try running from a desktop environment or with X11 forwarding")
        print("\n✅ Alternative demos that work in any environment:")
        print("   python3 picar_x_text_demo.py        # ASCII visualization")
        print("   python3 picar_x_2d_debug_demo.py    # 2D pygame (if GUI available)")
        return
    
    try:
        demo = PiCarXBattlezoneDemo(world_size=50)
        demo.run_demo(duration_seconds=60)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        if "opengl" in str(e).lower() or "display" in str(e).lower():
            print("💡 This demo requires a GUI environment with OpenGL support")
            print("💡 Try running from a desktop environment")
        elif "pyglet" in str(e).lower():
            print("💡 Install pyglet: pip install pyglet")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()