#!/usr/bin/env python3
"""
Socket-Based Embodied Free Energy 3D Demo - Distributed Intelligence

Interactive 3D visualization showcasing the socket-based embodied Free Energy system.
This demo demonstrates truly distributed intelligence with realistic network communication.

IMPORTANT: This demo requires a running brain server!
Start server first: python3 server/brain_server.py

Features:
- Real-time socket communication with brain server
- Embodied Free Energy action selection on robot client
- Hardware state visualization (battery, temperature)
- Network latency and reliability tracking
- Physics-grounded behavior emergence
- Interactive camera controls and HUD

The robot's behavior emerges purely from:
- Battery depletion → Energy-seeking behavior
- Motor heating → Thermal management  
- Memory pressure → Simplified actions
- Sensor noise → Conservative behavior

Controls:
- Mouse: Orbit camera around scene
- Scroll: Zoom in/out
- Right-click drag: Pan camera
- R: Reset camera to default position
- Space: Pause simulation
- C: Connect to brain server
- B: Toggle battery drain speed (for demo)
- V: Toggle verbose embodied Free Energy logging
- ESC: Exit demo

Run with: python3 demo_runner.py 3d_embodied
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

# Import socket-based embodied brainstem
try:
    from .picar_x_simulation.socket_embodied_brainstem import SocketEmbodiedBrainstem
except ImportError:
    from picar_x_simulation.socket_embodied_brainstem import SocketEmbodiedBrainstem


class Camera3D:
    """3D camera with mouse orbit controls."""
    
    def __init__(self):
        self.distance = 300.0
        self.rotation_x = -20.0
        self.rotation_y = 45.0
        self.target = [0.0, 0.0, 0.0]
        self.mouse_sensitivity = 0.5
        self.zoom_sensitivity = 10.0
        
    def update_from_mouse(self, mouse_rel, mouse_buttons):
        """Update camera from mouse input."""
        if mouse_buttons[0]:  # Left mouse button - orbit
            self.rotation_y += mouse_rel[0] * self.mouse_sensitivity
            self.rotation_x += mouse_rel[1] * self.mouse_sensitivity
            self.rotation_x = max(-89, min(89, self.rotation_x))
    
    def zoom(self, zoom_delta):
        """Zoom camera in/out."""
        self.distance += zoom_delta * self.zoom_sensitivity
        self.distance = max(50, min(1000, self.distance))
    
    def apply_transform(self):
        """Apply camera transformation to OpenGL."""
        glLoadIdentity()
        glTranslatef(0, 0, -self.distance)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glTranslatef(-self.target[0], -self.target[1], -self.target[2])
    
    def reset(self):
        """Reset camera to default position."""
        self.distance = 300.0
        self.rotation_x = -20.0
        self.rotation_y = 45.0


class HUD:
    """Heads-up display for embodied Free Energy information."""
    
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
    
    def render_text_3d(self, text, x, y, color=(1, 1, 1)):
        """Render text in 3D space."""
        # Switch to 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for text
        glDisable(GL_DEPTH_TEST)
        
        # Render text using pygame
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
        glRasterPos2f(x, y)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Restore 3D rendering
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def draw_embodied_hud(self, brainstem, cycle_info):
        """Draw the embodied Free Energy HUD."""
        
        y_offset = 20
        line_height = 25
        
        # Get embodied statistics
        stats = brainstem.get_embodied_statistics()
        telemetry = brainstem.hardware_interface.get_telemetry()
        
        # Title
        self.render_text_3d("🧬 EMBODIED FREE ENERGY ROBOT", 20, y_offset)
        y_offset += line_height * 1.5
        
        # Hardware state
        battery_color = (1, 0, 0) if telemetry.battery_percentage < 0.2 else (1, 1, 0) if telemetry.battery_percentage < 0.5 else (0, 1, 0)
        self.render_text_3d(f"🔋 Battery: {telemetry.battery_percentage:.1%} ({telemetry.battery_voltage:.1f}V)", 20, y_offset, battery_color)
        y_offset += line_height
        
        motor_temp = max(telemetry.motor_temperatures.values())
        temp_color = (1, 0, 0) if motor_temp > 60 else (1, 1, 0) if motor_temp > 45 else (0, 1, 0)
        self.render_text_3d(f"🌡️ Motor Temp: {motor_temp:.1f}°C", 20, y_offset, temp_color)
        y_offset += line_height
        
        self.render_text_3d(f"🧠 Memory: {telemetry.memory_usage_percentage:.1f}%", 20, y_offset)
        y_offset += line_height * 1.5
        
        # Current action
        if 'selected_action' in cycle_info:
            action = cycle_info['selected_action']
            action_str = f"{action.get('type', 'unknown')}"
            if action.get('speed'):
                action_str += f" (speed: {action['speed']:.1f})"
            if action.get('urgency'):
                action_str += f" ({action['urgency']})"
            self.render_text_3d(f"🎯 Action: {action_str}", 20, y_offset)
            y_offset += line_height
        
        # Precision weights
        embodied_stats = stats.get('embodied_system_stats', {})
        precision_weights = embodied_stats.get('current_precision_weights', {})
        
        self.render_text_3d("⚖️ Precision Weights:", 20, y_offset)
        y_offset += line_height
        
        for prior_name, weight in precision_weights.items():
            display_name = prior_name.replace('_', ' ').title()
            color = (1, 0.5, 0) if weight > 10 else (1, 1, 0) if weight > 5 else (1, 1, 1)
            self.render_text_3d(f"   {display_name}: {weight:.1f}", 20, y_offset, color)
            y_offset += line_height * 0.8
        
        y_offset += line_height * 0.5
        
        # Network statistics (for socket-based brainstem)
        if hasattr(brainstem, 'network_latencies') and brainstem.network_latencies:
            avg_latency = sum(brainstem.network_latencies) / len(brainstem.network_latencies)
            reliability = stats.get('network_reliability', 0) * 100
            
            self.render_text_3d("🌐 Network Performance:", 20, y_offset)
            y_offset += line_height
            
            latency_color = (1, 0, 0) if avg_latency > 0.1 else (1, 1, 0) if avg_latency > 0.05 else (0, 1, 0)
            self.render_text_3d(f"   Latency: {avg_latency*1000:.1f}ms", 20, y_offset, latency_color)
            y_offset += line_height * 0.8
            
            reliability_color = (1, 0, 0) if reliability < 80 else (1, 1, 0) if reliability < 95 else (0, 1, 0)
            self.render_text_3d(f"   Reliability: {reliability:.1f}%", 20, y_offset, reliability_color)
            y_offset += line_height * 0.8
            
            self.render_text_3d(f"   Socket errors: {stats.get('socket_errors', 0)}", 20, y_offset)
            y_offset += line_height * 1.2
        
        # Behavior statistics
        self.render_text_3d("📊 Embodied Behavior:", 20, y_offset)
        y_offset += line_height
        
        decisions = stats.get('embodied_decisions', 0)
        energy_actions = stats.get('energy_seeking_actions', 0)
        energy_rate = (energy_actions / max(1, decisions)) * 100
        
        self.render_text_3d(f"   Decisions: {decisions}", 20, y_offset)
        y_offset += line_height * 0.8
        self.render_text_3d(f"   Energy-seeking: {energy_rate:.1f}%", 20, y_offset)
        y_offset += line_height * 1.5
        
        # Robot physical state
        self.render_text_3d("🤖 Robot State:", 20, y_offset)
        y_offset += line_height
        
        robot_stats = cycle_info.get('robot_stats', {})
        position = robot_stats.get('position', [0, 0, 0])
        self.render_text_3d(f"   Pos: ({position[0]:.1f}, {position[1]:.1f})", 20, y_offset)
        y_offset += line_height * 0.8
        self.render_text_3d(f"   Distance: {robot_stats.get('ultrasonic_distance', 0):.1f}cm", 20, y_offset)
        y_offset += line_height * 0.8
        self.render_text_3d(f"   Speed: {robot_stats.get('motor_speed', 0):.1f}", 20, y_offset)
        y_offset += line_height * 1.5
        
        # Connection status
        if hasattr(brainstem, 'connected'):
            connection_status = "🔗 Connected" if brainstem.connected else "❌ Disconnected"
            connection_color = (0, 1, 0) if brainstem.connected else (1, 0, 0)
            self.render_text_3d(f"Brain Server: {connection_status}", 20, self.screen_height - 110, connection_color)
        
        # Controls
        self.render_text_3d("Controls: Space=Pause, B=Battery, V=Verbose, C=Connect", 20, self.screen_height - 60)
        self.render_text_3d("Mouse=Camera, Scroll=Zoom, R=Reset, ESC=Exit", 20, self.screen_height - 35)


class Scene3D:
    """3D scene renderer for embodied robot demo."""
    
    def __init__(self):
        self.setup_lighting()
    
    def setup_lighting(self):
        """Setup 3D lighting."""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        
        # Light parameters
        light_pos = [100.0, 100.0, 100.0, 1.0]
        light_color = [1.0, 1.0, 1.0, 1.0]
        ambient = [0.3, 0.3, 0.3, 1.0]
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_color)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
    
    def draw_ground(self):
        """Draw ground plane."""
        glColor3f(0.4, 0.4, 0.4)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-200, 0, -200)
        glVertex3f(200, 0, -200)
        glVertex3f(200, 0, 200)
        glVertex3f(-200, 0, 200)
        glEnd()
        
        # Grid lines
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        for i in range(-200, 201, 20):
            glVertex3f(i, 0.1, -200)
            glVertex3f(i, 0.1, 200)
            glVertex3f(-200, 0.1, i)
            glVertex3f(200, 0.1, i)
        glEnd()
    
    def draw_robot(self, brainstem):
        """Draw the PiCar-X robot."""
        pos = brainstem.position
        
        glPushMatrix()
        glTranslatef(pos[0], 5, pos[1])
        glRotatef(math.degrees(pos[2]), 0, 1, 0)
        
        # Robot body - color indicates battery level
        battery_level = brainstem.hardware_interface.battery_level
        if battery_level > 0.5:
            glColor3f(0, 1, 0)  # Green when healthy
        elif battery_level > 0.2:
            glColor3f(1, 1, 0)  # Yellow when moderate
        else:
            glColor3f(1, 0, 0)  # Red when low
        
        # Main body
        glBegin(GL_QUADS)
        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(-10, 5, -15)
        glVertex3f(10, 5, -15)
        glVertex3f(10, 5, 15)
        glVertex3f(-10, 5, 15)
        # Sides
        glNormal3f(0, 0, 1)
        glVertex3f(-10, 0, 15)
        glVertex3f(10, 0, 15)
        glVertex3f(10, 5, 15)
        glVertex3f(-10, 5, 15)
        glEnd()
        
        # Wheels
        glColor3f(0.2, 0.2, 0.2)
        for wheel_pos in [(-8, 0, -10), (8, 0, -10), (-8, 0, 10), (8, 0, 10)]:
            glPushMatrix()
            glTranslatef(*wheel_pos)
            # Simple wheel representation
            glBegin(GL_QUADS)
            glVertex3f(-2, 0, -3)
            glVertex3f(2, 0, -3)
            glVertex3f(2, 3, -3)
            glVertex3f(-2, 3, -3)
            glEnd()
            glPopMatrix()
        
        # Ultrasonic sensor
        glColor3f(0, 0, 1)
        glPushMatrix()
        glTranslatef(0, 3, 18)
        glBegin(GL_QUADS)
        glVertex3f(-3, -1, -2)
        glVertex3f(3, -1, -2)
        glVertex3f(3, 1, -2)
        glVertex3f(-3, 1, -2)
        glEnd()
        glPopMatrix()
        
        glPopMatrix()
    
    def draw_ultrasonic_beam(self, brainstem):
        """Draw ultrasonic sensor beam."""
        pos = brainstem.position
        distance = brainstem.ultrasonic_distance
        heading = pos[2]
        
        # Beam color based on distance
        if distance < 20:
            glColor4f(1, 0, 0, 0.3)  # Red for close
        elif distance < 50:
            glColor4f(1, 1, 0, 0.3)  # Yellow for medium
        else:
            glColor4f(0, 1, 0, 0.3)  # Green for far
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Draw beam cone
        beam_length = min(distance, 100)
        cone_width = beam_length * 0.2
        
        start_x = pos[0] + 18 * math.cos(heading)
        start_z = pos[1] + 18 * math.sin(heading)
        end_x = start_x + beam_length * math.cos(heading)
        end_z = start_z + beam_length * math.sin(heading)
        
        glBegin(GL_TRIANGLES)
        # Beam cone
        for i in range(8):
            angle1 = i * math.pi / 4
            angle2 = (i + 1) * math.pi / 4
            
            x1 = end_x + cone_width * math.cos(angle1)
            z1 = end_z + cone_width * math.sin(angle1)
            x2 = end_x + cone_width * math.cos(angle2)
            z2 = end_z + cone_width * math.sin(angle2)
            
            glVertex3f(start_x, 8, start_z)
            glVertex3f(x1, 8, z1)
            glVertex3f(x2, 8, z2)
        glEnd()
        
        glDisable(GL_BLEND)
    
    def draw_obstacles(self):
        """Draw environment obstacles."""
        obstacles = [
            (80, 0, 60, 20, 20, 20),   # x, y, z, w, h, d
            (-60, 0, -40, 15, 25, 15),
            (40, 0, -80, 30, 15, 30),
            (-100, 0, 80, 25, 30, 25)
        ]
        
        glColor3f(0.6, 0.3, 0.1)  # Brown obstacles
        for obs in obstacles:
            x, y, z, w, h, d = obs
            glPushMatrix()
            glTranslatef(x, y + h/2, z)
            glBegin(GL_QUADS)
            # Draw simple box
            # Front
            glVertex3f(-w/2, -h/2, d/2)
            glVertex3f(w/2, -h/2, d/2)
            glVertex3f(w/2, h/2, d/2)
            glVertex3f(-w/2, h/2, d/2)
            # Back  
            glVertex3f(-w/2, -h/2, -d/2)
            glVertex3f(-w/2, h/2, -d/2)
            glVertex3f(w/2, h/2, -d/2)
            glVertex3f(w/2, -h/2, -d/2)
            # Top
            glVertex3f(-w/2, h/2, -d/2)
            glVertex3f(-w/2, h/2, d/2)
            glVertex3f(w/2, h/2, d/2)
            glVertex3f(w/2, h/2, -d/2)
            glEnd()
            glPopMatrix()


class EmbodiedDemo3D:
    """Main demo application."""
    
    def __init__(self):
        pygame.init()
        
        # Screen setup
        self.screen_width = 1200
        self.screen_height = 800
        pygame.display.set_mode((self.screen_width, self.screen_height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Embodied Free Energy Robot - 3D Demo")
        
        # 3D setup
        gluPerspective(45, self.screen_width / self.screen_height, 0.1, 1000.0)
        
        # Components
        self.camera = Camera3D()
        self.hud = HUD(self.screen_width, self.screen_height)
        self.scene = Scene3D()
        
        # Socket-based embodied robot
        self.brainstem = SocketEmbodiedBrainstem()
        self.brain_connected = False
        
        # Demo state
        self.paused = False
        self.verbose = False
        self.fast_battery_drain = False
        self.clock = pygame.time.Clock()
        self.last_cycle_info = {}
        
        print("🧬 Socket-Based Embodied Free Energy 3D Demo initialized")
        print("   Distributed intelligence - brain server required!")
        print("   Watch how behavior emerges from physical constraints!")
    
    def handle_events(self):
        """Handle pygame events."""
        mouse_rel = pygame.mouse.get_rel()
        mouse_buttons = pygame.mouse.get_pressed()
        
        # Update camera from mouse
        self.camera.update_from_mouse(mouse_rel, mouse_buttons)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"Demo {'paused' if self.paused else 'resumed'}")
                elif event.key == pygame.K_r:
                    self.camera.reset()
                    print("Camera reset")
                elif event.key == pygame.K_v:
                    self.verbose = not self.verbose
                    self.brainstem.set_verbose_embodied(self.verbose)
                    print(f"Verbose embodied logging {'enabled' if self.verbose else 'disabled'}")
                elif event.key == pygame.K_b:
                    self.fast_battery_drain = not self.fast_battery_drain
                    print(f"Fast battery drain {'enabled' if self.fast_battery_drain else 'disabled'}")
                elif event.key == pygame.K_c:
                    if not self.brain_connected:
                        print("🔗 Attempting to connect to brain server...")
                        if self.brainstem.connect_to_brain_server():
                            self.brain_connected = True
                            print("✅ Connected to brain server!")
                        else:
                            print("❌ Failed to connect - is brain server running?")
                    else:
                        print("Already connected to brain server")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.camera.zoom(-1)
                elif event.button == 5:  # Scroll down
                    self.camera.zoom(1)
        
        return True
    
    def update(self):
        """Update simulation."""
        if not self.paused:
            # Run robot control cycle
            cycle_info = self.brainstem.control_cycle()
            self.last_cycle_info = cycle_info
            
            # Simulate faster battery drain for demo purposes
            if self.fast_battery_drain:
                self.brainstem.hardware_interface.battery_level = max(
                    0.0, 
                    self.brainstem.hardware_interface.battery_level - 0.01
                )
    
    def render(self):
        """Render the 3D scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Apply camera transform
        self.camera.apply_transform()
        
        # Draw scene
        self.scene.draw_ground()
        self.scene.draw_obstacles()
        self.scene.draw_robot(self.brainstem)
        self.scene.draw_ultrasonic_beam(self.brainstem)
        
        # Draw HUD
        self.hud.draw_embodied_hud(self.brainstem, self.last_cycle_info)
        
        pygame.display.flip()
    
    def run(self):
        """Main demo loop."""
        print("🚀 Starting socket-based embodied Free Energy demo...")
        print("⚠️  NOTE: This demo requires a running brain server!")
        print("   Start server: python3 server/brain_server.py")
        print("   Press 'C' in demo to connect to brain server")
        print("Watch how the robot's behavior emerges from physical constraints!")
        
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(30)  # 30 FPS
        
        # Disconnect from brain server
        if self.brain_connected:
            self.brainstem.disconnect_from_brain_server()
        
        # Print final report
        print("\\n" + "="*60)
        if hasattr(self.brainstem, 'print_socket_report'):
            self.brainstem.print_socket_report()
        else:
            print("Socket-based embodied demo completed")
        
        pygame.quit()


def main():
    """Run the embodied 3D demo."""
    try:
        demo = EmbodiedDemo3D()
        demo.run()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()