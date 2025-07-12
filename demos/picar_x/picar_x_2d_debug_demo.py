#!/usr/bin/env python3
"""
PiCar-X 2D Grid Debug Demo

A simple 2D top-down grid visualization for debugging robot navigation errors.
Shows the robot's path, decision points, and learning progress clearly.

Features:
- Large grid cells for easy error spotting
- Color-coded robot states (learning, confident, stuck)
- Trail visualization showing path history
- Brain state indicators
- Collision and error highlighting
- Real-time statistics overlay
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
from typing import List, Tuple

# Handle both direct execution and module import
try:
    from .picar_x_brainstem import PiCarXBrainstem
except ImportError:
    from picar_x_brainstem import PiCarXBrainstem


class GridDebugRenderer:
    """
    2D grid-based renderer for debugging robot navigation errors.
    """
    
    def __init__(self, width=800, height=800, grid_size=20):
        self.width = width
        self.height = height
        self.grid_size = grid_size  # Size of each grid cell in pixels
        self.world_size = min(width, height) // grid_size  # World size in grid units
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PiCar-X 2D Debug View - Grid Navigation Analysis")
        self.clock = pygame.time.Clock()
        
        # Colors for different elements
        self.colors = {
            'background': (20, 20, 20),          # Dark gray
            'grid': (60, 60, 60),                # Grid lines
            'obstacle': (150, 50, 50),           # Dark red obstacles
            'robot_learning': (255, 165, 0),     # Orange (learning)
            'robot_confident': (50, 255, 50),    # Green (confident)
            'robot_stuck': (255, 50, 50),        # Red (stuck/confused)
            'trail_recent': (100, 100, 255),     # Blue trail (recent)
            'trail_old': (50, 50, 150),          # Dark blue (old)
            'collision': (255, 255, 0),          # Yellow collision
            'sensor_beam': (255, 255, 150),      # Light yellow sensor
            'text': (255, 255, 255),             # White text
            'prediction_good': (0, 255, 0),      # Green prediction
            'prediction_bad': (255, 100, 100),   # Red prediction
        }
        
        # Font for text overlay
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        print(f"🎯 Grid Debug Renderer initialized")
        print(f"   Window: {width}x{height}")
        print(f"   Grid size: {grid_size}px")
        print(f"   World size: {self.world_size}x{self.world_size} cells")
    
    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen pixels."""
        screen_x = int(world_x * self.grid_size)
        screen_y = int(world_y * self.grid_size)
        return screen_x, screen_y
    
    def clear_screen(self):
        """Clear screen with background color."""
        self.screen.fill(self.colors['background'])
    
    def draw_grid(self):
        """Draw grid lines for easy coordinate reference."""
        
        # Vertical lines
        for x in range(0, self.width + 1, self.grid_size):
            pygame.draw.line(self.screen, self.colors['grid'], (x, 0), (x, self.height), 1)
        
        # Horizontal lines
        for y in range(0, self.height + 1, self.grid_size):
            pygame.draw.line(self.screen, self.colors['grid'], (0, y), (self.width, y), 1)
        
        # Grid coordinates (every 5th line)
        for x in range(0, self.world_size + 1, 5):
            for y in range(0, self.world_size + 1, 5):
                screen_x, screen_y = self.world_to_screen(x, y)
                if screen_x < self.width and screen_y < self.height:
                    text = self.small_font.render(f"{x},{y}", True, self.colors['grid'])
                    self.screen.blit(text, (screen_x + 2, screen_y + 2))
    
    def draw_obstacle(self, world_x, world_y, size):
        """Draw an obstacle at world coordinates."""
        screen_x, screen_y = self.world_to_screen(world_x - size/2, world_y - size/2)
        width = int(size * self.grid_size)
        height = int(size * self.grid_size)
        
        # Draw filled rectangle
        pygame.draw.rect(self.screen, self.colors['obstacle'], 
                        (screen_x, screen_y, width, height))
        
        # Draw border
        pygame.draw.rect(self.screen, (255, 100, 100), 
                        (screen_x, screen_y, width, height), 2)
    
    def draw_robot(self, world_x, world_y, heading, brain_state):
        """Draw robot with color indicating its learning state."""
        
        screen_x, screen_y = self.world_to_screen(world_x, world_y)
        
        # Choose color based on brain state
        confidence = brain_state.get('prediction_confidence', 0)
        method = brain_state.get('prediction_method', 'unknown')
        
        if method == 'bootstrap_random':
            color = self.colors['robot_learning']  # Orange - still learning
        elif confidence > 0.8:
            color = self.colors['robot_confident']  # Green - confident
        else:
            color = self.colors['robot_stuck']      # Red - uncertain/stuck
        
        # Draw robot body (circle)
        robot_radius = self.grid_size // 3
        pygame.draw.circle(self.screen, color, (screen_x, screen_y), robot_radius)
        
        # Draw direction arrow
        arrow_length = self.grid_size // 2
        heading_rad = math.radians(heading)
        end_x = screen_x + arrow_length * math.cos(heading_rad)
        end_y = screen_y + arrow_length * math.sin(heading_rad)
        
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (screen_x, screen_y), (end_x, end_y), 3)
        
        # Draw confidence indicator (circle around robot)
        confidence_radius = robot_radius + int(confidence * 10)
        pygame.draw.circle(self.screen, color, (screen_x, screen_y), confidence_radius, 2)
    
    def draw_trail(self, trail_points):
        """Draw robot's trail with fading colors."""
        
        if len(trail_points) < 2:
            return
        
        for i in range(1, len(trail_points)):
            # Fade color based on age
            age_factor = i / len(trail_points)
            
            if age_factor > 0.7:  # Recent trail
                color = self.colors['trail_recent']
            else:  # Older trail
                color = self.colors['trail_old']
            
            # Draw line segment
            start_x, start_y = self.world_to_screen(trail_points[i-1][0], trail_points[i-1][1])
            end_x, end_y = self.world_to_screen(trail_points[i][0], trail_points[i][1])
            
            pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 2)
            
            # Draw direction dots
            if i % 3 == 0:  # Every 3rd point
                pygame.draw.circle(self.screen, color, (end_x, end_y), 3)
    
    def draw_sensor_beam(self, robot_x, robot_y, heading, distance, max_distance=20):
        """Draw ultrasonic sensor beam."""
        
        if distance >= max_distance:
            return
        
        screen_start_x, screen_start_y = self.world_to_screen(robot_x, robot_y)
        
        heading_rad = math.radians(heading)
        beam_end_x = robot_x + (distance / 5.0) * math.cos(heading_rad)  # Scale distance
        beam_end_y = robot_y + (distance / 5.0) * math.sin(heading_rad)
        
        screen_end_x, screen_end_y = self.world_to_screen(beam_end_x, beam_end_y)
        
        # Draw beam line
        pygame.draw.line(self.screen, self.colors['sensor_beam'], 
                        (screen_start_x, screen_start_y), (screen_end_x, screen_end_y), 2)
        
        # Draw detection point
        pygame.draw.circle(self.screen, (255, 255, 0), (screen_end_x, screen_end_y), 5)
    
    def draw_collision_marker(self, world_x, world_y):
        """Draw collision marker."""
        screen_x, screen_y = self.world_to_screen(world_x, world_y)
        
        # Draw X marker
        size = self.grid_size // 4
        pygame.draw.line(self.screen, self.colors['collision'], 
                        (screen_x - size, screen_y - size), (screen_x + size, screen_y + size), 3)
        pygame.draw.line(self.screen, self.colors['collision'], 
                        (screen_x + size, screen_y - size), (screen_x - size, screen_y + size), 3)
    
    def draw_stats_overlay(self, stats):
        """Draw real-time statistics overlay."""
        
        y_offset = 10
        line_height = 25
        
        stat_lines = [
            f"Step: {stats.get('step', 0)}",
            f"Position: ({stats.get('robot_x', 0):.1f}, {stats.get('robot_y', 0):.1f})",
            f"Heading: {stats.get('robot_heading', 0):.1f}°",
            f"Brain Experiences: {stats.get('total_experiences', 0)}",
            f"Prediction Method: {stats.get('prediction_method', 'unknown')}",
            f"Confidence: {stats.get('prediction_confidence', 0):.3f}",
            f"Consensus Rate: {stats.get('consensus_rate', 0):.2%}",
            f"Collisions: {stats.get('collisions', 0)}",
            f"Cycle Time: {stats.get('cycle_time', 0):.1f}ms",
        ]
        
        # Draw semi-transparent background
        overlay_height = len(stat_lines) * line_height + 20
        overlay_surface = pygame.Surface((250, overlay_height))
        overlay_surface.set_alpha(180)
        overlay_surface.fill((0, 0, 0))
        self.screen.blit(overlay_surface, (10, 10))
        
        # Draw text
        for i, line in enumerate(stat_lines):
            text_surface = self.font.render(line, True, self.colors['text'])
            self.screen.blit(text_surface, (20, y_offset + i * line_height))


class PiCarX2DDebugDemo:
    """
    Complete 2D grid debug demo for analyzing robot navigation errors.
    """
    
    def __init__(self, world_size=30):
        self.world_size = world_size
        
        # Create robot brainstem
        self.robot = PiCarXBrainstem(
            enable_camera=True,
            enable_ultrasonics=True,
            enable_line_tracking=True
        )
        
        # Scale robot to grid world
        scale_factor = world_size / 50.0  # Original world was 50x50
        self.robot.position[0] *= scale_factor
        self.robot.position[1] *= scale_factor
        
        # Create 2D debug renderer
        self.renderer = GridDebugRenderer(width=800, height=800, grid_size=25)
        
        # Environment obstacles (scaled to grid)
        self.obstacles = [
            (8, 8, 3),     # x, y, size
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
        self.step_count = 0
        self.trail_points = []
        self.collision_points = []
        self.stats = {}
        
        print(f"🎯 2D Debug Demo initialized")
        print(f"   Grid world: {world_size}x{world_size}")
        print(f"   Obstacles: {len(self.obstacles)}")
        print(f"   Robot scale: {scale_factor:.2f}")
    
    def check_collision(self):
        """Check if robot collided with obstacles."""
        robot_x, robot_y = self.robot.position[0], self.robot.position[1]
        
        for obs_x, obs_y, obs_size in self.obstacles:
            distance = math.sqrt((robot_x - obs_x)**2 + (robot_y - obs_y)**2)
            if distance < obs_size / 2 + 1:  # Robot size ~1 unit
                return True
        return False
    
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
                elif event.key == pygame.K_r:
                    # Reset robot position
                    self.robot.position = [self.world_size/2, self.world_size/2, 0]
                    self.trail_points.clear()
                    self.collision_points.clear()
                    print("🔄 Robot position reset")
    
    def update_stats(self, cycle_result):
        """Update statistics for overlay."""
        brain_state = cycle_result['brain_state']
        
        self.stats = {
            'step': self.step_count,
            'robot_x': self.robot.position[0],
            'robot_y': self.robot.position[1],
            'robot_heading': self.robot.position[2],
            'total_experiences': brain_state['total_experiences'],
            'prediction_method': brain_state['prediction_method'],
            'prediction_confidence': brain_state['prediction_confidence'],
            'consensus_rate': self.robot.get_robot_status()['brain']['consensus_rate'],
            'collisions': len(self.collision_points),
            'cycle_time': cycle_result.get('cycle_time_ms', 0),
        }
    
    def render_frame(self):
        """Render one frame of the debug view."""
        
        # Clear screen
        self.renderer.clear_screen()
        
        # Draw grid
        self.renderer.draw_grid()
        
        # Draw obstacles
        for obs_x, obs_y, obs_size in self.obstacles:
            self.renderer.draw_obstacle(obs_x, obs_y, obs_size)
        
        # Draw robot trail
        self.renderer.draw_trail(self.trail_points)
        
        # Draw collision markers
        for collision_x, collision_y in self.collision_points:
            self.renderer.draw_collision_marker(collision_x, collision_y)
        
        # Draw ultrasonic sensor beam
        if self.robot.ultrasonic_distance < 50:
            self.renderer.draw_sensor_beam(
                self.robot.position[0], self.robot.position[1],
                self.robot.position[2], self.robot.ultrasonic_distance
            )
        
        # Draw robot
        self.renderer.draw_robot(
            self.robot.position[0], self.robot.position[1],
            self.robot.position[2], self.stats
        )
        
        # Draw statistics overlay
        self.renderer.draw_stats_overlay(self.stats)
        
        # Update display
        pygame.display.flip()
    
    def run_demo(self, duration_seconds=120):
        """Run the 2D debug demo."""
        
        print(f"\n🎯 Starting 2D Grid Debug Demo ({duration_seconds}s)")
        print("   Controls: ESC=quit, SPACE=pause, R=reset robot")
        print("   Watch for navigation errors and learning patterns...")
        
        start_time = time.time()
        
        while self.running and (time.time() - start_time) < duration_seconds:
            step_start = time.time()
            
            # Handle events
            self.handle_events()
            
            if not self.running:
                break
            
            # Robot control cycle
            cycle_result = self.robot.control_cycle()
            self.step_count += 1
            
            # Record trail
            pos = [self.robot.position[0], self.robot.position[1]]
            self.trail_points.append(pos)
            
            # Keep trail manageable
            if len(self.trail_points) > 200:
                self.trail_points.pop(0)
            
            # Check for collisions
            if self.check_collision():
                self.collision_points.append(pos.copy())
                print(f"💥 Collision #{len(self.collision_points)} at step {self.step_count}")
            
            # Keep robot in bounds
            self.robot.position[0] = max(1, min(self.world_size - 1, self.robot.position[0]))
            self.robot.position[1] = max(1, min(self.world_size - 1, self.robot.position[1]))
            
            # Update statistics
            self.update_stats(cycle_result)
            
            # Render frame
            self.render_frame()
            
            # Control frame rate
            self.renderer.clock.tick(20)  # 20 FPS
            
            # Progress updates
            if self.step_count % 100 == 0:
                print(f"📍 Step {self.step_count}: Grid({pos[0]:.1f}, {pos[1]:.1f}) | "
                      f"Brain: {self.stats['total_experiences']} exp | "
                      f"Confidence: {self.stats['prediction_confidence']:.3f}")
        
        # Demo complete
        pygame.quit()
        
        print(f"\n✅ 2D Debug demo complete!")
        print(f"   Steps: {self.step_count}")
        print(f"   Collisions: {len(self.collision_points)}")
        print(f"   Final experiences: {self.stats.get('total_experiences', 0)}")
        print(f"   Final confidence: {self.stats.get('prediction_confidence', 0):.3f}")


def main():
    """Run the PiCar-X 2D debug demonstration."""
    
    print("🎯 PiCar-X 2D Grid Debug Demo")
    print("   Large grid view for spotting navigation errors")
    print("   Color-coded robot states show learning progress")
    
    try:
        demo = PiCarX2DDebugDemo(world_size=30)
        demo.run_demo(duration_seconds=120)
        
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