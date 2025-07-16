#!/usr/bin/env python3
"""
Multi-Modal Long-Term Learning Demo

Enhanced 3D demo that fully exercises the brain's potential:
- Multi-modal sensory fusion (vision, ultrasonic, IMU, temperature)
- Long-term learning with 1000+ experiences
- Complex environment with multiple goals
- Adaptive behavior emergence over time
- Real-time brain analytics

This demo pushes the brain to its limits to demonstrate:
- Hierarchical experience indexing at scale
- Natural attention mechanisms across modalities
- Similarity learning adaptation
- Meta-learning parameter evolution
- Memory consolidation effects

Run: python3 demo_runner.py multimodal
"""

import sys
import os
import math
import time
import random
from typing import Dict, List, Any, Tuple

current_dir = os.path.dirname(__file__)
brain_dir = os.path.dirname(current_dir)
sys.path.insert(0, brain_dir)

import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

try:
    from .picar_x_simulation.socket_embodied_brainstem import SocketEmbodiedBrainstem
    from .demo_3d_embodied import Camera3D, Scene3D, HUD
except ImportError:
    from picar_x_simulation.socket_embodied_brainstem import SocketEmbodiedBrainstem
    from demo_3d_embodied import Camera3D, Scene3D, HUD


class MultiModalSensorSuite:
    """Enhanced sensor suite with multiple modalities."""
    
    def __init__(self, brainstem):
        self.brainstem = brainstem
        self.vision_history = []
        self.imu_history = []
        self.temperature_sensors = {'ambient': 25.0, 'cpu': 35.0}
        
    def get_enhanced_sensory_vector(self) -> List[float]:
        """Get 16D multi-modal sensory vector."""
        
        # Base sensors (4D)
        base = self.brainstem.get_sensory_vector()
        
        # Vision processing (4D) - simulate edge detection, color analysis
        vision = self._process_vision()
        
        # IMU data (4D) - acceleration, gyroscope
        imu = self._process_imu()
        
        # Environmental (4D) - temperature, humidity, light, sound
        environmental = self._process_environmental()
        
        return base + vision + imu + environmental
    
    def _process_vision(self) -> List[float]:
        """Simulate vision processing."""
        pos = self.brainstem.position
        
        # Simulate visual features based on position and obstacles
        edge_density = abs(math.sin(pos[0] * 0.1)) * 0.5 + 0.3
        color_variance = abs(math.cos(pos[1] * 0.1)) * 0.4 + 0.2
        motion_blur = min(1.0, abs(self.brainstem.motor_speed) / 50.0)
        brightness = 0.7 + 0.3 * math.sin(time.time() * 0.1)  # Simulated lighting
        
        vision = [edge_density, color_variance, motion_blur, brightness]
        self.vision_history.append(vision)
        if len(self.vision_history) > 10:
            self.vision_history.pop(0)
            
        return vision
    
    def _process_imu(self) -> List[float]:
        """Simulate IMU processing."""
        # Simulate acceleration and rotation based on movement
        accel_x = self.brainstem.motor_speed * 0.01 + random.gauss(0, 0.02)
        accel_y = abs(self.brainstem.steering_angle) * 0.005 + random.gauss(0, 0.01)
        gyro_z = self.brainstem.steering_angle * 0.1 + random.gauss(0, 0.05)
        vibration = abs(self.brainstem.motor_speed) * 0.008 + random.gauss(0, 0.01)
        
        imu = [accel_x, accel_y, gyro_z, vibration]
        self.imu_history.append(imu)
        if len(self.imu_history) > 10:
            self.imu_history.pop(0)
            
        return imu
    
    def _process_environmental(self) -> List[float]:
        """Simulate environmental sensors."""
        # Temperature varies with motor activity
        motor_heat = abs(self.brainstem.motor_speed) * 0.2
        self.temperature_sensors['ambient'] = 25.0 + motor_heat + random.gauss(0, 1.0)
        self.temperature_sensors['cpu'] = 35.0 + motor_heat * 1.5 + random.gauss(0, 2.0)
        
        # Simulate other environmental factors
        humidity = 0.6 + 0.2 * math.sin(time.time() * 0.05)
        light_level = 0.8 + 0.2 * math.sin(time.time() * 0.02)
        sound_level = 0.3 + abs(self.brainstem.motor_speed) * 0.01
        
        return [
            self.temperature_sensors['ambient'] / 50.0,  # Normalize
            humidity,
            light_level, 
            sound_level
        ]


class ComplexEnvironment:
    """Complex environment with multiple goals and challenges."""
    
    def __init__(self):
        self.goals = [
            {'pos': (100, 100), 'type': 'energy', 'reward': 0.8, 'active': True},
            {'pos': (-80, 120), 'type': 'data', 'reward': 0.6, 'active': True},
            {'pos': (150, -90), 'type': 'shelter', 'reward': 0.7, 'active': True},
            {'pos': (-120, -60), 'type': 'social', 'reward': 0.5, 'active': True}
        ]
        
        self.dynamic_obstacles = [
            {'pos': [0, 80], 'velocity': [10, 0], 'size': 20},
            {'pos': [60, -40], 'velocity': [-5, 15], 'size': 15}
        ]
        
        self.environmental_changes = {
            'weather': 'clear',
            'time_of_day': 0.5,
            'difficulty': 1.0
        }
    
    def update(self, dt: float):
        """Update dynamic environment."""
        # Move dynamic obstacles
        for obs in self.dynamic_obstacles:
            obs['pos'][0] += obs['velocity'][0] * dt
            obs['pos'][1] += obs['velocity'][1] * dt
            
            # Bounce off boundaries
            if abs(obs['pos'][0]) > 200:
                obs['velocity'][0] *= -1
            if abs(obs['pos'][1]) > 200:
                obs['velocity'][1] *= -1
        
        # Environmental changes
        self.environmental_changes['time_of_day'] = (time.time() * 0.01) % 1.0
        
        # Randomly activate/deactivate goals
        if random.random() < 0.01:  # 1% chance per update
            goal = random.choice(self.goals)
            goal['active'] = not goal['active']
    
    def get_nearest_goal(self, robot_pos: Tuple[float, float]) -> Dict:
        """Get nearest active goal."""
        active_goals = [g for g in self.goals if g['active']]
        if not active_goals:
            return None
            
        distances = []
        for goal in active_goals:
            dist = math.sqrt((robot_pos[0] - goal['pos'][0])**2 + 
                           (robot_pos[1] - goal['pos'][1])**2)
            distances.append((dist, goal))
        
        return min(distances, key=lambda x: x[0])[1]


class EnhancedMultiModalBrainstem(SocketEmbodiedBrainstem):
    """Enhanced brainstem with multi-modal sensors and long-term learning."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_suite = MultiModalSensorSuite(self)
        self.environment = ComplexEnvironment()
        self.learning_session_start = time.time()
        self.experience_milestones = [100, 500, 1000, 2000]
        self.reached_milestones = set()
        
    def get_sensory_vector(self) -> List[float]:
        """Override to use multi-modal sensors."""
        return self.sensor_suite.get_enhanced_sensory_vector()
    
    def control_cycle(self) -> Dict[str, Any]:
        """Enhanced control cycle with environment updates."""
        # Update environment
        self.environment.update(0.033)  # 30 FPS
        
        # Run base control cycle
        result = super().control_cycle()
        
        # Add multi-modal information
        if result.get('success'):
            result['multimodal_sensors'] = {
                'vision_features': len(self.sensor_suite.vision_history),
                'imu_stability': np.std([h[3] for h in self.sensor_suite.imu_history[-5:]]) if len(self.sensor_suite.imu_history) >= 5 else 0,
                'temperature_gradient': self.sensor_suite.temperature_sensors['cpu'] - self.sensor_suite.temperature_sensors['ambient'],
                'environmental_complexity': len([g for g in self.environment.goals if g['active']])
            }
            
            # Check for learning milestones
            if hasattr(self, 'brain_client') and self.brain_client:
                self._check_learning_milestones()
        
        return result
    
    def _check_learning_milestones(self):
        """Check and report learning milestones."""
        # This would require brain server to report experience count
        # For now, estimate based on time and cycles
        estimated_experiences = self.total_control_cycles * 0.8  # Rough estimate
        
        for milestone in self.experience_milestones:
            if milestone not in self.reached_milestones and estimated_experiences >= milestone:
                self.reached_milestones.add(milestone)
                print(f"🎯 LEARNING MILESTONE: {milestone} experiences reached!")
                print(f"   Session time: {time.time() - self.learning_session_start:.1f}s")
                print(f"   Cycles completed: {self.total_control_cycles}")


class EnhancedHUD(HUD):
    """Enhanced HUD with multi-modal and learning analytics."""
    
    def draw_multimodal_hud(self, brainstem, cycle_info):
        """Draw enhanced HUD with multi-modal information."""
        # Draw base embodied HUD
        self.draw_embodied_hud(brainstem, cycle_info)
        
        # Add multi-modal section
        y_offset = 400
        line_height = 20
        
        self.render_text_3d("🔬 MULTI-MODAL SENSORS", self.screen_width - 300, y_offset)
        y_offset += line_height * 1.5
        
        multimodal = cycle_info.get('multimodal_sensors', {})
        
        # Vision system
        vision_features = multimodal.get('vision_features', 0)
        self.render_text_3d(f"👁️ Vision: {vision_features} features", self.screen_width - 300, y_offset)
        y_offset += line_height
        
        # IMU stability
        imu_stability = multimodal.get('imu_stability', 0)
        stability_color = (0, 1, 0) if imu_stability < 0.1 else (1, 1, 0) if imu_stability < 0.2 else (1, 0, 0)
        self.render_text_3d(f"📐 IMU Stability: {imu_stability:.3f}", self.screen_width - 300, y_offset, stability_color)
        y_offset += line_height
        
        # Temperature gradient
        temp_gradient = multimodal.get('temperature_gradient', 0)
        self.render_text_3d(f"🌡️ Temp Δ: {temp_gradient:.1f}°C", self.screen_width - 300, y_offset)
        y_offset += line_height
        
        # Environment complexity
        complexity = multimodal.get('environmental_complexity', 0)
        self.render_text_3d(f"🌍 Active Goals: {complexity}", self.screen_width - 300, y_offset)
        y_offset += line_height * 2
        
        # Learning progress
        self.render_text_3d("📈 LEARNING PROGRESS", self.screen_width - 300, y_offset)
        y_offset += line_height * 1.5
        
        session_time = time.time() - brainstem.learning_session_start
        estimated_exp = brainstem.total_control_cycles * 0.8
        
        self.render_text_3d(f"⏱️ Session: {session_time/60:.1f}min", self.screen_width - 300, y_offset)
        y_offset += line_height
        self.render_text_3d(f"🧠 Est. Experiences: {estimated_exp:.0f}", self.screen_width - 300, y_offset)
        y_offset += line_height
        self.render_text_3d(f"🎯 Milestones: {len(brainstem.reached_milestones)}/4", self.screen_width - 300, y_offset)


class MultiModalDemo3D:
    """Enhanced 3D demo with multi-modal sensors and long-term learning."""
    
    def __init__(self):
        pygame.init()
        
        self.screen_width = 1400
        self.screen_height = 900
        pygame.display.set_mode((self.screen_width, self.screen_height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Multi-Modal Long-Term Learning Demo")
        
        gluPerspective(45, self.screen_width / self.screen_height, 0.1, 1000.0)
        
        self.camera = Camera3D()
        self.hud = EnhancedHUD(self.screen_width, self.screen_height)
        self.scene = Scene3D()
        
        self.brainstem = EnhancedMultiModalBrainstem()
        self.brain_connected = False
        
        self.paused = False
        self.verbose = False
        self.clock = pygame.time.Clock()
        self.last_cycle_info = {}
        
        print("🔬 Multi-Modal Long-Term Learning Demo initialized")
        print("   16D sensory input, complex environment, 1000+ experience target")
    
    def handle_events(self):
        """Handle events (same as base demo)."""
        mouse_rel = pygame.mouse.get_rel()
        mouse_buttons = pygame.mouse.get_pressed()
        self.camera.update_from_mouse(mouse_rel, mouse_buttons)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.camera.reset()
                elif event.key == pygame.K_v:
                    self.verbose = not self.verbose
                    self.brainstem.set_verbose_embodied(self.verbose)
                elif event.key == pygame.K_c:
                    if not self.brain_connected:
                        if self.brainstem.connect_to_brain_server():
                            self.brain_connected = True
                            print("✅ Connected to brain server!")
                        else:
                            print("❌ Failed to connect - is brain server running?")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.camera.zoom(-1)
                elif event.button == 5:
                    self.camera.zoom(1)
        
        return True
    
    def update(self):
        """Update simulation."""
        if not self.paused:
            cycle_info = self.brainstem.control_cycle()
            self.last_cycle_info = cycle_info
    
    def render(self):
        """Render enhanced scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.camera.apply_transform()
        
        # Draw enhanced scene
        self.scene.draw_ground()
        self.scene.draw_obstacles()
        self._draw_environment_goals()
        self.scene.draw_robot(self.brainstem)
        self.scene.draw_ultrasonic_beam(self.brainstem)
        
        # Draw enhanced HUD
        self.hud.draw_multimodal_hud(self.brainstem, self.last_cycle_info)
        
        pygame.display.flip()
    
    def _draw_environment_goals(self):
        """Draw environment goals."""
        for goal in self.brainstem.environment.goals:
            if not goal['active']:
                continue
                
            x, y = goal['pos']
            goal_type = goal['type']
            
            # Color by goal type
            colors = {
                'energy': (1, 1, 0),    # Yellow
                'data': (0, 0, 1),      # Blue  
                'shelter': (0, 1, 0),   # Green
                'social': (1, 0, 1)     # Magenta
            }
            
            glColor3f(*colors.get(goal_type, (1, 1, 1)))
            
            # Draw goal marker
            glPushMatrix()
            glTranslatef(x, 10, y)
            
            # Simple pyramid marker
            glBegin(GL_TRIANGLES)
            for i in range(4):
                angle1 = i * math.pi / 2
                angle2 = (i + 1) * math.pi / 2
                
                x1 = 8 * math.cos(angle1)
                z1 = 8 * math.sin(angle1)
                x2 = 8 * math.cos(angle2)
                z2 = 8 * math.sin(angle2)
                
                glVertex3f(0, 10, 0)    # Top
                glVertex3f(x1, 0, z1)   # Base
                glVertex3f(x2, 0, z2)   # Base
            glEnd()
            
            glPopMatrix()
    
    def run(self):
        """Run enhanced demo."""
        print("🚀 Starting Multi-Modal Long-Term Learning Demo...")
        print("⚠️  Requires brain server: python3 demo_runner.py server")
        print("🎯 Target: 1000+ experiences with multi-modal learning")
        
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(30)
        
        if self.brain_connected:
            self.brainstem.disconnect_from_brain_server()
        
        # Final learning report
        print("\n" + "="*60)
        print("🔬 MULTI-MODAL LEARNING SESSION COMPLETE")
        print(f"   Session duration: {(time.time() - self.brainstem.learning_session_start)/60:.1f} minutes")
        print(f"   Control cycles: {self.brainstem.total_control_cycles}")
        print(f"   Milestones reached: {len(self.brainstem.reached_milestones)}/4")
        print(f"   Estimated experiences: {self.brainstem.total_control_cycles * 0.8:.0f}")
        
        pygame.quit()


def main():
    """Run the multi-modal demo."""
    try:
        demo = MultiModalDemo3D()
        demo.run()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()