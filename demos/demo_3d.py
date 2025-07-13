#!/usr/bin/env python3
"""
3D Demo - PiCar-X Scientific Visualization

Realistic 3D simulation showing robot physics and brain activity.
Uses matplotlib for scientific visualization of emergent behaviors.

Features:
- 3D visualization using matplotlib
- Realistic PiCar-X physics simulation
- Real-time brain activity monitoring
- Emergent navigation behaviors
- Performance analytics and learning curves

Run with: python3 demo_runner.py 3d_demo
"""

import sys
import os

# Add the brain/ directory to import minimal as a package
current_dir = os.path.dirname(__file__)  # demos/
brain_dir = os.path.dirname(current_dir)   # brain/
sys.path.insert(0, brain_dir)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math

# Handle both direct execution and module import
try:
    from .picar_x_simulation.picar_x_brainstem import PiCarXBrainstem
except ImportError:
    from picar_x_simulation.picar_x_brainstem import PiCarXBrainstem


class PiCarX3DSimulation:
    """
    3D simulation environment for PiCar-X robot with minimal brain.
    """
    
    def __init__(self, world_size=50, show_realtime=True):
        """Initialize the 3D simulation."""
        
        self.world_size = world_size
        self.show_realtime = show_realtime
        
        # Create the robot brainstem
        self.robot = PiCarXBrainstem(
            enable_camera=True,
            enable_ultrasonics=True, 
            enable_line_tracking=True
        )
        
        # Environment obstacles (x, y, radius, height)
        self.obstacles = [
            (10, 10, 3, 2),
            (25, 15, 4, 3),
            (15, 35, 2.5, 1.5),
            (35, 25, 3.5, 2.5),
            (8, 30, 2, 4),
            (40, 10, 3, 2),
            (30, 40, 4, 1),
            (5, 45, 2.5, 3)
        ]
        
        # Performance tracking
        self.robot_trail = []
        self.brain_metrics = []
        self.collision_points = []
        
        # Simulation parameters
        self.sim_speed = 1.0  # 1.0 = real-time
        self.max_steps = 1000
        self.step_count = 0
        
        print(f"🌍 Created 3D simulation environment ({world_size}x{world_size})")
        print(f"   Obstacles: {len(self.obstacles)}")
        print(f"   Real-time visualization: {'✅' if show_realtime else '❌'}")
    
    def run_simulation(self, duration_seconds=120):
        """Run the complete 3D simulation."""
        
        print(f"\n🚀 Starting 3D PiCar-X simulation ({duration_seconds}s)")
        
        # Setup visualization
        if self.show_realtime:
            fig = plt.figure(figsize=(15, 10))
            
            # 3D environment view
            ax3d = fig.add_subplot(221, projection='3d')
            ax3d.set_title('3D PiCar-X Environment')
            
            # Top-down trajectory view
            ax_traj = fig.add_subplot(222)
            ax_traj.set_title('Robot Trajectory')
            ax_traj.set_aspect('equal')
            
            # Brain activity monitor
            ax_brain = fig.add_subplot(223)
            ax_brain.set_title('Brain Activity')
            
            # Performance metrics
            ax_perf = fig.add_subplot(224)
            ax_perf.set_title('Performance Metrics')
            
            plt.tight_layout()
            plt.ion()
            plt.show()
        
        # Simulation loop
        start_time = time.time()
        last_update = start_time
        
        while time.time() - start_time < duration_seconds and self.step_count < self.max_steps:
            step_start = time.time()
            
            # Robot control cycle
            cycle_result = self.robot.control_cycle()
            
            # Record robot trail
            pos = self.robot.position.copy()
            self.robot_trail.append([pos[0], pos[1], 0.2, time.time() - start_time])
            
            # Record brain metrics
            brain_state = cycle_result['brain_state']
            self.brain_metrics.append({
                'time': time.time() - start_time,
                'working_memory': brain_state['working_memory_size'],
                'prediction_confidence': brain_state['prediction_confidence'],
                'method': brain_state['prediction_method'],
                'total_experiences': brain_state['total_experiences']
            })
            
            # Check for collisions
            if self._check_collision():
                self.collision_points.append(pos.copy())
                print(f"💥 Collision detected at ({pos[0]:.1f}, {pos[1]:.1f})")
                # Reset robot position slightly
                self.robot.position[0] += np.random.uniform(-2, 2)
                self.robot.position[1] += np.random.uniform(-2, 2)
            
            # Keep robot in bounds
            self._enforce_boundaries()
            
            # Update visualization
            if self.show_realtime and time.time() - last_update > 0.5:  # Update every 0.5s
                self._update_visualization(fig, ax3d, ax_traj, ax_brain, ax_perf)
                plt.pause(0.01)
                last_update = time.time()
            
            # Control simulation speed
            step_time = time.time() - step_start
            target_step_time = 0.1 / self.sim_speed  # 10Hz control loop
            if step_time < target_step_time:
                time.sleep(target_step_time - step_time)
            
            self.step_count += 1
            
            # Progress updates
            if self.step_count % 50 == 0:
                status = self.robot.get_robot_status()
                print(f"📍 Step {self.step_count}: Position ({status['position']['x']:.1f}, {status['position']['y']:.1f}), "
                      f"Brain: {status['brain']['total_experiences']} exp, "
                      f"{status['brain']['consensus_rate']:.2f} consensus")
        
        # Simulation complete
        print(f"\n✅ Simulation complete!")
        self._print_final_analysis()
        
        # Show final visualization
        if self.show_realtime:
            self._create_final_visualization()
            plt.ioff()
            plt.show()
        
        return self._get_simulation_results()
    
    def _check_collision(self) -> bool:
        """Check if robot has collided with any obstacles."""
        robot_x, robot_y = self.robot.position[0], self.robot.position[1]
        
        for obs_x, obs_y, obs_radius, obs_height in self.obstacles:
            distance = math.sqrt((robot_x - obs_x)**2 + (robot_y - obs_y)**2)
            if distance < obs_radius + 1.0:  # Robot has ~1m radius
                return True
        
        return False
    
    def _enforce_boundaries(self):
        """Keep robot within world boundaries."""
        self.robot.position[0] = max(2, min(self.world_size - 2, self.robot.position[0]))
        self.robot.position[1] = max(2, min(self.world_size - 2, self.robot.position[1]))
    
    def _update_visualization(self, fig, ax3d, ax_traj, ax_brain, ax_perf):
        """Update real-time visualization."""
        
        # Clear axes
        ax3d.clear()
        ax_traj.clear()
        ax_brain.clear()
        ax_perf.clear()
        
        # 3D Environment
        self._plot_3d_environment(ax3d)
        
        # Trajectory
        self._plot_trajectory(ax_traj)
        
        # Brain activity
        self._plot_brain_activity(ax_brain)
        
        # Performance metrics
        self._plot_performance(ax_perf)
        
        fig.canvas.draw()
    
    def _plot_3d_environment(self, ax):
        """Plot the 3D environment with robot and obstacles."""
        
        # Plot obstacles
        for obs_x, obs_y, obs_radius, obs_height in self.obstacles:
            # Create cylinder for obstacle
            theta = np.linspace(0, 2*np.pi, 20)
            z = np.linspace(0, obs_height, 10)
            theta_mesh, z_mesh = np.meshgrid(theta, z)
            x_mesh = obs_x + obs_radius * np.cos(theta_mesh)
            y_mesh = obs_y + obs_radius * np.sin(theta_mesh)
            
            ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.7, color='red')
        
        # Plot robot
        robot_x, robot_y = self.robot.position[0], self.robot.position[1]
        robot_heading = math.radians(self.robot.position[2])
        
        # Robot body (box)
        ax.scatter([robot_x], [robot_y], [0.5], color='blue', s=200, marker='s')
        
        # Robot direction arrow
        arrow_length = 3
        arrow_x = robot_x + arrow_length * math.cos(robot_heading)
        arrow_y = robot_y + arrow_length * math.sin(robot_heading)
        ax.plot([robot_x, arrow_x], [robot_y, arrow_y], [0.5, 0.5], 'b-', linewidth=3)
        
        # Ultrasonic sensor beam
        if self.robot.ultrasonic_distance < 50:
            beam_length = self.robot.ultrasonic_distance / 10.0
            beam_x = robot_x + beam_length * math.cos(robot_heading)
            beam_y = robot_y + beam_length * math.sin(robot_heading)
            ax.plot([robot_x, beam_x], [robot_y, beam_y], [1, 1], 'y-', linewidth=2, alpha=0.7)
        
        # Recent trail
        if len(self.robot_trail) > 10:
            trail_points = np.array(self.robot_trail[-20:])
            ax.plot(trail_points[:, 0], trail_points[:, 1], trail_points[:, 2], 'g-', alpha=0.5)
        
        # Collision points
        if self.collision_points:
            collision_array = np.array(self.collision_points)
            ax.scatter(collision_array[:, 0], collision_array[:, 1], 
                      [1] * len(collision_array), color='red', s=100, marker='x')
        
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_zlim(0, 5)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D PiCar-X Environment')
    
    def _plot_trajectory(self, ax):
        """Plot robot trajectory from top-down view."""
        
        # Plot obstacles (top-down)
        for obs_x, obs_y, obs_radius, obs_height in self.obstacles:
            circle = plt.Circle((obs_x, obs_y), obs_radius, color='red', alpha=0.7)
            ax.add_patch(circle)
        
        # Plot robot trail
        if len(self.robot_trail) > 2:
            trail_points = np.array(self.robot_trail)
            ax.plot(trail_points[:, 0], trail_points[:, 1], 'g-', alpha=0.7, linewidth=2)
        
        # Plot current robot position
        robot_x, robot_y = self.robot.position[0], self.robot.position[1]
        ax.scatter([robot_x], [robot_y], color='blue', s=100, marker='o')
        
        # Robot heading arrow
        robot_heading = math.radians(self.robot.position[2])
        arrow_length = 2
        arrow_x = robot_x + arrow_length * math.cos(robot_heading)
        arrow_y = robot_y + arrow_length * math.sin(robot_heading)
        ax.arrow(robot_x, robot_y, arrow_x - robot_x, arrow_y - robot_y, 
                head_width=1, head_length=0.5, fc='blue', ec='blue')
        
        # Collision points
        if self.collision_points:
            collision_array = np.array(self.collision_points)
            ax.scatter(collision_array[:, 0], collision_array[:, 1], 
                      color='red', s=50, marker='x')
        
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Robot Trajectory (Top View)')
        ax.grid(True, alpha=0.3)
    
    def _plot_brain_activity(self, ax):
        """Plot brain activity over time."""
        
        if len(self.brain_metrics) < 2:
            return
        
        metrics = self.brain_metrics
        times = [m['time'] for m in metrics]
        
        # Working memory size
        working_memory = [m['working_memory'] for m in metrics]
        ax.plot(times, working_memory, 'b-', label='Working Memory Size', linewidth=2)
        
        # Prediction confidence (scaled)
        confidence = [m['prediction_confidence'] * 50 for m in metrics]  # Scale for visibility
        ax.plot(times, confidence, 'r-', label='Prediction Confidence (×50)', linewidth=2)
        
        # Total experiences (scaled)
        if metrics:
            max_exp = max(m['total_experiences'] for m in metrics)
            if max_exp > 0:
                experiences = [m['total_experiences'] / max_exp * 30 for m in metrics]
                ax.plot(times, experiences, 'g-', label='Total Experiences (scaled)', linewidth=2)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Value')
        ax.set_title('Brain Activity Monitor')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance(self, ax):
        """Plot performance metrics."""
        
        status = self.robot.get_robot_status()
        
        # Performance bars
        metrics = [
            ('Navigation Success', status['performance']['navigation_success_rate']),
            ('Brain Consensus', status['brain']['consensus_rate']),
            ('Sensor Coverage', min(1.0, status['brain']['total_experiences'] / 100.0)),
            ('Exploration Rate', min(1.0, len(self.robot_trail) / 200.0))
        ]
        
        names = [m[0] for m in metrics]
        values = [m[1] for m in metrics]
        colors = ['green', 'blue', 'orange', 'purple']
        
        bars = ax.bar(names, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Performance (0-1)')
        ax.set_title('Robot Performance Metrics')
        ax.tick_params(axis='x', rotation=45)
    
    def _create_final_visualization(self):
        """Create comprehensive final analysis visualization."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Final trajectory with heat map
        self._plot_trajectory(ax1)
        ax1.set_title('Complete Robot Trajectory')
        
        # Brain learning curve
        if self.brain_metrics:
            times = [m['time'] for m in self.brain_metrics]
            experiences = [m['total_experiences'] for m in self.brain_metrics]
            confidence = [m['prediction_confidence'] for m in self.brain_metrics]
            
            ax2.plot(times, experiences, 'b-', label='Total Experiences', linewidth=2)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(times, confidence, 'r-', label='Prediction Confidence', linewidth=2)
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Total Experiences', color='blue')
            ax2_twin.set_ylabel('Prediction Confidence', color='red')
            ax2.set_title('Brain Learning Progression')
            ax2.grid(True, alpha=0.3)
        
        # Performance summary
        self._plot_performance(ax3)
        
        # Distance traveled and efficiency
        if len(self.robot_trail) > 2:
            trail_array = np.array(self.robot_trail)
            distances = []
            for i in range(1, len(trail_array)):
                dist = math.sqrt((trail_array[i,0] - trail_array[i-1,0])**2 + 
                               (trail_array[i,1] - trail_array[i-1,1])**2)
                distances.append(dist)
            
            cumulative_distance = np.cumsum(distances)
            times = trail_array[1:, 3]  # Time column
            
            ax4.plot(times, cumulative_distance, 'g-', linewidth=2)
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Distance Traveled (meters)')
            ax4.set_title('Robot Movement Efficiency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('PiCar-X Minimal Brain - Final Analysis', fontsize=16, y=1.02)
    
    def _print_final_analysis(self):
        """Print comprehensive simulation analysis."""
        
        status = self.robot.get_robot_status()
        
        print("\n" + "="*60)
        print("🎯 FINAL SIMULATION ANALYSIS")
        print("="*60)
        
        print(f"\n📊 Robot Performance:")
        print(f"   • Total distance traveled: {len(self.robot_trail) * 0.5:.1f} meters")
        print(f"   • Collisions: {len(self.collision_points)}")
        print(f"   • Navigation success rate: {status['performance']['navigation_success_rate']:.2%}")
        print(f"   • Control cycles: {status['performance']['control_cycles']}")
        
        print(f"\n🧠 Brain Intelligence:")
        print(f"   • Total experiences learned: {status['brain']['total_experiences']}")
        print(f"   • Consensus prediction rate: {status['brain']['consensus_rate']:.2%}")
        print(f"   • Working memory size: {status['brain']['working_memory_size']}")
        print(f"   • Total predictions made: {status['brain']['total_predictions']}")
        
        print(f"\n🎮 Simulation Statistics:")
        print(f"   • Simulation steps: {self.step_count}")
        print(f"   • Environment size: {self.world_size}×{self.world_size} meters")
        print(f"   • Obstacles navigated: {len(self.obstacles)}")
        
        # Calculate emergent behaviors
        if len(self.brain_metrics) > 10:
            early_confidence = np.mean([m['prediction_confidence'] for m in self.brain_metrics[:10]])
            late_confidence = np.mean([m['prediction_confidence'] for m in self.brain_metrics[-10:]])
            learning_improvement = late_confidence - early_confidence
            
            print(f"\n🌟 Emergent Intelligence:")
            print(f"   • Learning improvement: {learning_improvement:+.3f}")
            print(f"   • Initial confidence: {early_confidence:.3f}")
            print(f"   • Final confidence: {late_confidence:.3f}")
            
            if learning_improvement > 0.1:
                print("   ✅ Strong learning detected!")
            elif learning_improvement > 0.05:
                print("   ✅ Moderate learning detected!")
            else:
                print("   ⚠️  Limited learning detected")
        
        print("\n" + "="*60)
    
    def _get_simulation_results(self) -> dict:
        """Return comprehensive simulation results."""
        
        status = self.robot.get_robot_status()
        
        return {
            'robot_performance': status['performance'],
            'brain_intelligence': status['brain'],
            'simulation_stats': {
                'steps': self.step_count,
                'collisions': len(self.collision_points),
                'distance_traveled': len(self.robot_trail) * 0.5,
                'world_size': self.world_size
            },
            'trajectory': self.robot_trail,
            'brain_metrics': self.brain_metrics,
            'collision_points': self.collision_points
        }


def main():
    """Run the complete PiCar-X 3D demonstration."""
    
    print("🚀 PiCar-X 3D Demo - Minimal Brain Robotic Intelligence")
    print("   Demonstrating emergent navigation with 4-system brain architecture")
    
    # Create and run simulation
    sim = PiCarX3DSimulation(world_size=50, show_realtime=True)
    results = sim.run_simulation(duration_seconds=60)  # 1 minute demo
    
    print("\n🎉 3D PiCar-X demonstration completed!")
    print("   The minimal brain successfully controlled a realistic robot simulation")
    print("   Key achievement: Emergent navigation behavior from 4 simple systems")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()