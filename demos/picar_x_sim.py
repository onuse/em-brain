#!/usr/bin/env python3
"""
PiCar-X Simulation Demo

Simple entry point for testing the PiCar-X robot simulation environment.
This connects a simulated PiCar-X robot to the brain server for intelligent navigation.

Usage:
    python3 demos/picar_x_sim.py

Requirements:
    - Brain server running on localhost:8765
    - pygame for visualization (optional)
"""

import sys
import os
import time
import threading
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demos.picar_x_simulation.brainstem import PiCarXBrainstem
from demos.picar_x_simulation.world.environment import SimulationWorld
from demos.picar_x_simulation.vehicle.picar_x_model import PiCarXVehicle
from demos.picar_x_simulation.visualization.renderer import SimulationRenderer

# Try to import 3D renderer
try:
    from demos.picar_x_simulation.visualization.renderer3d import Renderer3D
    RENDERER_3D_AVAILABLE = True
except ImportError:
    RENDERER_3D_AVAILABLE = False


class PiCarXSimulation:
    """
    Main simulation coordinator for PiCar-X robot testing.
    
    Manages the 3D world, vehicle model, brainstem intelligence,
    and visualization systems.
    """
    
    def __init__(self, brain_server_host="localhost", brain_server_port=9999, 
                 use_3d_renderer=True):
        self.brain_server_host = brain_server_host
        self.brain_server_port = brain_server_port
        self.use_3d_renderer = use_3d_renderer and RENDERER_3D_AVAILABLE
        
        # Simulation components
        self.world = None
        self.vehicle = None
        self.brainstem = None
        self.renderer = None
        
        # Simulation state
        self.running = False
        self.simulation_time = 0.0
        self.target_fps = 60
        self.physics_fps = 30
        
        print("🚗 PiCar-X Simulation initializing...")
    
    def initialize(self):
        """Initialize all simulation components."""
        try:
            # Create 3D world environment
            print("🌍 Creating simulation world...")
            self.world = SimulationWorld()
            
            # Create PiCar-X vehicle model
            print("🚙 Creating PiCar-X vehicle...")
            self.vehicle = PiCarXVehicle(
                position=[1.5, 2.0],  # Center of room
                orientation=0.0        # Facing forward
            )
            
            # Add vehicle to world
            self.world.add_vehicle(self.vehicle)
            
            # Create brainstem (robot's autonomous nervous system)
            print("🧠 Initializing brainstem client...")
            self.brainstem = PiCarXBrainstem(
                brain_server_host=self.brain_server_host,
                brain_server_port=self.brain_server_port,
                vehicle=self.vehicle,
                world=self.world
            )
            
            # Create visualization renderer (optional)
            try:
                if self.use_3d_renderer:
                    print("🎮 Initializing 3D wireframe renderer...")
                    self.renderer = Renderer3D(
                        world=self.world,
                        vehicle=self.vehicle,
                        window_size=(1024, 768)
                    )
                    # Give renderer access to simulation for brain stats
                    self.renderer._simulation_ref = self
                    print("📺 3D wireframe visualization enabled")
                else:
                    print("🎮 Initializing 2D renderer...")
                    self.renderer = SimulationRenderer(
                        world=self.world,
                        vehicle=self.vehicle,
                        window_size=(800, 600)
                    )
                    print("📺 2D visualization enabled")
            except ImportError as e:
                print(f"⚠️  Visualization not available: {e}")
                print("Running headless mode")
                self.renderer = None
            
            print("✅ PiCar-X simulation ready!")
            return True
            
        except Exception as e:
            print(f"❌ Simulation initialization failed: {e}")
            return False
    
    def run(self, duration_minutes: float = None):
        """
        Run the simulation.
        
        Args:
            duration_minutes: How long to run (None = infinite)
        """
        if not self.initialize():
            return
        
        print(f"🚀 Starting simulation...")
        print(f"   World: {self.world}")
        print(f"   Vehicle: {self.vehicle}")
        print(f"   Brainstem: Connected to brain server")
        print(f"   Renderer: {'Enabled' if self.renderer else 'Headless'}")
        print()
        
        self.running = True
        start_time = time.time()
        
        # Try initial connection to brain (non-blocking)
        self.brainstem.connect()
        if not self.brainstem.connected:
            print("⚠️ Brain server not available - running in standalone mode")
            print("   Vehicle will use basic obstacle avoidance behavior")
            print("   Start brain server to enable intelligent behavior")
        
        try:
            # Main simulation loop
            last_physics_update = time.time()
            last_render_update = time.time()
            frame_count = 0
            
            while self.running:
                current_time = time.time()
                self.simulation_time = current_time - start_time
                
                # Check duration limit
                if duration_minutes and self.simulation_time > duration_minutes * 60:
                    print(f"⏰ Simulation completed ({duration_minutes} minutes)")
                    break
                
                # Physics update (30 Hz)
                if current_time - last_physics_update >= 1.0 / self.physics_fps:
                    dt = current_time - last_physics_update
                    self.update_physics(dt)
                    last_physics_update = current_time
                
                # Brainstem update (autonomous operation)
                self.brainstem.update()
                
                # Rendering update (60 Hz)
                if self.renderer and current_time - last_render_update >= 1.0 / self.target_fps:
                    should_continue = self.renderer.update()
                    if not should_continue:
                        self.running = False
                    last_render_update = current_time
                
                # Status updates
                frame_count += 1
                if frame_count % 300 == 0:  # Every 5 seconds at 60fps
                    self.print_status()
                
                # Prevent CPU spinning
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n🛑 Simulation interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Simulation error: {e}")
        
        finally:
            self.cleanup()
    
    def update_physics(self, dt: float):
        """Update physics simulation."""
        # Update world physics
        self.world.update(dt)
        
        # Update vehicle dynamics
        self.vehicle.update_physics(dt)
        
        # Check collisions
        self.world.check_collisions()
    
    def print_status(self):
        """Print simulation status."""
        brain_stats = self.brainstem.get_brain_connection_stats()
        vehicle_stats = self.vehicle.get_status()
        
        print(f"⏱️  Time: {self.simulation_time:.1f}s | "
              f"Pos: ({vehicle_stats['position'][0]:.2f}, {vehicle_stats['position'][1]:.2f}) | "
              f"Speed: {vehicle_stats['speed']:.2f}m/s | "
              f"Brain: {brain_stats['experiences']} exp")
    
    def cleanup(self):
        """Clean up simulation resources."""
        print("\n🧹 Cleaning up simulation...")
        
        if self.brainstem:
            self.brainstem.disconnect()
        
        if self.renderer:
            self.renderer.cleanup()
        
        print("✅ Simulation cleanup complete")


def main():
    """Main entry point for PiCar-X simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PiCar-X Robot Simulation")
    parser.add_argument("--host", default="localhost", 
                       help="Brain server host (default: localhost)")
    parser.add_argument("--port", type=int, default=9999,
                       help="Brain server port (default: 9999)")
    parser.add_argument("--duration", type=float, default=None,
                       help="Simulation duration in minutes (default: infinite)")
    parser.add_argument("--headless", action="store_true",
                       help="Run without visualization")
    parser.add_argument("--2d", action="store_true", dest="use_2d",
                       help="Use 2D renderer instead of 3D")
    parser.add_argument("--3d", action="store_true", dest="use_3d",
                       help="Force 3D renderer (default if available)")
    
    args = parser.parse_args()
    
    # Determine renderer preference
    if args.use_2d:
        use_3d_renderer = False
    elif args.use_3d:
        use_3d_renderer = True
    else:
        use_3d_renderer = RENDERER_3D_AVAILABLE  # Use 3D if available
    
    print("🤖 PiCar-X Simulation Environment")
    print("================================")
    print()
    print("This simulation tests the robot brainstem intelligence")
    print("in a virtual environment before deploying to real hardware.")
    print()
    print("📋 Usage:")
    print("  1. Run simulation: python3 demos/picar_x_sim.py")
    print("  2. Optional: Start brain server for intelligent behavior")
    print("     python3 brain_server.py")
    print("  3. Watch the robot navigate and learn!")
    print()
    
    # Display renderer info
    if args.headless:
        print("🖥️  Running in headless mode (no visualization)")
    elif use_3d_renderer:
        print("🎮 3D wireframe visualization enabled")
        print("     Controls: WASD+mouse, 1-4 for views, R to reset")
    else:
        print("📺 2D visualization enabled")
        print("     Controls: R to reset, D for debug, ESC to exit")
    
    if not RENDERER_3D_AVAILABLE:
        print("💡 Install PyOpenGL for 3D visualization: pip install PyOpenGL PyOpenGL_accelerate")
    
    print()
    
    # Create and run simulation
    simulation = PiCarXSimulation(
        brain_server_host=args.host,
        brain_server_port=args.port,
        use_3d_renderer=use_3d_renderer and not args.headless
    )
    
    simulation.run(duration_minutes=args.duration)


if __name__ == "__main__":
    main()