#!/usr/bin/env python3
"""
Brain Monitor Demo - Shows both grid world and brain state
"""

import pygame
from core import WorldGraph
from simulation import GridWorldBrainstem
from visualization import IntegratedDisplay


def simple_learning_agent(state_dict) -> dict:
    """Simple reactive learning agent."""
    sensors = state_dict['sensors']
    
    # Extract sensor information
    distance_sensors = sensors[:4]  # front, left, right, back
    
    # Simple reactive behaviors
    front_distance = distance_sensors[0]
    left_distance = distance_sensors[1] 
    right_distance = distance_sensors[2]
    
    # Decision making
    if front_distance < 0.3:  # Obstacle ahead
        if left_distance > right_distance:
            return {'forward_motor': 0.0, 'turn_motor': -0.7, 'brake_motor': 0.3}  # Turn left
        else:
            return {'forward_motor': 0.0, 'turn_motor': 0.7, 'brake_motor': 0.3}   # Turn right
    else:
        # Move forward with occasional exploration
        import random
        turn = random.uniform(-0.2, 0.2) if random.random() < 0.1 else 0.0
        return {'forward_motor': 0.4, 'turn_motor': turn, 'brake_motor': 0.0}


def main():
    """Launch demo with brain monitoring."""
    print("🧠 Brain Monitor Demo")
    print("=" * 40)
    print("Showing grid world + brain state monitor")
    print()
    
    # Create smaller world to ensure window fits
    brainstem = GridWorldBrainstem(world_width=12, world_height=12, seed=42)
    brain_graph = WorldGraph()
    
    # Calculate window size info
    cell_size = 25
    grid_width = brainstem.simulation.width * cell_size
    brain_panel_width = 400
    total_width = grid_width + brain_panel_width
    
    print(f"Expected window size: {total_width}x{max(grid_width, 600)}")
    print()
    print("Controls:")
    print("• SPACE: Pause/Resume")
    print("• R: Reset")
    print("• S: Toggle sensor rays")
    print("• ESC: Exit")
    print("=" * 40)
    
    try:
        # Create display
        display = IntegratedDisplay(brainstem, cell_size=cell_size)
        display.set_brain_graph(brain_graph)
        display.set_learning_callback(simple_learning_agent)
        
        print(f"🖥️  Actual window: {display.window_width}x{display.window_height}")
        print("🚀 Launching GUI...")
        
        # Run visualization
        display.run(auto_step=True, step_delay=0.2)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Demo completed")


if __name__ == "__main__":
    main()