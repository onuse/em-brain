#!/usr/bin/env python3
"""
Simple GUI Demo - Direct Launch
Boots the visualization system immediately without prompts.
"""

import sys
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
    """Launch the GUI directly."""
    print("🎮 EMERGENT INTELLIGENCE ROBOT - GUI DEMO")
    print("=" * 50)
    print("Launching visualization system...")
    print()
    print("Controls:")
    print("- SPACE: Pause/Resume")
    print("- R: Reset robot")
    print("- S: Toggle sensor rays")
    print("- F1: Full help")
    print("- ESC: Exit")
    print("=" * 50)
    
    try:
        # Create components
        brainstem = GridWorldBrainstem(world_width=20, world_height=20, seed=42)
        brain_graph = WorldGraph()
        
        # Create visualization
        display = IntegratedDisplay(brainstem, cell_size=25)
        display.set_brain_graph(brain_graph)
        display.set_learning_callback(simple_learning_agent)
        
        print("🚀 Starting GUI... (Window should open)")
        
        # Run visualization
        display.run(auto_step=True, step_delay=0.15)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("👋 GUI Demo completed")


if __name__ == "__main__":
    main()