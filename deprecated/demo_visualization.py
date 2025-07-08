#!/usr/bin/env python3
"""
Visualization Demo for Emergent Intelligence Robot Brain

Demonstrates the complete multi-panel visualization system:
- Grid World (robot movement, environment, sensors)
- Brain State (memory graph, statistics, learning progress)
- Live Log (real-time events and learning activities)

This showcases the "controlled terminal output spam" approach with
a visual game interface for comprehensive system monitoring.
"""

import random
import time
from core import WorldGraph
from simulation import GridWorldBrainstem
from visualization import IntegratedDisplay


def simple_learning_agent(state_dict) -> dict:
    """
    A simple learning agent that demonstrates basic behavior patterns.
    This would be replaced by the actual emergent intelligence system.
    """
    sensors = state_dict['sensors']
    health = state_dict['health']
    energy = state_dict['energy']
    
    # Extract sensor information
    distance_sensors = sensors[:4]  # front, left, right, back
    vision_features = sensors[4:17]
    internal_state = sensors[17:22]
    
    # Simple reactive behaviors
    front_distance = distance_sensors[0]
    left_distance = distance_sensors[1]
    right_distance = distance_sensors[2]
    
    # Decision making
    forward_motor = 0.0
    turn_motor = 0.0
    brake_motor = 0.0
    
    # Obstacle avoidance
    if front_distance < 0.3:  # Obstacle ahead
        if left_distance > right_distance:
            turn_motor = -0.7  # Turn left
        else:
            turn_motor = 0.7   # Turn right
        brake_motor = 0.3
    else:
        # Move forward
        forward_motor = 0.4
        
        # Slight random turning to explore
        if random.random() < 0.1:
            turn_motor = random.uniform(-0.3, 0.3)
    
    # Emergency behaviors
    if health < 0.3:
        # Very cautious when low health
        forward_motor *= 0.5
        brake_motor = max(brake_motor, 0.2)
    
    if energy < 0.2:
        # Slow down when low energy
        forward_motor *= 0.3
    
    return {
        'forward_motor': forward_motor,
        'turn_motor': turn_motor,
        'brake_motor': brake_motor
    }


def random_exploration_agent() -> dict:
    """Simple random exploration for comparison."""
    return {
        'forward_motor': random.uniform(-0.2, 0.6),  # Bias toward forward
        'turn_motor': random.uniform(-0.4, 0.4),
        'brake_motor': random.uniform(0.0, 0.1)
    }


def demo_visualization_system():
    """Run the complete visualization demo."""
    print("🎮 EMERGENT INTELLIGENCE ROBOT - VISUALIZATION DEMO")
    print("=" * 60)
    print()
    print("Launching multi-panel visualization system:")
    print("- Left Panel: Grid World with real-time robot navigation")  
    print("- Right Panel: Brain State with memory graph and learning log")
    print("- Status Bar: System information and controls")
    print()
    print("This demo shows the 'controlled terminal output spam' approach")
    print("with visual game interface for comprehensive monitoring!")
    print()
    print("Press F1 in the game window for full controls help.")
    print("=" * 60)
    
    # Create simulation components
    print("\n🧠 Initializing brain and simulation...")
    brainstem = GridWorldBrainstem(world_width=20, world_height=20, seed=42)
    brain_graph = WorldGraph()
    
    # Create integrated display
    print("🎮 Setting up visualization system...")
    display = IntegratedDisplay(brainstem, cell_size=25)
    display.set_brain_graph(brain_graph)
    
    # Choose behavior mode
    print("\n🤖 Select robot behavior:")
    print("1. Simple Learning Agent (reactive behaviors)")
    print("2. Random Exploration (baseline)")
    print("3. Manual Control (use SPACE to pause and step)")
    
    choice = input("Enter choice (1-3, or just press Enter for #1): ").strip()
    
    if choice == "2":
        print("🎲 Using random exploration agent")
        display.set_action_generator(random_exploration_agent)
        step_delay = 0.2  # Faster for random
    elif choice == "3":
        print("⏸️  Manual control mode - use SPACE and ENTER to control")
        step_delay = 0.5
        # No action generator - will use built-in random
    else:
        print("🧠 Using simple learning agent")
        display.set_learning_callback(simple_learning_agent)
        step_delay = 0.15  # Moderate speed for learning
    
    print(f"\n🚀 Starting visualization with {step_delay}s step delay...")
    print("Window should open shortly. Press F1 for help, ESC to exit.")
    
    try:
        # Run the visualization
        display.run(auto_step=True, step_delay=step_delay)
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        display.cleanup()
        print("\n✅ Visualization demo completed")
        
        # Show final statistics
        final_state = display.get_simulation_state()
        print(f"\nFinal Statistics:")
        print(f"- Simulation steps: {final_state['brainstem_stats']['step_count']}")
        print(f"- Robot health: {final_state['brainstem_stats']['robot_health']:.2f}")
        print(f"- Robot energy: {final_state['brainstem_stats']['robot_energy']:.2f}")
        print(f"- Food consumed: {final_state['brainstem_stats']['total_food_consumed']}")
        print(f"- Collisions: {final_state['brainstem_stats']['total_collisions']}")
        print(f"- Memory nodes: {final_state['brain_nodes']}")
        
        if final_state['brain_stats']:
            print(f"- Memory merges: {final_state['brain_stats']['total_merges']}")
            print(f"- Average strength: {final_state['brain_stats']['avg_strength']:.2f}")


def quick_test():
    """Quick test to verify everything works without GUI."""
    print("🔧 Running quick functionality test...")
    
    # Test components
    brainstem = GridWorldBrainstem(world_width=10, world_height=10, seed=123)
    brain_graph = WorldGraph()
    
    # Test simulation steps
    for i in range(5):
        sensor_packet = brainstem.get_sensor_readings()
        action = simple_learning_agent({
            'sensors': sensor_packet.sensor_values,
            'position': brainstem.simulation.robot.position,
            'orientation': brainstem.simulation.robot.orientation,
            'health': brainstem.simulation.robot.health,
            'energy': brainstem.simulation.robot.energy
        })
        
        brainstem.execute_motor_commands(action)
        print(f"  Step {i+1}: Position {brainstem.simulation.robot.position}, "
              f"Health {brainstem.simulation.robot.health:.2f}")
    
    print("✅ Quick test passed - all components working!")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        try:
            demo_visualization_system()
        except ImportError as e:
            if "pygame" in str(e):
                print("❌ PyGame not available - running quick test instead")
                quick_test()
            else:
                raise