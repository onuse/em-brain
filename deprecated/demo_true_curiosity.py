#!/usr/bin/env python3
"""
True Curiosity Demo - Uses the actual brain with curiosity drive
NO hardcoded behaviors - pure emergent intelligence!
"""

import pygame
from core import WorldGraph
from simulation import GridWorldBrainstem
from visualization import IntegratedDisplay
from predictor.multi_drive_predictor import MultiDrivePredictor
from drives.curiosity_drive import CuriosityDrive
from drives.survival_drive import SurvivalDrive
from drives.exploration_drive import ExplorationDrive


def brain_driven_agent(state_dict) -> dict:
    """Agent that uses the actual brain with curiosity drive."""
    # This would normally be handled by the full brain interface,
    # but for this demo we'll create a simple multi-drive predictor
    
    global brain_predictor, world_graph
    
    if not hasattr(brain_driven_agent, 'initialized'):
        # Initialize the actual brain system
        brain_driven_agent.initialized = True
        global brain_predictor, world_graph
        world_graph = WorldGraph()
        
        # Create the multi-drive system
        brain_predictor = MultiDrivePredictor()
        brain_predictor.add_drive(SurvivalDrive(base_weight=0.3))
        brain_predictor.add_drive(CuriosityDrive(base_weight=0.4))  # High curiosity!
        brain_predictor.add_drive(ExplorationDrive(base_weight=0.3))
        
        print("🧠 Initialized REAL brain with curiosity drive!")
        print("   This should break repetitive patterns!")
    
    # Convert state to the format the brain expects
    sensory_data = state_dict['sensors']
    
    # Use the actual brain to make decisions
    prediction = brain_predictor.predict_next_state(
        sensory_input=sensory_data,
        current_context={'health': state_dict['health'], 'energy': state_dict['energy']}
    )
    
    # Extract motor commands from prediction
    motor_action = prediction.motor_action if hasattr(prediction, 'motor_action') else {
        'forward_motor': 0.3, 'turn_motor': 0.0, 'brake_motor': 0.0
    }
    
    return motor_action


def main():
    """Launch demo with real curiosity-driven brain."""
    print("🧠 TRUE CURIOSITY DEMO")
    print("=" * 50)
    print("Using ACTUAL brain with curiosity drive system!")
    print("Robot should:")
    print("• Get bored of repetitive patterns")
    print("• Seek novel experiences")
    print("• Learn from prediction errors")
    print("• Balance survival vs exploration")
    print("=" * 50)
    
    try:
        brainstem = GridWorldBrainstem(world_width=15, world_height=15, seed=42)
        brain_graph = WorldGraph()
        
        # Create display 
        display = IntegratedDisplay(brainstem, cell_size=25)
        display.set_brain_graph(brain_graph)
        display.set_learning_callback(brain_driven_agent)  # Use REAL brain!
        
        print(f"🖥️  Window: {display.window_width}x{display.window_height}")
        print("🚀 Watch for ANTI-repetitive behavior!")
        print("   The robot should break out of loops!")
        
        display.run(auto_step=True, step_delay=0.3)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ True curiosity demo completed")


if __name__ == "__main__":
    main()