#!/usr/bin/env python3
"""
Complete Brain System Visual Demo
Shows the complete brain system working in real-time with visual feedback:
- 2D Grid World with robot navigation
- Brain state monitoring (memory, adaptation, actuator discovery)
- Real-time learning visualization
"""

import time
from simulation.brainstem_sim import GridWorldBrainstem
from visualization.integrated_display import IntegratedDisplay


def run_complete_brain_visual_demo():
    """Run the complete brain system with visual interface."""
    print("🧠 Complete Brain System Visual Demo")
    print("==================================")
    print("Launching visual interface with:")
    print("• Unified emergent memory system")
    print("• Adaptive parameter tuning")
    print("• Universal actuator discovery")
    print("• Persistent memory (lifelong learning)")
    print("• Multi-drive motivation system")
    print()
    print("🎮 Controls:")
    print("• Watch the robot learn to navigate autonomously")
    print("• Brain state updates in real-time on the right panel")
    print("• Close window or press ESC to exit")
    print()
    
    # Create 2D world with complete brain system
    brainstem = GridWorldBrainstem(
        world_width=15, 
        world_height=15, 
        seed=42, 
        use_sockets=False  # Use local brain directly
    )
    
    # Start memory session for persistent learning
    session_id = brainstem.brain_client.start_memory_session("Visual Demo - Complete Brain System")
    print(f"📝 Started memory session: {session_id}")
    
    # Initialize visualization
    display = IntegratedDisplay(brainstem, cell_size=30)
    
    print("🚀 Launching visual interface...")
    print("   Watch the robot's brain learn in real-time!")
    
    try:
        # Run the visual simulation
        display.run(
            auto_step=True,
            step_delay=0.5  # Slower for better observation
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        # Save final brain state
        print("\n💾 Saving brain state...")
        save_result = brainstem.brain_client.save_current_state()
        if save_result:
            print(f"   Saved {save_result['experiences_count']} experiences")
        
        # End memory session
        session_summary = brainstem.brain_client.end_memory_session()
        if session_summary:
            print(f"   Session: {session_summary['session_id']}")
            print(f"   Total adaptations: {session_summary['total_adaptations']}")
        
        # Print final brain statistics
        stats = brainstem.brain_client.get_brain_statistics()
        print(f"\n📊 Final Brain Statistics:")
        print(f"   Experiences: {stats['graph_stats']['total_nodes']}")
        print(f"   Adaptations: {stats['adaptive_tuning_stats']['total_adaptations']}")
        print(f"   Actuators discovered: {stats['actuator_discovery_stats']['total_actuators_discovered']}")
        print(f"   Categories formed: {stats['actuator_discovery_stats']['emergent_categories_formed']}")
        
        # Show discovered actuator categories
        categories = brainstem.brain_client.get_discovered_actuator_categories()
        if categories:
            print(f"   Emergent actuator categories:")
            for category_id, category_data in categories.items():
                properties = category_data['emergent_properties']
                print(f"     {category_id}: {category_data['member_actuators']}")
                if properties['appears_spatial']:
                    print(f"       -> Appears to affect spatial movement")
                if properties['appears_manipulative']:
                    print(f"       -> Appears to affect object manipulation")
                if properties['appears_environmental']:
                    print(f"       -> Appears to affect environment")
        
        print(f"\n🧠 Brain learning complete! All experiences saved for next session.")


def main():
    """Launch the complete brain visual demo."""
    print("🌟 Complete Brain System Visual Demo")
    print("===================================")
    print()
    print("This demo shows all brain capabilities working together:")
    print("• Robot learns to navigate through experience")
    print("• Memory accumulates and creates emergent phenomena")
    print("• Parameters adapt based on prediction accuracy")
    print("• Actuator effects discovered through correlation")
    print("• All learning persists for future sessions")
    print()
    print("The visualization shows:")
    print("• Left: 2D grid world with robot, food, dangers")
    print("• Right: Brain state (memory, learning, statistics)")
    print()
    
    try:
        run_complete_brain_visual_demo()
        print("✅ Demo completed successfully!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()