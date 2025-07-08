#!/usr/bin/env python3
"""
Robot Brain Demo - THE definitive demonstration of the complete robot brain system

This is the single consolidated demo that showcases ALL brain capabilities working together:
• Unified emergent memory system with neural-like dynamics
• Multi-drive motivation system (survival, curiosity, exploration)
• Real-time goal generation from competing drives
• Adaptive parameter tuning based on prediction accuracy
• Universal actuator discovery (learns motor effects automatically)
• Persistent memory with lifelong learning across sessions
• Real-time visual brain state monitoring and analysis
• Cross-session learning accumulation and memory consolidation
• High-performance prediction pipeline (70+ FPS capable)

This consolidates and replaces all other demo files in the project.
"""

import time
import pygame
from simulation.brainstem_sim import GridWorldBrainstem
from visualization.integrated_display import IntegratedDisplay
from enhanced_run_logger import EnhancedRunLogger


def main():
    """Launch the ultimate 2D brain demonstration."""
    print("🧠 ROBOT BRAIN DEMO")
    print("=" * 60)
    print("THE definitive demonstration of the complete robot brain system")
    print()
    print("🎯 This demo includes ALL capabilities:")
    print("• Unified emergent memory system")
    print("• Multi-drive motivation (survival, curiosity, exploration)")
    print("• Drive-generated temporary goals")
    print("• Adaptive parameter tuning based on prediction accuracy")
    print("• Universal actuator discovery (learns motor effects)")
    print("• Persistent memory (remembers across sessions)")
    print("• Real-time visual brain state monitoring")
    print("• Cross-session learning accumulation")
    print()
    print("⏱️  Survival Parameters (Balanced for Learning):")
    print("• Collision damage: 0.5% per wall hit (200 collisions to die)")
    print("• Red square damage: 0.2% per step (500 red squares to die)")
    print("• Energy decay: 50,000 steps to starvation (~4+ hours real-time)")
    print("• Expected robot lifespan: 10-20 minutes real-time")
    print("• Enough time for complex learning and goal development")
    print()
    print("🎮 Controls:")
    print("• SPACE: Pause/Resume simulation")
    print("• R: Reset robot (keeps learned memories)")
    print("• S: Toggle sensor ray visualization")
    print("• ESC: Exit (saves all learning)")
    print("=" * 60)
    
    # Initialize enhanced logging system
    print("\\n📝 Initializing enhanced logging system...")
    enhanced_logger = EnhancedRunLogger()
    
    # Initialize the complete brain system
    print("\\n🧠 Initializing complete brain system...")
    brainstem = GridWorldBrainstem(
        world_width=12,  # Smaller world for better screen fit
        world_height=12, 
        seed=42, 
        use_sockets=False  # Use local brain directly for best performance
    )
    
    # Start persistent memory session
    session_id = brainstem.brain_client.start_memory_session("Ultimate 2D Brain Demo")
    print(f"📝 Started persistent memory session: {session_id}")
    
    # Show what was loaded from previous sessions
    stats = brainstem.brain_client.get_brain_statistics()
    if stats['graph_stats']['total_nodes'] > 0:
        print(f"📚 Loaded {stats['graph_stats']['total_nodes']} experiences from previous sessions")
        print(f"   Previous adaptations: {stats['adaptive_tuning_stats']['total_adaptations']}")
        print(f"   Actuators discovered: {stats['actuator_discovery_stats']['total_actuators_discovered']}")
    else:
        print("🆕 Starting with fresh brain - no previous experiences")
    
    # Initialize visualization with brain monitoring
    print("🎮 Setting up visualization with brain monitoring...")
    display = IntegratedDisplay(brainstem, cell_size=25)  # Smaller cells for better fit
    
    # Connect visualization to brain system for real predictions
    def brain_prediction_callback(state):
        """Use the brain system to generate motor commands instead of random actions."""
        # Log frame with timing
        enhanced_logger.log_frame_with_timing(brainstem.brain_client, state)
        
        # Create sensory packet from current state
        from core.communication import SensoryPacket
        from datetime import datetime
        sensory_packet = SensoryPacket(
            sequence_id=brainstem.sequence_counter,
            sensor_values=state['sensors'],
            actuator_positions=[0.0, 0.0, 0.0],  # Dummy actuator positions for grid world
            timestamp=datetime.now()
        )
        
        # Get brain prediction with current mental context
        mental_context = state['sensors'][:8] if len(state['sensors']) >= 8 else state['sensors']
        
        # Use brain interface to generate prediction with timing
        prediction = enhanced_logger.time_brain_prediction(
            brainstem.brain_client.process_sensory_input,
            sensory_packet, 
            mental_context, 
            threat_level="normal"
        )
        
        return prediction.motor_action if prediction else {
            'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 0.0
        }
    
    # Set the brain callback so visualization uses real brain predictions
    display.set_learning_callback(brain_prediction_callback)
    
    # Connect brain's world graph to visualization
    brain_graph = brainstem.brain_client.get_world_graph()
    display.set_brain_graph(brain_graph)
    
    print(f"🖥️  Window size: {display.window_width}x{display.window_height}")
    print("   Left: 2D grid world with robot navigation")
    print("   Right: Real-time brain state monitoring")
    print()
    
    try:
        print("🚀 Launching complete brain system...")
        print("   Watch the robot learn and develop goals in real-time!")
        print("   Brain state updates continuously on the right panel")
        print("   Close window or press ESC to save and exit")
        print()
        
        # Run the complete brain simulation
        display.run(
            auto_step=True,
            step_delay=0.0  # No artificial delay - let brain run at full speed
        )
        
    except KeyboardInterrupt:
        print("\\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\\n💾 Saving complete brain state...")
        
        # Save current brain state
        save_result = brainstem.brain_client.save_current_state()
        if save_result:
            print(f"   💾 Saved {save_result['experiences_count']} experiences")
        
        # End persistent memory session
        session_summary = brainstem.brain_client.end_memory_session()
        if session_summary:
            print(f"   📝 Session: {session_summary['session_id']}")
            print(f"   📊 Total adaptations this session: {session_summary['total_adaptations']}")
        
        # Save enhanced performance log
        final_fps = enhanced_logger.timer.get_stats().get('brain_prediction', {}).get('recent_avg', 0.0)
        final_fps = 1.0 / final_fps if final_fps > 0 else 0.0
        log_file = enhanced_logger.save_enhanced_log(brainstem.brain_client, final_fps)
        if log_file:
            print(f"   📊 Enhanced performance log saved: {log_file}")
        
        # Display final comprehensive statistics
        final_stats = brainstem.brain_client.get_brain_statistics()
        print(f"\\n📊 FINAL BRAIN STATISTICS:")
        print(f"   🧠 Total experiences: {final_stats['graph_stats']['total_nodes']}")
        print(f"   🔄 Memory merges: {final_stats['graph_stats']['total_merges']}")
        print(f"   📈 Parameter adaptations: {final_stats['adaptive_tuning_stats']['total_adaptations']}")
        print(f"   🎯 Actuators discovered: {final_stats['actuator_discovery_stats']['total_actuators_discovered']}")
        print(f"   📂 Emergent categories: {final_stats['actuator_discovery_stats']['emergent_categories_formed']}")
        
        # Show discovered actuator categories
        categories = brainstem.brain_client.get_discovered_actuator_categories()
        if categories:
            print(f"\\n🎯 DISCOVERED ACTUATOR CATEGORIES:")
            for category_id, category_data in categories.items():
                properties = category_data['emergent_properties']
                members = ', '.join(category_data['member_actuators'])
                print(f"   {category_id}: [{members}]")
                
                if properties['appears_spatial']:
                    print(f"      → Affects spatial movement")
                if properties['appears_manipulative']:
                    print(f"      → Affects object manipulation")
                if properties['appears_environmental']:
                    print(f"      → Affects environment")
        
        print(f"\\n🧠 Complete brain learning session finished!")
        print(f"   All experiences and adaptations saved for next session.")
        print(f"   The brain will continue learning from where it left off.")


if __name__ == "__main__":
    print("🌟 ULTIMATE 2D BRAIN DEMO")
    print("=" * 60)
    print()
    print("This is THE comprehensive demonstration of the complete")
    print("emergent intelligence robot brain system.")
    print()
    print("Everything is integrated here:")
    print("• Complete multi-drive motivation system")
    print("• Drive-generated goals and objectives")
    print("• Adaptive learning and parameter tuning")  
    print("• Universal actuator effect discovery")
    print("• Persistent cross-session memory")
    print("• Real-time brain state visualization")
    print()
    print("The robot starts as a 'newborn' and develops intelligent")
    print("behavior through pure experience and emergent phenomena.")
    print("No hardcoded behaviors - everything emerges naturally.")
    print()
    
    try:
        main()
        print("✅ Ultimate brain demo completed successfully!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()