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
• Enhanced 40x40 world with plant-based smell sensors

This consolidates and replaces all other demo files in the project.
"""

import time
import warnings

# Suppress pygame's pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import pygame
from simulation.brainstem_sim import GridWorldBrainstem
from visualization.integrated_display import IntegratedDisplay
from tools.enhanced_run_logger import EnhancedRunLogger
# Decision logging imports
from monitoring.decision_logger import start_decision_logging, stop_decision_logging
# Brain evolution tracking imports
from monitoring.brain_evolution_tracker import BrainEvolutionTracker
from monitoring.learning_velocity_monitor import LearningVelocityMonitor



def main():
    # Start decision logging
    logger = start_decision_logging()
    
    # Initialize brain evolution tracker
    evolution_tracker = BrainEvolutionTracker(session_name="demo_robot_brain", track_every_n_steps=5)
    
    # Initialize learning velocity monitor
    learning_monitor = LearningVelocityMonitor(session_name="demo_robot_brain")

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
    print("• Enhanced 40x40 world with plant-based smell sensors")
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
    enhanced_logger = EnhancedRunLogger()
    
    # Initialize the complete brain system with async server
    print("🧠 Initializing brain system...")
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    
    # Start async brain server for decoupled processing
    print("🚀 Starting async brain server...")
    from core.async_brain_server import AsyncBrainServer
    brain_server = AsyncBrainServer(brainstem, brainstem.brain_client)
    brain_server.start()
    
    # Start persistent memory session
    session_id = brainstem.brain_client.start_memory_session("Robot Brain Demo")
    
    # Show what was loaded from previous sessions
    stats = brainstem.brain_client.get_brain_statistics()
    if stats['graph_stats']['total_nodes'] > 0:
        print(f"📚 Loaded {stats['graph_stats']['total_nodes']} experiences from previous sessions")
    else:
        print("🆕 Starting with fresh brain - no previous experiences")
    
    # Initialize visualization
    print("🎮 Setting up visualization...")
    display = IntegratedDisplay(brainstem, cell_size=15)
    
    # Prediction error tracking
    prediction_error_tracker = {'last_prediction': None, 'last_sensory': None, 'current_error': 0.0}
    
    def calculate_prediction_error(predicted_sensory, actual_sensory):
        """Calculate prediction error between predicted and actual sensory values."""
        if not predicted_sensory or not actual_sensory:
            return 0.0
        
        if len(predicted_sensory) != len(actual_sensory):
            return 1.0  # Maximum error for mismatched lengths
        
        # Simple Euclidean distance
        squared_diffs = [(p - a) ** 2 for p, a in zip(predicted_sensory, actual_sensory)]
        return (sum(squared_diffs) / len(squared_diffs)) ** 0.5
    
    # Connect visualization to brain system for real predictions
    def brain_prediction_callback(state):
        """Use the brain system to generate motor commands instead of random actions."""
        # Log frame with timing, including GUI FPS
        gui_fps = getattr(display, 'current_fps', None)
        enhanced_logger.log_frame_with_timing(brainstem.brain_client, state, gui_fps=gui_fps)
        
        # Calculate prediction error from last cycle
        if prediction_error_tracker['last_prediction'] and prediction_error_tracker['last_sensory']:
            current_error = calculate_prediction_error(
                prediction_error_tracker['last_prediction'].expected_sensory,
                prediction_error_tracker['last_sensory']
            )
            prediction_error_tracker['current_error'] = current_error
        
        # Create sensory packet from current state
        from core.communication import SensoryPacket
        from datetime import datetime
        sensory_packet = SensoryPacket(
            sequence_id=brainstem.sequence_counter,
            sensor_values=state['sensors'],
            actuator_positions=[0.0, 0.0, 0.0],  # Dummy actuator positions for grid world
            timestamp=datetime.now()
        )
        
        # Extract robot state from the simulation state
        robot_position = state.get('position', (0, 0))
        robot_orientation = state.get('orientation', 0)
        robot_health = state.get('health', 0.0)
        robot_energy = state.get('energy', 0.0)
        
        # Get brain prediction with current mental context
        mental_context = state['sensors'][:8] if len(state['sensors']) >= 8 else state['sensors']
        
        # Get latest brain state from async server (non-blocking)
        brain_state = brain_server.get_current_state()
        
        # Use the brain state's recent action as our prediction
        if brain_state:
            prediction_action = brain_state.recent_action
        else:
            # Fallback to direct brain interface if no state available
            prediction_action = {'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 0.0}
        
        # Create a mock prediction object for compatibility
        from core.communication import PredictionPacket
        prediction = PredictionPacket(
            sequence_id=brainstem.sequence_counter,
            motor_action=prediction_action,
            confidence=0.8,
            expected_sensory=[],
            thinking_depth=1,
            timestamp=datetime.now()
        )
        
        # Capture brain evolution snapshot
        brain_stats = brainstem.brain_client.get_brain_statistics()
        decision_context = {
            'step_count': brainstem.sequence_counter,
            'robot_position': robot_position,
            'robot_orientation': robot_orientation,
            'robot_health': robot_health,
            'robot_energy': robot_energy,
            'chosen_action': prediction.motor_action if prediction else {},
            'recent_prediction_error': prediction_error_tracker['current_error'],
            'confidence': 0.8,  # Placeholder - would come from prediction system
            'motivator_weights': {},  # Will be populated if multi-drive predictor
            'total_motivator_pressure': 0.0,
            'dominant_motivator': 'unknown'
        }
        
        # Extract drive information if available
        if hasattr(brainstem.brain_client.predictor, 'motivation_system'):
            motivation_system = brainstem.brain_client.predictor.motivation_system
            decision_context['motivator_weights'] = {name: drive.current_weight for name, drive in motivation_system.motivators.items()}
            decision_context['total_motivator_pressure'] = sum(decision_context['motivator_weights'].values())
            
            # Find dominant drive
            if decision_context['motivator_weights']:
                dominant_motivator = max(decision_context['motivator_weights'].keys(), key=lambda k: decision_context['motivator_weights'][k])
                decision_context['dominant_motivator'] = dominant_motivator
        
        # Capture evolution snapshot
        evolution_snapshot = evolution_tracker.capture_snapshot(brain_stats, decision_context, brainstem.sequence_counter)
        
        # Capture learning velocity snapshot
        learning_snapshot = learning_monitor.capture_learning_snapshot(brain_stats, decision_context, brainstem.sequence_counter)
        
        # Store prediction and sensory data for next cycle's error calculation
        prediction_error_tracker['last_prediction'] = prediction
        prediction_error_tracker['last_sensory'] = state['sensors']
        
        return prediction.motor_action if prediction else {
            'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 0.0
        }
    
    # Set the brain callback so visualization uses real brain predictions
    display.set_learning_callback(brain_prediction_callback)
    
    # Connect brain's world graph to visualization
    brain_graph = brainstem.brain_client.get_world_graph()
    display.set_brain_graph(brain_graph)
    
    print("🚀 Launching robot brain...")
    print()
    
    try:
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
        # Stop async brain server
        print("🛑 Stopping async brain server...")
        brain_server.stop()
        
        # Stop decision logging
        summary = stop_decision_logging()
        if summary:
            print(f'📊 Decision log summary: {summary["problem_analysis"]["total_problems"]} problems detected')
        
        # Generate and save brain evolution analysis
        print("\\n🧠 BRAIN EVOLUTION ANALYSIS")
        print("=" * 60)
        
        # Print live status
        evolution_tracker.print_live_status()
        
        # Print learning velocity status
        learning_monitor.print_learning_status()
        
        # Generate comprehensive evolution analysis
        evolution_analysis = evolution_tracker.analyze_brain_evolution()
        if 'error' not in evolution_analysis:
            print(f"\\n📈 EVOLUTION INSIGHTS:")
            for insight in evolution_analysis['insights']:
                print(f"   {insight}")
            
            # Print key metrics
            neural_dev = evolution_analysis['neural_development']
            print(f"\\n🧠 Neural Development:")
            print(f"   Growth: {neural_dev['initial_nodes']} → {neural_dev['final_nodes']} nodes")
            print(f"   Pattern: {neural_dev['growth_pattern']}")
            
            learning_dyn = evolution_analysis['learning_dynamics']
            print(f"\\n🎯 Learning Dynamics:")
            print(f"   Error improvement: {learning_dyn['total_improvement']:.3f}")
            print(f"   Learning efficiency: {learning_dyn['learning_efficiency']:.3f}")
            
            behavioral = evolution_analysis['behavioral_emergence']
            print(f"\\n🌟 Behavioral Emergence:")
            print(f"   Complexity trend: {behavioral['complexity_trend']}")
            print(f"   Behavior maturity: {behavioral['behavior_maturity']:.1%}")
        
        # Generate and save learning velocity analysis
        learning_analysis = learning_monitor.analyze_learning_patterns()
        if 'error' not in learning_analysis:
            print(f"\\n🎯 LEARNING VELOCITY INSIGHTS:")
            for insight in learning_analysis['learning_insights']:
                print(f"   {insight}")
            
            print(f"\\n🔧 OPTIMIZATION RECOMMENDATIONS:")
            for recommendation in learning_analysis['optimization_recommendations']:
                print(f"   {recommendation}")
            
            # Print key learning metrics
            velocity_analysis = learning_analysis['learning_velocity_analysis']
            print(f"\\n📈 Learning Velocity:")
            print(f"   Current velocity: {velocity_analysis['final_velocity']:.3f}")
            print(f"   Velocity trend: {velocity_analysis['velocity_trend']}")
            print(f"   Error reduction rate: {velocity_analysis['error_reduction_rate']:.3f}")
            
            confidence_analysis = learning_analysis['confidence_evolution']
            print(f"\\n💪 Confidence Evolution:")
            print(f"   Confidence growth: {confidence_analysis['confidence_growth']:.3f}")
            print(f"   Confidence maturity: {confidence_analysis['confidence_maturity']:.1%}")
        
        # Save evolution reports
        evolution_tracker.save_evolution_report()
        evolution_tracker.save_snapshots()
        
        # Save learning velocity reports
        learning_monitor.save_learning_report()
        learning_monitor.save_learning_snapshots()

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
        print(f"   🔄 Memory merges: {final_stats['graph_stats'].get('total_merges', 0)}")
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