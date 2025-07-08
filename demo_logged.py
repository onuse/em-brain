#!/usr/bin/env python3
"""
Demo with comprehensive run logging
"""

import time
import pygame
from simulation.brainstem_sim import GridWorldBrainstem
from visualization.integrated_display import IntegratedDisplay
from run_logger import RunLogger


def main():
    """Demo with logging enabled."""
    print("🧠 LOGGED BRAIN DEMO")
    print("=" * 40)
    
    # Initialize logger
    logger = RunLogger()
    logger.log_event("demo_start", "Starting logged brain demo")
    
    try:
        # Initialize brain system
        brainstem = GridWorldBrainstem(
            world_width=12,
            world_height=12, 
            seed=42, 
            use_sockets=False
        )
        
        # Start session
        session_id = brainstem.brain_client.start_memory_session("Logged Demo")
        logger.log_event("session_start", f"Memory session started: {session_id}")
        
        # Get initial brain stats
        initial_stats = brainstem.brain_client.get_brain_statistics()
        logger.log_brain_state(initial_stats)
        logger.log_event("brain_loaded", f"Brain loaded with {initial_stats['graph_stats']['total_nodes']} nodes")
        
        # Initialize visualization
        display = IntegratedDisplay(brainstem, cell_size=25)
        
        # Enhanced callback with logging
        def logged_brain_callback(state):
            """Brain callback that logs performance data."""
            from core.communication import SensoryPacket
            from datetime import datetime
            
            frame_start = time.time()
            
            sensory_packet = SensoryPacket(
                sequence_id=brainstem.sequence_counter,
                sensor_values=state['sensors'],
                actuator_positions=[0.0, 0.0, 0.0],
                timestamp=datetime.now()
            )
            
            mental_context = state['sensors'][:8] if len(state['sensors']) >= 8 else state['sensors']
            
            prediction = brainstem.brain_client.process_sensory_input(
                sensory_packet, 
                mental_context, 
                threat_level="normal"
            )
            
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            
            # Log performance sample every 10 frames
            if brainstem.sequence_counter % 10 == 0:
                brain_stats = brainstem.brain_client.get_brain_statistics()
                nodes = brain_stats['graph_stats']['total_nodes']
                
                logger.log_performance_sample(fps, frame_time, nodes, brain_stats)
                
                # Log brain snapshot every 50 frames
                if brainstem.sequence_counter % 50 == 0:
                    robot_state = {
                        'health': state.get('health', 1.0),
                        'energy': state.get('energy', 1.0),
                        'position': state.get('position', [0, 0]),
                        'sensors': state['sensors']
                    }
                    logger.log_brain_state(brain_stats, robot_state)
            
            return prediction.motor_action if prediction else {
                'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 0.0
            }
        
        display.set_learning_callback(logged_brain_callback)
        brain_graph = brainstem.brain_client.get_world_graph()
        display.set_brain_graph(brain_graph)
        
        logger.log_event("demo_ready", "Demo initialized and ready to run")
        
        # Run demo
        display.run(auto_step=True, step_delay=0.0)
        
    except KeyboardInterrupt:
        logger.log_event("user_interrupt", "Demo interrupted by user")
        
        # Calculate final FPS estimate
        if logger.performance_data:
            recent_samples = logger.performance_data[-10:]  # Last 10 samples
            final_fps = sum(s['fps'] for s in recent_samples) / len(recent_samples)
        else:
            final_fps = 0.0
        
        print(f"\n⏹️  Demo interrupted - Final FPS: {final_fps:.1f}")
        
    except Exception as e:
        logger.log_event("error", f"Demo error: {e}")
        final_fps = 0.0
        print(f"\n❌ Demo error: {e}")
        
    finally:
        # Save comprehensive log
        final_fps = final_fps if 'final_fps' in locals() else 0.0
        
        # End memory session
        try:
            session_summary = brainstem.brain_client.end_memory_session()
            if session_summary:
                logger.log_event("session_end", f"Session ended: {session_summary}")
        except:
            pass
        
        # Save log
        logger.save_session_log(brainstem.brain_client, final_fps=final_fps)
        logger.print_session_summary(final_fps)


if __name__ == "__main__":
    main()
