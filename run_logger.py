#!/usr/bin/env python3
"""
Run logging system to capture and serialize session data
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class RunLogger:
    """Captures and logs run data for analysis"""
    
    def __init__(self, log_directory: str = "./run_logs"):
        self.log_dir = Path(log_directory)
        self.log_dir.mkdir(exist_ok=True)
        
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("run_%Y%m%d_%H%M%S")
        
        self.performance_data = []
        self.brain_snapshots = []
        self.events_log = []
        
        print(f"📝 Run Logger initialized: {self.session_id}")
    
    def log_performance_sample(self, fps: float, frame_time: float, brain_nodes: int = 0, 
                             brain_stats: Optional[Dict] = None):
        """Log a performance data point"""
        timestamp = time.time()
        elapsed = timestamp - self.session_start.timestamp()
        
        sample = {
            'timestamp': timestamp,
            'elapsed_seconds': elapsed,
            'fps': fps,
            'frame_time': frame_time,
            'brain_nodes': brain_nodes,
            'brain_stats': brain_stats
        }
        
        self.performance_data.append(sample)
    
    def log_brain_state(self, brain_stats: Dict[str, Any], robot_state: Optional[Dict] = None,
                       learning_progress: Optional[Dict] = None):
        """Log complete brain state snapshot"""
        timestamp = time.time()
        elapsed = timestamp - self.session_start.timestamp()
        
        snapshot = {
            'timestamp': timestamp,
            'elapsed_seconds': elapsed,
            'brain_stats': brain_stats,
            'robot_state': robot_state,
            'learning_progress': learning_progress
        }
        
        self.brain_snapshots.append(snapshot)
    
    def log_event(self, event_type: str, description: str, data: Optional[Dict] = None):
        """Log a specific event during the run"""
        timestamp = time.time()
        elapsed = timestamp - self.session_start.timestamp()
        
        event = {
            'timestamp': timestamp,
            'elapsed_seconds': elapsed,
            'type': event_type,
            'description': description,
            'data': data or {}
        }
        
        self.events_log.append(event)
        print(f"📋 Event logged: {event_type} - {description}")
    
    def capture_final_state(self, brain_client, simulation_state: Optional[Dict] = None,
                           final_fps: float = 0.0):
        """Capture final state data at session end"""
        try:
            # Get comprehensive brain statistics
            brain_stats = brain_client.get_brain_statistics()
            
            # Get similarity engine performance
            if hasattr(brain_client, 'brain_interface') and hasattr(brain_client.brain_interface, 'world_graph'):
                graph_stats = brain_client.brain_interface.world_graph.get_graph_statistics()
                similarity_stats = graph_stats.get('similarity_engine', {})
            else:
                similarity_stats = {}
            
            final_state = {
                'session_id': self.session_id,
                'session_start': self.session_start.isoformat(),
                'session_end': datetime.now().isoformat(),
                'duration_seconds': time.time() - self.session_start.timestamp(),
                'final_fps': final_fps,
                'brain_statistics': brain_stats,
                'similarity_engine_stats': similarity_stats,
                'simulation_state': simulation_state,
                'performance_samples': len(self.performance_data),
                'brain_snapshots': len(self.brain_snapshots),
                'events_logged': len(self.events_log)
            }
            
            return final_state
            
        except Exception as e:
            print(f"⚠️  Error capturing final state: {e}")
            return {
                'session_id': self.session_id,
                'error': str(e),
                'final_fps': final_fps
            }
    
    def save_session_log(self, brain_client, simulation_state: Optional[Dict] = None,
                        final_fps: float = 0.0):
        """Save complete session log to disk"""
        try:
            # Capture final state
            final_state = self.capture_final_state(brain_client, simulation_state, final_fps)
            
            # Complete log data
            log_data = {
                'session_info': final_state,
                'performance_data': self.performance_data,
                'brain_snapshots': self.brain_snapshots,
                'events_log': self.events_log
            }
            
            # Save to file
            log_file = self.log_dir / f"{self.session_id}.json"
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            print(f"💾 Session log saved: {log_file}")
            print(f"   Performance samples: {len(self.performance_data)}")
            print(f"   Brain snapshots: {len(self.brain_snapshots)}")
            print(f"   Events logged: {len(self.events_log)}")
            
            return log_file
            
        except Exception as e:
            print(f"❌ Error saving session log: {e}")
            return None
    
    def print_session_summary(self, final_fps: float = 0.0):
        """Print a summary of the session"""
        duration = time.time() - self.session_start.timestamp()
        
        print(f"\\n📊 SESSION SUMMARY: {self.session_id}")
        print("=" * 50)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Final FPS: {final_fps:.1f}")
        
        if self.performance_data:
            fps_values = [p['fps'] for p in self.performance_data if p['fps'] > 0]
            if fps_values:
                avg_fps = sum(fps_values) / len(fps_values)
                max_fps = max(fps_values)
                min_fps = min(fps_values)
                print(f"FPS stats: avg={avg_fps:.1f}, min={min_fps:.1f}, max={max_fps:.1f}")
        
        if self.brain_snapshots:
            print(f"Brain snapshots captured: {len(self.brain_snapshots)}")
            last_snapshot = self.brain_snapshots[-1]
            if 'brain_stats' in last_snapshot:
                stats = last_snapshot['brain_stats']
                if 'graph_stats' in stats:
                    nodes = stats['graph_stats'].get('total_nodes', 0)
                    print(f"Final brain size: {nodes} nodes")
        
        print(f"Events logged: {len(self.events_log)}")


def create_demo_with_logging():
    """Create a demo version that includes comprehensive logging"""
    
    demo_with_logging = '''#!/usr/bin/env python3
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
        
        print(f"\\n⏹️  Demo interrupted - Final FPS: {final_fps:.1f}")
        
    except Exception as e:
        logger.log_event("error", f"Demo error: {e}")
        final_fps = 0.0
        print(f"\\n❌ Demo error: {e}")
        
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
'''
    
    with open('/Users/jkarlsson/Documents/Projects/robot-project/brain/demo_logged.py', 'w') as f:
        f.write(demo_with_logging)
    
    print("📝 Created: demo_logged.py")
    print("   This version will capture comprehensive run data")

if __name__ == "__main__":
    create_demo_with_logging()