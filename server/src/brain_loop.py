"""
Decoupled Brain Loop

Separates brain processing from socket I/O by running the brain on its own schedule.
The brain reads from a sensor buffer and processes at 50ms intervals, enabling 
predictive operation and idle time for maintenance tasks.

This is Step 1 toward a fully predictive brain architecture.
"""

import time
import threading
from typing import Dict, List, Optional, Any

from .brain_factory import MinimalBrain
from .communication.sensor_buffer import get_sensor_buffer


class DecoupledBrainLoop:
    """
    Decoupled brain loop that runs independently of socket I/O.
    
    Features:
    - Reads latest sensor data from buffer (non-blocking)
    - Processes brain updates every 50ms
    - Performs maintenance tasks during idle time
    - Generates predictions proactively
    """
    
    def __init__(self, brain: MinimalBrain, cycle_time_ms: float = 50.0):
        """
        Initialize decoupled brain loop.
        
        Args:
            brain: The minimal brain instance
            cycle_time_ms: Base brain cycle time in milliseconds (adaptive)
        """
        self.brain = brain
        self.base_cycle_time_s = cycle_time_ms / 1000.0
        self.sensor_buffer = get_sensor_buffer()
        
        # No artificial timing - let cognitive load create natural delays
        self.emergent_timing = True
        
        # Loop state
        self.running = False
        self.thread = None
        self.total_cycles = 0
        self.active_cycles = 0
        self.idle_cycles = 0
        
        # Performance tracking
        self.cycle_times = []
        self.max_cycle_times = 100
        self.maintenance_tasks_performed = 0
        self.current_cognitive_mode = 'focused'
        self.cognitive_mode_changes = 0
        
        print(f"🧠 DecoupledBrainLoop initialized")
        if self.emergent_timing:
            print(f"   Emergent timing: Natural delays from cognitive load")
            print(f"   AUTOPILOT: minimal processing → fast cycles")
            print(f"   FOCUSED: moderate processing → medium cycles") 
            print(f"   DEEP_THINK: intensive processing → slow cycles")
        else:
            print(f"   Fixed cycle time: {cycle_time_ms}ms")
        print(f"   Brain: {brain}")
    
    def start(self):
        """Start the decoupled brain loop in a separate thread."""
        if self.running:
            print("⚠️ Brain loop already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._brain_loop, daemon=True)
        self.thread.start()
        
        print("🚀 Decoupled brain loop started")
    
    def stop(self):
        """Stop the decoupled brain loop."""
        if not self.running:
            return
        
        print("🛑 Stopping decoupled brain loop...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        print("✅ Decoupled brain loop stopped")
    
    def _track_cognitive_mode(self):
        """Track cognitive mode changes for monitoring."""
        if hasattr(self.brain, 'cognitive_autopilot') and self.brain.cognitive_autopilot:
            autopilot = self.brain.cognitive_autopilot
            mode = autopilot.current_mode.value
            
            # Track mode changes
            if mode != self.current_cognitive_mode:
                self.current_cognitive_mode = mode
                self.cognitive_mode_changes += 1
    
    def _brain_loop(self):
        """Main brain loop running on independent schedule."""
        print("🔄 Brain loop started - running independently")
        
        while self.running:
            cycle_start = time.time()
            
            try:
                # Track cognitive mode changes
                self._track_cognitive_mode()
                
                # Get latest sensor data from all clients
                sensor_data = self.sensor_buffer.get_all_latest_data()
                
                if sensor_data:
                    # Process each client's sensor input
                    # The cognitive load itself will determine cycle time
                    for client_id, data in sensor_data.items():
                        self._process_client_sensors(client_id, data)
                    
                    self.active_cycles += 1
                else:
                    # No sensor data - perform maintenance tasks
                    self._perform_maintenance_tasks()
                    self.idle_cycles += 1
                
                self.total_cycles += 1
                
                # Track cycle performance (no artificial delays)
                cycle_time = time.time() - cycle_start
                self.cycle_times.append(cycle_time)
                if len(self.cycle_times) > self.max_cycle_times:
                    self.cycle_times.pop(0)
                
                # No sleep - let cognitive load create natural timing
                
            except Exception as e:
                print(f"❌ Brain loop error: {e}")
                # Brief pause only on errors to prevent error spam
                time.sleep(0.001)
    
    def _process_client_sensors(self, client_id: str, sensor_data):
        """Process sensor input for a specific client."""
        try:
            # TODO: For now, skip brain processing to avoid blocking
            # Later we'll add: action_vector, brain_state = self.brain.process_sensory_input(...)
            # Just log that we received the data
            pass
            
        except Exception as e:
            print(f"⚠️ Error processing sensors for {client_id}: {e}")
    
    def _perform_maintenance_tasks(self):
        """Perform brain maintenance during idle cycles."""
        try:
            # Task 1: Cleanup old experiences if needed
            if self.total_cycles % 20 == 0:  # Every 1 second
                self._cleanup_old_experiences()
            
            # Task 2: Optimize similarity search structures
            if self.total_cycles % 100 == 0:  # Every 5 seconds
                self._optimize_similarity_structures()
            
            # Task 3: Update prediction models
            if self.total_cycles % 200 == 0:  # Every 10 seconds
                self._update_prediction_models()
            
            self.maintenance_tasks_performed += 1
            
        except Exception as e:
            print(f"⚠️ Maintenance task error: {e}")
    
    def _cleanup_old_experiences(self):
        """Clean up old experiences during maintenance."""
        # Let the brain's normal cleanup handle this
        # Future: More sophisticated cleanup strategies
        pass
    
    def _optimize_similarity_structures(self):
        """Optimize similarity search structures during idle time."""
        # Future: Rebuild similarity indexes, optimize memory layout, etc.
        pass
    
    def _update_prediction_models(self):
        """Update prediction models during idle time."""
        # Future: Retrain prediction engines, update weights, etc.
        pass
    
    def get_loop_statistics(self) -> Dict[str, Any]:
        """Get comprehensive brain loop statistics."""
        avg_cycle_time = 0.0
        max_cycle_time = 0.0
        
        if self.cycle_times:
            avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times)
            max_cycle_time = max(self.cycle_times)
        
        utilization = self.active_cycles / max(1, self.total_cycles)
        
        return {
            'running': self.running,
            'emergent_timing': self.emergent_timing,
            'current_cognitive_mode': self.current_cognitive_mode,
            'cognitive_mode_changes': self.cognitive_mode_changes,
            'actual_avg_cycle_time_ms': avg_cycle_time * 1000,
            'max_cycle_time_ms': max_cycle_time * 1000,
            'total_cycles': self.total_cycles,
            'active_cycles': self.active_cycles,
            'idle_cycles': self.idle_cycles,
            'utilization': utilization,
            'maintenance_tasks_performed': self.maintenance_tasks_performed,
            'sensor_buffer_stats': self.sensor_buffer.get_statistics()
        }
    
    def print_loop_report(self):
        """Print brain loop performance report."""
        stats = self.get_loop_statistics()
        
        print(f"\n🧠 DECOUPLED BRAIN LOOP REPORT")
        print(f"=" * 50)
        print(f"🔄 Loop Status: {'Running' if stats['running'] else 'Stopped'}")
        
        if stats['emergent_timing']:
            print(f"🧬 Cognitive Mode: {stats['current_cognitive_mode']}")
            print(f"🔄 Mode changes: {stats['cognitive_mode_changes']}")
            print(f"⚡ Emergent timing: Cycle speed determined by cognitive load")
        
        print(f"📊 Actual avg: {stats['actual_avg_cycle_time_ms']:.1f}ms")
        print(f"⚡ Max cycle: {stats['max_cycle_time_ms']:.1f}ms")
        print(f"🔢 Total cycles: {stats['total_cycles']:,}")
        print(f"🎯 Active cycles: {stats['active_cycles']:,} ({stats['utilization']:.1%})")
        print(f"💤 Idle cycles: {stats['idle_cycles']:,}")
        print(f"🔧 Maintenance tasks: {stats['maintenance_tasks_performed']:,}")
        
        buffer_stats = stats['sensor_buffer_stats']
        print(f"\n📡 Sensor Buffer:")
        print(f"   Active clients: {buffer_stats['active_clients']}")
        print(f"   Total inputs: {buffer_stats['total_inputs_received']:,}")
        print(f"   Discard rate: {buffer_stats['discard_rate']:.1%}")
    
    def is_running(self) -> bool:
        """Check if brain loop is running."""
        return self.running
    
    def __str__(self) -> str:
        if self.running:
            return f"DecoupledBrainLoop(running, {self.total_cycles} cycles)"
        else:
            return f"DecoupledBrainLoop(stopped)"
    
    def __repr__(self) -> str:
        return self.__str__()