"""
Decoupled Brain Loop

Separates brain processing from socket I/O by running the brain on its own schedule.
The brain reads from a sensor buffer and processes at 50ms intervals, enabling 
predictive operation and idle time for maintenance tasks.

This is Step 1 toward a fully predictive brain architecture.
"""

import time
import threading
import numpy as np
import torch
from typing import Dict, List, Optional, Any

from .core.interfaces import IBrain
from .communication.sensor_buffer import get_sensor_buffer
from .parameters.cognitive_config import get_cognitive_config


class DecoupledBrainLoop:
    """
    Decoupled brain loop that runs independently of socket I/O.
    
    Features:
    - Reads latest sensor data from buffer (non-blocking)
    - Processes brain updates every 50ms
    - Performs maintenance tasks during idle time
    - Generates predictions proactively
    """
    
    def __init__(self, brain: IBrain, cycle_time_ms: Optional[float] = None):
        """
        Initialize decoupled brain loop.
        
        Args:
            brain: The minimal brain instance
            cycle_time_ms: Base brain cycle time in milliseconds (adaptive)
        """
        self.brain = brain
        
        # Load cognitive configuration
        self.cognitive_config = get_cognitive_config()
        self.sensor_config = self.cognitive_config.sensor_config
        
        # Use temporal constants for cycle time if not specified
        if cycle_time_ms is None:
            temporal_config = self.cognitive_config.get_temporal_config()
            cycle_time_ms = temporal_config['control_cycle_target'] * 1000
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
        self.sensor_skip_cycles = 0
        
        # Confidence tracking for sensor decisions
        self.last_prediction_confidence = self.cognitive_config.brain_config.default_prediction_confidence
        self.confidence_history = []
        self.sensor_check_probability = 1.0
        
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
        # Check if we have a brain wrapper with the actual brain
        if hasattr(self.brain, 'brain') and hasattr(self.brain.brain, 'cognitive_autopilot'):
            autopilot = self.brain.brain.cognitive_autopilot
            mode = autopilot.current_mode.value
            
            # Track mode changes
            if mode != self.current_cognitive_mode:
                self.current_cognitive_mode = mode
                self.cognitive_mode_changes += 1
        elif hasattr(self.brain, 'cognitive_autopilot') and self.brain.cognitive_autopilot:
            autopilot = self.brain.cognitive_autopilot
            mode = autopilot.current_mode.value
            
            # Track mode changes
            if mode != self.current_cognitive_mode:
                self.current_cognitive_mode = mode
                self.cognitive_mode_changes += 1
    
    def _should_check_sensors(self) -> bool:
        """Decide whether to check sensor buffer based on cognitive state."""
        # Always check if we don't have cognitive autopilot
        has_autopilot = False
        if hasattr(self.brain, 'brain') and hasattr(self.brain.brain, 'cognitive_autopilot'):
            has_autopilot = True
        elif hasattr(self.brain, 'cognitive_autopilot'):
            has_autopilot = True
            
        if not has_autopilot:
            return True
        
        # Get current confidence from brain if available
        if hasattr(self.brain, 'brain') and hasattr(self.brain.brain, '_current_prediction_confidence'):
            self.last_prediction_confidence = self.brain.brain._current_prediction_confidence
            self.confidence_history.append(self.last_prediction_confidence)
            if len(self.confidence_history) > 100:
                self.confidence_history.pop(0)
        elif hasattr(self.brain, '_current_prediction_confidence'):
            self.last_prediction_confidence = self.brain._current_prediction_confidence
            self.confidence_history.append(self.last_prediction_confidence)
            if len(self.confidence_history) > 100:
                self.confidence_history.pop(0)
        
        # Decision based on cognitive mode and confidence
        mode = self.current_cognitive_mode
        confidence = self.last_prediction_confidence
        
        if mode == 'autopilot' and confidence > self.sensor_config.autopilot_threshold:
            # High confidence autopilot - check sensors rarely
            self.sensor_check_probability = self.sensor_config.autopilot_sensor_probability
        elif mode == 'focused' and confidence > self.sensor_config.focused_threshold:
            # Focused mode - check sensors moderately
            self.sensor_check_probability = self.sensor_config.focused_sensor_probability
        else:
            # Deep think or low confidence - check sensors frequently
            self.sensor_check_probability = self.sensor_config.deep_think_sensor_probability
        
        # Stochastic decision with smooth transitions
        check_sensors = np.random.random() < self.sensor_check_probability
        
        if not check_sensors:
            self.sensor_skip_cycles += 1
            if self.total_cycles % 100 == 0:
                print(f"🧠 BRAIN_LOOP: Skipping sensors (confidence: {confidence:.2f}, mode: {mode})")
        
        return check_sensors
    
    def _process_internal_only(self):
        """Process one cycle without sensor input - pure internal dynamics."""
        try:
            # Run brain with neutral/zero input to let spontaneous dynamics dominate
            expected_dim = None
            if hasattr(self.brain, 'brain') and hasattr(self.brain.brain, 'expected_sensory_dim'):
                expected_dim = self.brain.brain.expected_sensory_dim
            elif hasattr(self.brain, 'expected_sensory_dim'):
                expected_dim = self.brain.expected_sensory_dim
                
            if expected_dim:
                neutral_value = self.cognitive_config.brain_config.default_prediction_confidence
                neutral_input = [neutral_value] * (expected_dim - 1) + [0.0]  # No reward
            else:
                neutral_value = self.cognitive_config.brain_config.default_prediction_confidence
                neutral_input = [neutral_value] * 16 + [0.0]  # Default neutral input
            
            # Process with neutral input
            # Check if we have a brain wrapper with the actual brain
            if hasattr(self.brain, 'brain') and hasattr(self.brain.brain, 'process_robot_cycle'):
                action_vector, brain_state = self.brain.brain.process_robot_cycle(neutral_input)
            else:
                # Fallback to field dynamics interface
                field_input = torch.tensor(neutral_input, dtype=torch.float32)
                field_output = self.brain.process_field_dynamics(field_input)
                action_vector = field_output.tolist()[:4]  # Extract motor commands
                brain_state = self.brain.get_state()
            
            # Log occasionally
            if self.total_cycles % 50 == 0:
                print(f"🌀 BRAIN_LOOP: Internal processing - energy: {brain_state.get('field_energy', 0):.4f}")
        
        except Exception as e:
            print(f"⚠️ Error in internal processing: {e}")
    
    def _brain_loop(self):
        """Main brain loop running on independent schedule."""
        print("🔄 Brain loop started - running independently")
        
        while self.running:
            cycle_start = time.time()
            
            try:
                # Track cognitive mode changes
                self._track_cognitive_mode()
                
                # Decide whether to check sensors based on cognitive state
                should_check_sensors = self._should_check_sensors()
                
                if should_check_sensors:
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
                else:
                    # Brain chose to skip sensors - pure internal processing
                    self._process_internal_only()
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
            # SMART PROCESSING: Only process if enough time has passed since last brain cycle
            # This prevents the 50ms brain loop from blocking on 350ms brain processing
            current_time = time.time()
            
            # Track last processing time per client to avoid overwhelming brain
            if not hasattr(self, '_last_brain_processing'):
                self._last_brain_processing = {}
            
            last_processing = self._last_brain_processing.get(client_id, 0.0)
            time_since_processing = current_time - last_processing
            
            # Only process if enough time passed (respect brain's natural cycle time)
            min_processing_interval = 0.2  # 200ms minimum between brain processing
            
            if time_since_processing >= min_processing_interval:
                if hasattr(sensor_data, 'vector') and sensor_data.vector:
                    # Process through brain
                    # Check if we have a brain wrapper with the actual brain
                    if hasattr(self.brain, 'brain') and hasattr(self.brain.brain, 'process_robot_cycle'):
                        action_vector, brain_state = self.brain.brain.process_robot_cycle(sensor_data.vector)
                    else:
                        # Fallback to field dynamics interface
                        field_input = torch.tensor(sensor_data.vector, dtype=torch.float32)
                        field_output = self.brain.process_field_dynamics(field_input)
                        action_vector = field_output.tolist()[:4]  # Extract motor commands
                        brain_state = self.brain.get_state()
                    
                    # Mark data as consumed
                    self.sensor_buffer.clear_client_data(client_id)
                    self._last_brain_processing[client_id] = current_time
                    
                    # Log occasionally with cognitive mode info
                    if self.total_cycles % 50 == 0:
                        confidence = brain_state.get('prediction_confidence', 0.0)
                        cycle_time_ms = brain_state.get('cycle_time_ms', 0.0)
                        cognitive_mode = brain_state.get('cognitive_autopilot', {}).get('cognitive_mode', 'unknown')
                        print(f"🧠 BRAIN_LOOP: Processed {client_id} -> confidence: {confidence:.2f}, cycle: {cycle_time_ms:.1f}ms, mode: {cognitive_mode}")
            else:
                # Don't process yet - data is fresh but we recently processed this client
                # This will eventually allow idle cycles when no new data arrives
                pass
            
        except Exception as e:
            print(f"⚠️ Error processing sensors for {client_id}: {e}")
    
    def _perform_maintenance_tasks(self):
        """Perform brain maintenance during idle cycles."""
        try:
            # Use brain maintenance interface if available
            if hasattr(self.brain, 'run_recommended_maintenance'):
                # Modern unified maintenance system
                performed = self.brain.run_recommended_maintenance()
                if any(performed.values()):
                    self.maintenance_tasks_performed += 1
                    # Log maintenance scheduling decision
                    performed_types = [k for k, v in performed.items() if v]
                    print(f"🔧 BRAIN_LOOP: Maintenance triggered -> {', '.join(performed_types)}")
                elif self.total_cycles % 100 == 0:  # Periodic logging
                    print(f"🔧 BRAIN_LOOP: No maintenance needed (cycle {self.total_cycles}, idle: {self.idle_cycles})")
            else:
                # Fallback to legacy maintenance tasks
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
            'sensor_skip_cycles': self.sensor_skip_cycles,
            'utilization': utilization,
            'sensor_check_probability': self.sensor_check_probability,
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0.5,
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
        print(f"🌀 Sensor skip cycles: {stats['sensor_skip_cycles']:,}")
        print(f"📊 Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"🎲 Sensor check probability: {stats['sensor_check_probability']:.1%}")
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