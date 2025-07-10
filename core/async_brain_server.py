#!/usr/bin/env python3
"""
Async Brain Server - Decoupled brain processing with observation interface.

This separates the brain thinking from GUI rendering, allowing:
- Brain to run at full speed (1000+ FPS)
- GUI to observe at comfortable rate (30 FPS)
- Multiple observers without interference
- Clean architecture for physical robots
"""

import asyncio
import threading
import time
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import queue


@dataclass
class BrainState:
    """Snapshot of brain state for observation."""
    timestamp: float
    step_count: int
    node_count: int
    prediction_error: float
    robot_position: tuple
    robot_orientation: float
    robot_health: float
    robot_energy: float
    recent_action: Dict[str, float]
    brain_statistics: Dict[str, Any]
    drive_states: Dict[str, Any]
    learning_rate: float
    fps: float


@dataclass
class ObservationRequest:
    """Request for brain state observation."""
    observer_id: str
    requested_fields: List[str]
    update_frequency: float  # Hz
    callback: Optional[Callable] = None


class AsyncBrainServer:
    """
    Async brain server that decouples brain thinking from observation.
    
    The brain runs in its own thread at maximum speed, while observers
    can subscribe to state updates at their preferred frequency.
    """
    
    def __init__(self, brainstem, brain_client):
        self.brainstem = brainstem
        self.brain_client = brain_client
        
        # State management
        self.current_state: Optional[BrainState] = None
        self.state_lock = threading.RLock()
        
        # Performance tracking
        self.brain_fps = 0.0
        self.step_count = 0
        self.last_fps_update = time.time()
        self.frame_times = []
        
        # Observer management
        self.observers: Dict[str, ObservationRequest] = {}
        self.observer_queues: Dict[str, queue.Queue] = {}
        
        # Control
        self.running = False
        self.brain_thread: Optional[threading.Thread] = None
        self.observer_threads: Dict[str, threading.Thread] = {}
        
        # Prediction tracking
        self.prediction_tracker = {
            'last_prediction': None,
            'last_sensory': None,
            'current_error': 0.0
        }
    
    def start(self):
        """Start the async brain server."""
        if self.running:
            return
        
        self.running = True
        
        # Start brain processing thread
        self.brain_thread = threading.Thread(target=self._brain_loop, daemon=True)
        self.brain_thread.start()
        
        print("🧠 Async Brain Server started - brain running at full speed")
    
    def stop(self):
        """Stop the async brain server."""
        self.running = False
        
        # Stop brain thread
        if self.brain_thread:
            self.brain_thread.join(timeout=1.0)
        
        # Stop observer threads
        for observer_id, thread in self.observer_threads.items():
            thread.join(timeout=0.5)
        
        print("🧠 Async Brain Server stopped")
    
    def _brain_loop(self):
        """Main brain processing loop - runs at maximum speed."""
        last_state_update = time.time()
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Perform one brain step
                self._perform_brain_step()
                
                # Update state snapshot every 10ms (100 Hz) or when significant changes occur
                if time.time() - last_state_update >= 0.01 or self.step_count % 100 == 0:
                    self._update_state_snapshot()
                    last_state_update = time.time()
                
                # Update performance metrics
                self._update_performance_metrics(loop_start)
                
            except Exception as e:
                print(f"❌ Brain loop error: {e}")
                time.sleep(0.001)  # Brief pause on error
    
    def _perform_brain_step(self):
        """Perform one brain thinking step."""
        # Get current sensor readings
        sensor_packet = self.brainstem.get_sensor_readings()
        
        # Generate motor action using brain
        motor_commands = self._generate_brain_action(sensor_packet)
        
        # Execute action in simulation
        is_alive = self.brainstem.execute_motor_commands(motor_commands)
        
        # Get new sensor readings
        new_sensor_packet = self.brainstem.get_sensor_readings()
        
        # Calculate prediction error
        prediction_error = self._calculate_prediction_error(sensor_packet, new_sensor_packet)
        
        # Update internal state
        self.step_count += 1
        self.prediction_tracker['current_error'] = prediction_error
        
        # Handle robot death
        if not is_alive:
            self.brainstem.reset_robot()
    
    def _generate_brain_action(self, sensor_packet) -> Dict[str, float]:
        """Generate motor action using the brain system."""
        # Extract robot state
        robot_state = {
            'sensors': sensor_packet.sensor_values,
            'position': self.brainstem.simulation.robot.position,
            'orientation': self.brainstem.simulation.robot.orientation,
            'health': self.brainstem.simulation.robot.health,
            'energy': self.brainstem.simulation.robot.energy
        }
        
        # Create sensory packet
        from core.communication import SensoryPacket
        sensory_packet = SensoryPacket(
            sequence_id=self.step_count,
            sensor_values=robot_state['sensors'],
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        
        # Get brain prediction
        mental_context = robot_state['sensors'][:8] if len(robot_state['sensors']) >= 8 else robot_state['sensors']
        
        prediction = self.brain_client.process_sensory_input(
            sensory_packet,
            mental_context,
            threat_level="normal",
            robot_position=robot_state['position'],
            robot_orientation=robot_state['orientation']
        )
        
        # Store prediction for error calculation
        self.prediction_tracker['last_prediction'] = prediction
        self.prediction_tracker['last_sensory'] = robot_state['sensors']
        
        return prediction.motor_action if prediction else {
            'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 0.0
        }
    
    def _calculate_prediction_error(self, old_packet, new_packet) -> float:
        """Calculate prediction error between sensor readings."""
        if self.prediction_tracker['last_prediction'] and self.prediction_tracker['last_sensory']:
            old_sensors = self.prediction_tracker['last_sensory']
            new_sensors = new_packet.sensor_values
            
            if len(old_sensors) == len(new_sensors):
                error_sum = sum((new - old) ** 2 for old, new in zip(old_sensors, new_sensors))
                return (error_sum / len(old_sensors)) ** 0.5
        
        return 0.0
    
    def _update_state_snapshot(self):
        """Update the current state snapshot for observers."""
        try:
            # Get current robot state
            robot = self.brainstem.simulation.robot
            
            # Get brain statistics (use approximate for performance)
            brain_stats = self.brain_client.get_brain_statistics()
            
            # Get drive states if available
            drive_states = {}
            try:
                predictor = self.brain_client.get_predictor()
                if hasattr(predictor, 'motivation_system'):
                    drive_states = predictor.motivation_system.get_motivation_statistics()
            except:
                pass
            
            # Create state snapshot
            state = BrainState(
                timestamp=time.time(),
                step_count=self.step_count,
                node_count=brain_stats.get('graph_stats', {}).get('total_nodes', 0),
                prediction_error=self.prediction_tracker['current_error'],
                robot_position=robot.position,
                robot_orientation=robot.orientation,
                robot_health=robot.health,
                robot_energy=robot.energy,
                recent_action=getattr(robot, 'last_action', {}),
                brain_statistics=brain_stats,
                drive_states=drive_states,
                learning_rate=self._calculate_learning_rate(),
                fps=self.brain_fps
            )
            
            # Update state thread-safely
            with self.state_lock:
                self.current_state = state
            
            # Notify observers
            self._notify_observers(state)
            
        except Exception as e:
            print(f"❌ State update error: {e}")
    
    def _calculate_learning_rate(self) -> float:
        """Calculate current learning rate (nodes per second)."""
        if len(self.frame_times) < 10:
            return 0.0
        
        # Use recent frame times to estimate learning rate
        recent_time = sum(self.frame_times[-10:])
        if recent_time > 0:
            return 10.0 / recent_time  # 10 frames in recent_time seconds
        
        return 0.0
    
    def _update_performance_metrics(self, loop_start: float):
        """Update brain performance metrics."""
        frame_time = time.time() - loop_start
        self.frame_times.append(frame_time)
        
        # Keep only recent frame times
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-50:]
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.brain_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            self.last_fps_update = current_time
    
    def _notify_observers(self, state: BrainState):
        """Notify all registered observers of state update."""
        for observer_id, request in self.observers.items():
            if observer_id in self.observer_queues:
                try:
                    # Non-blocking queue put
                    self.observer_queues[observer_id].put_nowait(state)
                except queue.Full:
                    # Drop old states if queue is full
                    try:
                        self.observer_queues[observer_id].get_nowait()
                        self.observer_queues[observer_id].put_nowait(state)
                    except queue.Empty:
                        pass
    
    def register_observer(self, observer_id: str, update_frequency: float = 30.0, 
                         requested_fields: List[str] = None) -> queue.Queue:
        """
        Register an observer for brain state updates.
        
        Args:
            observer_id: Unique identifier for the observer
            update_frequency: Desired update frequency in Hz
            requested_fields: Specific fields to observe (None = all)
            
        Returns:
            Queue for receiving state updates
        """
        if requested_fields is None:
            requested_fields = []
        
        # Create observation request
        request = ObservationRequest(
            observer_id=observer_id,
            requested_fields=requested_fields,
            update_frequency=update_frequency
        )
        
        # Create queue for this observer
        obs_queue = queue.Queue(maxsize=10)  # Buffer up to 10 states
        
        # Register observer
        self.observers[observer_id] = request
        self.observer_queues[observer_id] = obs_queue
        
        print(f"📡 Observer '{observer_id}' registered at {update_frequency} Hz")
        
        return obs_queue
    
    def unregister_observer(self, observer_id: str):
        """Unregister an observer."""
        if observer_id in self.observers:
            del self.observers[observer_id]
        if observer_id in self.observer_queues:
            del self.observer_queues[observer_id]
        if observer_id in self.observer_threads:
            self.observer_threads[observer_id].join(timeout=0.5)
            del self.observer_threads[observer_id]
        
        print(f"📡 Observer '{observer_id}' unregistered")
    
    def get_current_state(self) -> Optional[BrainState]:
        """Get the current brain state snapshot."""
        with self.state_lock:
            return self.current_state
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        with self.state_lock:
            return {
                'brain_fps': self.brain_fps,
                'step_count': self.step_count,
                'observers_count': len(self.observers),
                'average_frame_time': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0,
                'min_frame_time': min(self.frame_times) if self.frame_times else 0,
                'max_frame_time': max(self.frame_times) if self.frame_times else 0
            }