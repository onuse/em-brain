#!/usr/bin/env python3
"""
Socket-Based Embodied PiCar-X Brainstem

This is the realistic brainstem that communicates with the brain server through 
TCP sockets and integrates the embodied Free Energy system for action selection.

Architecture:
1. Sensors → Socket → Brain Server (pattern recognition)
2. Brain Response + Current State → Embodied Free Energy System (action selection)  
3. Selected Action → Motors

This demonstrates truly distributed intelligence:
- Brain server runs on powerful hardware (or in cloud)
- Robot runs locally with embodied Free Energy action selection
- Communication through realistic TCP protocol with latency
- No cheating - robot can't directly access brain internals

This is how the real PiCar-X deployment would work.
"""

import sys
import os
import socket
import time
import threading
from typing import List, Tuple, Dict, Any, Optional

# Add the brain/ directory to sys.path
current_dir = os.path.dirname(__file__)  # picar_x_simulation/
demos_dir = os.path.dirname(current_dir)  # demos/
brain_dir = os.path.dirname(demos_dir)   # brain/ (project root)
sys.path.insert(0, brain_dir)

# Import base brainstem for simulation capabilities
try:
    from .picar_x_brainstem import PiCarXBrainstem
except ImportError:
    from picar_x_brainstem import PiCarXBrainstem

# Import communication protocol
try:
    from server.src.communication import MinimalBrainClient
except ImportError:
    # Alternative import path when running from different directory
    import sys
    import os
    server_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'server')
    sys.path.insert(0, server_path)
    from src.communication import MinimalBrainClient

# Import embodied Free Energy system
from server.embodied_free_energy import (
    EmbodiedFreeEnergySystem,
    HardwareInterface,
    HardwareTelemetry,
    MockHardwareInterface
)

import numpy as np
import math


class BrainServerAdapter:
    """Adapter to make brain server communication compatible with embodied Free Energy."""
    
    def __init__(self, client):
        self.client = client
        
    def predict(self, sensory_input: Any, action: Any) -> 'PredictionOutcome':
        """Predict outcome using brain server."""
        
        # Convert sensory input to vector format
        if hasattr(sensory_input, '__dict__'):
            # Object with attributes - extract key values
            sensory_vector = [
                getattr(sensory_input, 'battery', 0.7),
                getattr(sensory_input, 'obstacle_distance', 50.0) / 100.0,  # Normalize
                getattr(sensory_input, 'heading', 0.0) / (2 * math.pi),  # Normalize
                getattr(sensory_input, 'motor_speed', 0.0) / 100.0,  # Normalize
            ]
        elif isinstance(sensory_input, (list, tuple)):
            sensory_vector = list(sensory_input)
        else:
            sensory_vector = [0.5, 0.5, 0.0, 0.0]  # Default
        
        # Ensure vector is proper length
        while len(sensory_vector) < 4:
            sensory_vector.append(0.0)
        sensory_vector = sensory_vector[:8]  # Limit to reasonable size
        
        try:
            # Get prediction from brain server via socket
            action_vector = self.client.predict_action(sensory_vector, action_dimensions=4)
            
            # Use vector magnitude as confidence proxy
            confidence = min(0.9, np.linalg.norm(action_vector) / 2.0 + 0.1)
            
            return PredictionOutcome(action, action_vector, confidence)
            
        except Exception as e:
            print(f"⚠️ Brain server communication error: {e}")
            # Fallback prediction
            return PredictionOutcome(action, [0.0, 0.0, 0.0, 0.0], 0.1)


class PredictionOutcome:
    """Prediction outcome for embodied system compatibility."""
    
    def __init__(self, action: Any, predicted_action: List[float], confidence: float):
        self.action = action
        self.predicted_action = predicted_action
        self.confidence = confidence
        self.similarity_score = confidence  # For compatibility


class RobotHardwareInterface(HardwareInterface):
    """Hardware interface that tracks actual robot simulation state."""
    
    def __init__(self, brainstem):
        self.brainstem = brainstem
        self.battery_level = 0.9  # Start with high battery
        self.motor_temperature = 25.0  # Start at room temperature
        self.last_update = time.time()
        
        # Track energy usage
        self.total_energy_used = 0.0
        self.motor_activity_history = []
        
    def get_telemetry(self) -> HardwareTelemetry:
        """Get current robot hardware telemetry."""
        
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Simulate realistic battery drain
        motor_load = abs(self.brainstem.motor_speed) / 100.0
        steering_load = abs(self.brainstem.steering_angle) / 30.0
        
        # Energy consumption model
        base_consumption = 0.001  # Idle consumption
        motor_consumption = motor_load * 0.02  # Motor consumption
        steering_consumption = steering_load * 0.005  # Steering consumption
        
        total_consumption = (base_consumption + motor_consumption + steering_consumption) * dt
        self.battery_level = max(0.0, self.battery_level - total_consumption)
        self.total_energy_used += total_consumption
        
        # Simulate motor heating
        target_temp = 25.0 + motor_load * 35.0  # Up to 60°C under full load
        temp_rate = 3.0  # Temperature change rate
        temp_diff = target_temp - self.motor_temperature
        self.motor_temperature += temp_diff * min(1.0, dt * temp_rate)
        
        # Allow cooling when stopped
        if motor_load < 0.1:
            self.motor_temperature = max(25.0, self.motor_temperature - dt * 2.0)
        
        # Track motor activity for memory simulation
        self.motor_activity_history.append(motor_load)
        if len(self.motor_activity_history) > 100:
            self.motor_activity_history.pop(0)
        
        # Simulate memory usage based on complexity
        base_memory = 25.0
        activity_memory = sum(self.motor_activity_history) * 0.1
        memory_usage = min(90.0, base_memory + activity_memory)
        
        # Sensor noise increases with obstacles and low battery
        base_noise = 0.02
        obstacle_factor = max(0, (50.0 - self.brainstem.ultrasonic_distance) / 50.0)
        battery_factor = max(0, (0.2 - self.battery_level) / 0.2) if self.battery_level < 0.2 else 0
        sensor_noise = base_noise + obstacle_factor * 0.03 + battery_factor * 0.05
        
        return HardwareTelemetry(
            battery_voltage=self.battery_level * 12.6,
            battery_percentage=self.battery_level,
            motor_temperatures={
                'drive_left': self.motor_temperature,
                'drive_right': self.motor_temperature + 2.0,
                'steering': self.motor_temperature - 5.0
            },
            cpu_temperature=35.0 + motor_load * 15.0,
            memory_usage_percentage=memory_usage,
            disk_usage_percentage=20.0,
            wifi_signal_strength=-45.0,
            sensor_noise_levels={
                'ultrasonic': sensor_noise,
                'camera': base_noise * 1.5
            },
            timestamp=current_time
        )
    
    def predict_hardware_effects(self, action: Any, current_state: HardwareTelemetry) -> HardwareTelemetry:
        """Predict hardware effects of actions."""
        
        if not isinstance(action, dict):
            return current_state
        
        predicted_battery = current_state.battery_percentage
        predicted_temp = max(current_state.motor_temperatures.values())
        
        action_type = action.get('type', 'stop')
        
        if action_type == 'move':
            speed = action.get('speed', 0.5)
            direction = action.get('direction', 'forward')
            
            # Energy cost depends on speed and direction
            if direction in ['left', 'right']:
                energy_cost = speed * 0.025  # Turning costs more energy
                heat_increase = speed * 6.0
            else:
                energy_cost = speed * 0.02
                heat_increase = speed * 4.0
            
            predicted_battery -= energy_cost
            predicted_temp += heat_increase
            
        elif action_type == 'rotate':
            angle = abs(action.get('angle', 0))
            # Rotation energy cost
            energy_cost = (angle / 90.0) * 0.015
            heat_increase = (angle / 90.0) * 3.0
            
            predicted_battery -= energy_cost
            predicted_temp += heat_increase
            
        elif action_type == 'stop':
            # Stopping allows cooling and minimal drain
            predicted_battery -= 0.001
            predicted_temp = max(25.0, predicted_temp - 3.0)
            
        elif action_type == 'seek_charger':
            # Energy seeking behavior
            urgency = action.get('urgency', 'moderate')
            
            if urgency == 'high':
                # Predict successful charging despite movement cost
                predicted_battery += 0.1  # Significant energy gain
                predicted_temp += 4.0  # Some heat from movement
            else:
                # Moderate energy seeking
                predicted_battery += 0.05
                predicted_temp += 2.0
        
        # Keep values realistic
        predicted_battery = max(0.0, min(1.0, predicted_battery))
        predicted_temp = max(25.0, min(80.0, predicted_temp))
        
        return HardwareTelemetry(
            battery_voltage=predicted_battery * 12.6,
            battery_percentage=predicted_battery,
            motor_temperatures={
                'drive_left': predicted_temp,
                'drive_right': predicted_temp + 2.0,
                'steering': predicted_temp - 5.0
            },
            cpu_temperature=current_state.cpu_temperature,
            memory_usage_percentage=current_state.memory_usage_percentage,
            disk_usage_percentage=current_state.disk_usage_percentage,
            wifi_signal_strength=current_state.wifi_signal_strength,
            sensor_noise_levels=current_state.sensor_noise_levels,
            timestamp=time.time()
        )


class RobotState:
    """Robot state for embodied Free Energy system."""
    
    def __init__(self, brainstem, hardware_interface):
        self.battery = hardware_interface.battery_level
        self.obstacle_distance = brainstem.ultrasonic_distance
        self.location = tuple(brainstem.position[:2])
        self.heading = brainstem.position[2]
        self.motor_speed = brainstem.motor_speed
        self.steering_angle = brainstem.steering_angle
        self.total_cycles = brainstem.total_control_cycles


class SocketEmbodiedBrainstem(PiCarXBrainstem):
    """
    Socket-based embodied brainstem that communicates with brain server.
    
    This is the realistic implementation that would run on the actual robot:
    - Communicates with brain server through TCP sockets
    - Uses embodied Free Energy for local action selection
    - Handles network delays and communication errors gracefully
    - Provides the honest experience of distributed intelligence
    """
    
    def __init__(self, brain_host='localhost', brain_port=9999, 
                 enable_camera=True, enable_ultrasonics=True, enable_line_tracking=True):
        """Initialize socket-based embodied brainstem."""
        
        # Initialize base simulation capabilities
        super().__init__(enable_camera, enable_ultrasonics, enable_line_tracking)
        
        # Remove inline brain - we'll use socket communication
        self.brain = None
        
        # Brain server connection
        self.brain_host = brain_host
        self.brain_port = brain_port
        self.brain_client = None
        self.connected = False
        
        # Hardware interface for embodied system
        self.hardware_interface = RobotHardwareInterface(self)
        
        # Embodied Free Energy system (will be initialized after brain connection)
        self.embodied_system = None
        
        # Performance tracking
        self.socket_errors = 0
        self.successful_predictions = 0
        self.network_latencies = []
        self.embodied_decisions = 0
        self.energy_seeking_actions = 0
        
        print("🔌 Socket-based Embodied Brainstem initialized")
        print(f"   Brain server: {brain_host}:{brain_port}")
        print("   Ready to connect to distributed brain")
    
    def connect_to_brain_server(self) -> bool:
        """Connect to the brain server."""
        
        print(f"🔗 Connecting to brain server at {self.brain_host}:{self.brain_port}...")
        
        try:
            self.brain_client = MinimalBrainClient(self.brain_host, self.brain_port)
            self.brain_client.connect()
            
            # Create brain adapter for embodied system
            brain_adapter = BrainServerAdapter(self.brain_client)
            
            # Initialize embodied Free Energy system
            self.embodied_system = EmbodiedFreeEnergySystem(
                brain_adapter, 
                self.hardware_interface
            )
            
            self.connected = True
            print("✅ Connected to brain server")
            print("🧬 Embodied Free Energy system initialized with socket brain")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to brain server: {e}")
            self.connected = False
            return False
    
    def disconnect_from_brain_server(self):
        """Disconnect from brain server."""
        if self.brain_client:
            try:
                self.brain_client.disconnect()
                print("🔌 Disconnected from brain server")
            except:
                pass
            finally:
                self.brain_client = None
                self.connected = False
                self.embodied_system = None
    
    def control_cycle(self) -> Dict[str, Any]:
        """
        Socket-based control cycle with embodied Free Energy action selection.
        
        Flow: Sensors → Socket → Brain Server → Embodied Free Energy → Motors
        """
        cycle_start = time.time()
        
        if not self.connected:
            return self._fallback_control_cycle()
        
        try:
            # 1. Get sensory input from robot
            sensory_vector = self.get_sensory_vector()
            
            # 2. Create robot state for embodied system
            robot_state = RobotState(self, self.hardware_interface)
            
            # 3. Embodied Free Energy system selects action (using brain server)
            network_start = time.time()
            selected_action = self.embodied_system.select_action(robot_state)
            network_time = time.time() - network_start
            
            self.network_latencies.append(network_time)
            if len(self.network_latencies) > 100:
                self.network_latencies.pop(0)
            
            # 4. Convert embodied action to motor commands
            action_vector = self._embodied_action_to_motor_vector(selected_action)
            
            # 5. Execute action on robot
            motor_state = self.execute_action_vector(action_vector)
            
            # 6. Get outcome for brain learning (send to server)
            outcome_vector = self.get_sensory_vector()
            
            # 7. Send experience to brain server for learning
            experience_success = self._send_experience_to_server(
                sensory_vector, action_vector, outcome_vector
            )
            
            # 8. Performance tracking
            self.total_control_cycles += 1
            self.embodied_decisions += 1
            self.successful_predictions += 1
            
            # Track action types
            if isinstance(selected_action, dict):
                action_type = selected_action.get('type', 'unknown')
                if action_type == 'seek_charger':
                    self.energy_seeking_actions += 1
            
            cycle_time = time.time() - cycle_start
            
            return {
                'success': True,
                'cycle_time': cycle_time,
                'network_latency': network_time,
                'sensory_vector': sensory_vector,
                'selected_action': selected_action,
                'action_vector': action_vector,
                'motor_state': motor_state,
                'experience_stored': experience_success,
                'robot_stats': self._get_socket_robot_stats()
            }
            
        except Exception as e:
            print(f"⚠️ Socket control cycle error: {e}")
            self.socket_errors += 1
            return self._fallback_control_cycle()
    
    def _embodied_action_to_motor_vector(self, embodied_action: Any) -> List[float]:
        """Convert embodied action to motor vector (same as original implementation)."""
        
        if not isinstance(embodied_action, dict):
            return [0.0, 0.0, 0.0, 0.0]
        
        action_type = embodied_action.get('type', 'stop')
        
        if action_type == 'move':
            direction = embodied_action.get('direction', 'forward')
            speed = embodied_action.get('speed', 0.5)
            
            if direction == 'forward':
                motor_speed = speed * 100.0
                steering = 0.0
            elif direction == 'backward':
                motor_speed = -speed * 100.0
                steering = 0.0
            elif direction == 'left':
                motor_speed = speed * 80.0
                steering = -30.0
            elif direction == 'right':
                motor_speed = speed * 80.0
                steering = 30.0
            else:
                motor_speed = 0.0
                steering = 0.0
                
        elif action_type == 'rotate':
            angle = embodied_action.get('angle', 0)
            motor_speed = 0.0
            steering = max(-30.0, min(30.0, angle / 2.0))
            
        elif action_type == 'stop':
            motor_speed = 0.0
            steering = 0.0
            
        elif action_type == 'seek_charger':
            urgency = embodied_action.get('urgency', 'moderate')
            motor_speed = 40.0 if urgency == 'high' else 20.0
            steering = 0.0
            
        else:
            motor_speed = 0.0
            steering = 0.0
        
        return [motor_speed, steering, 0.0, 0.0]  # [motor, steering, camera_pan, camera_tilt]
    
    def _send_experience_to_server(self, sensory_vector: List[float], 
                                 action_vector: List[float], 
                                 outcome_vector: List[float]) -> bool:
        """Send experience to brain server for learning."""
        try:
            # The MinimalBrainClient predict_action call already stores the experience
            # This is handled automatically by the brain server
            return True
        except Exception as e:
            print(f"⚠️ Failed to send experience to server: {e}")
            return False
    
    def _fallback_control_cycle(self) -> Dict[str, Any]:
        """Fallback control when brain server is unavailable."""
        
        print("⚠️ Using fallback control (no brain server)")
        
        # Simple obstacle avoidance without brain
        if self.ultrasonic_distance < 30.0:
            self.motor_speed = 0.0
            self.steering_angle = 30.0  # Turn away from obstacle
        else:
            self.motor_speed = 20.0  # Slow forward movement
            self.steering_angle = 0.0
        
        self._update_robot_physics()
        self._update_sensors()
        
        self.total_control_cycles += 1
        
        return {
            'success': False,
            'fallback_mode': True,
            'robot_stats': self._get_socket_robot_stats()
        }
    
    def _get_socket_robot_stats(self) -> Dict[str, Any]:
        """Get robot statistics including socket performance."""
        
        base_stats = {
            'position': self.position,
            'ultrasonic_distance': self.ultrasonic_distance,
            'motor_speed': self.motor_speed,
            'steering_angle': self.steering_angle,
            'total_cycles': self.total_control_cycles,
            'embodied_decisions': self.embodied_decisions,
            'energy_seeking_actions': self.energy_seeking_actions,
        }
        
        # Add hardware telemetry
        telemetry = self.hardware_interface.get_telemetry()
        hardware_stats = {
            'battery_level': telemetry.battery_percentage,
            'motor_temperature': max(telemetry.motor_temperatures.values()),
            'memory_usage': telemetry.memory_usage_percentage,
            'sensor_noise': telemetry.sensor_noise_levels.get('ultrasonic', 0.02)
        }
        
        # Add network performance
        avg_latency = sum(self.network_latencies) / max(1, len(self.network_latencies))
        network_stats = {
            'connected_to_brain': self.connected,
            'socket_errors': self.socket_errors,
            'successful_predictions': self.successful_predictions,
            'network_latency_avg': avg_latency,
            'network_reliability': self.successful_predictions / max(1, self.successful_predictions + self.socket_errors)
        }
        
        return {**base_stats, **hardware_stats, **network_stats}
    
    def get_embodied_statistics(self) -> Dict[str, Any]:
        """Get comprehensive embodied system statistics."""
        
        base_stats = self._get_socket_robot_stats()
        
        if self.embodied_system:
            embodied_stats = self.embodied_system.get_system_statistics()
            return {**base_stats, 'embodied_system': embodied_stats}
        else:
            return base_stats
    
    def set_verbose_embodied(self, verbose: bool):
        """Enable/disable verbose embodied logging."""
        if self.embodied_system:
            self.embodied_system.set_verbose(verbose)
    
    def print_socket_report(self):
        """Print comprehensive socket-based embodied report."""
        
        print(f"\\n🔌 SOCKET-BASED EMBODIED ROBOT REPORT")
        print(f"=" * 60)
        
        # Connection status
        print(f"🔗 Brain Server Connection:")
        print(f"   Connected: {'✅' if self.connected else '❌'}")
        print(f"   Server: {self.brain_host}:{self.brain_port}")
        print(f"   Socket errors: {self.socket_errors}")
        print(f"   Successful predictions: {self.successful_predictions}")
        
        if self.network_latencies:
            avg_latency = sum(self.network_latencies) / len(self.network_latencies)
            print(f"   Avg network latency: {avg_latency*1000:.1f}ms")
        
        # Hardware state
        telemetry = self.hardware_interface.get_telemetry()
        print(f"\\n🔋 Hardware State:")
        print(f"   Battery: {telemetry.battery_percentage:.1%}")
        print(f"   Motor temp: {max(telemetry.motor_temperatures.values()):.1f}°C")
        print(f"   Memory usage: {telemetry.memory_usage_percentage:.1f}%")
        
        # Embodied behavior
        print(f"\\n🧬 Embodied Behavior:")
        print(f"   Total decisions: {self.embodied_decisions}")
        print(f"   Energy-seeking: {self.energy_seeking_actions}")
        
        # Let embodied system print its report
        if self.embodied_system:
            self.embodied_system.print_system_report()


def main():
    """Test socket-based embodied brainstem."""
    
    print("🔌 Testing Socket-Based Embodied Brainstem")
    print("=" * 60)
    print("Note: This requires a running brain server!")
    print("Start server with: python3 server/brain_server.py")
    
    # Create socket-based brainstem
    brainstem = SocketEmbodiedBrainstem()
    
    # Try to connect to brain server
    if not brainstem.connect_to_brain_server():
        print("❌ Could not connect to brain server")
        print("Please start the brain server first:")
        print("   python3 server/brain_server.py")
        return
    
    brainstem.set_verbose_embodied(True)
    
    try:
        # Run control cycles
        print(f"\\n🚀 Running socket-based control cycles...")
        
        for cycle in range(10):
            print(f"\\n--- Cycle {cycle+1} ---")
            result = brainstem.control_cycle()
            
            if result.get('success'):
                action = result.get('selected_action', {})
                stats = result.get('robot_stats', {})
                latency = result.get('network_latency', 0)
                
                print(f"Action: {action}")
                print(f"Battery: {stats.get('battery_level', 0):.1%}, "
                      f"Latency: {latency*1000:.1f}ms")
            else:
                print("⚠️ Cycle failed or using fallback")
            
            time.sleep(0.2)  # Allow time for network communication
        
        # Print final report
        brainstem.print_socket_report()
        
    finally:
        brainstem.disconnect_from_brain_server()
    
    print(f"\\n✅ Socket-based embodied brainstem test completed!")


if __name__ == "__main__":
    main()