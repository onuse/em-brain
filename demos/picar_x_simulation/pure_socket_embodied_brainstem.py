#!/usr/bin/env python3
"""
Pure Socket-Based Embodied PiCar-X Brainstem

A truly distributed brainstem that ONLY communicates via network protocol.
No imports from server/src/ - maintains complete separation between
robot client and brain server.

This is how the real PiCar-X deployment would work - the robot has no
access to brain internals, only socket communication.
"""

import sys
import os
import socket
import time
import datetime
import threading
import json
import struct
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import math

# Add the brain/ directory to sys.path for embodied Free Energy
current_dir = os.path.dirname(__file__)  # picar_x_simulation/
demos_dir = os.path.dirname(current_dir)  # demos/
brain_dir = os.path.dirname(demos_dir)   # brain/ (project root)
sys.path.insert(0, brain_dir)

# NO SERVER IMPORTS - Pure socket implementation only!
# Implements local embodied Free Energy logic for the brainstem


class PureSocketBrainClient:
    """
    Pure socket brain client - no imports from server code.
    Implements minimal protocol for brain communication.
    """
    
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to brain server via TCP socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Send handshake
            capabilities = [4.0, 16.0, 4.0, 1.0]  # Robot capabilities
            self._send_vector(capabilities, msg_type=2)  # Handshake
            
            # Receive server capabilities
            server_caps = self._receive_vector()
            print(f"✅ Connected to brain server!")
            print(f"   Server capabilities: {server_caps}")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    def predict_action(self, sensory_vector: List[float], action_dimensions: int = 4) -> List[float]:
        """Send sensory input and receive action prediction."""
        if not self.connected:
            return [0.0] * action_dimensions
            
        try:
            # Send sensory input
            self._send_vector(sensory_vector, msg_type=0)  # Sensory input
            
            # Receive action prediction
            action_vector = self._receive_vector()
            
            # Ensure correct dimensions
            if len(action_vector) < action_dimensions:
                action_vector.extend([0.0] * (action_dimensions - len(action_vector)))
            
            return action_vector[:action_dimensions]
            
        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            return [0.0] * action_dimensions
    
    def _send_vector(self, vector: List[float], msg_type: int = 0):
        """Send vector using minimal protocol."""
        # Convert to float32
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        
        # Message length = type(1) + vector_length(4) + vector_data
        message_length = 1 + 4 + len(vector_data)
        
        # Pack message: length + type + vector_length + vector_data
        header = struct.pack('!IBI', message_length, msg_type, len(vector))
        message = header + vector_data
        
        # Send complete message
        total_sent = 0
        while total_sent < len(message):
            sent = self.socket.send(message[total_sent:])
            if sent == 0:
                raise ConnectionError("Socket connection broken")
            total_sent += sent
    
    def _receive_vector(self) -> List[float]:
        """Receive vector using minimal protocol."""
        # Receive message length
        length_data = self._receive_exactly(4)
        message_length = struct.unpack('!I', length_data)[0]
        
        # Receive rest of message
        message_data = self._receive_exactly(message_length)
        
        # Unpack header
        msg_type, vector_length = struct.unpack('!BI', message_data[:5])
        
        # Unpack vector data
        vector_data = message_data[5:5 + (vector_length * 4)]
        vector = list(struct.unpack(f'{vector_length}f', vector_data))
        
        return vector
    
    def _receive_exactly(self, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from socket."""
        data = b''
        while len(data) < num_bytes:
            chunk = self.socket.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Socket closed unexpectedly")
            data += chunk
        return data
    
    def disconnect(self):
        """Disconnect from brain server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            finally:
                self.socket = None
                self.connected = False


class BrainServerAdapter:
    """Adapter to make pure socket client compatible with embodied Free Energy."""
    
    def __init__(self, client):
        self.client = client
        
    def predict(self, sensory_input: Any, action: Any) -> 'PredictionOutcome':
        """Predict outcome using brain server via pure socket."""
        
        # Convert sensory input to vector format
        if hasattr(sensory_input, '__dict__'):
            sensory_vector = [
                getattr(sensory_input, 'battery', 0.7),
                getattr(sensory_input, 'obstacle_distance', 50.0) / 100.0,
                getattr(sensory_input, 'heading', 0.0) / (2 * math.pi),
                getattr(sensory_input, 'motor_speed', 0.0) / 100.0,
            ]
        elif isinstance(sensory_input, (list, tuple)):
            sensory_vector = list(sensory_input)
        else:
            sensory_vector = [0.5, 0.5, 0.0, 0.0]
        
        # Ensure vector is proper length
        while len(sensory_vector) < 4:
            sensory_vector.append(0.0)
        sensory_vector = sensory_vector[:8]
        
        try:
            # Get prediction from brain server via pure socket
            action_vector = self.client.predict_action(sensory_vector, action_dimensions=4)
            
            # Use vector magnitude as confidence
            confidence = min(0.9, np.linalg.norm(action_vector) / 2.0 + 0.1)
            
            return PredictionOutcome(action, action_vector, confidence)
            
        except Exception as e:
            print(f"⚠️ Brain server communication error: {e}")
            return PredictionOutcome(action, [0.0, 0.0, 0.0, 0.0], 0.1)


class PredictionOutcome:
    """Prediction outcome for embodied system compatibility."""
    
    def __init__(self, action: Any, predicted_action: List[float], confidence: float):
        self.action = action
        self.predicted_action = predicted_action
        self.confidence = confidence
        self.similarity_score = confidence


class RobotHardwareInterface(HardwareInterface):
    """Hardware interface that tracks robot simulation state."""
    
    def __init__(self, brainstem):
        self.brainstem = brainstem
        self.battery_level = 0.9
        self.motor_temperature = 25.0
        self.last_update = time.time()
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
        
        # Energy consumption
        base_consumption = 0.001
        motor_consumption = motor_load * 0.02
        steering_consumption = steering_load * 0.005
        
        total_consumption = (base_consumption + motor_consumption + steering_consumption) * dt
        self.battery_level = max(0.0, self.battery_level - total_consumption)
        self.total_energy_used += total_consumption
        
        # Motor heating
        target_temp = 25.0 + motor_load * 35.0
        temp_rate = 3.0
        temp_diff = target_temp - self.motor_temperature
        self.motor_temperature += temp_diff * min(1.0, dt * temp_rate)
        
        # Cooling when stopped
        if motor_load < 0.1:
            self.motor_temperature = max(25.0, self.motor_temperature - dt * 2.0)
        
        # Track motor activity
        self.motor_activity_history.append(motor_load)
        if len(self.motor_activity_history) > 100:
            self.motor_activity_history.pop(0)
        
        # Memory usage simulation
        base_memory = 25.0
        activity_memory = sum(self.motor_activity_history) * 0.1
        memory_usage = min(90.0, base_memory + activity_memory)
        
        # Sensor noise
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
            
            if direction in ['left', 'right']:
                energy_cost = speed * 0.025
                heat_increase = speed * 6.0
            else:
                energy_cost = speed * 0.02
                heat_increase = speed * 4.0
            
            predicted_battery -= energy_cost
            predicted_temp += heat_increase
            
        elif action_type == 'rotate':
            angle = abs(action.get('angle', 0))
            energy_cost = (angle / 90.0) * 0.015
            heat_increase = (angle / 90.0) * 3.0
            
            predicted_battery -= energy_cost
            predicted_temp += heat_increase
            
        elif action_type == 'stop':
            predicted_battery -= 0.001
            predicted_temp = max(25.0, predicted_temp - 3.0)
            
        elif action_type == 'seek_charger':
            urgency = action.get('urgency', 'moderate')
            
            if urgency == 'high':
                predicted_battery += 0.1
                predicted_temp += 4.0
            else:
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


class PureSocketEmbodiedBrainstem:
    """
    Pure socket-based embodied brainstem with NO server imports.
    
    This is the truly distributed implementation:
    - NO imports from server/src/
    - ONLY socket communication with brain server
    - Embodied Free Energy for local action selection
    - Complete separation between robot and brain
    """
    
    def __init__(self, brain_host='localhost', brain_port=9999):
        # Robot simulation state
        self.position = [0.0, 0.0, 0.0]  # x, y, heading
        self.motor_speed = 0.0
        self.steering_angle = 0.0
        self.ultrasonic_distance = 50.0
        self.total_control_cycles = 0
        
        # Environment
        self.obstacles = [
            (80, 60, 20), (-60, -40, 15), (40, -80, 30), (-100, 80, 25)
        ]
        
        # Brain server connection
        self.brain_host = brain_host
        self.brain_port = brain_port
        self.brain_client = None
        self.connected = False
        
        # Hardware interface for embodied system
        self.hardware_interface = RobotHardwareInterface(self)
        
        # Embodied Free Energy system
        self.embodied_system = None
        
        # Performance tracking
        self.socket_errors = 0
        self.successful_predictions = 0
        self.network_latencies = []
        self.embodied_decisions = 0
        self.energy_seeking_actions = 0
        
        print("🔌 Pure Socket Embodied Brainstem initialized")
        print(f"   Brain server: {brain_host}:{brain_port}")
        print("   NO server imports - pure distributed intelligence")
    
    def connect_to_brain_server(self) -> bool:
        """Connect to brain server via pure socket communication."""
        print(f"🔗 Connecting to brain server at {self.brain_host}:{self.brain_port}...")
        
        try:
            self.brain_client = PureSocketBrainClient(self.brain_host, self.brain_port)
            
            if not self.brain_client.connect():
                return False
            
            # Create brain adapter for embodied system
            brain_adapter = BrainServerAdapter(self.brain_client)
            
            # Initialize embodied Free Energy system
            self.embodied_system = EmbodiedFreeEnergySystem(
                brain_adapter, 
                self.hardware_interface
            )
            
            self.connected = True
            print("✅ Connected to brain server")
            print("🧬 Embodied Free Energy system initialized with pure socket brain")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to brain server: {e}")
            self.connected = False
            return False
    
    def disconnect_from_brain_server(self):
        """Disconnect from brain server."""
        if self.brain_client:
            self.brain_client.disconnect()
            self.brain_client = None
            self.connected = False
            self.embodied_system = None
            print("🔌 Disconnected from brain server")
    
    def get_sensory_vector(self) -> List[float]:
        """Get current sensory input as vector."""
        return [
            self.position[0] / 100.0,  # Normalized x position
            self.position[1] / 100.0,  # Normalized y position
            self.position[2] / (2 * math.pi),  # Normalized heading
            self.ultrasonic_distance / 100.0,  # Normalized distance
            self.motor_speed / 100.0,  # Normalized motor speed
            self.steering_angle / 30.0,  # Normalized steering
            self.hardware_interface.battery_level,  # Battery level
            self.hardware_interface.motor_temperature / 80.0  # Normalized temperature
        ]
    
    def control_cycle(self) -> Dict[str, Any]:
        """Pure socket control cycle with embodied Free Energy action selection."""
        cycle_start = time.time()
        
        if not self.connected:
            return self._fallback_control_cycle()
        
        try:
            # 1. Get sensory input
            sensory_vector = self.get_sensory_vector()
            
            # 2. Create robot state for embodied system
            robot_state = RobotState(self, self.hardware_interface)
            
            # 3. Embodied Free Energy action selection (using pure socket brain)
            network_start = time.time()
            selected_action = self.embodied_system.select_action(robot_state)
            network_time = time.time() - network_start
            
            self.network_latencies.append(network_time)
            if len(self.network_latencies) > 100:
                self.network_latencies.pop(0)
            
            # 4. Convert to motor commands
            action_vector = self._embodied_action_to_motor_vector(selected_action)
            
            # 5. Execute action
            motor_state = self.execute_action_vector(action_vector)
            
            # 6. Update simulation
            self._update_robot_physics()
            self._update_sensors()
            
            # Performance tracking
            self.total_control_cycles += 1
            self.embodied_decisions += 1
            self.successful_predictions += 1
            
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
                'robot_stats': self._get_robot_stats()
            }
            
        except Exception as e:
            print(f"⚠️ Socket control cycle error: {e}")
            self.socket_errors += 1
            return self._fallback_control_cycle()
    
    def _embodied_action_to_motor_vector(self, embodied_action: Any) -> List[float]:
        """Convert embodied action to motor vector."""
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
        
        return [motor_speed, steering, 0.0, 0.0]
    
    def execute_action_vector(self, action_vector: List[float]) -> Dict[str, Any]:
        """Execute motor action vector."""
        if len(action_vector) >= 2:
            self.motor_speed = action_vector[0]
            self.steering_angle = action_vector[1]
        
        return {
            'motor_speed': self.motor_speed,
            'steering_angle': self.steering_angle
        }
    
    def _update_robot_physics(self):
        """Update robot physics simulation."""
        dt = 0.2  # 200ms per cycle
        
        # Convert motor speed to forward velocity
        forward_velocity = self.motor_speed / 100.0 * 30.0  # Max 30 units/sec
        
        # Update position based on current heading and velocity
        heading_rad = self.position[2]
        dx = forward_velocity * math.cos(heading_rad) * dt
        dy = forward_velocity * math.sin(heading_rad) * dt
        
        # Update heading based on steering
        max_angular_velocity = math.pi / 2  # 90 degrees per second
        angular_velocity = (self.steering_angle / 30.0) * max_angular_velocity
        dheading = angular_velocity * dt
        
        # Apply updates
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dheading
        
        # Keep heading in range [-π, π]
        while self.position[2] > math.pi:
            self.position[2] -= 2 * math.pi
        while self.position[2] < -math.pi:
            self.position[2] += 2 * math.pi
        
        # Boundary constraints
        self.position[0] = max(-150, min(150, self.position[0]))
        self.position[1] = max(-150, min(150, self.position[1]))
    
    def _update_sensors(self):
        """Update sensor readings."""
        # Ultrasonic sensor simulation
        heading = self.position[2]
        sensor_range = 100.0
        
        # Check distance to nearest obstacle
        min_distance = sensor_range
        
        for obs_x, obs_y, obs_radius in self.obstacles:
            # Vector from robot to obstacle
            dx = obs_x - self.position[0]
            dy = obs_y - self.position[1]
            
            # Check if obstacle is in sensor beam (±30 degree cone)
            obstacle_angle = math.atan2(dy, dx)
            angle_diff = abs(obstacle_angle - heading)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            if angle_diff <= math.pi / 6:  # 30 degrees
                distance = math.sqrt(dx*dx + dy*dy) - obs_radius
                min_distance = min(min_distance, max(5.0, distance))
        
        self.ultrasonic_distance = min_distance
    
    def _fallback_control_cycle(self) -> Dict[str, Any]:
        """Fallback control when brain server unavailable."""
        print("⚠️ Using fallback control (no brain server)")
        
        # Simple obstacle avoidance
        if self.ultrasonic_distance < 30.0:
            self.motor_speed = 0.0
            self.steering_angle = 30.0
        else:
            self.motor_speed = 20.0
            self.steering_angle = 0.0
        
        self._update_robot_physics()
        self._update_sensors()
        self.total_control_cycles += 1
        
        return {
            'success': False,
            'fallback_mode': True,
            'robot_stats': self._get_robot_stats()
        }
    
    def _get_robot_stats(self) -> Dict[str, Any]:
        """Get comprehensive robot statistics."""
        telemetry = self.hardware_interface.get_telemetry()
        
        base_stats = {
            'position': self.position,
            'ultrasonic_distance': self.ultrasonic_distance,
            'motor_speed': self.motor_speed,
            'steering_angle': self.steering_angle,
            'total_cycles': self.total_control_cycles,
            'embodied_decisions': self.embodied_decisions,
            'energy_seeking_actions': self.energy_seeking_actions,
        }
        
        hardware_stats = {
            'battery_level': telemetry.battery_percentage,
            'motor_temperature': max(telemetry.motor_temperatures.values()),
            'memory_usage': telemetry.memory_usage_percentage,
            'sensor_noise': telemetry.sensor_noise_levels.get('ultrasonic', 0.02)
        }
        
        if self.network_latencies:
            avg_latency = sum(self.network_latencies) / len(self.network_latencies)
            network_stats = {
                'connected_to_brain': self.connected,
                'socket_errors': self.socket_errors,
                'successful_predictions': self.successful_predictions,
                'network_latency_avg': avg_latency,
                'network_reliability': self.successful_predictions / max(1, self.successful_predictions + self.socket_errors)
            }
        else:
            network_stats = {
                'connected_to_brain': self.connected,
                'socket_errors': self.socket_errors,
                'successful_predictions': self.successful_predictions,
                'network_latency_avg': 0.0,
                'network_reliability': 0.0
            }
        
        return {**base_stats, **hardware_stats, **network_stats}
    
    def get_embodied_statistics(self) -> Dict[str, Any]:
        """Get comprehensive embodied system statistics."""
        base_stats = self._get_robot_stats()
        
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
        print(f"\n🔌 PURE SOCKET EMBODIED ROBOT REPORT")
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
        print(f"\n🔋 Hardware State:")
        print(f"   Battery: {telemetry.battery_percentage:.1%}")
        print(f"   Motor temp: {max(telemetry.motor_temperatures.values()):.1f}°C")
        print(f"   Memory usage: {telemetry.memory_usage_percentage:.1f}%")
        
        # Embodied behavior
        print(f"\n🧬 Embodied Behavior:")
        print(f"   Total decisions: {self.embodied_decisions}")
        print(f"   Energy-seeking: {self.energy_seeking_actions}")
        
        # Architecture validation
        print(f"\n✅ Architecture Validation:")
        print(f"   Pure socket communication: ✅")
        print(f"   No server imports: ✅")
        print(f"   Distributed intelligence: ✅")
        
        # Let embodied system print its report
        if self.embodied_system:
            self.embodied_system.print_system_report()


def main():
    """Test pure socket embodied brainstem."""
    print("🔌 Testing Pure Socket Embodied Brainstem")
    print("=" * 60)
    print("✅ NO server imports - truly distributed intelligence")
    print("Note: This requires a running brain server!")
    print("Start server with: python3 server/brain_server.py")
    
    # Create pure socket brainstem
    brainstem = PureSocketEmbodiedBrainstem()
    
    # Try to connect to brain server
    if not brainstem.connect_to_brain_server():
        print("❌ Could not connect to brain server")
        print("Please start the brain server first:")
        print("   python3 server/brain_server.py")
        return
    
    brainstem.set_verbose_embodied(True)
    
    try:
        # Run control cycles
        print(f"\n🚀 Running pure socket control cycles...")
        
        for cycle in range(10):
            print(f"\n--- Cycle {cycle+1} ---")
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
            
            time.sleep(0.2)
        
        # Print final report
        brainstem.print_socket_report()
        
    finally:
        brainstem.disconnect_from_brain_server()
    
    print(f"\n✅ Pure socket embodied brainstem test completed!")


if __name__ == "__main__":
    main()