"""
Brainstem Interface Classes
Provides unified interface for both simulation and real hardware
"""

import time
import json
import socket
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datetime import datetime


class BrainstemInterface(ABC):
    """
    Abstract base class for brainstem communication
    Allows switching between simulation and real hardware
    """
    
    @abstractmethod
    def get_sensor_readings(self) -> Dict[str, Any]:
        """Get current sensor data"""
        pass
    
    @abstractmethod
    def send_motor_commands(self, commands: Dict[str, float]) -> None:
        """Send motor commands to actuators"""
        pass
    
    @abstractmethod
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """Get list of available sensors and actuators"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean shutdown"""
        pass


class SimulationBrainstemInterface(BrainstemInterface):
    """
    Interface to Grid World simulation
    Acts as virtual brainstem for testing brain implementation
    """
    
    def __init__(self):
        from grid_world_simulation import GridWorldBrainstem
        self.simulation = GridWorldBrainstem()
        self.connected = True
        self.sequence_id = 0
        
    def get_sensor_readings(self) -> Dict[str, Any]:
        """Get sensor data from simulation"""
        if not self.connected:
            raise ConnectionError("Simulation not connected")
            
        self.sequence_id += 1
        return {
            'message_type': 'sensory_data',
            'brainstem_id': 'simulation_001',
            'sequence_id': self.sequence_id,
            'timestamp': time.time(),
            'sensor_values': self.simulation.get_sensor_readings(),
            'actuator_positions': self.simulation.get_actuator_positions(),
            'network_latency': 0.001  # Simulated minimal latency
        }
    
    def send_motor_commands(self, commands: Dict[str, float]) -> None:
        """Send motor commands to simulation"""
        if not self.connected:
            raise ConnectionError("Simulation not connected")
            
        command_packet = {
            'message_type': 'motor_commands',
            'brain_id': 'brain_001',
            'sequence_id': self.sequence_id + 1,
            'timestamp': time.time(),
            'actuator_commands': commands
        }
        
        self.simulation.execute_motor_commands(commands)
    
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """Get simulated hardware capabilities"""
        return {
            'message_type': 'hardware_capabilities',
            'brainstem_id': 'simulation_001',
            'timestamp': time.time(),
            'sensors': [
                {
                    'id': 'distance_sensors',
                    'type': 'distance',
                    'data_size': 4,
                    'update_frequency': 50
                },
                {
                    'id': 'vision_features', 
                    'type': 'camera_features',
                    'data_size': 13,
                    'update_frequency': 30
                },
                {
                    'id': 'internal_state',
                    'type': 'internal',
                    'data_size': 5,
                    'update_frequency': 50
                }
            ],
            'actuators': [
                {
                    'id': 'forward_motor',
                    'type': 'motor',
                    'range': [-1.0, 1.0],
                    'safe_default': 0.0
                },
                {
                    'id': 'turn_motor',
                    'type': 'motor', 
                    'range': [-1.0, 1.0],
                    'safe_default': 0.0
                },
                {
                    'id': 'brake_motor',
                    'type': 'motor',
                    'range': [0.0, 1.0],
                    'safe_default': 0.0
                }
            ]
        }
    
    def is_connected(self) -> bool:
        """Check if simulation is running"""
        return self.connected
    
    def close(self) -> None:
        """Shutdown simulation"""
        self.connected = False
        if hasattr(self.simulation, 'close'):
            self.simulation.close()


class NetworkBrainstemInterface(BrainstemInterface):
    """
    Interface to real brainstem over network
    For use with actual PiCar-X hardware
    """
    
    def __init__(self, brainstem_ip: str, base_port: int = 5000):
        self.brainstem_ip = brainstem_ip
        self.base_port = base_port
        
        # Separate sockets for different message types
        self.sensor_socket = None      # UDP for high-frequency sensor data
        self.command_socket = None     # TCP for reliable motor commands
        self.status_socket = None      # TCP for status and capabilities
        
        self.connected = False
        self.sequence_id = 0
        
        self._establish_connections()
    
    def _establish_connections(self):
        """Establish network connections to brainstem"""
        try:
            # UDP socket for sensor data (high frequency, can tolerate loss)
            self.sensor_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sensor_socket.settimeout(0.1)  # 100ms timeout
            
            # TCP socket for motor commands (reliable delivery)
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.command_socket.connect((self.brainstem_ip, self.base_port + 1))
            self.command_socket.settimeout(0.1)
            
            # TCP socket for status and capabilities
            self.status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.status_socket.connect((self.brainstem_ip, self.base_port + 2))
            self.status_socket.settimeout(1.0)  # Longer timeout for status
            
            self.connected = True
            print(f"Connected to brainstem at {self.brainstem_ip}")
            
        except Exception as e:
            print(f"Failed to connect to brainstem: {e}")
            self.connected = False
            self._cleanup_sockets()
    
    def get_sensor_readings(self) -> Dict[str, Any]:
        """Receive sensor data via UDP"""
        if not self.connected:
            raise ConnectionError("Not connected to brainstem")
        
        try:
            # Request sensor data
            request = json.dumps({'request': 'sensor_data'}).encode()
            self.sensor_socket.sendto(request, (self.brainstem_ip, self.base_port))
            
            # Receive response
            data, addr = self.sensor_socket.recvfrom(4096)
            return json.loads(data.decode())
            
        except socket.timeout:
            raise TimeoutError("Timeout waiting for sensor data")
        except Exception as e:
            raise ConnectionError(f"Error receiving sensor data: {e}")
    
    def send_motor_commands(self, commands: Dict[str, float]) -> None:
        """Send motor commands via TCP"""
        if not self.connected:
            raise ConnectionError("Not connected to brainstem")
        
        try:
            self.sequence_id += 1
            command_packet = {
                'message_type': 'motor_commands',
                'brain_id': 'brain_001',
                'sequence_id': self.sequence_id,
                'timestamp': time.time(),
                'actuator_commands': commands
            }
            
            message = json.dumps(command_packet).encode()
            self.command_socket.send(len(message).to_bytes(4, 'big') + message)
            
        except Exception as e:
            raise ConnectionError(f"Error sending motor commands: {e}")
    
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """Get hardware capabilities via TCP"""
        if not self.connected:
            raise ConnectionError("Not connected to brainstem")
        
        try:
            request = {
                'message_type': 'capability_request',
                'brain_id': 'brain_001',
                'timestamp': time.time()
            }
            
            message = json.dumps(request).encode()
            self.status_socket.send(len(message).to_bytes(4, 'big') + message)
            
            # Receive response
            length_bytes = self.status_socket.recv(4)
            message_length = int.from_bytes(length_bytes, 'big')
            
            response_data = b''
            while len(response_data) < message_length:
                chunk = self.status_socket.recv(message_length - len(response_data))
                response_data += chunk
            
            return json.loads(response_data.decode())
            
        except Exception as e:
            raise ConnectionError(f"Error getting capabilities: {e}")
    
    def is_connected(self) -> bool:
        """Check if all connections are active"""
        return self.connected and all([
            self.sensor_socket,
            self.command_socket, 
            self.status_socket
        ])
    
    def close(self) -> None:
        """Close all network connections"""
        self.connected = False
        self._cleanup_sockets()
    
    def _cleanup_sockets(self):
        """Close all sockets safely"""
        for sock in [self.sensor_socket, self.command_socket, self.status_socket]:
            if sock:
                try:
                    sock.close()
                except:
                    pass
        
        self.sensor_socket = None
        self.command_socket = None
        self.status_socket = None


def create_brainstem_interface(mode: str = "simulation", **kwargs) -> BrainstemInterface:
    """
    Factory function to create appropriate brainstem interface
    
    Args:
        mode: "simulation" or "network"
        **kwargs: Additional arguments for specific interfaces
    
    Returns:
        BrainstemInterface instance
    """
    if mode == "simulation":
        return SimulationBrainstemInterface()
    
    elif mode == "network":
        brainstem_ip = kwargs.get('brainstem_ip', '192.168.1.100')
        base_port = kwargs.get('base_port', 5000)
        return NetworkBrainstemInterface(brainstem_ip, base_port)
    
    else:
        raise ValueError(f"Unknown brainstem mode: {mode}")


# Utility functions for brainstem communication
def receive_sensory_data(brainstem_interface: BrainstemInterface) -> Dict[str, Any]:
    """Wrapper function for receiving sensor data"""
    return brainstem_interface.get_sensor_readings()


def send_motor_commands(brainstem_interface: BrainstemInterface, commands: Dict[str, float]) -> None:
    """Wrapper function for sending motor commands"""
    brainstem_interface.send_motor_commands(commands)


def receive_hardware_capabilities(brainstem_interface: BrainstemInterface) -> Dict[str, Any]:
    """Wrapper function for getting hardware capabilities"""
    return brainstem_interface.get_hardware_capabilities()


# Connection management
def connect_to_brainstem_or_simulation(mode: str = None) -> BrainstemInterface:
    """
    Connect to brainstem based on environment or explicit mode
    """
    import os
    
    if mode is None:
        mode = os.getenv('ROBOT_MODE', 'simulation')
    
    print(f"Connecting in {mode} mode...")
    
    if mode == 'simulation':
        return create_brainstem_interface('simulation')
    elif mode == 'network':
        brainstem_ip = os.getenv('BRAINSTEM_IP', '192.168.1.100')
        return create_brainstem_interface('network', brainstem_ip=brainstem_ip)
    else:
        raise ValueError(f"Unknown robot mode: {mode}")


def send_brain_ready_signal(brainstem_interface: BrainstemInterface) -> None:
    """Signal to brainstem that brain is ready to start"""
    if hasattr(brainstem_interface, 'send_status_message'):
        brainstem_interface.send_status_message({
            'status': 'brain_ready',
            'timestamp': time.time()
        })
    # For simulation, no explicit ready signal needed
