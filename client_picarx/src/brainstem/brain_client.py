#!/usr/bin/env python3
"""
Simple Brain Client

Minimal TCP client for brain server communication.
Using the same MessageProtocol as the server.
"""

import socket
import struct
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class BrainServerConfig:
    """Brain server connection config."""
    host: str = "localhost"
    port: int = 9999
    timeout: float = 0.05  # 50ms for 20Hz operation
    sensory_dimensions: int = 24
    action_dimensions: int = 4


class MessageProtocol:
    """Message protocol matching server's protocol.py"""
    
    # Message types
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1  
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    def __init__(self):
        self.max_vector_size = 1024
    
    def encode_sensory_input(self, sensory_vector: List[float]) -> bytes:
        """Encode sensory input vector for transmission."""
        return self._encode_vector(sensory_vector, self.MSG_SENSORY_INPUT)
    
    def encode_handshake(self, capabilities: List[float]) -> bytes:
        """Encode handshake message with robot capabilities."""
        return self._encode_vector(capabilities, self.MSG_HANDSHAKE)
    
    def _encode_vector(self, vector: List[float], msg_type: int) -> bytes:
        """Encode a vector with message header."""
        if len(vector) > self.max_vector_size:
            raise ValueError(f"Vector too large: {len(vector)} > {self.max_vector_size}")
        
        # Convert to float32 for consistent precision
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        
        # Message length = type(1) + vector_length(4) + vector_data
        message_length = 1 + 4 + len(vector_data)
        
        # Pack message: length + type + vector_length + vector_data
        header = struct.pack('!IBI', message_length, msg_type, len(vector))
        
        return header + vector_data
    
    def decode_message(self, data: bytes) -> Tuple[int, List[float]]:
        """Decode received message."""
        if len(data) < 9:  # Minimum message size
            raise ValueError("Message too short")
        
        # Unpack header
        message_length, msg_type, vector_length = struct.unpack('!IBI', data[:9])
        
        # Validate message
        expected_length = 1 + 4 + (vector_length * 4)
        if message_length != expected_length:
            raise ValueError(f"Invalid message length: {message_length} != {expected_length}")
        
        if vector_length > self.max_vector_size:
            raise ValueError(f"Vector too large: {vector_length}")
        
        # Extract vector data
        vector_data = data[9:9 + (vector_length * 4)]
        if len(vector_data) != vector_length * 4:
            raise ValueError("Incomplete vector data")
        
        # Unpack vector
        vector = list(struct.unpack(f'{vector_length}f', vector_data))
        
        return msg_type, vector
    
    def receive_message(self, sock: socket.socket, timeout: float = 5.0) -> Tuple[int, List[float]]:
        """Receive and decode a complete message from socket."""
        sock.settimeout(timeout)
        
        # First read the message length (4 bytes)
        length_data = sock.recv(4)
        if len(length_data) < 4:
            raise ValueError("Failed to read message length")
        
        message_length = struct.unpack('!I', length_data)[0]
        
        # Read the rest of the message
        remaining = message_length
        message_data = b''
        while remaining > 0:
            chunk = sock.recv(min(remaining, 4096))
            if not chunk:
                raise ValueError("Connection closed while reading message")
            message_data += chunk
            remaining -= len(chunk)
        
        # Decode the complete message
        full_data = length_data + message_data
        return self.decode_message(full_data)


class BrainClient:
    """
    Simple TCP client for brain server.
    
    Uses MessageProtocol for communication with server.
    """
    
    def __init__(self, config: BrainServerConfig = None):
        """Initialize client."""
        self.config = config or BrainServerConfig()
        self.socket = None
        self.connected = False
        self.protocol = MessageProtocol()
        
    def connect(self) -> bool:
        """Connect to brain server and perform handshake."""
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.config.timeout)
            
            # Connect
            self.socket.connect((self.config.host, self.config.port))
            
            # Perform handshake to negotiate dimensions
            if self._handshake():
                self.connected = True
                print(f"✅ Connected to brain at {self.config.host}:{self.config.port}")
                print(f"   Negotiated: {self.config.sensory_dimensions} sensory, {self.config.action_dimensions} motor")
                return True
            else:
                print("❌ Handshake failed")
                self.socket.close()
                return False
                
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    def _handshake(self) -> bool:
        """Perform handshake with brain server."""
        try:
            # Prepare handshake data
            # [robot_version, sensory_size, action_size, hardware_type, capabilities_mask]
            capabilities = 0
            capabilities |= 1  # Visual (we have camera support)
            capabilities |= 2  # Audio (we have mic/speaker)
            
            handshake_data = [
                1.0,  # Robot version
                float(self.config.sensory_dimensions),  # Sensory dimensions
                float(self.config.action_dimensions),   # Action dimensions
                1.0,  # Hardware type (1.0 = PiCar-X)
                float(capabilities)  # Capabilities mask
            ]
            
            # Send handshake using MessageProtocol
            message = self.protocol.encode_handshake(handshake_data)
            self.socket.sendall(message)
            
            # Receive brain's handshake response
            msg_type, brain_handshake = self.protocol.receive_message(self.socket, timeout=5.0)
            
            if msg_type == MessageProtocol.MSG_HANDSHAKE and len(brain_handshake) >= 5:
                brain_version = brain_handshake[0]
                accepted_sensory = int(brain_handshake[1])
                accepted_action = int(brain_handshake[2])
                gpu_available = brain_handshake[3] > 0
                brain_capabilities = int(brain_handshake[4]) if len(brain_handshake) > 4 else 0
                
                # Debug: Show what brain sent
                if accepted_sensory != self.config.sensory_dimensions or accepted_action != self.config.action_dimensions:
                    print(f"⚠️ Dimension mismatch in handshake!")
                    print(f"   Sent: {self.config.sensory_dimensions}s/{self.config.action_dimensions}m")
                    print(f"   Brain returned: {accepted_sensory}s/{accepted_action}m")
                    print(f"   Full brain response: {brain_handshake}")
                
                # DON'T update our config - use what we originally configured
                # The brain should accept our dimensions or reject them
                # self.config.sensory_dimensions = accepted_sensory
                # self.config.action_dimensions = accepted_action
                
                print(f"🤝 Handshake complete: Brain v{brain_version:.1f}, GPU={'✓' if gpu_available else '✗'}")
                return True
                
            return False
            
        except Exception as e:
            print(f"Handshake error: {e}")
            return False
    
    def process_sensors(self, sensor_data: List[float]) -> Optional[Dict[str, Any]]:
        """
        Send sensor data, receive motor commands.
        
        Returns dict with 'motor_commands' or None on error.
        """
        if not self.connected:
            return None
        
        try:
            # Ensure correct dimensions
            if len(sensor_data) < self.config.sensory_dimensions:
                sensor_data.extend([0.5] * (self.config.sensory_dimensions - len(sensor_data)))
            elif len(sensor_data) > self.config.sensory_dimensions:
                sensor_data = sensor_data[:self.config.sensory_dimensions]
            
            # Send sensor data using MessageProtocol
            message = self.protocol.encode_sensory_input(sensor_data)
            self.socket.sendall(message)
            
            # Receive motor commands
            msg_type, motor_commands = self.protocol.receive_message(self.socket, timeout=self.config.timeout)
            
            if msg_type == MessageProtocol.MSG_ACTION_OUTPUT:
                return {'motor_commands': motor_commands}
            
            return None
            
        except socket.timeout:
            # Timeout is ok, just means no response yet
            return None
        except Exception as e:
            print(f"❌ Communication error: {e}")
            self.connected = False
            return None
    
    def disconnect(self):
        """Disconnect from brain server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        print("🔌 Disconnected from brain")


def test_connection():
    """Test brain connection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test brain connection')
    parser.add_argument('--host', default='localhost', help='Brain server host')
    parser.add_argument('--port', type=int, default=9999, help='Brain server port')
    
    args = parser.parse_args()
    
    print(f"Testing connection to {args.host}:{args.port}...")
    
    config = BrainServerConfig(host=args.host, port=args.port)
    client = BrainClient(config)
    
    if client.connect():
        print("✅ Connection successful!")
        
        # Test sending data
        test_sensors = [0.5] * 24
        response = client.process_sensors(test_sensors)
        
        if response:
            print(f"✅ Got response: {response}")
        else:
            print("⚠️  No response (brain might be processing)")
        
        client.disconnect()
    else:
        print("❌ Connection failed")


if __name__ == "__main__":
    test_connection()