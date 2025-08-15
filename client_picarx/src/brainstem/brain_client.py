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


class MessageProtocolError(Exception):
    """Exception raised for protocol-related errors."""
    pass


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
    
    # Protocol constants
    MAGIC_BYTES = 0xDEADBEEF
    
    # Validation limits
    MIN_VECTOR_LENGTH = 0
    # Must support vision data: 640x480 = 307,200 pixels + sensors = 307,212 values
    # For 4K: 3840x2160 = 8,294,400 pixels + sensors = ~8.3M values
    MAX_REASONABLE_VECTOR_LENGTH = 10_000_000  # Support up to 4K resolution
    MAX_MESSAGE_SIZE_BYTES = 50_000_000     # 50MB absolute maximum
    
    def __init__(self, max_vector_size: int = None):
        # Support up to 4K resolution (3840×2160 = 8,294,400 pixels)
        # Plus some overhead for other sensors
        # This allows ~32MB messages (8.3M * 4 bytes)
        # Can be overridden for specific use cases
        if max_vector_size is None:
            max_vector_size = 10_000_000  # 10 million values (~40MB max)
        self.max_vector_size = max_vector_size
    
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
        
        # Message length = type(1) + vector_length(4) + vector_data (excludes magic)
        message_length = 1 + 4 + len(vector_data)
        
        # Pack message: magic + length + type + vector_length + vector_data
        header = struct.pack('!IIBI', self.MAGIC_BYTES, message_length, msg_type, len(vector))
        
        return header + vector_data
    
    def decode_message(self, data: bytes) -> Tuple[int, List[float]]:
        """Decode received message."""
        if len(data) < 13:  # Minimum message size with magic bytes
            raise ValueError("Message too short")
        
        # Unpack header with magic bytes
        magic, message_length, msg_type, vector_length = struct.unpack('!IIBI', data[:13])
        
        # Validate magic bytes
        if magic != self.MAGIC_BYTES:
            raise MessageProtocolError(f"Invalid magic bytes: 0x{magic:08X} != 0x{self.MAGIC_BYTES:08X}")
        
        # Validate message length consistency
        expected_length = 1 + 4 + (vector_length * 4)
        if message_length != expected_length:
            raise MessageProtocolError(f"Message length mismatch: {message_length} != {expected_length} (type+vec_len+data)")
        
        # Enhanced vector validation
        if vector_length < self.MIN_VECTOR_LENGTH:
            raise MessageProtocolError(f"Vector length negative: {vector_length}")
        if vector_length > self.MAX_REASONABLE_VECTOR_LENGTH:
            raise MessageProtocolError(f"Vector length unreasonable: {vector_length} > {self.MAX_REASONABLE_VECTOR_LENGTH}")
        if vector_length > self.max_vector_size:
            raise MessageProtocolError(f"Vector too large: {vector_length} > {self.max_vector_size}")
        
        # Additional sanity checks
        if msg_type not in [self.MSG_SENSORY_INPUT, self.MSG_ACTION_OUTPUT, self.MSG_HANDSHAKE, self.MSG_ERROR]:
            raise MessageProtocolError(f"Unknown message type: {msg_type}")
        
        # Extract vector data
        vector_data = data[13:13 + (vector_length * 4)]
        if len(vector_data) != vector_length * 4:
            raise ValueError("Incomplete vector data")
        
        # Unpack vector
        vector = list(struct.unpack(f'{vector_length}f', vector_data))
        
        return msg_type, vector
    
    def receive_message(self, sock: socket.socket, timeout: float = 5.0) -> Tuple[int, List[float]]:
        """Receive and decode a complete message from socket."""
        sock.settimeout(timeout)
        
        try:
            # Receive magic bytes and message length first
            header_data = self._receive_exactly(sock, 8)  # magic + length
            magic, message_length = struct.unpack('!II', header_data)
            
            # Validate magic bytes first
            if magic != self.MAGIC_BYTES:
                # Try to resynchronize stream
                self._resync_stream(sock, timeout)
                raise MessageProtocolError(f"Invalid magic bytes: 0x{magic:08X}, stream may be corrupt")
            
            # Validate message length with multiple checks
            if message_length < 5:  # Minimum: type(1) + vector_length(4)
                raise MessageProtocolError(f"Message too small: {message_length} bytes")
            if message_length > self.MAX_MESSAGE_SIZE_BYTES:
                raise MessageProtocolError(f"Message too large: {message_length} bytes > {self.MAX_MESSAGE_SIZE_BYTES}")
            if message_length > (self.max_vector_size * 4 + 5):
                raise MessageProtocolError(f"Message larger than max vector: {message_length} bytes")
            
            # Receive the rest of the message
            message_data = self._receive_exactly(sock, message_length)
            
            # Decode complete message
            return self.decode_message(header_data + message_data)
            
        except socket.timeout:
            raise TimeoutError("Message receive timeout")
    
    def _receive_exactly(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from socket."""
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Socket closed unexpectedly")
            data += chunk
        return data
    
    def _resync_stream(self, sock: socket.socket, timeout: float):
        """
        Attempt to resynchronize the stream by looking for magic bytes.
        
        This is called when we detect stream corruption to try to recover.
        """
        sock.settimeout(min(timeout, 2.0))  # Shorter timeout for resync
        
        buffer = b''
        max_search_bytes = 1024  # Don't search forever
        searched = 0
        
        try:
            while searched < max_search_bytes:
                # Read one byte at a time looking for magic
                byte = sock.recv(1)
                if not byte:
                    raise ConnectionError("Connection closed during resync")
                
                buffer += byte
                searched += 1
                
                # Keep only last 4 bytes for magic detection
                if len(buffer) > 4:
                    buffer = buffer[-4:]
                
                # Check if we found magic bytes
                if len(buffer) == 4:
                    potential_magic = struct.unpack('!I', buffer)[0]
                    if potential_magic == self.MAGIC_BYTES:
                        # Found magic bytes! Stream should be synchronized now
                        return
                        
        except socket.timeout:
            pass  # Give up on resync
        
        # Could not resynchronize
        raise MessageProtocolError("Could not resynchronize stream")


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
            # Close any existing socket first
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            
            # Create fresh socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.config.timeout)
            # Disable Nagle's algorithm for real-time communication
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Try to increase send buffer (system may limit this)
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)  # Request 2MB
            except OSError:
                pass  # System doesn't allow, use default
            
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
            # Use sendall to ensure complete message is sent
            # This handles partial sends automatically
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
            # IMPORTANT: Close the socket to prevent corruption
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
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