#!/usr/bin/env python3
"""
Brain Server Client

Handles TCP binary communication between the PiCar-X robot and the brain server.
Implements the proper binary protocol for sending sensor data and receiving
motor commands.
"""

import socket
import struct
import time
import threading
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass

try:
    from ..config.brainstem_config import BrainstemConfig, get_config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.brainstem_config import BrainstemConfig, get_config


@dataclass
class BrainServerConfig:
    """Configuration for brain server connection."""
    host: str = "localhost"
    port: int = 9999  # Standard brain server port
    timeout: float = 5.0
    retry_attempts: int = 3
    sensory_dimensions: int = 24
    action_dimensions: int = 4
    brainstem_config: BrainstemConfig = None
    
    def __post_init__(self):
        """Initialize from brainstem configuration."""
        if self.brainstem_config is None:
            self.brainstem_config = get_config()
        
        # Use values from config if not explicitly set
        if self.host is None:
            self.host = self.brainstem_config.network.brain_host
        if self.port is None:
            self.port = self.brainstem_config.network.brain_port
        if self.timeout is None:
            self.timeout = self.brainstem_config.network.connection_timeout
        if self.retry_attempts is None:
            self.retry_attempts = self.brainstem_config.network.max_reconnect_attempts
        if self.sensory_dimensions is None:
            self.sensory_dimensions = self.brainstem_config.sensors.brain_input_dimensions
        if self.action_dimensions is None:
            self.action_dimensions = self.brainstem_config.motors.brain_output_dimensions


class MessageProtocol:
    """Binary message protocol for brain communication."""
    
    # Message types
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    def __init__(self, config: BrainstemConfig = None):
        self.config = config or get_config()
        self.max_vector_size = self.config.network.max_vector_size
    
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
    
    def receive_message(self, sock: socket.socket, timeout: float = 5.0) -> Tuple[int, List[float]]:
        """
        Receive and decode a complete message from socket.
        
        Returns:
            Tuple of (message_type, vector_data)
        """
        sock.settimeout(timeout)
        
        try:
            # Receive message length first
            length_data = self._receive_exactly(sock, 4)
            message_length = struct.unpack('!I', length_data)[0]
            
            if message_length > (self.max_vector_size * 4 + 5):
                raise ValueError(f"Message too large: {message_length}")
            
            # Receive the rest of the message
            message_data = self._receive_exactly(sock, message_length)
            
            # Decode message
            msg_type, vector_length = struct.unpack('!BI', message_data[:5])
            vector_data = message_data[5:]
            
            # Unpack vector
            vector = list(struct.unpack(f'{vector_length}f', vector_data))
            
            return msg_type, vector
            
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
    
    def send_message(self, sock: socket.socket, message_data: bytes) -> bool:
        """Send complete message over socket."""
        try:
            total_sent = 0
            while total_sent < len(message_data):
                sent = sock.send(message_data[total_sent:])
                if sent == 0:
                    return False
                total_sent += sent
            return True
        except (socket.error, BrokenPipeError):
            return False


class BrainServerClient:
    """Client for TCP binary communication with brain server."""
    
    def __init__(self, config: BrainServerConfig):
        """Initialize brain server client."""
        self.config = config
        self.bsc = config.brainstem_config  # Shorthand
        self.protocol = MessageProtocol(self.bsc)
        self.socket = None
        
        # Connection state
        self.connected = False
        self.last_communication_time = 0.0
        self.communication_errors = 0
        
        # Latest brain response
        self.latest_motor_commands = None
        self.latest_vocal_commands = None
        
        # Connection lock for thread safety
        self._lock = threading.Lock()
        
        print(f"🧠 Brain client initialized: {config.host}:{config.port}")
    
    def connect(self) -> bool:
        """Establish TCP connection and perform handshake."""
        with self._lock:
            try:
                # Create socket
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.config.timeout)
                
                # Connect to brain server
                print(f"Connecting to brain server {self.config.host}:{self.config.port}...")
                self.socket.connect((self.config.host, self.config.port))
                
                # Enable TCP_NODELAY for low latency if configured
                if self.bsc.network.tcp_nodelay:
                    self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Enable keepalive if configured  
                if self.bsc.network.socket_keepalive:
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Send handshake
                handshake_vector = [
                    1.0,  # Robot version
                    float(self.bsc.sensors.brain_input_dimensions),  # Sensory vector size
                    float(self.bsc.motors.brain_output_dimensions),   # Action vector size
                    1.0,  # Hardware type (1.0 = PiCar-X)
                    3.0   # Capabilities (1=visual, 2=audio)
                ]
                
                handshake_msg = self.protocol.encode_handshake(handshake_vector)
                success = self.protocol.send_message(self.socket, handshake_msg)
                
                if not success:
                    raise ConnectionError("Failed to send handshake")
                
                # Receive handshake response
                msg_type, response_vector = self.protocol.receive_message(self.socket, self.config.timeout)
                
                if msg_type != self.protocol.MSG_HANDSHAKE:
                    raise ConnectionError(f"Expected handshake response, got message type {msg_type}")
                
                # Parse handshake response
                brain_version = response_vector[0]
                accepted_sensory = int(response_vector[1])
                accepted_action = int(response_vector[2])
                gpu_available = bool(response_vector[3])
                
                print(f"✅ Brain handshake successful:")
                print(f"   Brain version: {brain_version}")
                print(f"   Sensory dimensions: {accepted_sensory} (requested {self.config.sensory_dimensions})")
                print(f"   Action dimensions: {accepted_action} (requested {self.config.action_dimensions})")
                print(f"   GPU available: {gpu_available}")
                
                # Update dimensions based on negotiation
                self.config.sensory_dimensions = accepted_sensory
                self.config.action_dimensions = accepted_action
                
                self.connected = True
                self.communication_errors = 0
                self.last_communication_time = time.time()
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to connect to brain server: {e}")
                self.connected = False
                if self.socket:
                    self.socket.close()
                    self.socket = None
                return False
    
    def send_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """
        Send sensor data to brain server and receive action response.
        
        Args:
            sensor_data: Dictionary containing sensor readings
            
        Returns:
            bool: True if communication successful, False otherwise
        """
        with self._lock:
            if not self.connected or not self.socket:
                return False
            
            try:
                # Extract normalized sensor vector
                if 'normalized_sensors' in sensor_data:
                    sensory_vector = sensor_data['normalized_sensors']
                else:
                    # Fallback to raw sensors if needed
                    sensory_vector = sensor_data.get('raw_sensors', [0.5] * self.config.sensory_dimensions)
                
                # Ensure correct dimensions
                if len(sensory_vector) != self.config.sensory_dimensions:
                    # Pad or truncate to match expected dimensions
                    if len(sensory_vector) < self.config.sensory_dimensions:
                        sensory_vector.extend([0.5] * (self.config.sensory_dimensions - len(sensory_vector)))
                    else:
                        sensory_vector = sensory_vector[:self.config.sensory_dimensions]
                
                # Encode and send sensory input
                sensory_msg = self.protocol.encode_sensory_input(sensory_vector)
                success = self.protocol.send_message(self.socket, sensory_msg)
                
                if not success:
                    raise ConnectionError("Failed to send sensory data")
                
                # Receive action response
                msg_type, action_vector = self.protocol.receive_message(self.socket, self.config.timeout)
                
                if msg_type == self.protocol.MSG_ACTION_OUTPUT:
                    # Store motor commands in expected format
                    if len(action_vector) >= 4:
                        self.latest_motor_commands = {
                            'motor_x': action_vector[0],  # Forward/backward
                            'motor_y': action_vector[1],  # Left/right steering
                            'motor_z': action_vector[2],  # Camera pan
                            'motor_w': action_vector[3]   # Camera tilt / auxiliary
                        }
                        
                        # Add any additional motor channels if available
                        if len(action_vector) > 4:
                            self.latest_motor_commands['motor_extra'] = action_vector[4:]
                    
                    self.last_communication_time = time.time()
                    self.communication_errors = 0
                    return True
                    
                elif msg_type == self.protocol.MSG_ERROR:
                    error_code = action_vector[0] if action_vector else -1
                    print(f"⚠️ Brain server error: {error_code}")
                    self.communication_errors += 1
                    return False
                    
                else:
                    print(f"⚠️ Unexpected message type: {msg_type}")
                    self.communication_errors += 1
                    return False
                
            except Exception as e:
                print(f"❌ Error in brain communication: {e}")
                self.communication_errors += 1
                
                # Close connection on repeated errors
                if self.communication_errors > self.config.retry_attempts:
                    self.connected = False
                    if self.socket:
                        self.socket.close()
                        self.socket = None
                
                return False
    
    def send_status_update(self, status_data: Dict[str, Any]) -> bool:
        """
        Send robot status update. 
        
        Note: The binary protocol doesn't have a dedicated status message,
        so we'll include status info in the sensory data when relevant.
        """
        return True  # Status updates are handled through sensory data
    
    def get_latest_motor_commands(self) -> Optional[Dict[str, Any]]:
        """Get latest motor commands from brain server."""
        commands = self.latest_motor_commands
        self.latest_motor_commands = None  # Clear after reading
        return commands
    
    def get_latest_vocal_commands(self) -> Optional[Dict[str, Any]]:
        """Get latest vocal commands. Not implemented in binary protocol."""
        # Vocal commands could be encoded in additional action channels if needed
        return None
    
    def is_connected(self) -> bool:
        """Check if connected to brain server."""
        with self._lock:
            # Consider disconnected if no communication for configured timeout
            timeout_threshold = self.bsc.network.connection_timeout * 6  # 6x connection timeout
            if self.connected and time.time() - self.last_communication_time > timeout_threshold:
                self.connected = False
                print(f"⚠️ Brain server connection timeout (>{timeout_threshold:.1f}s)")
                if self.socket:
                    self.socket.close()
                    self.socket = None
            
            return self.connected
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'connected': self.connected,
            'last_communication': self.last_communication_time,
            'communication_errors': self.communication_errors,
            'server_address': f"{self.config.host}:{self.config.port}",
            'sensory_dimensions': self.config.sensory_dimensions,
            'action_dimensions': self.config.action_dimensions
        }
    
    def close(self):
        """Close the connection to brain server."""
        with self._lock:
            self.connected = False
            if self.socket:
                self.socket.close()
                self.socket = None
                print("🔌 Brain server connection closed")


class MockBrainServerClient(BrainServerClient):
    """Mock brain client for testing without real server."""
    
    def __init__(self, config: Optional[BrainServerConfig] = None):
        """Initialize mock brain client."""
        if config is None:
            config = BrainServerConfig()
        
        # Initialize parent without socket connection
        self.config = config
        self.protocol = MessageProtocol()
        self.socket = None
        
        # Connection state
        self.connected = False
        self.last_communication_time = 0.0
        self.communication_errors = 0
        
        # Latest brain response
        self.latest_motor_commands = None
        self.latest_vocal_commands = None
        
        # Connection lock for thread safety
        self._lock = threading.Lock()
        
        print("🧠 Mock brain client initialized - simulating server responses")
    
    def connect(self) -> bool:
        """Mock connection always succeeds."""
        with self._lock:
            self.connected = True
            self.last_communication_time = time.time()
            print("✅ Mock brain server connection established")
            print(f"   Sensory dimensions: {self.config.sensory_dimensions}")
            print(f"   Action dimensions: {self.config.action_dimensions}")
            return True
    
    def send_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """Mock sensor data sending with simulated responses."""
        with self._lock:
            if not self.connected:
                return False
            
            self.last_communication_time = time.time()
            
            # Extract sensor data
            if 'normalized_sensors' in sensor_data:
                sensors = sensor_data['normalized_sensors']
            elif 'raw_sensors' in sensor_data:
                # Use first few raw sensors for basic decisions
                sensors = sensor_data['raw_sensors']
            else:
                sensors = [0.5] * 16
            
            # Generate mock motor commands based on sensor data
            # Use first sensor as distance (from ultrasonic/proximity)
            if len(sensors) > 0:
                distance_sensor = sensors[0]
                # If normalized, convert back to distance estimate
                if distance_sensor <= 1.0:
                    estimated_distance = distance_sensor * 2.0  # rough meters
                else:
                    estimated_distance = distance_sensor / 100.0  # convert cm to m
            else:
                estimated_distance = 1.0
            
            if estimated_distance < 0.2:  # Close obstacle
                # Back up and turn
                self.latest_motor_commands = {
                    'motor_x': -0.3,   # Backward
                    'motor_y': 0.4,    # Turn right
                    'motor_z': 0.0,    # No camera pan
                    'motor_w': 0.0     # No camera tilt
                }
                print(f"🧠 Mock brain: obstacle at {estimated_distance:.2f}m - backing up")
                
            elif estimated_distance < 0.5:  # Moderate distance
                # Slow forward with slight avoidance
                self.latest_motor_commands = {
                    'motor_x': 0.2,    # Slow forward
                    'motor_y': 0.1,    # Slight turn
                    'motor_z': 0.0,    # No camera pan
                    'motor_w': 0.0     # No camera tilt
                }
                print(f"🧠 Mock brain: caution at {estimated_distance:.2f}m - slow forward")
                
            else:  # Clear path
                # Normal forward motion
                self.latest_motor_commands = {
                    'motor_x': 0.4,    # Forward
                    'motor_y': 0.0,    # Straight
                    'motor_z': 0.0,    # No camera pan  
                    'motor_w': 0.0     # No camera tilt
                }
                print(f"🧠 Mock brain: clear path at {estimated_distance:.2f}m - moving forward")
            
            return True
    
    def send_status_update(self, status_data: Dict[str, Any]) -> bool:
        """Mock status update always succeeds."""
        with self._lock:
            if not self.connected:
                return False
            
            self.last_communication_time = time.time()
            return True
    
    def close(self):
        """Close mock connection."""
        with self._lock:
            self.connected = False
            print("🔌 Mock brain server connection closed")