"""
Minimal Brain Client

Client-side communication for robot brainstems to connect to the minimal brain server.
This runs on the robot hardware (Pi Zero) and communicates with the brain server.
"""

import socket
import time
import datetime
from typing import List, Optional, Tuple, Dict, Any

from .protocol import MessageProtocol, MessageProtocolError


class MinimalBrainClient:
    """
    Client for robot brainstems to communicate with the minimal brain server.
    
    This handles the network communication so robot brainstems can focus
    on hardware interface logic.
    """
    
    def __init__(self, server_host: str = 'localhost', server_port: int = 9999):
        """
        Initialize brain client.
        
        Args:
            server_host: Brain server hostname/IP
            server_port: Brain server port
        """
        self.server_host = server_host
        self.server_port = server_port
        
        # Connection state
        self.socket = None
        self.connected = False
        self.server_capabilities = None
        
        # Protocol
        self.protocol = MessageProtocol()
        
        # Performance tracking
        self.requests_sent = 0
        self.responses_received = 0
        self.total_request_time = 0.0
        self.connection_time = None
        
        print(f"🤖 MinimalBrainClient initialized for {server_host}:{server_port}")
    
    def connect(self, robot_capabilities: Optional[List[float]] = None) -> bool:
        """
        Connect to the brain server and perform handshake.
        
        Args:
            robot_capabilities: Optional robot capability vector
            
        Returns:
            True if connected successfully
        """
        try:
            # Create socket connection
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # 10 second connection timeout
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] 🔌 Connecting to brain server at {self.server_host}:{self.server_port}...")
            self.socket.connect((self.server_host, self.server_port))
            
            # Perform handshake
            if robot_capabilities is None:
                robot_capabilities = [
                    1.0,  # Robot version
                    16.0, # Sensory vector size
                    4.0,  # Action vector size
                    1.0   # Hardware type (1.0 = PiCar-X)
                ]
            
            # Send handshake
            handshake_msg = self.protocol.encode_handshake(robot_capabilities)
            if not self.protocol.send_message(self.socket, handshake_msg):
                raise ConnectionError("Failed to send handshake")
            
            # Receive server response
            msg_type, server_caps = self.protocol.receive_message(self.socket, timeout=5.0)
            
            if msg_type != MessageProtocol.MSG_HANDSHAKE:
                raise MessageProtocolError(f"Expected handshake response, got {msg_type}")
            
            # Store server capabilities
            self.server_capabilities = server_caps
            self.connected = True
            self.connection_time = time.time()
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] ✅ Connected to brain server!")
            print(f"   Server capabilities: {server_caps}")
            
            return True
            
        except Exception as e:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] ❌ Failed to connect to brain server: {e}")
            self._cleanup_connection()
            return False
    
    def get_action(self, sensory_vector: List[float], timeout: float = 1.0) -> Optional[List[float]]:
        """
        Send sensory input to brain and get action response.
        
        Args:
            sensory_vector: Current robot sensor readings
            timeout: Request timeout in seconds
            
        Returns:
            Action vector from brain, or None if failed
        """
        if not self.connected:
            print("⚠️  Not connected to brain server")
            return None
        
        try:
            request_start = time.time()
            
            # Send sensory input
            sensory_msg = self.protocol.encode_sensory_input(sensory_vector)
            if not self.protocol.send_message(self.socket, sensory_msg):
                raise ConnectionError("Failed to send sensory input")
            
            self.requests_sent += 1
            
            # Receive action response
            msg_type, action_vector = self.protocol.receive_message(self.socket, timeout=timeout)
            
            request_time = time.time() - request_start
            self.total_request_time += request_time
            self.responses_received += 1
            
            if msg_type == MessageProtocol.MSG_ACTION_OUTPUT:
                return action_vector
            elif msg_type == MessageProtocol.MSG_ERROR:
                error_code = action_vector[0] if action_vector else 0
                print(f"⚠️  Brain server error: {error_code}")
                return None
            else:
                print(f"⚠️  Unexpected response type: {msg_type}")
                return None
                
        except TimeoutError:
            print("⏰ Brain request timed out")
            return None
        except (ConnectionError, MessageProtocolError) as e:
            print(f"💔 Communication error: {e}")
            self._cleanup_connection()
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def ping(self) -> Optional[float]:
        """
        Ping the brain server to measure latency.
        
        Returns:
            Round-trip time in seconds, or None if failed
        """
        ping_start = time.time()
        
        # Send empty sensory vector as ping
        ping_vector = [0.0]
        response = self.get_action(ping_vector, timeout=2.0)
        
        if response is not None:
            return time.time() - ping_start
        else:
            return None
    
    def disconnect(self):
        """Disconnect from brain server."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] 👋 Disconnecting from brain server...")
        self._cleanup_connection()
    
    def _cleanup_connection(self):
        """Clean up connection resources."""
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client performance statistics."""
        uptime = time.time() - self.connection_time if self.connection_time else 0
        avg_request_time = self.total_request_time / max(1, self.responses_received)
        
        return {
            'connection': {
                'connected': self.connected,
                'server_host': self.server_host,
                'server_port': self.server_port,
                'uptime_seconds': uptime,
                'server_capabilities': self.server_capabilities
            },
            'performance': {
                'requests_sent': self.requests_sent,
                'responses_received': self.responses_received,
                'success_rate': self.responses_received / max(1, self.requests_sent),
                'avg_request_time': avg_request_time,
                'requests_per_second': self.responses_received / max(1, uptime),
                'total_request_time': self.total_request_time
            }
        }
    
    def is_connected(self) -> bool:
        """Check if connected to brain server."""
        return self.connected
    
    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the connected server."""
        if not self.connected or not self.server_capabilities:
            return None
        
        return {
            'brain_version': self.server_capabilities[0] if len(self.server_capabilities) > 0 else 0,
            'expected_sensory_size': int(self.server_capabilities[1]) if len(self.server_capabilities) > 1 else 16,
            'action_size': int(self.server_capabilities[2]) if len(self.server_capabilities) > 2 else 4,
            'gpu_available': self.server_capabilities[3] > 0 if len(self.server_capabilities) > 3 else False
        }
    
    def __str__(self) -> str:
        if self.connected:
            return f"MinimalBrainClient(connected to {self.server_host}:{self.server_port})"
        else:
            return f"MinimalBrainClient(disconnected)"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()