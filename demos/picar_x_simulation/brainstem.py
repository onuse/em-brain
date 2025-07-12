"""
PiCar-X Brainstem Client

The robot's autonomous nervous system - coordinates sensors, motor control,
and communication with the brain server using the minimal binary protocol.

This is the critical interface between the physical/simulated robot
and the intelligent brain system.
"""

import socket
import struct
import time
import threading
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SensoryInput:
    """Structured sensory input from all robot sensors."""
    timestamp: float
    position: List[float]       # [x, y] position
    orientation: float          # heading in degrees
    speed: float               # current speed
    ultrasonic_distance: float # distance to obstacle
    camera_data: List[float]   # normalized camera features
    battery_voltage: float     # battery level
    temperature: float         # system temperature


@dataclass
class MotorCommands:
    """Motor commands from brain to actuators."""
    timestamp: float
    left_motor: float          # -1 to 1
    right_motor: float         # -1 to 1
    steering_angle: float      # -30 to 30 degrees
    camera_pan: float          # -90 to 90 degrees


class MinimalBrainProtocol:
    """
    Implementation of the minimal binary protocol for brain communication.
    
    This matches the actual protocol implemented in src/communication/protocol.py
    """
    
    # Message types (from actual implementation)
    MSG_SENSORY_INPUT = 0
    MSG_ACTION_OUTPUT = 1
    MSG_HANDSHAKE = 2
    MSG_ERROR = 255
    
    def __init__(self):
        self.max_vector_size = 1024
    
    def encode_message(self, vector: List[float], msg_type: int) -> bytes:
        """Encode vector message for transmission."""
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
    
    def receive_message(self, sock: socket.socket, timeout: float = 5.0) -> Tuple[int, List[float]]:
        """Receive and decode a complete message from socket."""
        sock.settimeout(timeout)
        
        try:
            # Receive message length first
            length_data = self._receive_exactly(sock, 4)
            message_length = struct.unpack('!I', length_data)[0]
            
            if message_length > (self.max_vector_size * 4 + 5):  # Sanity check
                raise ValueError(f"Message too large: {message_length}")
            
            # Receive the rest of the message
            message_data = self._receive_exactly(sock, message_length)
            
            # Decode complete message
            return self.decode_message(length_data + message_data)
            
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


class PiCarXBrainstem:
    """
    PiCar-X Robot Brainstem - Autonomous Nervous System
    
    Core responsibilities:
    1. Collect and fuse sensor data into 16-float vector
    2. Communicate with brain server via minimal binary protocol
    3. Execute 4-float motor commands from brain
    4. Maintain real-time operation (20-60 Hz)
    
    This is the robot's "spinal cord" - handles low-level coordination
    while the brain provides high-level intelligence.
    """
    
    def __init__(self, brain_server_host: str, brain_server_port: int,
                 vehicle, world, update_rate_hz: float = 30.0):
        """
        Initialize brainstem client.
        
        Args:
            brain_server_host: Brain server IP address
            brain_server_port: Brain server port (should be 9999 for actual brain)
            vehicle: Vehicle simulation model
            world: World simulation environment
            update_rate_hz: Brainstem update frequency
        """
        self.brain_host = brain_server_host
        self.brain_port = brain_server_port
        self.vehicle = vehicle
        self.world = world
        self.update_rate = update_rate_hz
        self.update_interval = 1.0 / update_rate_hz
        
        # Protocol implementation
        self.protocol = MinimalBrainProtocol()
        
        # Communication
        self.brain_socket = None
        self.connected = False
        self.server_capabilities = None
        self.last_successful_communication = 0.0
        self.last_sensory_update = 0.0
        self.last_motor_update = 0.0
        
        # Current motor commands
        self.current_motor_commands = MotorCommands(
            timestamp=time.time(),
            left_motor=0.0,
            right_motor=0.0,
            steering_angle=0.0,
            camera_pan=0.0
        )
        
        # Performance monitoring
        self.total_sensory_cycles = 0
        self.total_motor_cycles = 0
        self.communication_errors = 0
        
        # Thread safety
        self.sensor_lock = threading.Lock()
        self.motor_lock = threading.Lock()
        
        print(f"🦾 PiCarXBrainstem initialized (target: {update_rate_hz}Hz, protocol: binary)")
    
    def connect(self) -> bool:
        """Attempt to connect to brain server using minimal protocol."""
        try:
            print(f"🔌 Attempting to connect to brain server at {self.brain_host}:{self.brain_port}...")
            
            self.brain_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.brain_socket.settimeout(3.0)
            self.brain_socket.connect((self.brain_host, self.brain_port))
            
            # Send handshake with robot capabilities
            robot_capabilities = [
                1.0,  # Robot version
                16.0, # Sensory vector size we'll send
                4.0,  # Action vector size we expect
                1.0   # Hardware type (1.0 = PiCar-X simulation)
            ]
            
            handshake_msg = self.protocol.encode_message(robot_capabilities, self.protocol.MSG_HANDSHAKE)
            if not self.protocol.send_message(self.brain_socket, handshake_msg):
                raise ConnectionError("Failed to send handshake")
            
            # Receive server response
            msg_type, server_caps = self.protocol.receive_message(self.brain_socket, timeout=2.0)
            
            if msg_type != self.protocol.MSG_HANDSHAKE:
                print(f"⚠️ Expected handshake response, got message type {msg_type}")
                self._cleanup_failed_connection()
                return False
            
            # Store server capabilities
            self.server_capabilities = server_caps
            self.connected = True
            self.last_successful_communication = time.time()
            
            print("✅ Connected to brain server!")
            print(f"   Server capabilities: {server_caps}")
            return True
            
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            # Expected when server not running - don't spam errors
            if not hasattr(self, '_connection_attempts'):
                self._connection_attempts = 0
            self._connection_attempts += 1
            
            # Only print connection attempts occasionally
            if self._connection_attempts % 30 == 1:
                print(f"🔄 Brain server not available (attempt {self._connection_attempts})")
            
            self._cleanup_failed_connection()
            return False
        except Exception as e:
            print(f"❌ Unexpected connection error: {e}")
            self._cleanup_failed_connection()
            return False
    
    def _cleanup_failed_connection(self):
        """Clean up after failed connection attempt."""
        if self.brain_socket:
            try:
                self.brain_socket.close()
            except:
                pass
            self.brain_socket = None
        self.connected = False
    
    def disconnect(self):
        """Disconnect from brain server."""
        if self.connected and self.brain_socket:
            try:
                print("📴 Disconnecting from brain server")
                self.brain_socket.close()
            except Exception as e:
                print(f"⚠️ Error during disconnect: {e}")
            finally:
                self.brain_socket = None
                self.connected = False
    
    def update(self):
        """Main brainstem update cycle."""
        current_time = time.time()
        
        # Try to connect if not connected (non-blocking)
        if not self.connected:
            self._try_reconnect()
        
        # Check if it's time for sensory update
        if current_time - self.last_sensory_update >= self.update_interval:
            self._sensory_motor_cycle()
            self.last_sensory_update = current_time
    
    def _try_reconnect(self):
        """Attempt to reconnect to brain server (non-blocking)."""
        current_time = time.time()
        
        # Only try reconnecting every 1 second to avoid spam
        if not hasattr(self, '_last_reconnect_attempt'):
            self._last_reconnect_attempt = 0
        
        if current_time - self._last_reconnect_attempt >= 1.0:
            self._last_reconnect_attempt = current_time
            self.connect()
    
    def _sensory_motor_cycle(self):
        """Complete sensory-motor cycle using minimal protocol."""
        try:
            # 1. Collect sensor data into 16-float vector
            sensory_vector = self._collect_sensory_vector()
            
            # 2. Send sensory input to brain
            if self.connected:
                action_vector = self._send_sensory_get_action(sensory_vector)
                
                if action_vector:
                    # 3. Parse and execute motor commands
                    motor_commands = self._parse_action_vector(action_vector)
                    self._execute_motor_commands(motor_commands)
                    self.last_successful_communication = time.time()
                else:
                    # No response from brain - use safe behavior
                    self._execute_safe_motor_commands()
            else:
                # Not connected - operate in standalone mode
                self._execute_standalone_behavior()
            
            self.total_sensory_cycles += 1
            
        except Exception as e:
            print(f"⚠️ Brainstem cycle error: {e}")
            self.communication_errors += 1
            self._execute_safe_motor_commands()
    
    def _collect_sensory_vector(self) -> List[float]:
        """
        Collect sensor data into standardized 16-float vector.
        
        Vector format (as documented in protocol):
        [0] Position X (0-10 meters)
        [1] Position Y (0-10 meters)
        [2] Heading (0-360 degrees)
        [3] Speed (0-2 m/s)
        [4] Ultrasonic distance (0-3 meters)
        [5-12] Camera data (0-1 normalized)
        [13] Battery voltage (0-15 volts)
        [14] Temperature (0-100 celsius)
        [15] Reserved (0)
        """
        # Get vehicle state
        vehicle_state = self.vehicle.get_sensor_data()
        
        # Position (normalize to 0-10 range for room)
        pos_x = np.clip(vehicle_state["position"][0], 0.0, 10.0)
        pos_y = np.clip(vehicle_state["position"][1], 0.0, 10.0)
        
        # Heading (0-360 degrees)
        heading = vehicle_state["orientation"] % 360.0
        
        # Speed (clip to reasonable range)
        velocity = vehicle_state["velocity"]
        speed = np.clip(np.linalg.norm(velocity), 0.0, 2.0)
        
        # Ultrasonic distance
        ultrasonic_distance = np.clip(self._measure_ultrasonic_distance(), 0.0, 3.0)
        
        # Camera data (simplified to 8 values)
        camera_data = self._extract_camera_features()
        
        # Environmental data
        battery_voltage = 12.0  # Simulated battery level
        temperature = 25.0      # Simulated temperature
        
        # Assemble 16-float vector
        sensory_vector = [
            pos_x,                    # 0
            pos_y,                    # 1
            heading,                  # 2
            speed,                    # 3
            ultrasonic_distance,      # 4
            camera_data[0],           # 5
            camera_data[1],           # 6
            camera_data[2],           # 7
            camera_data[3],           # 8
            camera_data[4],           # 9
            camera_data[5],           # 10
            camera_data[6],           # 11
            camera_data[7],           # 12
            battery_voltage,          # 13
            temperature,              # 14
            0.0                       # 15 (reserved)
        ]
        
        return sensory_vector
    
    def _extract_camera_features(self) -> List[float]:
        """Extract simplified camera features (8 floats)."""
        # Simplified camera processing - in real implementation would
        # extract features from actual camera image
        
        # Get camera pose for feature extraction
        camera_pose = self.vehicle.get_camera_pose()
        pan_angle = self.vehicle.camera_pan_angle
        tilt_angle = self.vehicle.camera_tilt_angle
        
        # Simulate basic visual features (normalized 0-1)
        features = [
            (pan_angle + 90) / 180.0,     # Pan angle normalized
            (tilt_angle + 30) / 60.0,     # Tilt angle normalized
            0.5,                          # Brightness
            0.3,                          # Contrast
            0.2,                          # Edge density
            0.1,                          # Motion
            0.4,                          # Object presence
            0.6                           # Scene complexity
        ]
        
        return features
    
    def _measure_ultrasonic_distance(self) -> float:
        """Measure ultrasonic distance using world ray casting."""
        # Get ultrasonic sensor pose
        ultrasonic_pose = self.vehicle.get_ultrasonic_pose()
        sensor_pos = ultrasonic_pose["position"][:2]
        sensor_orientation = ultrasonic_pose["orientation"]
        
        # Cast ray in world
        if hasattr(self.world, 'cast_ray'):
            distance = self.world.cast_ray(sensor_pos, sensor_orientation, 3.0, 30.0)
        else:
            distance = 3.0  # Max range if no world ray casting
        
        # Add realistic sensor noise
        noise = np.random.normal(0, 0.003)  # ±3mm noise
        return max(0.02, distance + noise)  # Min 2cm range
    
    def _send_sensory_get_action(self, sensory_vector: List[float]) -> Optional[List[float]]:
        """Send sensory input and receive action using minimal protocol."""
        try:
            # Encode and send sensory input
            sensory_msg = self.protocol.encode_message(sensory_vector, self.protocol.MSG_SENSORY_INPUT)
            if not self.protocol.send_message(self.brain_socket, sensory_msg):
                raise ConnectionError("Failed to send sensory input")
            
            # Receive action response
            msg_type, action_vector = self.protocol.receive_message(self.brain_socket, timeout=0.1)
            
            if msg_type == self.protocol.MSG_ACTION_OUTPUT:
                return action_vector
            elif msg_type == self.protocol.MSG_ERROR:
                error_code = action_vector[0] if action_vector else 0
                print(f"⚠️ Brain server error: {error_code}")
                return None
            else:
                print(f"⚠️ Unexpected response type: {msg_type}")
                return None
                
        except socket.timeout:
            return None
        except (ConnectionError, ValueError) as e:
            print(f"💔 Communication error: {e}")
            self._cleanup_failed_connection()
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def _parse_action_vector(self, action_vector: List[float]) -> MotorCommands:
        """
        Parse 4-float action vector into motor commands.
        
        Action vector format:
        [0] Left motor speed (-1 to 1)
        [1] Right motor speed (-1 to 1)
        [2] Steering angle (-30 to 30 degrees)
        [3] Camera pan (-90 to 90 degrees)
        """
        if len(action_vector) < 4:
            return self._get_safe_motor_commands()
        
        return MotorCommands(
            timestamp=time.time(),
            left_motor=np.clip(action_vector[0], -1.0, 1.0),
            right_motor=np.clip(action_vector[1], -1.0, 1.0),
            steering_angle=np.clip(action_vector[2], -30.0, 30.0),
            camera_pan=np.clip(action_vector[3], -90.0, 90.0)
        )
    
    def _execute_motor_commands(self, motor_commands: MotorCommands):
        """Execute motor commands on vehicle."""
        try:
            with self.motor_lock:
                self.current_motor_commands = motor_commands
                
                # Execute drive motor commands
                self.vehicle.set_motor_speeds(motor_commands.left_motor, motor_commands.right_motor)
                
                # Execute steering command
                self.vehicle.set_steering_angle(motor_commands.steering_angle)
                
                # Execute camera servo command
                self.vehicle.set_camera_angles(motor_commands.camera_pan, 0.0)  # No tilt in minimal protocol
                
                self.total_motor_cycles += 1
                
        except Exception as e:
            print(f"⚠️ Motor execution error: {e}")
            self._execute_safe_motor_commands()
    
    def _execute_standalone_behavior(self):
        """Execute simple standalone behavior when not connected to brain."""
        # Simple obstacle avoidance
        distance = self._measure_ultrasonic_distance()
        
        if distance > 0.5:  # 50cm clearance
            # Move forward slowly
            motor_commands = MotorCommands(
                timestamp=time.time(),
                left_motor=0.2,
                right_motor=0.2,
                steering_angle=0.0,
                camera_pan=0.0
            )
        elif distance > 0.2:  # 20cm clearance
            # Turn slowly
            motor_commands = MotorCommands(
                timestamp=time.time(),
                left_motor=0.1,
                right_motor=-0.1,
                steering_angle=15.0,
                camera_pan=15.0
            )
        else:
            # Stop and back up
            motor_commands = MotorCommands(
                timestamp=time.time(),
                left_motor=-0.1,
                right_motor=-0.1,
                steering_angle=0.0,
                camera_pan=0.0
            )
        
        self._execute_motor_commands(motor_commands)
    
    def _execute_safe_motor_commands(self):
        """Execute safe default motor commands (stop everything)."""
        safe_commands = self._get_safe_motor_commands()
        self._execute_motor_commands(safe_commands)
    
    def _get_safe_motor_commands(self) -> MotorCommands:
        """Get safe default motor commands."""
        return MotorCommands(
            timestamp=time.time(),
            left_motor=0.0,
            right_motor=0.0,
            steering_angle=0.0,
            camera_pan=0.0
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get brainstem performance statistics."""
        current_time = time.time()
        
        return {
            "total_sensory_cycles": self.total_sensory_cycles,
            "total_motor_cycles": self.total_motor_cycles,
            "communication_errors": self.communication_errors,
            "error_rate": self.communication_errors / max(1, self.total_sensory_cycles),
            "time_since_last_communication": current_time - self.last_successful_communication,
            "update_rate_hz": self.update_rate,
            "connected": self.connected,
            "protocol": "minimal_binary"
        }
    
    def get_brain_connection_stats(self) -> Dict[str, Any]:
        """Get brain connection statistics for status display."""
        connection_status = "Connected" if self.connected else "Standalone Mode"
        
        # Extract brain stats from server capabilities if available
        experiences = 0
        predictions = 0
        if self.server_capabilities and len(self.server_capabilities) >= 2:
            # Server capabilities don't include counts, so we estimate based on uptime
            if self.connected:
                uptime = time.time() - self.last_successful_communication
                predictions = int(uptime * self.update_rate)  # Rough estimate
        
        return {
            "connected": self.connected,
            "status": connection_status,
            "experiences": experiences,
            "predictions": predictions,
            "server_capabilities": self.server_capabilities
        }
    
    def __str__(self) -> str:
        status = "Connected" if self.connected else "Disconnected"
        return f"PiCarXBrainstem({status}, {self.total_sensory_cycles} cycles, {self.update_rate}Hz, binary protocol)"