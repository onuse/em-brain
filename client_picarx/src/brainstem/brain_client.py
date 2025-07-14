#!/usr/bin/env python3
"""
Brain Server Client

Handles communication between the PiCar-X robot and the brain server.
Implements the API contract for sending sensor data and receiving
motor commands and vocal expressions.
"""

import json
import time
import requests
from typing import Dict, Optional, Any
import threading
from dataclasses import dataclass


@dataclass
class BrainServerConfig:
    """Configuration for brain server connection."""
    host: str = "localhost"
    port: int = 8000
    api_key: str = "development_key"
    timeout: float = 5.0
    retry_attempts: int = 3


class BrainServerClient:
    """Client for communicating with the brain server."""
    
    def __init__(self, config: BrainServerConfig):
        """Initialize brain server client."""
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json'
        })
        
        # Connection state
        self.connected = False
        self.last_communication_time = 0.0
        self.communication_errors = 0
        
        # Latest brain response
        self.latest_motor_commands = None
        self.latest_vocal_commands = None
        
        print(f"🧠 Brain client initialized: {self.base_url}")
    
    def connect(self) -> bool:
        """Establish connection to brain server."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.config.timeout)
            if response.status_code == 200:
                self.connected = True
                self.communication_errors = 0
                print("✅ Connected to brain server")
                return True
            else:
                print(f"❌ Brain server returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to brain server: {e}")
            self.connected = False
            return False
    
    def send_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """
        Send sensor data to brain server.
        
        Args:
            sensor_data: Dictionary containing all sensor readings
            
        Returns:
            bool: True if data sent successfully, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            # Add timestamp
            sensor_data['timestamp'] = time.time()
            
            response = self.session.post(
                f"{self.base_url}/brain/sensors",
                json=sensor_data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                self.last_communication_time = time.time()
                self.communication_errors = 0
                
                # Check for motor commands in response
                if response.content:
                    brain_response = response.json()
                    if 'motor_commands' in brain_response:
                        self.latest_motor_commands = brain_response['motor_commands']
                    if 'vocal_commands' in brain_response:
                        self.latest_vocal_commands = brain_response['vocal_commands']
                
                return True
            else:
                print(f"⚠️ Brain server sensor upload failed: {response.status_code}")
                self.communication_errors += 1
                return False
                
        except Exception as e:
            print(f"❌ Error sending sensor data: {e}")
            self.communication_errors += 1
            if self.communication_errors > self.config.retry_attempts:
                self.connected = False
            return False
    
    def send_status_update(self, status_data: Dict[str, Any]) -> bool:
        """
        Send robot status update to brain server.
        
        Args:
            status_data: Dictionary containing robot status information
            
        Returns:
            bool: True if status sent successfully, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            status_data['timestamp'] = time.time()
            
            response = self.session.post(
                f"{self.base_url}/brain/status",
                json=status_data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                self.last_communication_time = time.time()
                return True
            else:
                print(f"⚠️ Brain server status update failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending status update: {e}")
            return False
    
    def get_latest_motor_commands(self) -> Optional[Dict[str, Any]]:
        """Get latest motor commands from brain server."""
        commands = self.latest_motor_commands
        self.latest_motor_commands = None  # Clear after reading
        return commands
    
    def get_latest_vocal_commands(self) -> Optional[Dict[str, Any]]:
        """Get latest vocal commands from brain server."""
        commands = self.latest_vocal_commands
        self.latest_vocal_commands = None  # Clear after reading
        return commands
    
    def is_connected(self) -> bool:
        """Check if connected to brain server."""
        # Consider disconnected if no communication for 30 seconds
        if self.connected and time.time() - self.last_communication_time > 30.0:
            self.connected = False
            print("⚠️ Brain server connection timeout")
        return self.connected
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'connected': self.connected,
            'last_communication': self.last_communication_time,
            'communication_errors': self.communication_errors,
            'server_url': self.base_url
        }


class MockBrainServerClient(BrainServerClient):
    """Mock brain client for testing without real server."""
    
    def __init__(self, config: Optional[BrainServerConfig] = None):
        """Initialize mock brain client."""
        if config is None:
            config = BrainServerConfig()
        super().__init__(config)
        
        # Mock state
        self.mock_commands = {
            'motor_speed': 0.0,
            'steering_angle': 0.0,
            'camera_pan': 0.0,
            'camera_tilt': 0.0
        }
        self.mock_vocal_state = 'contentment'
        
        print("🧠 Mock brain client initialized - simulating server responses")
    
    def connect(self) -> bool:
        """Mock connection always succeeds."""
        self.connected = True
        print("✅ Mock brain server connection established")
        return True
    
    def send_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """Mock sensor data sending with simulated responses."""
        if not self.connected:
            return False
        
        self.last_communication_time = time.time()
        
        # Generate mock motor commands based on sensor data
        ultrasonic_distance = sensor_data.get('ultrasonic_distance', 100.0)
        
        if ultrasonic_distance < 20.0:
            # Obstacle detected - turn away
            self.latest_motor_commands = {
                'motor_speed': -20.0,  # Back up
                'steering_angle': 15.0,  # Turn
                'camera_pan': 0.0,
                'camera_tilt': 0.0
            }
            self.latest_vocal_commands = {
                'emotional_state': 'confusion',
                'intensity': 0.6,
                'duration': 0.3
            }
        else:
            # Clear path - move forward
            self.latest_motor_commands = {
                'motor_speed': 30.0,  # Forward
                'steering_angle': 0.0,  # Straight
                'camera_pan': 0.0,
                'camera_tilt': 0.0
            }
            if sensor_data.get('exploration_mode', False):
                self.latest_vocal_commands = {
                    'emotional_state': 'curiosity',
                    'intensity': 0.3,
                    'duration': 0.2
                }
        
        print(f"🧠 Mock brain processed sensors: distance={ultrasonic_distance:.1f}cm")
        return True
    
    def send_status_update(self, status_data: Dict[str, Any]) -> bool:
        """Mock status update always succeeds."""
        if not self.connected:
            return False
        
        self.last_communication_time = time.time()
        print(f"🧠 Mock brain received status: {status_data.get('operational_status', 'OK')}")
        return True