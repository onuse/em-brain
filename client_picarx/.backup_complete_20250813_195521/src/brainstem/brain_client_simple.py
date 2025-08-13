#!/usr/bin/env python3
"""
Simple Brain Client

Minimal TCP client for brain server communication.
No external dependencies, just what's needed.
"""

import socket
import struct
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class BrainServerConfig:
    """Brain server connection config."""
    host: str = "localhost"
    port: int = 9999
    timeout: float = 0.05  # 50ms for 20Hz operation
    sensory_dimensions: int = 24
    action_dimensions: int = 4


class BrainClient:
    """
    Simple TCP client for brain server.
    
    Protocol:
    - Send: 9-byte header + sensor data (float32)
    - Receive: 9-byte header + motor commands (float32)
    """
    
    def __init__(self, config: BrainServerConfig = None):
        """Initialize client."""
        self.config = config or BrainServerConfig()
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to brain server."""
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.config.timeout)
            
            # Connect
            self.socket.connect((self.config.host, self.config.port))
            self.connected = True
            
            print(f"✅ Connected to brain at {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
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
            
            # Pack sensor data
            header = struct.pack('<cII', b'S', self.config.sensory_dimensions, 0)
            body = struct.pack(f'<{self.config.sensory_dimensions}f', *sensor_data)
            
            # Send
            self.socket.sendall(header + body)
            
            # Receive response
            response_header = self.socket.recv(9)
            if len(response_header) < 9:
                return None
            
            msg_type, dim, _ = struct.unpack('<cII', response_header)
            
            if msg_type == b'M' and dim > 0:
                # Receive motor commands
                body_size = dim * 4  # float32
                body_data = self.socket.recv(body_size)
                
                if len(body_data) == body_size:
                    motor_commands = list(struct.unpack(f'<{dim}f', body_data))
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