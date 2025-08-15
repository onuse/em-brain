#!/usr/bin/env python3
"""
Test client for the dynamic brain server.

This client tests that brains are created dynamically based on
robot capabilities announced during handshake.
"""

import socket
import struct
import time
import sys


class TestDynamicClient:
    """Test client that connects with different robot configurations."""
    
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.sock = None
    
    def connect(self):
        """Connect to brain server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"✅ Connected to {self.host}:{self.port}")
    
    def encode_message(self, vector, msg_type):
        """Encode message in brain protocol format."""
        vector_data = struct.pack(f'{len(vector)}f', *vector)
        message_length = 1 + 4 + len(vector_data)
        header = struct.pack('!IBI', message_length, msg_type, len(vector))
        return header + vector_data
    
    def decode_message(self, data):
        """Decode message from brain protocol format."""
        if len(data) < 9:
            raise ValueError("Message too short")
        
        message_length, msg_type, vector_length = struct.unpack('!IBI', data[:9])
        vector_data = data[9:9 + (vector_length * 4)]
        vector = list(struct.unpack(f'{vector_length}f', vector_data))
        
        return msg_type, vector
    
    def receive_response(self):
        """Receive and decode response."""
        # Read message length
        length_data = self.sock.recv(4)
        message_length = struct.unpack('!I', length_data)[0]
        
        # Read rest of message
        message_data = self.sock.recv(message_length)
        
        # Decode
        return self.decode_message(length_data + message_data)
    
    def test_robot_config(self, name, sensory_dim, motor_dim, robot_type=1.0):
        """Test a specific robot configuration."""
        
        print(f"\n🤖 Testing {name} robot:")
        print(f"   Sensory: {sensory_dim}D, Motor: {motor_dim}D")
        
        # Send handshake
        capabilities = [
            1.0,                    # Version
            float(sensory_dim),     # Sensory dimensions
            float(motor_dim),       # Motor dimensions
            robot_type,             # Hardware type (1.0 = PiCar-X)
            3.0                     # Capabilities (visual + audio)
        ]
        
        handshake_msg = self.encode_message(capabilities, 2)  # Type 2 = handshake
        self.sock.sendall(handshake_msg)
        
        # Receive response
        msg_type, response = self.receive_response()
        print(f"   Response: {response}")
        
        # Verify dimensions were accepted
        if len(response) >= 3:
            accepted_sensory = int(response[1])
            accepted_motor = int(response[2])
            print(f"   ✓ Brain accepted: {accepted_sensory}D sensors, {accepted_motor}D motors")
        
        # Send a few sensory inputs
        for i in range(3):
            # Create sensory data
            sensory_data = [0.5] * sensory_dim
            sensory_msg = self.encode_message(sensory_data, 0)  # Type 0 = sensory
            
            start_time = time.time()
            self.sock.sendall(sensory_msg)
            
            # Get motor response
            msg_type, motor_response = self.receive_response()
            cycle_time = (time.time() - start_time) * 1000
            
            print(f"   Cycle {i+1}: {len(motor_response)} motor commands in {cycle_time:.1f}ms")
    
    def disconnect(self):
        """Disconnect from server."""
        if self.sock:
            self.sock.close()
            print("👋 Disconnected")


def main():
    """Test different robot configurations."""
    
    print("🧪 Dynamic Brain Server Test")
    print("=" * 40)
    
    # Test different robot configurations
    configs = [
        ("Minimal", 8, 2),      # Minimal robot
        ("PiCar-X", 16, 5),     # Standard PiCar-X
        ("Advanced", 32, 8),    # Advanced robot with many sensors
        ("Ultra", 48, 12),      # Ultra-complex robot
    ]
    
    for name, sensory, motor in configs:
        client = TestDynamicClient()
        
        try:
            client.connect()
            client.test_robot_config(name, sensory, motor)
            client.disconnect()
            
            time.sleep(1)  # Brief pause between tests
            
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()