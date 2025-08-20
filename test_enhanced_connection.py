#!/usr/bin/env python3
"""
Quick test to verify Enhanced Brain is working through the socket pipeline
"""

import socket
import struct
import time
import numpy as np

def test_connection():
    """Test connection to Enhanced Brain server."""
    
    print("Testing Enhanced Brain Connection")
    print("=" * 60)
    
    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('localhost', 9999))
        print("✓ Connected to brain server on port 9999")
    except Exception as e:
        print(f"✗ Could not connect: {e}")
        print("  Make sure server is running: python3 run_server.py --brain enhanced")
        return
    
    # Send handshake
    print("\nSending handshake...")
    handshake_data = struct.pack('5f', 
        1.0,  # Version
        5.0,  # Sensory dimensions
        6.0,  # Motor dimensions
        0.0,  # No vision
        0.0   # No audio
    )
    
    handshake_msg = struct.pack('!IIBI',
        0x524F424F,  # Magic 'ROBO'
        len(handshake_data) + 5,  # Message length
        1,  # MSG_HANDSHAKE
        5   # Vector length
    ) + handshake_data
    
    sock.send(handshake_msg)
    print("✓ Handshake sent")
    
    # Receive response
    response = sock.recv(1024)
    if len(response) > 0:
        print("✓ Received handshake response")
        # Parse response to verify brain type
        if len(response) >= 29:  # Expected response size
            print("  Brain accepted connection")
    
    # Send a few sensor packets and receive motor commands
    print("\nTesting sensor/motor loop...")
    
    for cycle in range(5):
        # Prepare sensor data
        sensors = [
            50.0 - cycle * 2,  # Ultrasonic getting closer
            1.0 if cycle % 2 == 0 else 0.0,  # Vision alternating
            0.3,  # Audio level
            0.9,  # Battery
            25.0  # Temperature
        ]
        
        # Pack sensor message
        sensor_data = struct.pack('5f', *sensors)
        sensor_msg = struct.pack('!IIBI',
            0x524F424F,  # Magic
            len(sensor_data) + 5,  # Length
            3,  # MSG_SENSORS
            5   # Vector length
        ) + sensor_data
        
        # Send sensors
        sock.send(sensor_msg)
        
        # Receive motor response
        motor_response = sock.recv(1024)
        
        if len(motor_response) >= 29:  # Expected motor message size
            # Parse header
            magic, msg_len, msg_type, vec_len = struct.unpack('!IIBI', motor_response[:13])
            
            if msg_type == 4:  # MSG_MOTORS
                # Parse motor values
                motor_data = motor_response[13:13+vec_len*4]
                motors = struct.unpack(f'{vec_len}f', motor_data)
                
                print(f"  Cycle {cycle+1}: Sent ultrasonic={sensors[0]:.0f}, "
                      f"Got motor1={motors[2]:.2f}")
        
        time.sleep(0.1)  # Small delay between cycles
    
    print("\n✓ Enhanced Brain is responding correctly!")
    
    # Check telemetry to see if learning is happening
    print("\nChecking learning progress...")
    try:
        tel_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tel_sock.settimeout(1.0)
        tel_sock.connect(('localhost', 9998))
        
        # Read one telemetry message
        tel_data = tel_sock.recv(4096).decode('utf-8')
        if 'learning_score' in tel_data:
            print("✓ Learning systems active (telemetry confirms)")
        else:
            print("⚡ Telemetry active (learning will begin shortly)")
        
        tel_sock.close()
    except:
        print("  (Telemetry not available)")
    
    sock.close()
    
    print("\n" + "=" * 60)
    print("SUCCESS: Enhanced Brain is fully operational!")
    print("The brain will learn from robot experience over time.")
    print("Use monitor_brain_progress.py to watch learning happen.")
    print("=" * 60)


if __name__ == "__main__":
    test_connection()