#!/usr/bin/env python3
"""
Test that PureFieldBrain can connect to robot without issues.
Simulates a basic robot connection and verifies proper operation.
"""

import socket
import json
import time
import struct
import numpy as np
import sys

def send_message(sock, msg_type, data):
    """Send a message to the brain server."""
    message = {
        'type': msg_type,
        'data': data
    }
    msg_bytes = json.dumps(message).encode('utf-8')
    msg_len = struct.pack('>I', len(msg_bytes))
    sock.sendall(msg_len + msg_bytes)

def receive_message(sock):
    """Receive a message from the brain server."""
    # Read message length
    msg_len_bytes = sock.recv(4)
    if not msg_len_bytes:
        return None
    msg_len = struct.unpack('>I', msg_len_bytes)[0]
    
    # Read message
    msg_bytes = b''
    while len(msg_bytes) < msg_len:
        chunk = sock.recv(min(msg_len - len(msg_bytes), 4096))
        if not chunk:
            return None
        msg_bytes += chunk
    
    return json.loads(msg_bytes.decode('utf-8'))

def test_robot_connection():
    """Test connecting to the brain server as a robot."""
    
    print("=" * 70)
    print("🤖 ROBOT CONNECTION TEST")
    print("=" * 70)
    
    # Connection parameters
    host = 'localhost'
    port = 9999
    
    print(f"📡 Connecting to brain server at {host}:{port}...")
    
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print("✅ Connected successfully!")
        
        # Send robot profile
        print("\n📋 Sending robot profile...")
        profile = {
            'robot_type': 'test_robot',
            'capabilities': {
                'sensors': {
                    'ultrasonic': {'dim': 1, 'range': [0, 300]},
                    'camera': {'dim': 9, 'range': [-1, 1]},
                },
                'motors': {
                    'speed': {'dim': 2, 'range': [-1, 1]},
                    'steering': {'dim': 1, 'range': [-30, 30]},
                    'actuator': {'dim': 1, 'range': [0, 1]}
                }
            },
            'metadata': {
                'version': '1.0',
                'test_mode': True
            }
        }
        
        send_message(sock, 'profile', profile)
        
        # Wait for acknowledgment
        response = receive_message(sock)
        if response and response['type'] == 'profile_ack':
            print(f"✅ Profile accepted: {response['data']['status']}")
            print(f"   Brain dimensions: {response['data']['sensory_dim']} → {response['data']['motor_dim']}")
        else:
            print("❌ Unexpected response to profile")
            return False
        
        # Run test cycles
        print("\n🔄 Running test cycles...")
        print("-" * 50)
        
        test_cycles = 100
        start_time = time.time()
        
        for cycle in range(test_cycles):
            # Create sensory data
            sensory_data = {
                'ultrasonic': [150.0 + np.sin(cycle * 0.1) * 50],
                'camera': np.random.randn(9).tolist(),
            }
            
            # Send sensory data
            send_message(sock, 'sense', sensory_data)
            
            # Receive motor response
            response = receive_message(sock)
            
            if response and response['type'] == 'action':
                motor_data = response['data']
                
                # Print progress every 20 cycles
                if (cycle + 1) % 20 == 0:
                    speed = motor_data.get('speed', [0, 0])
                    steering = motor_data.get('steering', [0])
                    actuator = motor_data.get('actuator', [0])
                    
                    print(f"Cycle {cycle+1:3d}: "
                          f"Speed=[{speed[0]:+.2f}, {speed[1]:+.2f}] | "
                          f"Steer=[{steering[0]:+.2f}] | "
                          f"Act=[{actuator[0]:.2f}]")
                    
                    # Check brain state if provided
                    if 'brain_state' in response:
                        state = response['brain_state']
                        if 'field_energy' in state:
                            print(f"         Brain: Energy={state['field_energy']:.3f}, "
                                  f"Cycle={state.get('cycle', 0)}")
            else:
                print(f"❌ Failed to receive response at cycle {cycle}")
                return False
        
        elapsed = time.time() - start_time
        avg_cycle_time = elapsed / test_cycles * 1000
        
        print("-" * 50)
        print(f"✅ Test completed successfully!")
        print(f"   Total time: {elapsed:.2f} seconds")
        print(f"   Average cycle: {avg_cycle_time:.2f} ms")
        
        # Send disconnect
        print("\n👋 Disconnecting...")
        send_message(sock, 'disconnect', {})
        sock.close()
        
        print("✅ Clean disconnect")
        return True
        
    except ConnectionRefusedError:
        print("❌ Cannot connect to brain server")
        print("   Make sure to start the server first:")
        print("   python3 brain.py")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            sock.close()
        except:
            pass

if __name__ == "__main__":
    print("🚀 Starting robot connection test...")
    print("   This test requires the brain server to be running.")
    print()
    
    success = test_robot_connection()
    
    print()
    print("=" * 70)
    if success:
        print("✅ ALL TESTS PASSED - Brain is ready for robot operation!")
        sys.exit(0)
    else:
        print("❌ Test failed - Check server logs for details")
        sys.exit(1)