#!/usr/bin/env python3
"""Test actual brain cycle performance in server mode."""

import sys
import os
import time
import socket
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

def test_server_performance():
    """Connect to brain server and measure cycle times."""
    
    print("Testing Brain Server Performance")
    print("="*60)
    
    # Wait for server to start
    time.sleep(3)
    
    # Connect to server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 9999))
        print("✓ Connected to brain server on port 9999")
        
        # Send handshake
        handshake = {
            "type": "handshake",
            "version": "1.0",
            "robot_profile": "test_robot",
            "dimensions": {
                "sensory": 16,
                "motor": 3,
                "hardware": {
                    "cpu_cores": 8,
                    "ram_gb": 16,
                    "gpu_type": "mps"
                }
            }
        }
        
        sock.send((json.dumps(handshake) + '\n').encode())
        
        # Read response
        response = sock.recv(4096).decode()
        print(f"✓ Handshake response: {response[:100]}...")
        
        # Measure cycle times
        cycle_times = []
        print("\nMeasuring cycle times...")
        
        for i in range(20):
            # Prepare sensory data
            sensory_data = {
                "type": "sensory",
                "timestamp": time.time(),
                "data": [0.5] * 16
            }
            
            # Send and time the round trip
            start_time = time.perf_counter()
            sock.send((json.dumps(sensory_data) + '\n').encode())
            
            # Wait for motor response
            response = sock.recv(4096).decode()
            cycle_time = (time.perf_counter() - start_time) * 1000
            
            cycle_times.append(cycle_time)
            
            if i % 5 == 0:
                print(f"  Cycle {i+1}: {cycle_time:.1f}ms")
        
        # Statistics
        print(f"\nPerformance Statistics:")
        print(f"  Average cycle time: {sum(cycle_times)/len(cycle_times):.1f}ms")
        print(f"  Min cycle time: {min(cycle_times):.1f}ms")
        print(f"  Max cycle time: {max(cycle_times):.1f}ms")
        print(f"  Std deviation: {(sum((x - sum(cycle_times)/len(cycle_times))**2 for x in cycle_times) / len(cycle_times))**0.5:.1f}ms")
        
        sock.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Is the brain server running? Start with: python3 server/brain.py")

if __name__ == "__main__":
    test_server_performance()