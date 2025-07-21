#!/usr/bin/env python3
"""
Test Real TCP Server Field Energy Variance
Start actual TCP server and connect to test field energy variance issue
"""

import sys
import os
import socket
import struct
import time
import threading
import subprocess
import signal
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

def start_tcp_server():
    """Start the brain server in a separate process"""
    print("🚀 Starting TCP brain server...")
    
    # Clean memory for fresh test
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
    
    # Start server process
    server_proc = subprocess.Popen([
        'python3', 'brain_server.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
       preexec_fn=os.setsid)
    
    # Give server time to start
    print("⏳ Waiting for server to start...")
    time.sleep(5)
    
    return server_proc

def send_receive_data(host='localhost', port=9999, sensor_data=None):
    """Send sensor data to server and receive action response"""
    if sensor_data is None:
        sensor_data = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2  # 16D
    
    try:
        # Connect to server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect((host, port))
        
        # Send sensor data using the protocol: [length, data...]
        sensor_count = len(sensor_data)
        message = struct.pack(f'<I{sensor_count}f', sensor_count, *sensor_data)
        sock.send(message)
        
        # Receive action response
        # First get the action count
        action_count_data = sock.recv(4)
        if len(action_count_data) != 4:
            raise Exception("Failed to receive action count")
        
        action_count = struct.unpack('<I', action_count_data)[0]
        
        # Then get the action data
        action_data = sock.recv(action_count * 4)
        if len(action_data) != action_count * 4:
            raise Exception(f"Expected {action_count * 4} bytes, got {len(action_data)}")
        
        actions = list(struct.unpack(f'<{action_count}f', action_data))
        
        sock.close()
        return actions
        
    except Exception as e:
        print(f"❌ TCP communication error: {e}")
        if 'sock' in locals():
            sock.close()
        return None

def test_server_with_monitoring(host='localhost', monitor_port=9998):
    """Connect to monitoring server to get brain state"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, monitor_port))
        
        # Send request for brain stats
        request = b"GET_BRAIN_STATS"
        sock.send(struct.pack('<I', len(request)))
        sock.send(request)
        
        # Receive response
        response_length = struct.unpack('<I', sock.recv(4))[0]
        response_data = sock.recv(response_length)
        
        sock.close()
        
        # Parse JSON response
        import json
        brain_stats = json.loads(response_data.decode())
        return brain_stats
        
    except Exception as e:
        print(f"❌ Monitoring connection error: {e}")
        return None

def test_tcp_server_field_energy():
    """Test field energy variance through real TCP server"""
    print("🔍 Testing REAL TCP SERVER field energy variance...")
    
    # Start server
    server_proc = start_tcp_server()
    
    try:
        # Wait for server to fully initialize
        print("⏳ Allowing server initialization time...")
        time.sleep(8)
        
        # Test connection first
        print("🔗 Testing TCP connection...")
        test_response = send_receive_data()
        if test_response is None:
            print("❌ Could not connect to TCP server")
            return
        
        print(f"✅ Connection successful, received: {test_response}")
        
        # Collect field energy data over multiple cycles
        field_energies = []
        
        print("\n📊 Collecting field energy data over 20 cycles...")
        for i in range(20):
            # Vary the input pattern slightly (like a real robot would)
            base_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2
            varied_pattern = [x + 0.1 * np.sin(i * 0.5) + 0.05 * np.random.randn() for x in base_pattern]
            
            # Send data and get response
            actions = send_receive_data(sensor_data=varied_pattern)
            if actions is None:
                print(f"❌ Failed at cycle {i}")
                continue
            
            # Try to get brain stats via monitoring
            brain_stats = test_server_with_monitoring()
            if brain_stats:
                field_energy = brain_stats.get('field_brain', {}).get('field_energy', 0.0)
                field_energies.append(field_energy)
                
                if i < 5 or i % 5 == 0:
                    evolution_cycles = brain_stats.get('field_brain', {}).get('field_evolution_cycles', 0)
                    print(f"   Cycle {i}: field_energy={field_energy:.6f}, evolution_cycles={evolution_cycles}")
            else:
                print(f"   Cycle {i}: Could not get brain stats")
            
            # Small delay between cycles
            time.sleep(0.5)
        
        # Calculate variance
        if field_energies:
            energy_variance = np.var(field_energies)
            energy_mean = np.mean(field_energies)
            energy_range = max(field_energies) - min(field_energies)
            
            print(f"\n📊 TCP SERVER Results:")
            print(f"   Samples collected: {len(field_energies)}")
            print(f"   Field energy mean: {energy_mean:.6f}")
            print(f"   Field energy variance: {energy_variance:.9f}")
            print(f"   Field energy range: {min(field_energies):.6f} to {max(field_energies):.6f}")
            print(f"   Energy spread: {energy_range:.6f}")
            
            if energy_variance == 0.0:
                print("❌ CONFIRMED: TCP server shows ZERO variance")
                print("   This is the core issue!")
            elif energy_variance < 0.000001:
                print("⚠️ Very low variance - possible precision issue")
            else:
                print("✅ TCP server shows normal variance")
        else:
            print("❌ No field energy data collected")
    
    finally:
        # Shutdown server
        print("\n🛑 Shutting down TCP server...")
        if server_proc:
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
                server_proc.wait(timeout=10)
            except:
                # Force kill if needed
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
        
        print("✅ Server shutdown complete")

def main():
    """Run TCP server energy variance test"""
    print("🔍 Real TCP Server Field Energy Variance Test")
    print("=" * 80)
    
    try:
        test_tcp_server_field_energy()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()