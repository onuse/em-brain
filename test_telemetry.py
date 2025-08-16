#!/usr/bin/env python3
"""
Test brain telemetry broadcasting.

Starts a brain server with telemetry, simulates a robot connection,
and monitors the telemetry stream.
"""

import sys
import os
import socket
import struct
import threading
import time
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.core.simple_brain_service import SimpleBrainService


def simulate_robot(port=9999, duration=10):
    """Simulate a robot sending sensor data."""
    time.sleep(2)  # Let server start
    
    print("🤖 Robot simulator starting...")
    
    try:
        # Connect to brain
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        print("🤖 Connected to brain")
        
        start = time.time()
        cycle = 0
        
        while time.time() - start < duration:
            cycle += 1
            
            # Generate sensor data
            if cycle < 50:
                # Low input (starvation)
                sensors = [np.random.randn() * 0.1 for _ in range(16)]
            elif cycle < 100:
                # Uniform (boredom)
                sensors = [0.5] * 16
            else:
                # Varied (should be interesting but brain finds it boring!)
                t = time.time() - start
                sensors = [
                    np.sin(t * 2 + i * 0.5) * np.cos(t * 0.5 + i * 0.2)
                    for i in range(16)
                ]
            
            # Send sensors
            header = b'BRAIN' + struct.pack('>BI', 1, len(sensors))
            data = struct.pack(f'>{len(sensors)}f', *sensors)
            sock.sendall(header + data)
            
            # Receive motors
            response = sock.recv(1024)
            
            time.sleep(0.05)  # ~20 Hz
        
        sock.close()
        print(f"🤖 Robot simulator finished ({cycle} cycles)")
        
    except Exception as e:
        print(f"🤖 Robot error: {e}")


def monitor_telemetry(port=9998, duration=10):
    """Connect to telemetry stream and display data."""
    time.sleep(3)  # Let server and robot start
    
    print("📊 Telemetry monitor starting...")
    
    try:
        # Connect to telemetry
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        print("📊 Connected to telemetry stream")
        
        buffer = b''
        start = time.time()
        count = 0
        
        motivation_counts = {}
        
        while time.time() - start < duration:
            # Receive data
            data = sock.recv(4096)
            if not data:
                break
            
            buffer += data
            
            # Parse JSON lines
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                try:
                    msg = json.loads(line.decode('utf-8'))
                    count += 1
                    
                    # Track motivation
                    if 'motivation' in msg:
                        mot = msg['motivation'].split(' - ')[0]
                        motivation_counts[mot] = motivation_counts.get(mot, 0) + 1
                    
                    # Display every 20th message
                    if count % 20 == 0:
                        print(f"📊 Cycle {msg.get('cycle', '?')}: "
                              f"{msg.get('motivation', 'unknown')[:30]}, "
                              f"Energy: {msg.get('energy', 0):.3f}, "
                              f"Comfort: {msg.get('comfort', 0):.2f}")
                    
                except Exception as e:
                    pass
        
        sock.close()
        
        print(f"\n📊 Telemetry monitor finished ({count} messages)")
        print("\nMotivation distribution:")
        total = sum(motivation_counts.values())
        for mot, cnt in motivation_counts.items():
            pct = (cnt / total * 100) if total > 0 else 0
            print(f"  {mot:15} {pct:5.1f}%")
        
    except Exception as e:
        print(f"📊 Monitor error: {e}")


def main():
    print("="*60)
    print("TELEMETRY TEST")
    print("="*60)
    
    # Create brain service
    service = SimpleBrainService(
        port=9999,
        brain_config='speed',  # Small brain for fast testing
        telemetry_port=9998
    )
    
    # Start threads
    robot_thread = threading.Thread(
        target=simulate_robot, 
        args=(9999, 8),
        daemon=True
    )
    
    monitor_thread = threading.Thread(
        target=monitor_telemetry,
        args=(9998, 7),
        daemon=True
    )
    
    robot_thread.start()
    monitor_thread.start()
    
    # Start brain service (blocks)
    try:
        service.start(enable_vision=False)  # No vision for test
    except KeyboardInterrupt:
        pass
    
    service.stop()
    
    # Wait for threads
    robot_thread.join(timeout=1)
    monitor_thread.join(timeout=1)
    
    print("\n✅ Telemetry test complete!")


if __name__ == "__main__":
    main()