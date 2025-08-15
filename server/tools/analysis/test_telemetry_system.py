#!/usr/bin/env python3
"""
Test the brain telemetry system

Shows how to access brain internals through the monitoring interface.
"""

import socket
import json
import time
import sys
import os
from pathlib import Path

# Add paths
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))
testing_path = Path(__file__).parent.parent / 'testing'
sys.path.insert(0, str(testing_path))

from behavioral_test_dynamic import DynamicBehavioralTestFramework


def test_monitoring_connection():
    """Test connection to monitoring server"""
    print("🔍 Testing Monitoring Server Connection")
    print("=" * 60)
    
    # Connect to monitoring server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    
    try:
        sock.connect(('localhost', 9998))
        print("✅ Connected to monitoring server")
        
        # Receive welcome message
        welcome = sock.recv(1024).decode('utf-8').strip()
        welcome_data = json.loads(welcome)
        print(f"\nServer: {welcome_data['server']}")
        print(f"Version: {welcome_data['version']}")
        print(f"Available commands: {', '.join(welcome_data['commands'])}")
        
        # Test telemetry command
        print("\n📊 Testing telemetry command...")
        sock.send(b"telemetry\n")
        
        response = sock.recv(4096).decode('utf-8').strip()
        telemetry_data = json.loads(response)
        
        if telemetry_data['status'] == 'success':
            print("✅ Telemetry response received")
            if telemetry_data['data']:
                for session_id, telemetry in telemetry_data['data'].items():
                    print(f"\nSession: {session_id}")
                    for key, value in telemetry.items():
                        print(f"  {key}: {value}")
            else:
                print("  (No active sessions)")
        
        sock.close()
        
    except ConnectionRefusedError:
        print("❌ Could not connect to monitoring server on port 9998")
        print("   Make sure the brain server is running")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_telemetry_with_brain():
    """Test telemetry while brain is processing"""
    print("\n🧠 Testing Telemetry with Active Brain")
    print("=" * 60)
    
    # Create brain and robot
    framework = DynamicBehavioralTestFramework(quiet_mode=True)
    framework.setup_virtual_robot()
    
    print("✅ Brain created, starting processing...")
    
    # Process some patterns
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    for i in range(10):
        framework.connection_handler.handle_sensory_input(
            framework.client_id, pattern
        )
    
    # Now connect to monitoring and check telemetry
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    
    try:
        sock.connect(('localhost', 9998))
        
        # Skip welcome
        sock.recv(1024)
        
        # Get telemetry
        sock.send(b"telemetry\n")
        response = sock.recv(4096).decode('utf-8').strip()
        telemetry_data = json.loads(response)
        
        if telemetry_data['status'] == 'success' and telemetry_data['data']:
            print("\n📊 Current Brain Telemetry:")
            for session_id, telemetry in telemetry_data['data'].items():
                print(f"\nSession: {session_id}")
                print(f"  Cycles: {telemetry.get('cycles', 0)}")
                print(f"  Energy: {telemetry.get('energy', 0)}")
                print(f"  Confidence: {telemetry.get('confidence', 0)}")
                print(f"  Mode: {telemetry.get('mode', 'unknown')}")
                print(f"  Phase: {telemetry.get('phase', 'unknown')}")
                print(f"  Memory regions: {telemetry.get('memory_regions', 0)}")
                print(f"  Constraints: {telemetry.get('constraints', 0)}")
                print(f"  Blend: {telemetry.get('blend', 'unknown')}")
                
                # Get detailed telemetry
                print(f"\n📈 Getting detailed telemetry for {session_id}...")
                sock.send(f"telemetry {session_id}\n".encode('utf-8'))
                
                detailed = sock.recv(4096).decode('utf-8').strip()
                detailed_data = json.loads(detailed)
                
                if detailed_data['status'] == 'success':
                    data = detailed_data['data']
                    print(f"  Prediction confidence: {data.get('prediction_confidence', 0):.3f}")
                    print(f"  Prediction error: {data.get('prediction_error', 'N/A')}")
                    print(f"  Improvement rate: {data.get('improvement_rate', 0):.3f}")
                    
                    history = data.get('prediction_history', [])
                    if history:
                        print(f"  Prediction history: {[f'{h:.3f}' for h in history[-5:]]}")
        
        sock.close()
        
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
    
    # Cleanup
    framework.cleanup()


def main():
    """Run telemetry tests"""
    # First test basic monitoring connection
    test_monitoring_connection()
    
    # Then test with active brain
    test_telemetry_with_brain()
    
    print("\n✅ Telemetry test complete")


if __name__ == "__main__":
    main()