#!/usr/bin/env python3
"""
Test monitoring server functionality.
"""

import sys
import os
import time
import json
import socket
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.core.monitoring_server import DynamicMonitoringServer


def test_monitoring_client(host='localhost', port=9998):
    """Connect to monitoring server and test commands."""
    
    print("\n📊 Testing Monitoring Client")
    print("=" * 40)
    
    try:
        # Connect to monitoring server
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        client.settimeout(5.0)
        
        # Receive welcome message
        welcome = client.recv(1024).decode('utf-8')
        welcome_data = json.loads(welcome)
        print(f"\n✅ Connected to monitoring server")
        print(f"   Server: {welcome_data.get('server')}")
        print(f"   Commands: {', '.join(welcome_data.get('commands', []))}")
        
        # Test each command
        commands = [
            'brain_stats',
            'session_info',
            'connection_stats',
            'active_brains',
            'performance_metrics'
        ]
        
        for cmd in commands:
            print(f"\n📋 Testing command: {cmd}")
            client.send((cmd + "\n").encode('utf-8'))
            
            response = client.recv(4096).decode('utf-8')
            data = json.loads(response)
            
            if data['status'] == 'success':
                print(f"   ✅ Success")
                if cmd == 'performance_metrics':
                    metrics = data['data']
                    print(f"   Total cycles: {metrics.get('total_cycles', 0)}")
                    print(f"   Active sessions: {metrics.get('active_sessions', 0)}")
                    print(f"   Avg cycle time: {metrics.get('average_cycle_time_ms', 0):.1f}ms")
                elif cmd == 'active_brains':
                    brains = data['data']
                    print(f"   Active brains: {len(brains)}")
                    for brain in brains:
                        print(f"     - {brain['profile']}: {brain['field_dimensions']}D")
            else:
                print(f"   ❌ Error: {data.get('error')}")
        
        # Test unknown command
        print(f"\n📋 Testing unknown command")
        client.send("unknown_command\n".encode('utf-8'))
        response = client.recv(1024).decode('utf-8')
        data = json.loads(response)
        print(f"   Expected error: {data.get('error')}")
        
        client.close()
        print("\n✅ Monitoring client test complete")
        
    except Exception as e:
        print(f"❌ Monitoring client error: {e}")


def test_monitoring_server():
    """Test monitoring server functionality."""
    
    print("🧪 Testing Monitoring Server")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
    
    robot_registry = RobotRegistry()
    brain_factory = DynamicBrainFactory({
        'quiet_mode': True,
        'use_simple_brain': True
    })
    
    brain_pool = BrainPool(brain_factory)
    adapter_factory = AdapterFactory()
    brain_service = BrainService(
        brain_pool, 
        adapter_factory,
        enable_persistence=False
    )
    
    connection_handler = ConnectionHandler(robot_registry, brain_service)
    
    # Create monitoring server
    monitoring_server = DynamicMonitoringServer(
        brain_service=brain_service,
        connection_handler=connection_handler,
        host='localhost',
        port=9998
    )
    
    print("   ✓ Components initialized")
    
    # Start monitoring server
    print("\n2. Starting monitoring server...")
    monitoring_server.start()
    time.sleep(0.5)  # Let it start
    
    # Simulate some robot activity
    print("\n3. Creating robot sessions...")
    
    # Connect first robot
    client_id1 = "test_robot_1"
    capabilities1 = [1.0, 16.0, 5.0, 1.0, 3.0]  # PiCar-X
    response1 = connection_handler.handle_handshake(client_id1, capabilities1)
    print(f"   ✓ Robot 1 connected")
    
    # Process some data
    for i in range(5):
        sensory_data = [0.5] * 16
        connection_handler.handle_sensory_input(client_id1, sensory_data)
    
    # Connect second robot
    client_id2 = "test_robot_2"
    capabilities2 = [1.0, 8.0, 2.0, 0.0, 0.0]  # Simple robot
    response2 = connection_handler.handle_handshake(client_id2, capabilities2)
    print(f"   ✓ Robot 2 connected")
    
    # Process some data
    for i in range(3):
        sensory_data = [0.3] * 8
        connection_handler.handle_sensory_input(client_id2, sensory_data)
    
    # Test monitoring client in a thread
    print("\n4. Testing monitoring client...")
    client_thread = threading.Thread(target=test_monitoring_client)
    client_thread.start()
    client_thread.join()
    
    # Shutdown
    print("\n5. Shutting down...")
    monitoring_server.stop()
    print("   ✓ Monitoring server stopped")
    
    print("\n✅ Monitoring server test complete!")


if __name__ == "__main__":
    test_monitoring_server()