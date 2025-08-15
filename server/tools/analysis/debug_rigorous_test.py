#!/usr/bin/env python3
"""Debug why rigorous test hangs"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("Starting debug...")

# Test 1: Basic initialization
print("\n1. Testing basic initialization...")
from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory
print("   ✓ All imports successful")

# Test 2: Create components
print("\n2. Creating components...")
robot_registry = RobotRegistry()
print("   ✓ RobotRegistry")

brain_config = {'quiet_mode': True, 'spatial_resolution': 4}
brain_factory = DynamicBrainFactory(brain_config)
print("   ✓ BrainFactory")

brain_pool = BrainPool(brain_factory)
print("   ✓ BrainPool")

adapter_factory = AdapterFactory()
print("   ✓ AdapterFactory")

brain_service = BrainService(brain_pool, adapter_factory)
print("   ✓ BrainService")

connection_handler = ConnectionHandler(robot_registry, brain_service)
print("   ✓ ConnectionHandler")

# Test 3: Try monitoring server
print("\n3. Testing monitoring server...")
try:
    from src.core.monitoring_server import DynamicMonitoringServer
    monitoring_server = DynamicMonitoringServer(
        brain_service=brain_service,
        connection_handler=connection_handler,
        host='localhost',
        port=9998
    )
    print("   ✓ MonitoringServer created")
    
    monitoring_server.start()
    print("   ✓ MonitoringServer started")
except Exception as e:
    print(f"   ✗ MonitoringServer error: {e}")

# Test 4: Create a session
print("\n4. Testing handshake...")
try:
    response = connection_handler.handle_handshake('test_robot', [1.0, 16.0, 4.0, 0.0, 0.0])
    print(f"   ✓ Handshake successful: {response}")
except Exception as e:
    print(f"   ✗ Handshake error: {e}")

print("\n✅ Debug complete!")