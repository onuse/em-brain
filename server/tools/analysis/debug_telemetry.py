#!/usr/bin/env python3
"""
Debug telemetry issue - why are we getting default values?
"""

import sys
import os
from pathlib import Path
import time

# Add paths
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.core.monitoring_server import DynamicMonitoringServer
from src.core.telemetry_client import TelemetryClient


def debug_telemetry():
    """Debug telemetry flow"""
    print("🔍 Debugging Telemetry System")
    print("=" * 60)
    
    # Create components
    robot_registry = RobotRegistry()
    brain_config = {
        'quiet_mode': True,
        'use_simple_brain': False,
        'spatial_resolution': 4
    }
    brain_factory = DynamicBrainFactory(brain_config)
    brain_pool = BrainPool(brain_factory)
    adapter_factory = AdapterFactory()
    brain_service = BrainService(brain_pool, adapter_factory)
    connection_handler = ConnectionHandler(robot_registry, brain_service)
    
    # Start monitoring server
    monitoring_server = DynamicMonitoringServer(
        brain_service=brain_service,
        connection_handler=connection_handler,
        host='localhost',
        port=9998
    )
    monitoring_server.start()
    
    # Create virtual robot
    client_id = "debug_robot"
    capabilities = [1.0, 16.0, 4.0, 0.0, 0.0]
    
    print("\n1. Creating session...")
    response = connection_handler.handle_handshake(client_id, capabilities)
    print(f"   Handshake response: {response}")
    
    # Get session info
    sessions = brain_service.get_all_sessions()
    print(f"\n2. Active sessions: {list(sessions.keys())}")
    
    if sessions:
        session_id = list(sessions.keys())[0]
        print(f"   Using session: {session_id}")
        
        # Get the actual session object
        session_obj = brain_service.get_session(session_id)
        if session_obj:
            print(f"\n3. Session object found")
            print(f"   Brain class: {type(session_obj.brain).__name__}")
            print(f"   Brain cycles: {getattr(session_obj.brain, 'brain_cycles', 'NOT FOUND')}")
            print(f"   Has telemetry adapter: {hasattr(session_obj, 'telemetry_adapter')}")
            
            # Process some data
            print("\n4. Processing sensory input...")
            for i in range(5):
                sensory_input = [0.5] * 16
                motor_output = connection_handler.handle_sensory_input(client_id, sensory_input)
                print(f"   Cycle {i}: motor output = {motor_output[:2]}")
            
            # Check brain state after processing
            print(f"\n5. Brain state after processing:")
            print(f"   Brain cycles: {getattr(session_obj.brain, 'brain_cycles', 'NOT FOUND')}")
            
            # Get telemetry through brain service
            print("\n6. Getting telemetry through brain service...")
            telemetry_summary = brain_service.get_session_telemetry(session_id)
            print(f"   Summary: {telemetry_summary}")
            
            detailed_telemetry = brain_service.get_detailed_telemetry(session_id)
            print(f"   Detailed: {detailed_telemetry}")
            
            # Direct access to telemetry adapter
            if hasattr(session_obj, 'telemetry_adapter'):
                print("\n7. Direct telemetry adapter access...")
                adapter = session_obj.telemetry_adapter
                print(f"   Adapter brain reference: {adapter.brain}")
                print(f"   Same brain? {adapter.brain is session_obj.brain}")
                
                # Get brain state directly
                if hasattr(adapter.brain, 'get_brain_state'):
                    brain_state = adapter.brain.get_brain_state()
                    print(f"   Direct brain state: {brain_state}")
                
                # Get telemetry
                telemetry = adapter.get_telemetry()
                print(f"   Telemetry cycles: {telemetry.brain_cycles}")
                print(f"   Telemetry confidence: {telemetry.prediction_confidence}")
    
    # Test through telemetry client
    print("\n8. Testing through telemetry client...")
    telemetry_client = TelemetryClient()
    telemetry_client.connect()
    
    sessions = telemetry_client.get_all_sessions()
    print(f"   Sessions from client: {sessions}")
    
    if sessions:
        session_telemetry = telemetry_client.get_session_telemetry(sessions[0])
        if session_telemetry:
            print(f"   Client telemetry: cycles={session_telemetry.cycles}, confidence={session_telemetry.confidence}")
    
    # Cleanup
    telemetry_client.disconnect()
    connection_handler.handle_disconnect(client_id)
    monitoring_server.stop()
    
    print("\n✅ Debug complete")


if __name__ == "__main__":
    debug_telemetry()