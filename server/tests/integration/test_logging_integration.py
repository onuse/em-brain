#!/usr/bin/env python3
"""
Test logging integration in new architecture.
"""

import sys
import os
import time
import json
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory


def test_logging():
    """Test logging functionality in new architecture."""
    
    print("🧪 Testing Logging Integration")
    print("=" * 60)
    
    # Create test log directory
    test_log_dir = "./test_logs"
    os.makedirs(test_log_dir, exist_ok=True)
    
    # Initialize components
    print("\n1. Initializing components with logging...")
    
    robot_registry = RobotRegistry()
    brain_factory = DynamicBrainFactory({
        'quiet_mode': True,
        'use_simple_brain': True
    })
    
    brain_pool = BrainPool(brain_factory)
    adapter_factory = AdapterFactory()
    
    # Create brain service with logging
    brain_service = BrainService(
        brain_pool, 
        adapter_factory,
        enable_persistence=False,  # Disable for this test
        enable_logging=True,
        log_dir=test_log_dir
    )
    
    connection_handler = ConnectionHandler(robot_registry, brain_service)
    print("   ✓ Components initialized with logging")
    
    # Simulate robot connection
    print("\n2. Connecting robot and processing cycles...")
    
    client_id = "test_logging_robot"
    capabilities = [1.0, 8.0, 2.0, 0.0, 0.0]  # Simple 8D sensors, 2D motors
    
    response = connection_handler.handle_handshake(client_id, capabilities)
    print(f"   ✓ Robot connected: {response}")
    
    # Process cycles to generate logs
    print("\n3. Processing cycles to generate logs...")
    
    for i in range(20):
        sensory_data = [0.5 + 0.1 * (i % 5)] * 8
        motor_response = connection_handler.handle_sensory_input(client_id, sensory_data)
        
        if i % 5 == 0 and motor_response:
            print(f"   Cycle {i}: Motor[0]={motor_response[0]:.3f}")
    
    print("   ✓ Processed 20 cycles")
    
    # Check log files
    print("\n4. Checking log files...")
    
    log_path = Path(test_log_dir)
    log_files = list(log_path.glob("**/*.log"))
    json_files = list(log_path.glob("**/*.jsonl"))
    
    print(f"   Found {len(log_files)} .log files")
    print(f"   Found {len(json_files)} .jsonl files")
    
    # Show some log files
    all_files = log_files + json_files
    if all_files:
        print("\n   Log files created:")
        for f in all_files[:5]:  # Show first 5
            print(f"     - {f.relative_to(log_path)}")
    
    # Check log content
    if json_files:
        print("\n5. Checking log content...")
        
        # Read first few lines from a JSONL file
        first_jsonl = json_files[0]
        with open(first_jsonl, 'r') as f:
            lines = f.readlines()[:3]
        
        print(f"   Sample from {first_jsonl.name}:")
        for line in lines:
            try:
                data = json.loads(line)
                log_type = data.get('type', 'unknown')
                timestamp = data.get('timestamp', 0)
                print(f"     - {log_type} at {timestamp:.2f}")
            except:
                pass
    
    # Get session summary
    print("\n6. Getting session summary...")
    
    if hasattr(brain_service.logging_service, 'get_session_summary'):
        # Find session ID
        sessions = brain_service.get_all_sessions()
        if sessions:
            session_id = list(sessions.keys())[0]
            summary = brain_service.logging_service.get_session_summary(session_id)
            
            if summary:
                print(f"   Session {session_id} summary:")
                print(f"     Total cycles: {summary['total_cycles']}")
                print(f"     Avg cycle time: {summary['avg_cycle_time_ms']:.2f}ms")
                print(f"     Max cycle time: {summary['max_cycle_time_ms']:.2f}ms")
    
    # Shutdown
    print("\n7. Shutting down...")
    connection_handler.handle_disconnect(client_id)
    brain_service.shutdown()
    
    # Cleanup
    print("\n8. Cleaning up test logs...")
    import shutil
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)
        print("   ✓ Test logs cleaned")
    
    print("\n✅ Logging integration test complete!")


if __name__ == "__main__":
    test_logging()