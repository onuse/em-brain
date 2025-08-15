#!/usr/bin/env python3
"""
Test experience storage in new architecture.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory


def test_experience_storage():
    """Test experience storage functionality."""
    
    print("🧪 Testing Experience Storage")
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
        enable_persistence=False  # Disable for this test
    )
    
    connection_handler = ConnectionHandler(robot_registry, brain_service)
    print("   ✓ Components initialized")
    
    # Simulate robot connection
    print("\n2. Connecting robot...")
    
    client_id = "test_experience_robot"
    capabilities = [1.0, 8.0, 2.0, 0.0, 0.0]  # Simple 8D sensors, 2D motors
    
    response = connection_handler.handle_handshake(client_id, capabilities)
    print(f"   ✓ Robot connected: {response}")
    
    # Process multiple cycles to generate experiences
    print("\n3. Processing cycles to generate experiences...")
    
    for i in range(10):
        # Vary sensory input
        sensory_data = [0.5 + 0.1 * (i % 3)] * 8
        motor_response = connection_handler.handle_sensory_input(client_id, sensory_data)
        
        if i < 5:
            print(f"   Cycle {i}: Sensory[0]={sensory_data[0]:.1f} → Motor[0]={motor_response[0]:.3f}")
    
    # Get session statistics
    print("\n4. Checking experience tracking...")
    
    stats = connection_handler.get_stats()
    print(f"   Total connections: {stats['total_connections']}")
    print(f"   Total messages: {stats['total_messages']}")
    
    # Get detailed session info
    sessions = brain_service.get_all_sessions()
    for session_id, session_stats in sessions.items():
        print(f"\n   Session {session_id}:")
        print(f"     Robot: {session_stats['robot_type']}")
        print(f"     Cycles: {session_stats['cycles_processed']}")
        print(f"     Experiences: {session_stats['total_experiences']}")
        print(f"     Avg cycle time: {session_stats['average_cycle_time_ms']:.1f}ms")
    
    # Verify experiences were tracked
    if sessions:
        first_session = list(sessions.values())[0]
        expected_experiences = first_session['cycles_processed'] - 1  # First cycle has no previous
        actual_experiences = first_session['total_experiences']
        
        print(f"\n5. Experience validation:")
        print(f"   Expected experiences: {expected_experiences}")
        print(f"   Actual experiences: {actual_experiences}")
        
        if actual_experiences == expected_experiences:
            print("   ✅ Experience tracking working correctly!")
        else:
            print("   ❌ Experience count mismatch")
    
    print("\n✅ Experience storage test complete!")


if __name__ == "__main__":
    test_experience_storage()