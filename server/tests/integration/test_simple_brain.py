#!/usr/bin/env python3
"""
Test the dynamic architecture with SimpleFieldBrain.
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


def test_with_simple_brain():
    """Test using the simple field brain implementation."""
    
    print("🧪 Testing Dynamic Architecture with Simple Field Brain")
    print("=" * 60)
    
    # Initialize components with simple brain
    print("\n1. Initializing with SimpleFieldBrain...")
    
    profiles_dir = Path(__file__).parent.parent / "client_picarx"
    robot_registry = RobotRegistry(profiles_dir=profiles_dir)
    
    # Enable simple brain
    brain_factory = DynamicBrainFactory({
        'quiet_mode': True,
        'use_simple_brain': True
    })
    
    brain_pool = BrainPool(brain_factory)
    adapter_factory = AdapterFactory()
    brain_service = BrainService(brain_pool, adapter_factory)
    connection_handler = ConnectionHandler(robot_registry, brain_service)
    
    print("   ✓ Components initialized with simple brain")
    
    # Test different robots
    test_configs = [
        ("Minimal Robot", "test_minimal", [1.0, 8.0, 2.0, 2.0, 0.0]),
        ("PiCar-X", "test_picarx", [1.0, 16.0, 5.0, 1.0, 3.0]),
        ("Advanced Robot", "test_advanced", [1.0, 32.0, 8.0, 2.0, 7.0]),
    ]
    
    for robot_name, client_id, capabilities in test_configs:
        print(f"\n2. Testing {robot_name}...")
        print(f"   Capabilities: {int(capabilities[1])}D sensors, {int(capabilities[2])}D motors")
        
        # Handshake
        response = connection_handler.handle_handshake(client_id, capabilities)
        print(f"   ✓ Handshake response: {response}")
        
        # Test processing
        sensory_dim = int(capabilities[1])
        sensory_data = [0.5] * sensory_dim
        
        print(f"\n   Processing {sensory_dim}D sensory input...")
        motor_response = connection_handler.handle_sensory_input(client_id, sensory_data)
        
        print(f"   ✓ Motor response: {len(motor_response)}D")
        print(f"   Values: {[f'{v:.3f}' for v in motor_response[:5]]}...")
        
        # Multiple cycles to see field evolution
        print("\n   Running 5 cycles to observe field evolution:")
        for i in range(5):
            # Vary input slightly
            sensory_data = [0.5 + 0.1 * (i % 2)] * sensory_dim
            motor_response = connection_handler.handle_sensory_input(client_id, sensory_data)
            max_motor = max(abs(v) for v in motor_response)
            print(f"     Cycle {i+1}: Max motor output = {max_motor:.3f}")
    
    # Show final statistics
    print("\n3. Final Statistics:")
    stats = connection_handler.get_stats()
    print(f"   Total connections: {stats['total_connections']}")
    print(f"   Total messages: {stats['total_messages']}")
    
    # Show brain configurations
    print("\n4. Brain Configurations:")
    for profile_key, brain in brain_pool.get_active_brains().items():
        config = brain_pool.get_brain_config(profile_key)
        print(f"   {profile_key}: {brain.get_field_dimensions()}D field, "
              f"{config['sensory_dim']}→{config['motor_dim']} interface")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_with_simple_brain()