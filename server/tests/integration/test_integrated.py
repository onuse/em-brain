#!/usr/bin/env python3
"""
Integrated test of the dynamic brain architecture.
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


def test_full_flow():
    """Test the full flow from handshake to sensory processing."""
    
    print("🧪 Testing Full Dynamic Brain Flow")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing components...")
    
    profiles_dir = Path(__file__).parent.parent / "client_picarx"
    robot_registry = RobotRegistry(profiles_dir=profiles_dir)
    brain_factory = DynamicBrainFactory({'quiet_mode': True})
    brain_pool = BrainPool(brain_factory)
    adapter_factory = AdapterFactory()
    brain_service = BrainService(brain_pool, adapter_factory)
    connection_handler = ConnectionHandler(robot_registry, brain_service)
    
    print("   ✓ All components initialized")
    
    # Test handshake
    print("\n2. Testing handshake...")
    
    client_id = "test_client_001"
    capabilities = [1.0, 16.0, 5.0, 1.0, 3.0]  # PiCar-X
    
    try:
        response = connection_handler.handle_handshake(client_id, capabilities)
        print(f"   ✓ Handshake successful: {response}")
    except Exception as e:
        print(f"   ❌ Handshake failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test sensory processing
    print("\n3. Testing sensory processing...")
    
    # Create sensory data (16D for PiCar-X)
    sensory_data = [0.5] * 16
    
    try:
        motor_response = connection_handler.handle_sensory_input(client_id, sensory_data)
        print(f"   ✓ Sensory processing successful")
        print(f"     Input: {len(sensory_data)}D sensory")
        print(f"     Output: {len(motor_response)}D motor")
        print(f"     Motor values: {[f'{v:.3f}' for v in motor_response]}")
    except Exception as e:
        print(f"   ❌ Sensory processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test with different robot
    print("\n4. Testing with different robot...")
    
    client_id2 = "test_client_002"
    capabilities2 = [1.0, 32.0, 8.0, 2.0, 7.0]  # Advanced robot
    
    try:
        response2 = connection_handler.handle_handshake(client_id2, capabilities2)
        print(f"   ✓ Second robot handshake: {response2}")
        
        sensory_data2 = [0.3] * 32
        motor_response2 = connection_handler.handle_sensory_input(client_id2, sensory_data2)
        print(f"   ✓ Second robot processing: {len(motor_response2)}D motor output")
    except Exception as e:
        print(f"   ❌ Second robot failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Show statistics
    print("\n5. Statistics:")
    stats = connection_handler.get_stats()
    print(f"   Total connections: {stats['total_connections']}")
    print(f"   Active connections: {stats['active_connections']}")
    print(f"   Total messages: {stats['total_messages']}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_full_flow()