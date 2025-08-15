#!/usr/bin/env python3
"""
Test persistence migration to new architecture.
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory


def test_persistence():
    """Test persistence functionality in new architecture."""
    
    print("🧪 Testing Persistence Migration")
    print("=" * 60)
    
    # Create test memory directory
    test_memory_path = "./test_memory"
    os.makedirs(test_memory_path, exist_ok=True)
    
    # Initialize components
    print("\n1. Initializing components with persistence...")
    
    profiles_dir = Path(__file__).parent.parent / "client_picarx"
    robot_registry = RobotRegistry(profiles_dir=profiles_dir)
    
    brain_factory = DynamicBrainFactory({
        'quiet_mode': True,
        'use_simple_brain': True
    })
    
    brain_pool = BrainPool(brain_factory)
    adapter_factory = AdapterFactory()
    
    # Create brain service with persistence
    brain_service = BrainService(
        brain_pool, 
        adapter_factory,
        enable_persistence=True,
        memory_path=test_memory_path
    )
    
    connection_handler = ConnectionHandler(robot_registry, brain_service)
    
    print("   ✓ Components initialized with persistence")
    
    # Simulate robot connection and processing
    print("\n2. Simulating robot connection...")
    
    client_id = "test_persistence_robot"
    capabilities = [1.0, 16.0, 5.0, 1.0, 3.0]  # PiCar-X
    
    response = connection_handler.handle_handshake(client_id, capabilities)
    print(f"   ✓ Handshake complete: {response}")
    
    # Process enough cycles to trigger persistence
    print("\n3. Processing cycles to trigger persistence saves...")
    
    for i in range(250):  # Should trigger 2 saves (at 100 and 200)
        sensory_data = [0.5 + 0.1 * (i % 10)] * 16
        motor_response = connection_handler.handle_sensory_input(client_id, sensory_data)
        
        if i % 50 == 0:
            print(f"   Cycle {i}: Motor output = {motor_response[0]:.3f}")
    
    print("   ✓ Processed 250 cycles")
    
    # Check if persistence files were created
    print("\n4. Checking persistence files...")
    
    incremental_dir = Path(test_memory_path) / "incremental"
    if incremental_dir.exists():
        files = list(incremental_dir.glob("*.json.gz"))
        print(f"   ✓ Found {len(files)} incremental save files")
        for f in files[:3]:  # Show first 3
            print(f"     - {f.name}")
    else:
        print("   ❌ No incremental directory found")
    
    # Shutdown and save final state
    print("\n5. Shutting down and saving final state...")
    brain_service.shutdown()
    
    # Check final save
    if incremental_dir.exists():
        final_files = list(incremental_dir.glob("*.json.gz"))
        if len(final_files) > len(files):
            print(f"   ✓ Final save created ({len(final_files) - len(files)} new files)")
    
    # Cleanup
    print("\n6. Cleaning up test directory...")
    import shutil
    if os.path.exists(test_memory_path):
        shutil.rmtree(test_memory_path)
        print("   ✓ Test directory cleaned")
    
    print("\n✅ Persistence test complete!")


if __name__ == "__main__":
    test_persistence()