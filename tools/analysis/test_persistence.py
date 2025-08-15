#!/usr/bin/env python3
"""Test persistence system functionality."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import json
from brain_factory import BrainFactory

print("=== PERSISTENCE SYSTEM TEST ===\n")

# Create configuration with persistence enabled
config = {
    'brain': {
        'type': 'unified_field',
        'sensory_dim': 24,
        'motor_dim': 4
    },
    'memory': {
        'enable_persistence': True,
        'persistent_memory_path': './test_memory'
    }
}

print("1. Creating brain with persistence...")
try:
    brain = BrainFactory(config)
    print(f"✅ Brain created")
    print(f"   Persistence enabled: {brain.persistence_manager is not None}")
    
    if brain.persistence_manager:
        print(f"   Memory path: {brain.persistence_manager.config.memory_root_path}")
        print(f"   Recovery result: {brain.persistence_manager.startup_recovery_result}")
    
except Exception as e:
    print(f"❌ Error creating brain: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing save functionality...")
try:
    # Process some cycles
    for i in range(5):
        brain.process_sensory_input([0.5] * 24)
    
    # Get state
    state = brain.get_brain_state_for_persistence()
    print(f"✅ State retrieved:")
    print(f"   Brain cycles: {state['brain_cycles']}")
    print(f"   Field dimensions: {state['field_dimensions']}")
    
    # Check if persistence manager is saving
    if brain.persistence_manager:
        stats = brain.persistence_manager.manager_stats
        print(f"\n3. Persistence stats:")
        print(f"   Total saves requested: {stats['total_saves_requested']}")
        print(f"   Total cycles processed: {stats['total_cycles_processed']}")
        
except Exception as e:
    print(f"❌ Error during operation: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
if brain:
    brain.shutdown()
    print("\n✅ Shutdown complete")