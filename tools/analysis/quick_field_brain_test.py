#!/usr/bin/env python3
"""
Quick Field Brain Configuration Test

Simple test to demonstrate field brain configuration and switching.
"""

import sys
import os
import json
from pathlib import Path

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

def test_field_brain_config():
    """Test field brain configuration loading."""
    print("🧠 Field Brain Configuration Test")
    print("=" * 40)
    
    try:
        from src.brain import MinimalBrain
        
        # Test 1: Sparse Goldilocks configuration
        print("\n1. Testing Sparse Goldilocks Brain:")
        sparse_config = {
            "brain": {
                "type": "sparse_goldilocks",
                "sensory_dim": 16,
                "motor_dim": 4
            }
        }
        
        sparse_brain = MinimalBrain(config=sparse_config, quiet_mode=True)
        print(f"   ✅ Type: {type(sparse_brain.vector_brain).__name__}")
        print(f"   ✅ Architecture: {sparse_brain.brain_type}")
        sparse_brain.finalize_session()
        
        # Test 2: Field Brain configuration
        print("\n2. Testing Field Brain:")
        field_config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4,
                "field_spatial_resolution": 20,
                "field_temporal_window": 10.0,
                "field_evolution_rate": 0.1,
                "constraint_discovery_rate": 0.15
            }
        }
        
        field_brain = MinimalBrain(config=field_config, quiet_mode=True)
        print(f"   ✅ Type: {type(field_brain.vector_brain).__name__}")
        print(f"   ✅ Architecture: {field_brain.brain_type}")
        
        # Quick processing test
        sensory_input = [0.1] * 16
        action, state = field_brain.process_sensory_input(sensory_input)
        print(f"   ✅ Processing: {len(sensory_input)}D → {len(action)}D")
        print(f"   ✅ Confidence: {state.get('prediction_confidence', 0.0):.3f}")
        
        field_brain.finalize_session()
        
        print("\n🎯 Configuration switching successful!")
        print("   Both brain types can be instantiated correctly")
        
        # Show how to switch
        print("\n📋 To switch to field brain in production:")
        print("   1. Update settings.json:")
        print('      "brain": { "type": "field", ... }')
        print("   2. Restart brain_server.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_field_brain_config()
    if success:
        print("\n✅ Field brain integration ready!")
    else:
        print("\n❌ Field brain integration needs work")
    sys.exit(0 if success else 1)