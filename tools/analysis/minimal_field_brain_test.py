#!/usr/bin/env python3
"""
Minimal Field Brain Test - Find the Hang
"""

import sys
import os
import time
from pathlib import Path

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

def test_brain_instantiation():
    """Test just brain creation to see if that's where it hangs."""
    print("🧠 Testing brain instantiation...")
    
    try:
        import json
        
        # Minimal config
        config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4
            },
            "memory": {"enable_persistence": False}  # Disable persistence to avoid complexity
        }
        
        print("   Loading MinimalBrain class...")
        from src.brain import MinimalBrain
        
        print("   Creating brain instance...")
        start_time = time.time()
        brain = MinimalBrain(config=config, quiet_mode=True)
        creation_time = time.time() - start_time
        
        print(f"   ✅ Brain created in {creation_time:.3f}s")
        print(f"   Type: {type(brain.vector_brain).__name__}")
        
        # Quick cleanup
        brain.finalize_session()
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_processing():
    """Test a single processing cycle."""
    print("\n🔄 Testing single processing cycle...")
    
    try:
        import json
        from src.brain import MinimalBrain
        
        config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4
            },
            "memory": {"enable_persistence": False}
        }
        
        print("   Creating brain (with timeout protection)...")
        brain = MinimalBrain(config=config, quiet_mode=True)
        
        print("   Testing single sensory input...")
        sensory_input = [0.1] * 16
        
        start_time = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        processing_time = time.time() - start_time
        
        print(f"   ✅ Processing completed in {processing_time:.3f}s")
        print(f"   Output: {len(action)}D action vector")
        
        brain.finalize_session()
        return True
        
    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sparse_brain_comparison():
    """Test sparse brain to make sure it's not a general issue."""
    print("\n🧪 Testing sparse brain for comparison...")
    
    try:
        import json
        from src.brain import MinimalBrain
        
        config = {
            "brain": {
                "type": "sparse_goldilocks",
                "sensory_dim": 16,
                "motor_dim": 4
            },
            "memory": {"enable_persistence": False}
        }
        
        print("   Creating sparse brain...")
        start_time = time.time()
        brain = MinimalBrain(config=config, quiet_mode=True)
        creation_time = time.time() - start_time
        
        print(f"   ✅ Sparse brain created in {creation_time:.3f}s")
        
        print("   Testing sparse brain processing...")
        sensory_input = [0.1] * 16
        
        start_time = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        processing_time = time.time() - start_time
        
        print(f"   ✅ Sparse processing in {processing_time:.3f}s")
        
        brain.finalize_session()
        return True
        
    except Exception as e:
        print(f"   ❌ Sparse brain failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 MINIMAL FIELD BRAIN DEBUG")
    print("=" * 40)
    
    tests = [
        test_brain_instantiation,
        test_single_processing,
        test_sparse_brain_comparison
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}]", end=" ")
        try:
            success = test()
            if not success:
                print(f"\n❌ Test {i} failed - stopping here")
                break
        except Exception as e:
            print(f"\n💥 Test {i} crashed: {e}")
            break
    else:
        print(f"\n✅ All tests completed!")