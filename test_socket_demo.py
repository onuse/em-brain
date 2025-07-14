#!/usr/bin/env python3
"""
Quick test for socket-based embodied brainstem
"""

import sys
import os
import time

# Add the brain/ directory to sys.path
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from demos.picar_x_simulation.socket_embodied_brainstem import SocketEmbodiedBrainstem
        print("✅ Socket embodied brainstem import successful")
    except Exception as e:
        print(f"❌ Socket embodied brainstem import failed: {e}")
        return False
    
    try:
        from server.brain_server import MinimalBrainServer
        print("✅ Brain server import successful")
    except Exception as e:
        print(f"❌ Brain server import failed: {e}")
        return False
    
    return True

def test_socket_brainstem_creation():
    """Test creating a socket-based embodied brainstem."""
    print("\n🧪 Testing socket brainstem creation...")
    
    try:
        from demos.picar_x_simulation.socket_embodied_brainstem import SocketEmbodiedBrainstem
        brainstem = SocketEmbodiedBrainstem()
        print("✅ Socket embodied brainstem created successfully")
        print(f"   Brain server target: {brainstem.brain_host}:{brainstem.brain_port}")
        print(f"   Connected: {brainstem.connected}")
        return True
    except Exception as e:
        print(f"❌ Socket embodied brainstem creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_mode():
    """Test brainstem fallback mode (without brain server)."""
    print("\n🧪 Testing fallback mode (no brain server)...")
    
    try:
        from demos.picar_x_simulation.socket_embodied_brainstem import SocketEmbodiedBrainstem
        brainstem = SocketEmbodiedBrainstem()
        
        # Try a control cycle without connecting to brain server
        result = brainstem.control_cycle()
        
        if result.get('success') == False and result.get('fallback_mode') == True:
            print("✅ Fallback mode working correctly")
            return True
        else:
            print(f"❌ Unexpected fallback result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Fallback mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔌 Socket-Based Embodied Brainstem Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_socket_brainstem_creation,
        test_fallback_mode
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            print("❌ Test failed - stopping")
            break
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n✅ All tests passed! Socket-based embodied brainstem is working.")
        print("\nTo test with brain server:")
        print("1. Terminal 1: python3 server/brain_server.py")
        print("2. Terminal 2: python3 demo_runner.py 3d_embodied")
        print("3. Press 'C' in demo to connect to brain server")
    else:
        print("\n❌ Some tests failed. Check error messages above.")

if __name__ == "__main__":
    main()