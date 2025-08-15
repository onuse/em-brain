#!/usr/bin/env python3
"""
Test brain server startup and basic functionality
"""

import sys
import os
import time
import threading

# Add server directory to path
# From tests/ we need to go up one level to brain/, then into server/
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server_dir = os.path.join(brain_root, 'server')
sys.path.insert(0, server_dir)

def test_brain_server_import():
    """Test if brain server can be imported."""
    print("🧪 Testing brain server import...")
    
    try:
        from brain_server import MinimalBrainServer
        print("✅ Brain server import successful")
        return True
    except Exception as e:
        print(f"❌ Brain server import failed: {e}")
        return False

def test_brain_client_import():
    """Test if brain client can be imported."""
    print("🧪 Testing brain client import...")
    
    try:
        from src.communication import MinimalBrainClient
        print("✅ Brain client import successful")
        return True
    except Exception as e:
        print(f"❌ Brain client import failed: {e}")
        return False

def test_server_creation():
    """Test creating brain server (without starting)."""
    print("🧪 Testing brain server creation...")
    
    try:
        from brain_server import MinimalBrainServer
        server = MinimalBrainServer(host='localhost', port=9999)
        print(f"✅ Brain server created successfully on {server.host}:{server.port}")
        return True
    except Exception as e:
        print(f"❌ Brain server creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_client_creation():
    """Test creating brain client (without connecting)."""
    print("🧪 Testing brain client creation...")
    
    try:
        from src.communication import MinimalBrainClient
        client = MinimalBrainClient()
        print("✅ Brain client created successfully")
        return True
    except Exception as e:
        print(f"❌ Brain client creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run brain server tests."""
    print("🧠 Brain Server Test")
    print("=" * 40)
    
    tests = [
        test_brain_server_import,
        test_brain_client_import,
        test_server_creation,
        test_client_creation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            print("❌ Test failed - stopping")
            break
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n✅ Brain server components working!")
        print("\nTo manually test:")
        print("1. Terminal 1: cd server && python3 brain_server.py")
        print("2. Terminal 2: cd server && python3 -c \"from src.communication import MinimalBrainClient; c = MinimalBrainClient(); c.connect(); print('Connected!')\"")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    main()