#!/usr/bin/env python3
"""
Test that error messages properly reach the terminal
"""

import sys
import os
import time
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server' / 'src'))
sys.path.insert(0, str(brain_root / 'server'))
sys.path.insert(0, str(brain_root / 'validation'))

from src.communication.client import MinimalBrainClient
from embodied_learning.environments.sensory_motor_world import SensoryMotorWorld

def test_error_display():
    """Test that brain errors are displayed to terminal."""
    print("🔍 Testing Error Display to Terminal")
    print("=" * 50)
    
    # Test 1: Generate errors by sending invalid input
    print("\n1. Testing error display with invalid input...")
    
    client = MinimalBrainClient()
    if client.connect():
        print("   ✅ Connected to brain server")
        
        # Test empty input (should trigger EMPTY_SENSORY_INPUT error)
        print("   🧪 Testing empty sensory input...")
        try:
            action = client.get_action([], timeout=3.0)
            if action is None:
                print("   ✅ Empty input handled (check [SERVER] messages above)")
            else:
                print(f"   ⚠️  Unexpected response: {action}")
        except Exception as e:
            print(f"   ⚠️  Exception: {e}")
        
        # Test oversized input (should trigger SENSORY_INPUT_TOO_LARGE error)
        print("   🧪 Testing oversized sensory input...")
        try:
            large_input = [0.5] * 50  # 50 dimensions, limit is 32
            action = client.get_action(large_input, timeout=3.0)
            if action is None:
                print("   ✅ Oversized input handled (check [SERVER] messages above)")
            else:
                print(f"   ⚠️  Unexpected response: {action}")
        except Exception as e:
            print(f"   ⚠️  Exception: {e}")
        
        # Test rapid-fire requests to potentially trigger memory pressure
        print("   🧪 Testing rapid requests (may trigger memory pressure)...")
        for i in range(10):
            try:
                action = client.get_action([0.5] * 16, timeout=1.0)
                if action is None:
                    print(f"   ⚠️  Request {i+1} failed (check [SERVER] messages above)")
                    break
            except Exception as e:
                print(f"   ⚠️  Request {i+1} exception: {e}")
                break
        
        client.disconnect()
    else:
        print("   ❌ Could not connect to brain server")
        print("   🔧 Make sure brain server is running")
        return False
    
    print("\n📋 What to Look For:")
    print("   • [SERVER] messages should appear above showing detailed error information")
    print("   • Error messages should include error codes (e.g., 3.0, 3.1)")
    print("   • Error messages should include descriptive names (e.g., EMPTY_SENSORY_INPUT)")
    print("   • Error messages should include resolution suggestions")
    print("   • Error messages should include context information")
    
    print("\n💡 Expected Error Messages:")
    print("   [SERVER] ❌ Client xxx: [ERROR] BrainError 3.0 (EMPTY_SENSORY_INPUT): ...")
    print("   [SERVER] ❌ Client xxx: [ERROR] BrainError 3.1 (SENSORY_INPUT_TOO_LARGE): ...")
    
    return True

if __name__ == "__main__":
    success = test_error_display()
    if success:
        print("\n✅ Error display test completed!")
        print("🔄 Check above for [SERVER] error messages")
    else:
        print("\n❌ Error display test failed!")