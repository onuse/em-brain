#!/usr/bin/env python3
"""
Test Robot Client Connection
=============================
Quick test to verify the robot client can import and connect to brain server.
"""

import sys
import os

# Add client path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'client_picarx'))

print("Testing robot client imports...")

try:
    # Test the imports
    print("1. Importing picarx_robot...")
    from client_picarx import picarx_robot
    print("   ✓ picarx_robot imported")
    
    print("2. Importing brainstem components...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'client_picarx', 'src'))
    from brainstem.integrated_brainstem import IntegratedBrainstem, BrainstemConfig
    from brainstem.brain_client import BrainServerConfig
    print("   ✓ Brainstem components imported")
    
    print("3. Creating config...")
    config = BrainServerConfig(
        host="localhost",
        port=9999,
        timeout=5.0
    )
    print(f"   ✓ Config created: {config.host}:{config.port}")
    
    print("\n✅ All imports successful!")
    print("\nTo run the robot client:")
    print("  cd client_picarx")
    print("  python3 picarx_robot.py --brain-host localhost")
    
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    print("\nDebug info:")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  Python path: {sys.path}")
    
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")