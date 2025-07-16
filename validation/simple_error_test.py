#!/usr/bin/env python3
"""Simple test to trigger and investigate error 5.0"""

import sys
from pathlib import Path

brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.communication.client import MinimalBrainClient

client = MinimalBrainClient()

if client.connect():
    print("Connected to brain server")
    
    # Try with very short timeout to trigger error
    try:
        print("Testing with 0.05s timeout...")
        action = client.get_action([0.5] * 16, timeout=0.05)
        print(f"Success: {action}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        
    # Try normal request
    try:
        print("\nTesting with normal timeout...")
        action = client.get_action([0.5] * 16, timeout=5.0)
        print(f"Success: {action}")
    except Exception as e:
        print(f"Error: {e}")
        
    client.disconnect()
else:
    print("Failed to connect")