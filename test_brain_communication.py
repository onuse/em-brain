#!/usr/bin/env python3
"""
Test brain-robot communication directly.
"""

import sys
import os
import time
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'client_picarx/src'))

print("=" * 60)
print("TESTING BRAIN-ROBOT COMMUNICATION")
print("=" * 60)

# Test 1: Create brain client
print("\n1. Creating brain client...")
from communication.client import BrainClient, BrainServerConfig

config = BrainServerConfig(
    host="localhost",
    port=9999,
    timeout=0.1,
    sensory_dimensions=307212,
    action_dimensions=6
)

client = BrainClient(config)
print(f"   Client created for {config.host}:{config.port}")
print(f"   Sensory: {config.sensory_dimensions} dims")
print(f"   Action: {config.action_dimensions} dims")

# Test 2: Try to connect
print("\n2. Attempting connection...")
if client.connect():
    print("   ✅ Connected to brain server!")
    
    # Test 3: Send sensory data
    print("\n3. Sending test sensory data...")
    
    # Create test sensory data (307,212 values)
    sensory_data = []
    sensory_data.extend([0.5, 0.5, 0.5])  # Grayscale
    sensory_data.append(0.5)  # Ultrasonic
    sensory_data.append(0.7)  # Battery
    sensory_data.extend([0.5] * 307200)  # Vision
    sensory_data.extend([0.0] * 7)  # Audio
    
    print(f"   Sending {len(sensory_data)} sensory values...")
    
    # Send and get response
    response = client.process_sensors(sensory_data)
    
    if response:
        print("   ✅ Got response from brain!")
        if 'motor_commands' in response:
            motor_commands = response['motor_commands']
            print(f"   Motor commands: {motor_commands}")
            print(f"     - Thrust: {motor_commands[0]:.3f}")
            print(f"     - Turn: {motor_commands[1]:.3f}")
            print(f"     - Steering: {motor_commands[2]:.3f}")
            print(f"     - Camera: {motor_commands[3]:.3f}")
            print(f"     - Audio freq: {motor_commands[4]:.3f}")
            print(f"     - Audio vol: {motor_commands[5]:.3f}")
    else:
        print("   ❌ No response from brain")
    
    # Disconnect
    client.disconnect()
    print("\n✅ Communication test successful!")
    
else:
    print("   ⚠️ Could not connect - brain server not running")
    print("\n   To start brain server:")
    print("     python3 server/brain.py --safe-mode")
    print("\n   Then run this test again")

print("\n" + "=" * 60)