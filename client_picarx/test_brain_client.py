#!/usr/bin/env python3
"""
Test brain client communication.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("TESTING BRAIN CLIENT")
print("=" * 60)

# Import brain client
from brainstem.brain_client import BrainClient, BrainServerConfig

# Create config
config = BrainServerConfig(
    host="localhost",
    port=9999,
    timeout=0.1,
    sensory_dimensions=307212,
    action_dimensions=6
)

print(f"\nConnecting to {config.host}:{config.port}")
print(f"Sensory dimensions: {config.sensory_dimensions}")
print(f"Action dimensions: {config.action_dimensions}")

# Create client
client = BrainClient(config)

# Try to connect
if client.connect():
    print("\n✅ Connected to brain server!")
    
    # Send test data
    print("\nSending test sensory data...")
    sensory_data = [0.5] * config.sensory_dimensions
    
    response = client.process_sensors(sensory_data)
    
    if response:
        print("✅ Got response from brain!")
        if 'motor_commands' in response:
            motor_commands = response['motor_commands']
            print(f"Motor commands: {[f'{x:.3f}' for x in motor_commands]}")
    else:
        print("❌ No response from brain")
    
    client.disconnect()
else:
    print("\n⚠️ Could not connect to brain server")
    print("\nTo test the complete system:")
    print("1. Start brain server: python3 server/brain.py --safe-mode")
    print("2. Run robot: python3 client_picarx/picarx_robot.py --brain-host localhost")

print("\n" + "=" * 60)