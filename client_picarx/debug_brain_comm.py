#!/usr/bin/env python3
"""
Debug brain communication - test what's actually being sent to the brain.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("DEBUG BRAIN COMMUNICATION")
print("=" * 60)

# Step 1: Create real sensor data
print("\n1. Creating sensor data...")
from hardware.raw_robot_hat_hal import create_hal

hal = create_hal()
sensors = hal.read_raw_sensors()
print(f"   Raw sensors collected")
print(f"   Vision: {len(sensors.vision_data)} values")
print(f"   Audio: {len(sensors.audio_features)} values")

# Step 2: Convert to brain format
print("\n2. Converting to brain format...")
from brainstem.brainstem import Brainstem

brainstem = Brainstem(enable_brain=False, enable_monitor=False)
brain_input = brainstem.sensors_to_brain_format(sensors)
print(f"   Brain input created: {len(brain_input)} values")

# Step 3: Test brain client directly
print("\n3. Testing brain client...")
from brainstem.brain_client import BrainClient, BrainServerConfig

config = BrainServerConfig(
    host="192.168.1.231",  # Your brain server
    port=9999,
    timeout=1.0,
    sensory_dimensions=307212,
    action_dimensions=6
)

print(f"   Configured for {config.host}:{config.port}")
print(f"   Expected sensory: {config.sensory_dimensions}")
print(f"   Actual sensory: {len(brain_input)}")

client = BrainClient(config)

# Step 4: Try to connect and send
print("\n4. Attempting connection...")
if client.connect():
    print("   ✅ Connected to brain server!")
    
    print("\n5. Sending sensor data...")
    print(f"   Sending {len(brain_input)} values...")
    
    # Debug what we're actually sending
    print(f"   First 5 values: {brain_input[:5]}")
    print(f"   Last 5 values: {brain_input[-5:]}")
    
    # Send the data
    try:
        response = client.process_sensors(brain_input)
        
        if response:
            print("   ✅ Got response from brain!")
            if 'motor_commands' in response:
                print(f"   Motor commands: {response['motor_commands']}")
            if 'error' in response:
                print(f"   ❌ Brain error: {response['error']}")
        else:
            print("   ❌ No response from brain")
            
    except Exception as e:
        print(f"   ❌ Send failed: {e}")
        import traceback
        traceback.print_exc()
    
    client.disconnect()
    print("\n6. Disconnected")
    
else:
    print("   ❌ Could not connect to brain server")
    print("   Check that brain server is running at 192.168.1.231:9999")

# Step 7: Test with minimal data to see if that works
print("\n7. Testing with minimal data (5 values)...")
client2 = BrainClient(BrainServerConfig(
    host="192.168.1.231",
    port=9999,
    timeout=1.0,
    sensory_dimensions=5,  # Only 5 values
    action_dimensions=6
))

if client2.connect():
    print("   Connected with 5-value config")
    minimal_data = [0.5, 0.5, 0.5, 0.5, 0.7]  # Just 5 values
    response = client2.process_sensors(minimal_data)
    if response:
        print(f"   ✅ Minimal data worked: {response}")
    else:
        print("   ❌ Minimal data failed")
    client2.disconnect()

print("\n" + "=" * 60)
print("Please copy this entire output!")
print("=" * 60)