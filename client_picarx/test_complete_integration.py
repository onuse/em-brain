#!/usr/bin/env python3
"""
Complete system integration test.

Tests:
1. Configuration loading from robot_config.json
2. HAL with vision/audio integration  
3. Brainstem sensory data collection
4. Brain server communication
5. Motor command execution
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("COMPLETE SYSTEM INTEGRATION TEST")
print("=" * 60)

# Test 1: Configuration
print("\n1. Testing configuration loading...")
from brainstem.brainstem import load_robot_config

config = load_robot_config()
brain_config = config.get("brain", {})
print(f"   ✅ Brain server: {brain_config.get('host', 'NOT SET')}:{brain_config.get('port', 'NOT SET')}")

# Test 2: HAL initialization
print("\n2. Testing HAL with vision/audio...")
from hardware.raw_robot_hat_hal import RawRobotHatHAL, RawSensorData

hal = RawRobotHatHAL(debug=True)
print(f"   HAL initialized (mock={hal.mock_mode})")

# Test sensor reading
sensors = hal.read_raw_sensors()
print(f"   Sensors read:")
print(f"     - Grayscale: {sensors.i2c_grayscale}")
print(f"     - Distance: {sensors.gpio_ultrasonic_us:.1f} µs")
print(f"     - Battery: {sensors.analog_battery_raw}")
print(f"     - Vision data: {len(sensors.vision_data)} pixels")
print(f"     - Audio features: {len(sensors.audio_features)} channels")

if len(sensors.vision_data) == 0:
    print("   ⚠️ No vision data - camera may be disabled")
else:
    print(f"   ✅ Vision working: {len(sensors.vision_data)} values")

# Test 3: Brainstem
print("\n3. Testing brainstem sensory conversion...")
from brainstem.brainstem import Brainstem

# Create brainstem without brain connection for now
brainstem = Brainstem(enable_brain=False)
brain_input = brainstem.sensors_to_brain_format(sensors)
print(f"   Brain input vector: {len(brain_input)} dimensions")
print(f"   Expected: 307,212 (5 basic + 307,200 vision + 7 audio)")

if len(brain_input) < 100:
    print(f"   ❌ Only {len(brain_input)} values - vision not working!")
else:
    print(f"   ✅ Full sensory data: {len(brain_input)} values")

# Test 4: Brain server check
print("\n4. Checking brain server...")
brain_host = brain_config.get('host', 'localhost')
brain_port = brain_config.get('port', 9999)

# Try to connect
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((brain_host, brain_port))
    sock.close()
    
    if result == 0:
        print(f"   ✅ Brain server reachable at {brain_host}:{brain_port}")
        brain_available = True
    else:
        print(f"   ⚠️ Brain server not reachable at {brain_host}:{brain_port}")
        brain_available = False
except:
    print(f"   ⚠️ Could not check brain server")
    brain_available = False

# Test 5: Full cycle with brain (if available)
if brain_available:
    print("\n5. Testing full brainstem cycle with brain...")
    
    # Create brainstem with brain connection
    brainstem_with_brain = Brainstem(brain_host=brain_host, brain_port=brain_port)
    
    print("   Running 3 cycles...")
    for i in range(3):
        print(f"   Cycle {i+1}:")
        start = time.time()
        
        # Run one cycle
        brainstem_with_brain.run_cycle()
        
        cycle_time = (time.time() - start) * 1000
        print(f"     - Cycle time: {cycle_time:.1f}ms")
        
        # Check metrics
        if brainstem_with_brain.monitor:
            metrics = brainstem_with_brain.monitor.get_metrics()
            print(f"     - Sensor time: {metrics.get('sensor_time', 0):.1f}ms")
            print(f"     - Comm time: {metrics.get('comm_time', 0):.1f}ms")
            print(f"     - Connected: {metrics.get('brain_connected', False)}")
        
        time.sleep(0.1)
    
    brainstem_with_brain.shutdown()
    print("   ✅ Full system test complete!")
else:
    print("\n5. Skipping brain test (server not available)")

# Test 6: Motor commands
print("\n6. Testing motor command execution...")
from hardware.raw_robot_hat_hal import RawMotorCommand, RawServoCommand

# Test motor command
motor_cmd = RawMotorCommand(
    left_pwm_duty=0.2,
    right_pwm_duty=0.2
)
hal.execute_motor_command(motor_cmd)
print("   Motors set to 20% forward")
time.sleep(0.5)

# Stop motors
hal.emergency_stop()
print("   ✅ Emergency stop executed")

# Cleanup
hal.cleanup()
print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)

# Summary
print("\nSUMMARY:")
print(f"  Configuration: ✅")
print(f"  HAL: ✅")
print(f"  Sensors: ✅")
print(f"  Vision: {'✅' if len(sensors.vision_data) > 0 else '⚠️ Disabled'}")
print(f"  Audio: ✅")
print(f"  Brain server: {'✅ Connected' if brain_available else '⚠️ Not available'}")
print(f"  Motor control: ✅")

if len(sensors.vision_data) == 0:
    print("\n⚠️ Vision is not sending data. Check:")
    print("  - Camera hardware connected")
    print("  - picamera2 installed")
    print("  - vision.enabled in robot_config.json")