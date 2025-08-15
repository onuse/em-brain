#!/usr/bin/env python3
"""
Diagnose why robot is only sending 5 values to brain.
Run this ON THE ROBOT to see what's happening.
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("ROBOT DIAGNOSTIC")
print("=" * 60)

# Test 1: Import checks
print("\n1. Import checks...")
errors = []

try:
    from hardware.raw_robot_hat_hal import RawRobotHatHAL
    print("   ✅ raw_robot_hat_hal imports")
except Exception as e:
    print(f"   ❌ raw_robot_hat_hal import failed: {e}")
    errors.append("raw_robot_hat_hal")

try:
    from hardware.vision_singleton import VisionSingleton
    print("   ✅ vision_singleton imports")
except Exception as e:
    print(f"   ❌ vision_singleton import failed: {e}")
    errors.append("vision_singleton")

try:
    from hardware.configurable_vision import ConfigurableVision
    print("   ✅ configurable_vision imports")
except Exception as e:
    print(f"   ❌ configurable_vision import failed: {e}")
    errors.append("configurable_vision")

# Test 2: HAL creation with debug
print("\n2. Creating HAL with debug enabled...")
from hardware.raw_robot_hat_hal import RawRobotHatHAL

hal = RawRobotHatHAL(debug=True)
print(f"   HAL created: {hal.__class__.__name__}")
print(f"   Mock mode: {hal.mock_mode}")

# Test 3: Read sensors and check data
print("\n3. Reading sensors...")
sensors = hal.read_raw_sensors()

print(f"   Basic sensors:")
print(f"     - Grayscale: {sensors.i2c_grayscale}")
print(f"     - Distance: {sensors.gpio_ultrasonic_us:.1f} µs")
print(f"     - Battery: {sensors.analog_battery_raw}")

print(f"   Vision/Audio:")
print(f"     - vision_data length: {len(sensors.vision_data)}")
print(f"     - audio_features length: {len(sensors.audio_features)}")

if len(sensors.vision_data) == 0:
    print("     ❌ VISION DATA IS EMPTY!")
else:
    print(f"     ✅ Vision has {len(sensors.vision_data)} pixels")

# Test 4: Brainstem conversion
print("\n4. Testing brainstem conversion...")
from brainstem.brainstem import Brainstem

brainstem = Brainstem(enable_brain=False)
brain_input = brainstem.sensors_to_brain_format(sensors)

print(f"   Total brain input: {len(brain_input)} values")
if len(brain_input) == 307212:
    print("   ✅ Correct number of values!")
else:
    print(f"   ❌ Wrong number! Expected 307212, got {len(brain_input)}")
    print(f"   Breakdown:")
    print(f"     - First 5 (basic sensors): {brain_input[:5] if len(brain_input) >= 5 else brain_input}")
    if len(brain_input) > 5:
        print(f"     - Remaining: {len(brain_input) - 5} values")

# Test 5: Direct vision test
print("\n5. Direct vision singleton test...")
try:
    from hardware.vision_singleton import VisionSingleton
    vision = VisionSingleton.get_instance()
    if vision:
        print(f"   Vision singleton created: {vision.__class__.__name__}")
        if hasattr(vision, 'enabled'):
            print(f"   Vision enabled: {vision.enabled}")
        if hasattr(vision, 'get_flattened_frame'):
            frame = vision.get_flattened_frame()
            print(f"   Direct frame call: {len(frame)} pixels")
        else:
            print("   ❌ No get_flattened_frame method!")
    else:
        print("   ❌ Vision singleton is None")
except Exception as e:
    print(f"   ❌ Vision test failed: {e}")

# Summary
print("\n" + "=" * 60)
print("DIAGNOSIS:")
if len(brain_input) < 100:
    print("❌ PROBLEM CONFIRMED: Only sending basic sensor data!")
    print("\nPossible causes:")
    if "vision_singleton" in errors:
        print("  - vision_singleton module can't be imported")
    if "configurable_vision" in errors:
        print("  - configurable_vision module can't be imported")
    if len(sensors.vision_data) == 0:
        print("  - HAL's _get_vision_data() returning empty list")
    print("\nCheck:")
    print("  1. picamera2 is installed")
    print("  2. Camera is connected")
    print("  3. Import paths are correct")
else:
    print("✅ System appears to be working correctly")
    print(f"   Sending {len(brain_input)} values as expected")