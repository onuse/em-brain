#!/usr/bin/env python3
"""
Debug why vision data isn't being sent.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("DEBUGGING VISION DATA ISSUE")
print("=" * 60)

# Test 1: Check which HAL is being used
print("\n1. Checking HAL import...")
try:
    from hardware.raw_robot_hat_hal import create_hal
    print("   ✅ Using raw_robot_hat_hal")
    hal_type = "raw_robot_hat"
except ImportError as e:
    print(f"   ❌ Can't import raw_robot_hat_hal: {e}")
    from hardware.bare_metal_hal import create_hal
    print("   ⚠️ Falling back to bare_metal_hal")
    hal_type = "bare_metal"

# Test 2: Create HAL and read sensors
print("\n2. Creating HAL and reading sensors...")
hal = create_hal()
print(f"   HAL type: {hal.__class__.__name__}")

raw_sensors = hal.read_raw_sensors()
print(f"   Raw sensor data fields:")
print(f"     - i2c_grayscale: {len(raw_sensors.i2c_grayscale)} values")
print(f"     - gpio_ultrasonic_us: {raw_sensors.gpio_ultrasonic_us}")
print(f"     - analog_battery_raw: {raw_sensors.analog_battery_raw}")

# Check if vision_data exists
if hasattr(raw_sensors, 'vision_data'):
    print(f"     - vision_data: {len(raw_sensors.vision_data)} values")
    if len(raw_sensors.vision_data) == 0:
        print("       ⚠️ vision_data is empty!")
else:
    print("     - vision_data: NOT PRESENT IN RAWSENSORDATA")

# Check if audio_features exists
if hasattr(raw_sensors, 'audio_features'):
    print(f"     - audio_features: {len(raw_sensors.audio_features)} values")
else:
    print("     - audio_features: NOT PRESENT IN RAWSENSORDATA")

# Test 3: Check brainstem conversion
print("\n3. Testing brainstem conversion...")
from brainstem.brainstem import Brainstem

brainstem = Brainstem(enable_brain=False)
brain_input = brainstem.sensors_to_brain_format(raw_sensors)
print(f"   Brain input: {len(brain_input)} values")

if len(brain_input) < 100:
    print(f"   ❌ PROBLEM: Only {len(brain_input)} values!")
    print("   First 10 values:", brain_input[:10] if len(brain_input) >= 10 else brain_input)
else:
    print(f"   ✅ Full sensory data")

# Test 4: Check vision singleton directly
print("\n4. Checking vision singleton...")
try:
    from src.hardware.vision_singleton import VisionSingleton
    vision = VisionSingleton.get_instance()
    if vision:
        print(f"   ✅ Vision singleton exists")
        if hasattr(vision, 'get_flattened_frame'):
            frame = vision.get_flattened_frame()
            print(f"   Vision frame: {len(frame)} pixels")
        else:
            print("   ❌ Vision has no get_flattened_frame method")
    else:
        print("   ❌ Vision singleton is None")
except Exception as e:
    print(f"   ❌ Vision singleton error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
if hal_type == "bare_metal":
    print("❌ Using bare_metal_hal which doesn't have vision/audio support!")
    print("   Fix: Ensure raw_robot_hat_hal can be imported")
elif len(brain_input) < 100:
    print("❌ Vision data not being collected by HAL")
    print("   Fix: Check _get_vision_data() method in HAL")
else:
    print("✅ System should be working")