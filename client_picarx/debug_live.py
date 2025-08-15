#!/usr/bin/env python3
"""
Live debugging script - run this on the robot to see what's happening.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("LIVE DEBUG - VISION DATA ISSUE")
print("=" * 60)

# Step 1: Check which HAL is actually being imported
print("\n1. HAL Import Test:")
try:
    # This is what brainstem.py does
    try:
        from hardware.raw_robot_hat_hal import (
            RawRobotHatHAL, create_hal,
            RawMotorCommand, RawServoCommand, RawSensorData
        )
        print("   ✅ Using raw_robot_hat_hal")
        hal_type = "raw_robot_hat"
    except ImportError as e:
        print(f"   ❌ raw_robot_hat_hal failed: {e}")
        from hardware.bare_metal_hal import (
            BareMetalHAL, create_hal,
            RawMotorCommand, RawServoCommand, RawSensorData
        )
        print("   ⚠️ Fallback to bare_metal_hal")
        hal_type = "bare_metal"
except Exception as e:
    print(f"   ❌ TOTAL FAILURE: {e}")
    sys.exit(1)

# Step 2: Create HAL and check what class it is
print("\n2. HAL Creation:")
hal = create_hal()
print(f"   HAL class: {hal.__class__.__name__}")
print(f"   HAL module: {hal.__class__.__module__}")

# Step 3: Check if vision/audio methods exist
print("\n3. Method Check:")
print(f"   Has _get_vision_data: {hasattr(hal, '_get_vision_data')}")
print(f"   Has _get_audio_features: {hasattr(hal, '_get_audio_features')}")

# Step 4: Read raw sensors
print("\n4. Raw Sensor Read:")
try:
    sensors = hal.read_raw_sensors()
    print(f"   ✅ Sensor read successful")
    
    # Check what fields exist
    print(f"   Has vision_data field: {hasattr(sensors, 'vision_data')}")
    print(f"   Has audio_features field: {hasattr(sensors, 'audio_features')}")
    
    if hasattr(sensors, 'vision_data'):
        print(f"   vision_data length: {len(sensors.vision_data)}")
        if len(sensors.vision_data) == 0:
            print("      ❌ VISION DATA IS EMPTY!")
    else:
        print("      ❌ NO VISION_DATA FIELD!")
        
    if hasattr(sensors, 'audio_features'):
        print(f"   audio_features length: {len(sensors.audio_features)}")
    else:
        print("      ❌ NO AUDIO_FEATURES FIELD!")
        
except Exception as e:
    print(f"   ❌ Sensor read failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Test brainstem conversion
print("\n5. Brainstem Conversion:")
try:
    from brainstem.brainstem import Brainstem
    brainstem = Brainstem(enable_brain=False, enable_monitor=False)
    brain_input = brainstem.sensors_to_brain_format(sensors)
    print(f"   Total brain input: {len(brain_input)} values")
    
    if len(brain_input) < 100:
        print(f"   ❌ ONLY {len(brain_input)} VALUES!")
        print(f"   Content: {brain_input}")
    else:
        print(f"   ✅ Expected size")
        
except Exception as e:
    print(f"   ❌ Brainstem conversion failed: {e}")
    import traceback
    traceback.print_exc()

# Step 6: Direct vision test
print("\n6. Direct Vision Test:")
try:
    # Try to manually call _get_vision_data if it exists
    if hasattr(hal, '_get_vision_data'):
        vision_data = hal._get_vision_data()
        print(f"   Direct _get_vision_data(): {len(vision_data)} values")
    else:
        print("   ❌ HAL has no _get_vision_data method")
except Exception as e:
    print(f"   ❌ Direct vision call failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
if hal_type == "bare_metal":
    print("❌ Using bare_metal_hal - this doesn't have vision support!")
elif not hasattr(sensors, 'vision_data'):
    print("❌ RawSensorData doesn't have vision_data field!")
elif len(sensors.vision_data) == 0:
    print("❌ vision_data field exists but is empty!")
else:
    print("🤔 Everything looks correct here...")

print("\nPlease copy and paste this entire output.")
print("=" * 60)