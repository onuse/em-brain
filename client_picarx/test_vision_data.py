#!/usr/bin/env python3
"""
Test vision data collection.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Testing vision data collection...\n")

try:
    from src.hardware.raw_robot_hat_hal import create_hal
    
    # Create HAL
    hal = create_hal()
    
    if hal.mock_mode:
        print("⚠️ Running in mock mode")
    
    # Read sensors
    print("Reading sensors...")
    sensors = hal.read_raw_sensors()
    
    print(f"\nSensor data:")
    print(f"  Grayscale: {sensors.i2c_grayscale}")
    print(f"  Ultrasonic: {sensors.gpio_ultrasonic_us:.1f}µs")
    print(f"  Battery: {sensors.analog_battery_raw}")
    print(f"  Vision data length: {len(sensors.vision_data)}")
    print(f"  Audio features length: {len(sensors.audio_features)}")
    
    # Check if vision data is actual data or just placeholder
    if len(sensors.vision_data) > 0:
        # Check if all values are the same (indicates placeholder)
        first_val = sensors.vision_data[0]
        all_same = all(v == first_val for v in sensors.vision_data[:100])
        if all_same:
            print(f"  ⚠️ Vision data is placeholder (all values = {first_val})")
        else:
            print(f"  ✅ Vision data appears to be real")
            print(f"     Sample values: {sensors.vision_data[:5]}")
    
    # Convert to brain format
    from src.brainstem.brainstem import Brainstem
    
    # Create temporary brainstem just for conversion
    brainstem = Brainstem(enable_brain=False, enable_monitor=False)
    brain_input = brainstem.sensors_to_brain_format(sensors)
    
    print(f"\nBrain input:")
    print(f"  Total length: {len(brain_input)}")
    print(f"  Expected: 307,212")
    print(f"  Match: {'YES' if len(brain_input) == 307212 else 'NO'}")
    
    if len(brain_input) < 100:
        print(f"  ❌ Only basic sensors! Data: {brain_input}")
    else:
        print(f"  ✅ Full sensor array including vision")
    
    # Cleanup
    hal.cleanup()
    brainstem.shutdown()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()