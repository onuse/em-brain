#!/usr/bin/env python3
"""
Test that dimension mismatch is fixed.
Verifies brainstem sends correct 3,084 values (not 307,212).
"""

import sys
import numpy as np

# Add paths
sys.path.insert(0, 'client_picarx/src')

from hardware.bare_metal_hal import RawSensorData


def test_dimension_calculation():
    """Test that brainstem calculates correct dimensions."""
    print("=" * 60)
    print("DIMENSION FIX TEST")
    print("=" * 60)
    
    # Expected dimensions
    basic_sensors = 5       # grayscale(3) + ultrasonic + battery
    vision_pixels = 3072    # 64x48 from config
    audio_features = 7      # audio channels
    
    expected_total = basic_sensors + vision_pixels + audio_features
    
    print(f"\nExpected dimensions:")
    print(f"  Basic sensors: {basic_sensors}")
    print(f"  Vision pixels: {vision_pixels:,} (64x48)")
    print(f"  Audio features: {audio_features}")
    print(f"  TOTAL: {expected_total:,}")
    
    # Create mock sensor data with correct dimensions
    raw = RawSensorData(
        i2c_grayscale=[1000, 2000, 3000],
        gpio_ultrasonic_us=5800,  # 100cm
        analog_battery_raw=3000,  # ~7.3V
        motor_current_raw=[0, 0],
        vision_data=[0.5] * vision_pixels,  # Correct size!
        audio_features=[0.0] * audio_features,
        timestamp_ns=0
    )
    
    print(f"\nMock sensor data created:")
    print(f"  Vision data length: {len(raw.vision_data):,}")
    print(f"  Audio features length: {len(raw.audio_features)}")
    
    # Import brainstem and test conversion
    from brainstem.brainstem import Brainstem
    
    # Create brainstem instance (without HAL)
    brainstem = Brainstem(hal=None)
    
    # Convert to brain format
    brain_input = brainstem.sensors_to_brain_format(raw)
    
    print(f"\nBrain input vector:")
    print(f"  Length: {len(brain_input):,}")
    print(f"  Expected: {expected_total:,}")
    
    # Verify structure
    print(f"\nVector structure verification:")
    print(f"  [0-2]: Grayscale = {brain_input[0:3]}")
    print(f"  [3]: Ultrasonic = {brain_input[3]:.3f}")
    print(f"  [4]: Battery = {brain_input[4]:.3f}")
    print(f"  [5-3076]: Vision (first 5) = {brain_input[5:10]}")
    print(f"  [3077-3083]: Audio = {brain_input[3077:3084]}")
    
    # Check size
    if len(brain_input) == expected_total:
        print(f"\n✅ SUCCESS! Dimension mismatch fixed!")
        print(f"   Brainstem sends {len(brain_input):,} values (not 307,212)")
        print(f"   This fits in TCP buffers without fragmentation")
        return True
    else:
        print(f"\n❌ FAILED! Wrong dimension: {len(brain_input):,}")
        return False


def test_message_size():
    """Calculate actual message sizes."""
    print("\n" + "=" * 60)
    print("MESSAGE SIZE ANALYSIS")
    print("=" * 60)
    
    # Calculate sizes
    values = 3084
    float32_bytes = values * 4
    
    # TCP message structure
    header_bytes = 9  # Type(1) + length(4) + CRC(4)
    total_bytes = header_bytes + float32_bytes
    
    print(f"\nTCP message size:")
    print(f"  Values: {values:,}")
    print(f"  Float32 data: {float32_bytes:,} bytes ({float32_bytes/1024:.1f}KB)")
    print(f"  Header: {header_bytes} bytes")
    print(f"  TOTAL: {total_bytes:,} bytes ({total_bytes/1024:.1f}KB)")
    
    # Compare to buffer sizes
    tcp_buffer_default = 16 * 1024  # 16KB default
    tcp_buffer_max = 64 * 1024      # 64KB typical max
    
    print(f"\nTCP buffer comparison:")
    print(f"  Default buffer: {tcp_buffer_default:,} bytes (16KB)")
    print(f"  Message size: {total_bytes:,} bytes ({total_bytes/1024:.1f}KB)")
    
    if total_bytes < tcp_buffer_default:
        print(f"  ✅ Fits in default buffer! No fragmentation needed")
    elif total_bytes < tcp_buffer_max:
        print(f"  ⚠️  Needs larger buffer but fits in 64KB")
    else:
        print(f"  ❌ Too large even for 64KB buffer")
    
    # Compare to old broken size
    old_values = 307212
    old_bytes = old_values * 4 + header_bytes
    
    print(f"\nOld (broken) message size:")
    print(f"  Values: {old_values:,}")
    print(f"  Size: {old_bytes:,} bytes ({old_bytes/1024/1024:.1f}MB)")
    print(f"  Reduction: {old_bytes/total_bytes:.1f}x smaller now!")


if __name__ == "__main__":
    # Run tests
    success = test_dimension_calculation()
    test_message_size()
    
    print("\n" + "=" * 60)
    if success:
        print("RESULT: System ready for TCP deployment!")
        print("Next step: Test actual robot connection")
    else:
        print("RESULT: Still has issues - check dimensions")
    print("=" * 60)