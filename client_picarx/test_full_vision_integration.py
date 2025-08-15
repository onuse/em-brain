#!/usr/bin/env python3
"""
Test Full Vision Integration

This test verifies that we're sending full resolution vision data 
(640×480 = 307,200 pixels) instead of just 14 feature values.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from hardware.bare_metal_hal import create_hal
from brainstem.brainstem import Brainstem

def test_vision_resolution():
    """Test that we're getting full resolution vision data."""
    print("🧪 Testing Full Vision Resolution Integration")
    print("=" * 60)
    
    # Create HAL (force mock mode for testing)
    hal = create_hal(force_mock=True)
    
    # Read raw sensor data
    raw_data = hal.read_raw_sensors()
    
    print(f"📊 Raw Vision Data Analysis:")
    print(f"   Vision data length: {len(raw_data.vision_data):,}")
    print(f"   Expected for 640×480: {640*480:,}")
    
    if len(raw_data.vision_data) == 640 * 480:
        print("   ✅ Correct! Full resolution vision data detected")
    else:
        print(f"   ❌ Wrong size! Expected {640*480:,}, got {len(raw_data.vision_data):,}")
    
    # Test brainstem conversion
    print(f"\n🧠 Brainstem Conversion Test:")
    brainstem = Brainstem("localhost", 9999)
    
    # Don't actually connect to brain server, just test conversion
    brain_input = brainstem.sensors_to_brain_format(raw_data)
    
    print(f"   Brain input dimensions: {len(brain_input):,}")
    
    expected_dims = 5 + 307200 + 7  # basic + vision + audio
    print(f"   Expected dimensions: {expected_dims:,}")
    
    if len(brain_input) == expected_dims:
        print("   ✅ Perfect! Brain input has correct dimensions")
    else:
        print(f"   ❌ Dimension mismatch! Expected {expected_dims:,}, got {len(brain_input):,}")
    
    # Analyze the structure
    print(f"\n📈 Brain Input Structure:")
    print(f"   Channels 0-2: Grayscale sensors = {brain_input[0:3]}")
    print(f"   Channel 3: Ultrasonic = {brain_input[3]}")
    print(f"   Channel 4: Battery = {brain_input[4]}")
    print(f"   Channels 5-307204: Vision pixels ({len(brain_input[5:307205]):,} values)")
    print(f"   Channels 307205-307211: Audio features = {brain_input[307205:307212]}")
    
    # Sample some vision pixels
    vision_start = 5
    vision_sample = brain_input[vision_start:vision_start+10]
    print(f"   Vision sample (first 10 pixels): {vision_sample}")
    
    # Check if vision pixels are in valid range (0-1)
    vision_pixels = brain_input[5:307205]
    min_pixel = min(vision_pixels)
    max_pixel = max(vision_pixels)
    print(f"   Vision pixel range: {min_pixel:.3f} to {max_pixel:.3f}")
    
    if 0 <= min_pixel <= 1 and 0 <= max_pixel <= 1:
        print("   ✅ Vision pixels properly normalized to 0-1 range")
    else:
        print("   ⚠️  Vision pixels outside expected 0-1 range")
    
    print(f"\n🎯 Summary:")
    if len(raw_data.vision_data) == 307200 and len(brain_input) == 307212:
        print("   ✅ SUCCESS: Full resolution vision integration working!")
        print("   🔥 Sending 307,200 pixels to brain for real vision intelligence")
    else:
        print("   ❌ FAILED: Still not sending full resolution vision data")
    
    # Cleanup
    hal.cleanup()

def test_configurable_vision():
    """Test the ConfigurableVision module directly."""
    print("\n📷 Testing ConfigurableVision Module")
    print("-" * 40)
    
    try:
        from hardware.configurable_vision import ConfigurableVision
        
        # Create vision module
        vision = ConfigurableVision()
        
        print(f"   Resolution: {vision.resolution}")
        print(f"   Pixels: {vision.pixels:,}")
        print(f"   Output dimension: {vision.output_dim:,}")
        print(f"   Enabled: {vision.enabled}")
        
        # Get frame data
        frame_data = vision.get_flattened_frame()
        
        print(f"   Frame data length: {len(frame_data):,}")
        print(f"   Sample values: {frame_data[:5]} ... {frame_data[-5:]}")
        
        if len(frame_data) == vision.output_dim:
            print("   ✅ ConfigurableVision working correctly")
        else:
            print("   ❌ ConfigurableVision dimension mismatch")
        
        vision.cleanup()
        
    except Exception as e:
        print(f"   ⚠️  ConfigurableVision test failed: {e}")

if __name__ == "__main__":
    test_configurable_vision()
    test_vision_resolution()
    
    print(f"\n{'='*60}")
    print("🎯 INTEGRATION TEST COMPLETE")
    print("If you see ✅ SUCCESS above, the brain will now receive")
    print("full 640×480 = 307,200 pixel resolution vision data!")
    print("This forces the brain to develop real vision intelligence.")
    print("{'='*60}")