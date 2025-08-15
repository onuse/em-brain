#!/usr/bin/env python3
"""
Test brainstem reconnection and audio features.

This verifies:
1. Reconnection logic when brain server disconnects
2. Audio output from brain commands
3. Full resolution vision (307,200 pixels)
"""

import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.brainstem.brainstem import Brainstem


def test_reconnection():
    """Test that brainstem handles disconnection gracefully."""
    
    print("=" * 60)
    print("BRAINSTEM RECONNECTION TEST")
    print("=" * 60)
    
    # Create brainstem (will try to connect)
    brainstem = Brainstem(brain_host="localhost", brain_port=9999)
    
    print("\n📊 Configuration:")
    print(f"   Reconnect interval: {brainstem.reconnect_interval}s")
    print(f"   Brain client: {'Connected' if brainstem.brain_client and brainstem.brain_client.connected else 'Not connected'}")
    
    if brainstem.hal.vision:
        print(f"   Vision resolution: {brainstem.hal.vision.resolution}")
        print(f"   Vision pixels: {brainstem.hal.vision.pixels:,}")
    
    if brainstem.hal.audio:
        print(f"   Audio: Enabled")
    
    print("\n🔄 Testing reconnection behavior...")
    print("   (Start/stop brain server to test reconnection)")
    
    # Run for 30 seconds, should attempt reconnection every 5 seconds
    start_time = time.time()
    while time.time() - start_time < 30:
        # Run one cycle
        brainstem.run_cycle()
        
        # Check connection status
        if brainstem.brain_client and brainstem.brain_client.connected:
            status = "✅ Connected"
        else:
            time_since_attempt = time.time() - brainstem.last_connect_attempt
            next_attempt = max(0, brainstem.reconnect_interval - time_since_attempt)
            status = f"❌ Disconnected (retry in {next_attempt:.1f}s)"
        
        # Print status every 2 seconds
        if int(time.time()) % 2 == 0:
            print(f"   [{time.time() - start_time:.1f}s] {status}")
        
        time.sleep(0.1)
    
    print("\n✅ Test complete!")
    brainstem.shutdown()


def test_audio_output():
    """Test that audio output works from brain commands."""
    
    print("\n" + "=" * 60)
    print("AUDIO OUTPUT TEST")
    print("=" * 60)
    
    brainstem = Brainstem()
    
    if not brainstem.hal.audio:
        print("❌ Audio not available")
        return
    
    print("🎵 Testing audio output...")
    
    # Simulate brain commands with different audio parameters
    test_patterns = [
        (0.0, 0.5),  # Low freq, medium volume
        (0.5, 0.5),  # Mid freq, medium volume
        (1.0, 0.5),  # High freq, medium volume
        (0.5, 0.0),  # No sound (volume = 0)
        (0.5, 1.0),  # Mid freq, max volume
    ]
    
    for i, (freq, vol) in enumerate(test_patterns):
        print(f"   Pattern {i+1}: freq={freq:.1f}, vol={vol:.1f}")
        
        # Create fake brain output with audio commands
        brain_output = [0.0, 0.0, 0.0, 0.0, freq, vol]
        
        # Process through brainstem
        motor_cmd, servo_cmd = brainstem.brain_to_hardware_commands(brain_output)
        
        time.sleep(0.5)
    
    print("✅ Audio test complete!")
    brainstem.shutdown()


def test_vision_bandwidth():
    """Test that full resolution vision is being sent."""
    
    print("\n" + "=" * 60)
    print("VISION BANDWIDTH TEST")
    print("=" * 60)
    
    brainstem = Brainstem()
    
    if not brainstem.hal.vision:
        print("❌ Vision not available")
        return
    
    print(f"📷 Vision configuration:")
    print(f"   Resolution: {brainstem.hal.vision.resolution}")
    print(f"   Pixels: {brainstem.hal.vision.pixels:,}")
    print(f"   Format: {brainstem.hal.vision.format}")
    print(f"   Output dimension: {brainstem.hal.vision.output_dim:,}")
    
    # Read sensors and check vision data size
    raw_sensors = brainstem.hal.read_raw_sensors()
    
    print(f"\n📊 Sensor data sizes:")
    print(f"   Vision data: {len(raw_sensors.vision_data):,} values")
    print(f"   Audio features: {len(raw_sensors.audio_features)} values")
    
    # Convert to brain format
    brain_input = brainstem.sensors_to_brain_format(raw_sensors)
    
    print(f"\n🧠 Brain input:")
    print(f"   Total channels: {len(brain_input):,}")
    print(f"   Expected: {5 + brainstem.hal.vision.output_dim + 7:,}")
    
    # Calculate bandwidth
    bytes_per_frame = len(brain_input) * 4  # float32
    mb_per_second = (bytes_per_frame * 20) / 1_000_000  # at 20Hz
    
    print(f"\n📡 Network bandwidth:")
    print(f"   Bytes per frame: {bytes_per_frame:,}")
    print(f"   At 20Hz: {mb_per_second:.1f} MB/s")
    
    brainstem.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test brainstem features')
    parser.add_argument('--test', choices=['reconnect', 'audio', 'vision', 'all'],
                        default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    if args.test in ['reconnect', 'all']:
        test_reconnection()
    
    if args.test in ['audio', 'all']:
        test_audio_output()
    
    if args.test in ['vision', 'all']:
        test_vision_bandwidth()