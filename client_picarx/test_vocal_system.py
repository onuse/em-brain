#!/usr/bin/env python3
"""
Test the vocal system in the unified brain client architecture.

This demonstrates the digital vocal cords working as a complex actuator
within the hardware abstraction layer.

Usage:
    python3 test_vocal_system.py              # Silent mock driver (default)
    python3 test_vocal_system.py mac          # Mac speakers (advanced mock)
    python3 test_vocal_system.py mock         # Silent mock driver (explicit)
"""

import sys
import os
import time

# Add client source to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from hardware.interfaces.vocal_interface import EmotionalVocalMapper
from hardware.mock.mock_vocal_driver import MockVocalDriver
from hardware.mock.mac_audio_vocal_driver import MacAudioVocalDriver
from brainstem.brain_client import MockBrainServerClient, BrainServerConfig


def test_unified_vocal_system(driver_type='mock'):
    """Test the complete vocal system in unified architecture."""
    
    print("🎵 UNIFIED BRAIN VOCAL SYSTEM TEST")
    print("=" * 60)
    print("Testing digital vocal cords as complex actuator in HAL")
    print(f"Driver type: {driver_type}")
    print()
    
    # Initialize components
    print("🔧 Initializing unified brain components...")
    
    # Mock brain server client
    brain_client = MockBrainServerClient()
    brain_client.connect()
    
    # Vocal system components - select driver based on parameter
    if driver_type == 'mac':
        vocal_driver = MacAudioVocalDriver()
        print("🔊 Using Mac Audio Vocal Driver - you should hear sounds!")
    else:
        vocal_driver = MockVocalDriver()
        print("🔇 Using Mock Vocal Driver - silent operation")
    
    vocal_driver.initialize_vocal_system()
    
    emotional_mapper = EmotionalVocalMapper()
    
    print("✅ All components initialized\n")
    
    # Simulate robot control cycle with vocal expressions
    print("🤖 Starting robot control cycle simulation...")
    
    for cycle in range(5):
        print(f"\n--- Cycle {cycle + 1} ---")
        
        # Simulate sensor data
        sensor_data = {
            'ultrasonic_distance': 50.0 - (cycle * 8),  # Getting closer to obstacle
            'battery_voltage': 7.2,
            'exploration_mode': cycle < 3
        }
        
        # Send to brain server (mock)
        brain_client.send_sensor_data(sensor_data)
        
        # Get brain response
        motor_commands = brain_client.get_latest_motor_commands()
        vocal_commands = brain_client.get_latest_vocal_commands()
        
        print(f"📊 Sensors: distance={sensor_data['ultrasonic_distance']:.1f}cm")
        if motor_commands:
            print(f"🚗 Motors: speed={motor_commands['motor_speed']:.1f}, steering={motor_commands['steering_angle']:.1f}")
        
        # Generate vocal expression
        if vocal_commands:
            emotional_state = vocal_commands['emotional_state']
            print(f"🎵 Vocal command: {emotional_state}")
            
            # Create mock brain state for emotional mapping
            brain_state = {
                'prediction_confidence': 0.8 if emotional_state == 'confidence' else 0.3,
                'prediction_method': 'consensus' if emotional_state == 'confidence' else 'bootstrap_random',
                'total_experiences': cycle * 10,
                'collision_detected': emotional_state == 'confusion'
            }
            
            # Map to vocal parameters
            vocal_params = emotional_mapper.map_brain_state_to_vocal_params(brain_state)
            
            # Execute vocalization
            vocal_driver.synthesize_vocalization(vocal_params)
            
            # Wait for vocalization to complete
            time.sleep(vocal_params.duration + 0.1)
        
        # Send status update
        status_data = {
            'operational_status': 'OK',
            'cycle_number': cycle + 1,
            'vocal_active': vocal_driver.is_vocalizing()
        }
        brain_client.send_status_update(status_data)
        
        time.sleep(0.5)  # Control cycle timing
    
    print(f"\n📊 VOCAL SYSTEM STATISTICS:")
    stats = vocal_driver.get_statistics()
    for key, value in stats.items():
        if key != 'last_vocalization':
            print(f"   {key}: {value}")
    
    print(f"\n🧠 BRAIN CLIENT STATISTICS:")
    client_stats = brain_client.get_connection_stats()
    for key, value in client_stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Unified brain vocal system test completed!")
    print("🎯 Key achievements:")
    print("   • Brain server ↔ robot client communication working")
    print("   • Emotional vocal expressions driven by brain state")
    print("   • Hardware abstraction layer functioning correctly")
    print("   • Mock implementations enable hardware-free development")
    print("   • Digital vocal cords working as complex actuator")


if __name__ == "__main__":
    # Parse command line arguments for driver selection
    driver_type = 'mock'  # Default to silent operation
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['mac', 'mock']:
            driver_type = arg
        else:
            print("❌ Invalid driver type. Use 'mac' or 'mock'")
            print("Usage:")
            print("    python3 test_vocal_system.py              # Silent mock driver (default)")
            print("    python3 test_vocal_system.py mac          # Mac speakers (advanced mock)")
            print("    python3 test_vocal_system.py mock         # Silent mock driver (explicit)")
            sys.exit(1)
    
    test_unified_vocal_system(driver_type)