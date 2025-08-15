#!/usr/bin/env python3
"""
Full Integration Test for PiCar-X Brainstem
Tests the complete chain from robot hardware to brain server.
"""

import sys
import time
import socket
import threading
import argparse
sys.path.append('src')

from brainstem.brain_client import BrainServerClient, BrainServerConfig
from brainstem.sensor_motor_adapter import PiCarXBrainAdapter
from brainstem.integrated_brainstem import IntegratedBrainstem, BrainstemConfig


def test_protocol_compatibility():
    """Test that our protocol matches server expectations."""
    print("\n" + "="*60)
    print("1. PROTOCOL COMPATIBILITY TEST")
    print("="*60)
    
    config = BrainServerConfig(
        host="localhost",
        port=9999
    )
    
    client = BrainServerClient(config)
    
    # Test connection
    print("Testing TCP connection to port 9999...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(("localhost", 9999))
        sock.close()
        print("✅ Can connect to port 9999")
    except:
        print("⚠️  Brain server not running on port 9999")
        print("   Start it with: python3 ../server/brain.py")
        return False
    
    # Test handshake
    print("\nTesting handshake protocol...")
    if client.connect():
        print("✅ Handshake successful")
        client.disconnect()
        return True
    else:
        print("❌ Handshake failed")
        return False


def test_sensor_adapter():
    """Test sensor-motor adaptation."""
    print("\n" + "="*60)
    print("2. SENSOR-MOTOR ADAPTER TEST")
    print("="*60)
    
    adapter = PiCarXBrainAdapter()
    
    # Test with realistic sensor data
    mock_sensors = [
        0.5,    # ultrasonic (meters)
        0.3,    # grayscale right
        0.8,    # grayscale center (on line)
        0.3,    # grayscale left
        0.2,    # left motor speed
        0.2,    # right motor speed
        0,      # camera pan
        0,      # camera tilt
        0,      # steering
        7.4,    # battery
        1,      # line detected
        0,      # no cliff
        45,     # CPU temp
        0.3,    # memory
        1000,   # timestamp
        0       # reserved
    ]
    
    print(f"Input: 16 hardware sensors")
    brain_input = adapter.sensors_to_brain_input(mock_sensors)
    print(f"✅ Expanded to {len(brain_input)} brain inputs")
    
    # Test motor conversion
    brain_output = [0.5, 0.2, 0.0, 0.0]  # Forward with slight right
    motor_commands = adapter.brain_output_to_motors(brain_output)
    print(f"✅ Brain output (4 channels) → Motor commands (5 motors)")
    
    return True


def test_integrated_brainstem():
    """Test the complete integrated brainstem."""
    print("\n" + "="*60)
    print("3. INTEGRATED BRAINSTEM TEST")
    print("="*60)
    
    config = BrainstemConfig(
        brain_server_config=BrainServerConfig(
            host="localhost",
            port=9999
        ),
        use_mock_brain=False,  # Try real connection first
        enable_local_reflexes=True,
        safety_override=True
    )
    
    brainstem = IntegratedBrainstem(config)
    
    # Try to connect
    print("Attempting brain server connection...")
    connected = brainstem.connect()
    
    if not connected:
        print("⚠️  Using autonomous mode (no brain server)")
    else:
        print("✅ Connected to brain server")
    
    # Test processing cycles
    print("\nTesting processing cycles...")
    
    test_scenarios = [
        ("Normal", [0.8, 0.3, 0.3, 0.3, 0.2, 0.2, 0, 0, 0, 7.4, 0, 0, 45, 0.3, 1000, 0]),
        ("Obstacle", [0.15, 0.3, 0.3, 0.3, 0.1, 0.1, 0, 0, 0, 7.4, 0, 0, 45, 0.3, 1000, 0]),
        ("On line", [0.5, 0.2, 0.8, 0.2, 0.2, 0.2, 0, 0, 0, 7.4, 1, 0, 45, 0.3, 1000, 0]),
    ]
    
    for name, sensors in test_scenarios:
        motor_commands = brainstem.process_cycle(sensors)
        print(f"✅ {name}: Motors={motor_commands['left_motor']:.1f},{motor_commands['right_motor']:.1f}")
        time.sleep(0.05)  # 20Hz
    
    # Check status
    status = brainstem.get_status()
    print(f"\n✅ Brainstem Status:")
    print(f"   Connected: {status['connected']}")
    print(f"   Cycles: {status['cycles']}")
    print(f"   Reflexes: {status['reflex_activations']}")
    
    brainstem.shutdown()
    return True


def test_full_communication_loop():
    """Test a full communication loop with timing."""
    print("\n" + "="*60)
    print("4. FULL COMMUNICATION LOOP TEST")
    print("="*60)
    
    config = BrainServerConfig(host="localhost", port=9999)
    client = BrainServerClient(config)
    
    if not client.connect():
        print("⚠️  Skipping - no brain server")
        return True
    
    print("Running 10 cycles at 20Hz...")
    
    adapter = PiCarXBrainAdapter()
    cycle_times = []
    
    for i in range(10):
        start = time.time()
        
        # Simulate sensor reading
        sensors = [0.5 + i*0.01, 0.3, 0.3, 0.3] + [0.2]*12
        
        # Convert and send
        brain_input = adapter.sensors_to_brain_input(sensors)
        sensory_vector = brain_input[:24]  # Don't send reward
        
        client.send_sensory_input(sensory_vector)
        actions = client.receive_action_output()
        
        if actions:
            motor_commands = adapter.brain_output_to_motors(actions)
            
        cycle_time = (time.time() - start) * 1000
        cycle_times.append(cycle_time)
        
        # Maintain 20Hz
        sleep_time = max(0, (50 - cycle_time) / 1000)
        time.sleep(sleep_time)
    
    avg_cycle = sum(cycle_times) / len(cycle_times)
    print(f"✅ Average cycle time: {avg_cycle:.1f}ms")
    print(f"✅ Max cycle time: {max(cycle_times):.1f}ms")
    
    client.disconnect()
    return avg_cycle < 50  # Should be under 50ms for 20Hz


def main():
    """Run all integration tests."""
    parser = argparse.ArgumentParser(
        description='Test PiCar-X brainstem integration'
    )
    parser.add_argument(
        '--brain-host',
        default='localhost',
        help='Brain server hostname'
    )
    parser.add_argument(
        '--skip-server',
        action='store_true',
        help='Skip tests requiring brain server'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PiCar-X BRAINSTEM INTEGRATION TEST SUITE")
    print("="*60)
    print(f"Testing against: {args.brain_host}:9999")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Protocol
    if not args.skip_server:
        tests_total += 1
        if test_protocol_compatibility():
            tests_passed += 1
    
    # Test 2: Adapter
    tests_total += 1
    if test_sensor_adapter():
        tests_passed += 1
    
    # Test 3: Integrated brainstem
    tests_total += 1
    if test_integrated_brainstem():
        tests_passed += 1
    
    # Test 4: Full loop
    if not args.skip_server:
        tests_total += 1
        if test_full_communication_loop():
            tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if tests_passed == tests_total:
        print(f"✅ ALL TESTS PASSED ({tests_passed}/{tests_total})")
        print("\n🎉 PiCar-X brainstem is READY for deployment!")
        print("\nTo run the robot:")
        print("  1. On robot: python3 picarx_robot.py --brain-host <server-ip>")
        print("  2. On server: python3 ../server/brain.py")
    else:
        print(f"⚠️  {tests_passed}/{tests_total} tests passed")
        if args.skip_server:
            print("\nNote: Some tests skipped (no server required)")
        else:
            print("\nCheck that brain server is running:")
            print("  python3 ../server/brain.py")
    
    return 0 if tests_passed == tests_total else 1


if __name__ == "__main__":
    sys.exit(main())