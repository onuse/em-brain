#!/usr/bin/env python3
"""
Brainstem Deployment Readiness Test

Tests all critical components before deployment to ensure nothing will break.
Run this on both development machine and Raspberry Pi.
"""

import sys
import time
import socket
from pathlib import Path

# Test results tracking
tests_passed = []
tests_failed = []

def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            try:
                print(f"\n🧪 Testing: {name}...")
                func()
                print(f"   ✅ PASS")
                tests_passed.append(name)
                return True
            except Exception as e:
                print(f"   ❌ FAIL: {e}")
                tests_failed.append((name, str(e)))
                return False
        return wrapper
    return decorator


@test("Import brainstem module")
def test_import_brainstem():
    from src.brainstem.brainstem import Brainstem
    assert Brainstem is not None


@test("Import brain client")
def test_import_brain_client():
    from src.brainstem.brain_client import BrainClient, MessageProtocol
    assert BrainClient is not None
    assert MessageProtocol is not None


@test("Import monitoring")
def test_import_monitor():
    from src.brainstem.brainstem_monitor import BrainstemMonitor
    assert BrainstemMonitor is not None


@test("Import HAL")
def test_import_hal():
    from src.hardware.bare_metal_hal import (
        BareMetalHAL, create_hal,
        RawMotorCommand, RawServoCommand, RawSensorData
    )
    assert create_hal is not None


@test("Load robot config")
def test_load_config():
    import json
    config_path = Path("config/robot_config.json")
    assert config_path.exists(), f"Config not found at {config_path}"
    
    with open(config_path) as f:
        config = json.load(f)
    
    assert "brain" in config
    assert "safety" in config
    assert config["safety"]["collision_distance_cm"] == 3  # Should be 3cm now
    print(f"   Config loaded: brain={config['brain']['host']}:{config['brain']['port']}")


@test("Create HAL instance")
def test_create_hal():
    from src.hardware.bare_metal_hal import create_hal
    hal = create_hal(force_mock=True)  # Force mock for testing
    assert hal is not None
    
    # Test critical methods exist
    assert hasattr(hal, 'read_raw_sensors')
    assert hasattr(hal, 'execute_motor_command')
    assert hasattr(hal, 'execute_servo_command')
    assert hasattr(hal, 'emergency_stop')


@test("MessageProtocol encode/decode")
def test_protocol():
    from src.brainstem.brain_client import MessageProtocol
    
    protocol = MessageProtocol()
    
    # Test encoding
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    encoded = protocol.encode_sensory_input(test_data)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 9  # Header + data
    
    # Test decoding
    msg_type, decoded = protocol.decode_message(encoded)
    assert msg_type == MessageProtocol.MSG_SENSORY_INPUT
    assert len(decoded) == len(test_data)
    assert all(abs(a - b) < 0.001 for a, b in zip(decoded, test_data))
    print(f"   Protocol test: encoded {len(test_data)} floats to {len(encoded)} bytes")


@test("Create brainstem instance")
def test_create_brainstem():
    from src.brainstem.brainstem import Brainstem
    
    # Create with monitoring disabled for test
    # Don't override config values - pass None or don't pass at all
    brainstem = Brainstem(
        enable_monitor=False  # Don't start monitor in test
    )
    
    assert brainstem is not None
    assert brainstem.safety.collision_distance_cm == 3  # Should be 3cm from config
    assert brainstem.control_loop_hz == 20
    
    # Check critical methods exist
    assert hasattr(brainstem, 'run_cycle')
    assert hasattr(brainstem, 'check_safety')
    assert hasattr(brainstem, 'sensors_to_brain_format')
    assert hasattr(brainstem, 'brain_to_hardware_commands')
    
    print(f"   Brainstem initialized with safety threshold: {brainstem.safety.collision_distance_cm}cm")


@test("Monitor server start/stop")
def test_monitor():
    from src.brainstem.brainstem_monitor import BrainstemMonitor
    
    # Use a random high port for testing
    test_port = 19997
    monitor = BrainstemMonitor(port=test_port)
    
    # Start server
    monitor.start()
    time.sleep(0.5)  # Let it start
    
    # Try to connect
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(('localhost', test_port))
        sock.close()
        print(f"   Monitor server accessible on port {test_port}")
    finally:
        monitor.stop()


@test("HAL sensor data structure")
def test_sensor_structure():
    from src.hardware.bare_metal_hal import RawSensorData
    
    # Create sample sensor data
    sensor_data = RawSensorData(
        i2c_grayscale=[2000, 2000, 2000],
        gpio_ultrasonic_us=5800.0,  # 100cm
        analog_battery_raw=2800,
        motor_current_raw=[0, 0],
        vision_data=[0.5] * 307200,  # 640x480 grayscale
        audio_features=[0.0] * 7,
        timestamp_ns=time.time_ns()
    )
    
    # Verify structure
    assert len(sensor_data.i2c_grayscale) == 3
    assert len(sensor_data.vision_data) == 307200
    assert len(sensor_data.audio_features) == 7
    print(f"   Sensor structure verified: {len(sensor_data.vision_data):,} vision pixels")


@test("Full cycle simulation")
def test_full_cycle():
    from src.brainstem.brainstem import Brainstem
    from src.hardware.bare_metal_hal import RawSensorData, RawMotorCommand, RawServoCommand
    
    # Create brainstem without monitor or brain connection
    # Don't pass brain_host/port - let it use config values
    brainstem = Brainstem(
        enable_monitor=False
    )
    
    # Create mock sensor data (safe values)
    mock_sensors = RawSensorData(
        i2c_grayscale=[2000, 2000, 2000],  # No cliff
        gpio_ultrasonic_us=5800.0,  # 100cm - safe distance
        analog_battery_raw=3000,  # Good battery
        motor_current_raw=[0, 0],
        vision_data=[0.5] * 307200,
        audio_features=[0.0] * 7,
        timestamp_ns=time.time_ns()
    )
    
    # Test safety check
    safe = brainstem.check_safety(mock_sensors)
    assert safe == True, "Safety check failed with safe values"
    
    # Test sensor conversion
    brain_input = brainstem.sensors_to_brain_format(mock_sensors)
    assert len(brain_input) == 307212  # 5 basic + 307200 vision + 7 audio
    
    # Test brain output conversion
    brain_output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Safe defaults
    motor_cmd, servo_cmd = brainstem.brain_to_hardware_commands(brain_output)
    
    assert isinstance(motor_cmd, RawMotorCommand)
    assert isinstance(servo_cmd, RawServoCommand)
    assert motor_cmd.left_pwm_duty == 0.0  # Should be stopped
    
    print(f"   Full cycle test completed successfully")


def main():
    """Run all deployment tests."""
    print("=" * 60)
    print("🚀 BRAINSTEM DEPLOYMENT READINESS TEST")
    print("=" * 60)
    
    # Run all test functions
    test_functions = [
        test_import_brainstem,
        test_import_brain_client,
        test_import_monitor,
        test_import_hal,
        test_load_config,
        test_create_hal,
        test_protocol,
        test_create_brainstem,
        test_monitor,
        test_sensor_structure,
        test_full_cycle
    ]
    
    for test_func in test_functions:
        test_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {len(tests_passed)}")
    print(f"❌ Failed: {len(tests_failed)}")
    
    if tests_failed:
        print("\nFailed tests:")
        for name, error in tests_failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print("\n🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        print("\nNext steps:")
        print("1. Run on Raspberry Pi: python3 test_brainstem_deployment.py")
        print("2. Start brain server: python3 server/brain.py")
        print("3. Start brainstem: python3 picarx_robot.py")
        sys.exit(0)


if __name__ == "__main__":
    main()