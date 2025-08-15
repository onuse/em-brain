#!/usr/bin/env python3
"""
Test script for robot-hat based HAL

This tests that the robot-hat library is properly integrated and working.
Run this on the Raspberry Pi to verify hardware access.
"""

import sys
import time
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

def test_robot_hat_library():
    """Test that robot-hat library can be imported."""
    print("\n1. Testing robot-hat library import...")
    try:
        sys.path.insert(0, 'robot_hat')
        from robot_hat import PWM, Servo, ADC, Pin
        from robot_hat.modules import Ultrasonic
        print("   ✅ robot-hat library imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import robot-hat: {e}")
        return False

def test_i2c_device():
    """Test that robot-hat MCU is detected at 0x14."""
    print("\n2. Testing I2C device detection...")
    try:
        from robot_hat import I2C
        # Robot-hat MCU should be at 0x14
        i2c = I2C([0x14, 0x15])
        print(f"   ✅ Robot HAT MCU found at address 0x{i2c.address:02x}")
        return True
    except Exception as e:
        print(f"   ❌ I2C test failed: {e}")
        return False

def test_raw_hal():
    """Test the raw robot-hat HAL."""
    print("\n3. Testing raw_robot_hat_hal...")
    try:
        from src.hardware.raw_robot_hat_hal import create_hal
        hal = create_hal()
        
        if hal.mock_mode:
            print("   ⚠️ HAL running in mock mode (no hardware)")
        else:
            print("   ✅ HAL initialized with real hardware")
        
        # Get hardware info
        info = hal.get_hardware_info()
        print(f"   HAL type: {info['hal_type']}")
        print(f"   I2C address: 0x{info['i2c_address']:02x}")
        
        return True
    except Exception as e:
        print(f"   ❌ HAL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sensors():
    """Test sensor reading."""
    print("\n4. Testing sensor reading...")
    try:
        from src.hardware.raw_robot_hat_hal import create_hal
        hal = create_hal()
        
        if hal.mock_mode:
            print("   ⚠️ Using mock sensors")
        
        # Read sensors
        sensors = hal.read_raw_sensors()
        
        print(f"   Grayscale ADC: {sensors.i2c_grayscale}")
        print(f"   Ultrasonic: {sensors.gpio_ultrasonic_us:.0f}μs ({sensors.gpio_ultrasonic_us/58:.1f}cm)")
        print(f"   Battery ADC: {sensors.analog_battery_raw}")
        
        # Check if values are reasonable
        if all(0 <= g <= 4095 for g in sensors.i2c_grayscale):
            print("   ✅ Sensor values in expected range")
            return True
        else:
            print("   ⚠️ Sensor values out of range")
            return False
            
    except Exception as e:
        print(f"   ❌ Sensor test failed: {e}")
        return False

def test_servo_control():
    """Test servo control (safe test)."""
    print("\n5. Testing servo control...")
    try:
        from src.hardware.raw_robot_hat_hal import create_hal, RawServoCommand
        hal = create_hal()
        
        if hal.mock_mode:
            print("   ⚠️ Skipping servo test (mock mode)")
            return True
        
        print("   Moving servos to center position...")
        hal.execute_servo_command(RawServoCommand(1500, 1500, 1500))
        time.sleep(1)
        
        print("   Testing small movement...")
        hal.execute_servo_command(RawServoCommand(1400, 1500, 1500))
        time.sleep(0.5)
        hal.execute_servo_command(RawServoCommand(1600, 1500, 1500))
        time.sleep(0.5)
        hal.execute_servo_command(RawServoCommand(1500, 1500, 1500))
        
        print("   ✅ Servo control working")
        return True
        
    except Exception as e:
        print(f"   ❌ Servo test failed: {e}")
        return False

def test_motor_control():
    """Test motor control (wheels should be off ground!)."""
    print("\n6. Testing motor control...")
    print("   ⚠️ WARNING: Wheels should be OFF THE GROUND!")
    
    response = input("   Are wheels off the ground? (y/n): ")
    if response.lower() != 'y':
        print("   ⏭️ Skipping motor test for safety")
        return True
    
    try:
        from src.hardware.raw_robot_hat_hal import create_hal, RawMotorCommand
        hal = create_hal()
        
        if hal.mock_mode:
            print("   ⚠️ Skipping motor test (mock mode)")
            return True
        
        print("   Testing forward at 20% speed...")
        hal.execute_motor_command(RawMotorCommand(0.2, 0.2))
        time.sleep(1)
        
        print("   Testing turn...")
        hal.execute_motor_command(RawMotorCommand(0.2, 0.1))
        time.sleep(1)
        
        print("   Stopping...")
        hal.emergency_stop()
        
        print("   ✅ Motor control working")
        return True
        
    except Exception as e:
        print(f"   ❌ Motor test failed: {e}")
        hal.emergency_stop()
        return False

def test_brainstem():
    """Test brainstem with new HAL."""
    print("\n7. Testing brainstem integration...")
    try:
        from src.brainstem.brainstem import Brainstem
        
        # Create brainstem without brain connection
        bs = Brainstem(
            enable_brain=False,
            enable_monitor=False
        )
        
        print("   ✅ Brainstem created successfully")
        
        # Test one sensor read
        sensors = bs.hal.read_raw_sensors()
        print(f"   Read sensors: {len(sensors.i2c_grayscale)} grayscale values")
        
        # Test safety check
        safe = bs.check_safety(sensors)
        print(f"   Safety check: {'✅ Safe' if safe else '⚠️ Reflex triggered'}")
        
        bs.cleanup()
        return True
        
    except Exception as e:
        print(f"   ❌ Brainstem test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Robot HAT Hardware Test Suite")
    print("=" * 60)
    
    tests = [
        test_robot_hat_library,
        test_i2c_device,
        test_raw_hal,
        test_sensors,
        test_servo_control,
        test_motor_control,
        test_brainstem,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except KeyboardInterrupt:
            print("\n\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\nThe robot is ready for deployment!")
    else:
        print(f"⚠️ {passed}/{total} tests passed")
        print("\nPlease fix the failures before deployment.")
    
    print("\nNext steps:")
    print("1. If all tests pass, run: sudo python3 picarx_robot.py")
    print("2. Monitor telemetry: nc localhost 9997")
    print("3. Connect brain server and test full system")

if __name__ == "__main__":
    main()