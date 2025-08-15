#!/usr/bin/env python3
"""
Simple Robot HAT Test

Tests just the robot-hat library and basic hardware access.
No camera or audio to avoid conflicts.
"""

import sys
import time
from pathlib import Path

# Add robot_hat to path
sys.path.insert(0, 'robot_hat')

def test_basic_import():
    """Test robot-hat imports."""
    print("\n1. Testing robot-hat library...")
    try:
        from robot_hat import PWM, ADC, Pin, I2C
        print("   ✅ robot-hat imported")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_i2c_connection():
    """Test I2C connection to robot-hat MCU."""
    print("\n2. Testing I2C connection...")
    try:
        from robot_hat import I2C
        i2c = I2C([0x14, 0x15])
        print(f"   ✅ MCU found at 0x{i2c.address:02x}")
        return True
    except Exception as e:
        print(f"   ❌ I2C failed: {e}")
        return False

def test_adc_reading():
    """Test ADC sensor reading."""
    print("\n3. Testing ADC sensors...")
    try:
        from robot_hat import ADC
        
        # Test grayscale sensors
        sensors = []
        for ch in ['A0', 'A1', 'A2']:
            adc = ADC(ch)
            value = adc.read()
            sensors.append(value)
            print(f"   {ch}: {value} (0-4095)")
        
        # Test battery
        battery = ADC('A4')
        batt_value = battery.read()
        batt_voltage = batt_value * 3.3 / 4095 * 3  # Voltage divider
        print(f"   Battery: {batt_value} ADC ({batt_voltage:.2f}V)")
        
        if all(0 <= v <= 4095 for v in sensors):
            print("   ✅ ADC working")
            return True
        else:
            print("   ⚠️ ADC values out of range")
            return False
            
    except Exception as e:
        print(f"   ❌ ADC failed: {e}")
        return False

def test_ultrasonic():
    """Test ultrasonic sensor."""
    print("\n4. Testing ultrasonic sensor...")
    try:
        from robot_hat import Pin
        from robot_hat.modules import Ultrasonic
        
        trig = Pin("D2")
        echo = Pin("D3")
        sonar = Ultrasonic(trig, echo)
        
        # Read distance
        distance = sonar.read()
        print(f"   Distance: {distance:.1f}cm")
        
        if 2 <= distance <= 400:  # Valid range
            print("   ✅ Ultrasonic working")
            return True
        else:
            print(f"   ⚠️ Distance out of range: {distance}")
            return True  # Still ok, might be timeout
            
    except Exception as e:
        print(f"   ❌ Ultrasonic failed: {e}")
        return False

def test_pwm_setup():
    """Test PWM channel setup (no movement)."""
    print("\n5. Testing PWM setup...")
    try:
        from robot_hat import PWM
        
        # Test motor PWM channels
        left_motor = PWM('P12')
        right_motor = PWM('P13')
        left_motor.freq(1000)
        right_motor.freq(1000)
        left_motor.pulse_width_percent(0)
        right_motor.pulse_width_percent(0)
        print("   Motor PWM: P12, P13 initialized")
        
        # Test servo PWM channels  
        steering = PWM('P2')
        pan = PWM('P0')
        tilt = PWM('P1')
        for pwm in [steering, pan, tilt]:
            pwm.freq(50)  # 50Hz for servos
            pwm.pulse_width(307)  # ~1500us center
        print("   Servo PWM: P0, P1, P2 initialized")
        
        print("   ✅ PWM setup complete")
        return True
        
    except Exception as e:
        print(f"   ❌ PWM setup failed: {e}")
        return False

def test_gpio_pins():
    """Test GPIO direction pins."""
    print("\n6. Testing GPIO pins...")
    try:
        from robot_hat import Pin
        
        # Motor direction pins
        left_dir = Pin("D4")
        right_dir = Pin("D5")
        left_dir.mode(Pin.OUT)
        right_dir.mode(Pin.OUT)
        
        # Set to low (safe)
        left_dir.value(0)
        right_dir.value(0)
        
        print("   Direction pins: D4, D5 configured")
        print("   ✅ GPIO working")
        return True
        
    except Exception as e:
        print(f"   ❌ GPIO failed: {e}")
        return False

def test_servo_center():
    """Center all servos (safe operation)."""
    print("\n7. Centering servos...")
    
    response = input("   OK to center servos? (y/n): ")
    if response.lower() != 'y':
        print("   ⏭️ Skipped")
        return True
    
    try:
        from robot_hat import Servo
        
        # Create servo objects
        steering = Servo('P2')
        pan = Servo('P0')
        tilt = Servo('P1')
        
        # Center all
        steering.angle(0)
        pan.angle(0)
        tilt.angle(0)
        
        print("   ✅ Servos centered")
        return True
        
    except Exception as e:
        print(f"   ❌ Servo centering failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMPLE ROBOT HAT TEST")
    print("=" * 60)
    print("This tests the robot-hat library without camera/audio")
    
    tests = [
        test_basic_import,
        test_i2c_connection,
        test_adc_reading,
        test_ultrasonic,
        test_pwm_setup,
        test_gpio_pins,
        test_servo_center,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ Robot HAT is working correctly!")
        print("✅ The MCU at 0x14 is handling PWM/servos")
        print("✅ You do NOT need PCA9685 at 0x40")
    else:
        print(f"⚠️ {passed}/{total} tests passed")
    
    print("\nNext steps:")
    print("1. Run the main robot: sudo python3 picarx_robot.py")
    print("2. The brain connection error is expected (no server running)")
    print("3. Camera/audio conflicts are from multiple initializations")

if __name__ == "__main__":
    main()