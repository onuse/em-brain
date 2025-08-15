#!/usr/bin/env python3
"""
Movement Test for PiCar-X

Tests servos and motors with safety prompts.
Run this to verify all actuators are working correctly.
"""

import sys
import time
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, 'robot_hat')

def test_servo_center():
    """Center all servos."""
    print("\n1. SERVO CENTERING TEST")
    print("-" * 40)
    
    response = input("Ready to center all servos? (y/n): ")
    if response.lower() != 'y':
        print("   Skipped")
        return
    
    try:
        from robot_hat import Servo
        
        print("   Creating servo objects...")
        steering = Servo('P2')
        pan = Servo('P0')
        tilt = Servo('P1')
        
        print("   Centering all servos to 0°...")
        steering.angle(0)
        pan.angle(0)
        tilt.angle(0)
        
        print("   ✅ All servos centered")
        time.sleep(1)
        
        return steering, pan, tilt
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None, None

def test_steering_servo(steering):
    """Test steering servo range."""
    print("\n2. STEERING SERVO TEST")
    print("-" * 40)
    
    if not steering:
        print("   Skipped (no servo object)")
        return
    
    response = input("Test steering servo? (y/n): ")
    if response.lower() != 'y':
        print("   Skipped")
        return
    
    try:
        print("   Testing steering range...")
        
        # Test sequence
        moves = [
            (0, "Center"),
            (-20, "Left 20°"),
            (-30, "Left 30° (max)"),
            (0, "Center"),
            (20, "Right 20°"),
            (30, "Right 30° (max)"),
            (0, "Center"),
        ]
        
        for angle, desc in moves:
            print(f"   {desc}...")
            steering.angle(angle)
            time.sleep(0.5)
        
        print("   ✅ Steering servo working")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_camera_servos(pan, tilt):
    """Test camera pan/tilt servos."""
    print("\n3. CAMERA SERVOS TEST")
    print("-" * 40)
    
    if not pan or not tilt:
        print("   Skipped (no servo objects)")
        return
    
    response = input("Test camera servos? (y/n): ")
    if response.lower() != 'y':
        print("   Skipped")
        return
    
    try:
        print("   Testing camera pan...")
        for angle in [0, -45, 0, 45, 0]:
            print(f"   Pan {angle}°...")
            pan.angle(angle)
            time.sleep(0.5)
        
        print("   Testing camera tilt...")
        for angle in [0, -20, 0, 20, 0]:
            print(f"   Tilt {angle}°...")
            tilt.angle(angle)
            time.sleep(0.5)
        
        print("   ✅ Camera servos working")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_motors():
    """Test motor control."""
    print("\n4. MOTOR TEST")
    print("-" * 40)
    print("⚠️  WARNING: Wheels should be OFF THE GROUND!")
    
    response = input("Are wheels OFF THE GROUND? (y/n): ")
    if response.lower() != 'y':
        print("   Skipped for safety")
        return
    
    try:
        from robot_hat import PWM, Pin
        
        print("   Setting up motor control...")
        
        # Motor PWM channels
        left_motor = PWM('P12')
        right_motor = PWM('P13')
        left_motor.freq(1000)
        right_motor.freq(1000)
        
        # Direction pins - mode is set in constructor
        left_dir = Pin("D4", mode=Pin.OUT)
        right_dir = Pin("D5", mode=Pin.OUT)
        
        print("\n   Note: One motor might be wired reversed.")
        print("   Watch carefully which wheel goes which direction.\n")
        
        # Test each motor individually first
        print("   === INDIVIDUAL MOTOR TEST ===")
        
        print("   LEFT motor only - FORWARD (5 seconds)...")
        left_dir.value(1)  # Forward
        right_dir.value(1)
        left_motor.pulse_width_percent(30)
        right_motor.pulse_width_percent(0)  # Right off
        time.sleep(5)
        
        print("   LEFT motor only - REVERSE (5 seconds)...")
        left_dir.value(0)  # Reverse
        left_motor.pulse_width_percent(30)
        right_motor.pulse_width_percent(0)
        time.sleep(5)
        
        print("   Stopping left motor...")
        left_motor.pulse_width_percent(0)
        time.sleep(2)
        
        print("   RIGHT motor only - FORWARD (5 seconds)...")
        left_dir.value(1)
        right_dir.value(1)  # Forward
        left_motor.pulse_width_percent(0)  # Left off
        right_motor.pulse_width_percent(30)
        time.sleep(5)
        
        print("   RIGHT motor only - REVERSE (5 seconds)...")
        right_dir.value(0)  # Reverse  
        left_motor.pulse_width_percent(0)
        right_motor.pulse_width_percent(30)
        time.sleep(5)
        
        print("   Stopping right motor...")
        right_motor.pulse_width_percent(0)
        time.sleep(2)
        
        # Now test both together
        print("\n   === COMBINED MOTOR TEST ===")
        
        # Test sequence with longer durations
        tests = [
            ("Both FORWARD slow", 1, 1, 20),
            ("Both FORWARD medium", 1, 1, 40),
            ("Stop", 1, 1, 0),
            ("Turn left (left slower)", 1, 1, 10, 30),
            ("Turn right (right slower)", 1, 1, 30, 10),
            ("Stop", 1, 1, 0),
            ("Both REVERSE slow", 0, 0, 20),
            ("Stop", 1, 1, 0),
        ]
        
        for test in tests:
            if len(test) == 4:
                desc, l_dir, r_dir, speed = test
                l_speed = r_speed = speed
            else:
                desc, l_dir, r_dir, l_speed, r_speed = test
            
            print(f"   {desc} (5 seconds)...")
            print(f"      L: dir={l_dir}, speed={l_speed}%")
            print(f"      R: dir={r_dir}, speed={r_speed}%")
            left_dir.value(l_dir)
            right_dir.value(r_dir)
            left_motor.pulse_width_percent(l_speed)
            right_motor.pulse_width_percent(r_speed)
            time.sleep(5)  # 5 seconds per test
        
        # Stop motors
        left_motor.pulse_width_percent(0)
        right_motor.pulse_width_percent(0)
        
        print("   ✅ Motors working")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Emergency stop
        try:
            left_motor.pulse_width_percent(0)
            right_motor.pulse_width_percent(0)
        except:
            pass

def test_ultrasonic():
    """Test ultrasonic sensor."""
    print("\n5. ULTRASONIC SENSOR TEST")
    print("-" * 40)
    
    try:
        from robot_hat import Pin
        from robot_hat.modules import Ultrasonic
        
        print("   Setting up ultrasonic...")
        trig = Pin("D2")
        echo = Pin("D3")
        sonar = Ultrasonic(trig, echo)
        
        print("   Reading distance (wave hand in front)...")
        for i in range(10):
            distance = sonar.read()
            print(f"   Distance: {distance:.1f} cm", end='\r')
            time.sleep(0.3)
        
        print("\n   ✅ Ultrasonic working")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_grayscale():
    """Test grayscale sensors."""
    print("\n6. GRAYSCALE SENSOR TEST")
    print("-" * 40)
    
    try:
        from robot_hat import ADC
        
        print("   Setting up ADC...")
        sensors = [ADC('A0'), ADC('A1'), ADC('A2')]
        
        print("   Reading grayscale (place on white/black surface)...")
        for i in range(10):
            values = [s.read() for s in sensors]
            print(f"   L:{values[0]:4d}  M:{values[1]:4d}  R:{values[2]:4d}", end='\r')
            time.sleep(0.3)
        
        print("\n   ✅ Grayscale sensors working")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_battery():
    """Test battery voltage reading."""
    print("\n7. BATTERY VOLTAGE TEST")
    print("-" * 40)
    
    try:
        from robot_hat import ADC
        
        battery = ADC('A4')
        
        for i in range(5):
            raw = battery.read()
            # Convert to voltage (assuming voltage divider)
            voltage = (raw / 4095.0) * 3.3 * 3  # Adjust multiplier as needed
            print(f"   Battery: {raw:4d} ADC = {voltage:.2f}V")
            time.sleep(0.5)
        
        if voltage < 6.5:
            print("   ⚠️ Battery low!")
        else:
            print("   ✅ Battery OK")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_full_system():
    """Test complete system with HAL."""
    print("\n8. FULL SYSTEM TEST (via HAL)")
    print("-" * 40)
    
    response = input("Test full system integration? (y/n): ")
    if response.lower() != 'y':
        print("   Skipped")
        return
    
    try:
        from src.hardware.raw_robot_hat_hal import create_hal, RawMotorCommand, RawServoCommand
        
        print("   Creating HAL...")
        hal = create_hal()
        
        if hal.mock_mode:
            print("   ⚠️ HAL in mock mode")
            return
        
        # Read sensors
        print("   Reading sensors...")
        sensors = hal.read_raw_sensors()
        print(f"   Grayscale: {sensors.i2c_grayscale}")
        print(f"   Distance: {sensors.gpio_ultrasonic_us/58:.1f}cm")
        print(f"   Battery: {sensors.analog_battery_raw}")
        
        # Test servos
        print("   Testing servo control via HAL...")
        hal.execute_servo_command(RawServoCommand(1500, 1500, 1500))  # Center
        time.sleep(0.5)
        hal.execute_servo_command(RawServoCommand(1200, 1500, 1500))  # Steer left
        time.sleep(0.5)
        hal.execute_servo_command(RawServoCommand(1800, 1500, 1500))  # Steer right
        time.sleep(0.5)
        hal.execute_servo_command(RawServoCommand(1500, 1500, 1500))  # Center
        
        # Test motors (if wheels off ground)
        if input("   Test motors? Wheels must be OFF ground (y/n): ").lower() == 'y':
            print("   Testing motors via HAL...")
            hal.execute_motor_command(RawMotorCommand(0.2, 0.2))  # Forward
            time.sleep(1)
            hal.execute_motor_command(RawMotorCommand(0.0, 0.0))  # Stop
        
        hal.cleanup()
        print("   ✅ Full system working")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("PICAR-X MOVEMENT TEST")
    print("=" * 60)
    print("This will test all servos and motors")
    print("Make sure:")
    print("  1. Robot is powered on")
    print("  2. Battery is charged")
    print("  3. Wheels are OFF THE GROUND for motor tests")
    print("=" * 60)
    
    # Run tests
    steering, pan, tilt = test_servo_center()
    test_steering_servo(steering)
    test_camera_servos(pan, tilt)
    test_motors()
    test_ultrasonic()
    test_grayscale()
    test_battery()
    test_full_system()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nIf all tests passed, your robot is ready!")
    print("Run: sudo python3 picarx_robot.py")

if __name__ == "__main__":
    main()