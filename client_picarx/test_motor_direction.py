#!/usr/bin/env python3
"""
Motor Direction Debug Test

Helps identify if motors are wired correctly or if one is reversed.
Tests each motor individually with clear visual feedback.
"""

import sys
import time
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, 'robot_hat')

def test_motor_wiring():
    """Test motor wiring and direction."""
    print("=" * 60)
    print("MOTOR DIRECTION DEBUG TEST")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Wheels MUST be OFF THE GROUND!")
    print("\nThis test will help identify if motors are wired correctly.")
    print("Watch which wheel spins and which direction.\n")
    
    response = input("Are wheels OFF THE GROUND? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Test aborted for safety")
        return
    
    try:
        from robot_hat import PWM, Pin
        
        print("\nSetting up motor control...")
        print("-" * 40)
        
        # Motor PWM channels
        left_motor = PWM('P12')
        right_motor = PWM('P13')
        left_motor.freq(1000)
        right_motor.freq(1000)
        
        # Direction pins
        left_dir = Pin("D4", mode=Pin.OUT)
        right_dir = Pin("D5", mode=Pin.OUT)
        
        print("✅ Motor control initialized")
        print("   Left motor: PWM P12, Direction D4")
        print("   Right motor: PWM P13, Direction D5")
        
        # Stop everything first
        left_motor.pulse_width_percent(0)
        right_motor.pulse_width_percent(0)
        left_dir.value(0)
        right_dir.value(0)
        
        print("\n" + "=" * 60)
        print("TEST 1: LEFT MOTOR ONLY")
        print("=" * 60)
        
        print("\n📍 LEFT motor - Direction pin HIGH (should be FORWARD)")
        print("   If wheel goes BACKWARD, left motor is reversed")
        input("   Press Enter to start (10 seconds)...")
        left_dir.value(1)  # HIGH = should be forward
        left_motor.pulse_width_percent(30)
        right_motor.pulse_width_percent(0)
        
        for i in range(10, 0, -1):
            print(f"   {i} seconds remaining...", end='\r')
            time.sleep(1)
        
        left_motor.pulse_width_percent(0)
        print("\n   Stopped")
        
        print("\n📍 LEFT motor - Direction pin LOW (should be REVERSE)")
        print("   If wheel goes FORWARD, left motor is reversed")
        input("   Press Enter to start (10 seconds)...")
        left_dir.value(0)  # LOW = should be reverse
        left_motor.pulse_width_percent(30)
        
        for i in range(10, 0, -1):
            print(f"   {i} seconds remaining...", end='\r')
            time.sleep(1)
        
        left_motor.pulse_width_percent(0)
        print("\n   Stopped")
        
        print("\n" + "=" * 60)
        print("TEST 2: RIGHT MOTOR ONLY")
        print("=" * 60)
        
        print("\n📍 RIGHT motor - Direction pin HIGH (should be FORWARD)")
        print("   If wheel goes BACKWARD, right motor is reversed")
        input("   Press Enter to start (10 seconds)...")
        right_dir.value(1)  # HIGH = should be forward
        right_motor.pulse_width_percent(30)
        left_motor.pulse_width_percent(0)
        
        for i in range(10, 0, -1):
            print(f"   {i} seconds remaining...", end='\r')
            time.sleep(1)
        
        right_motor.pulse_width_percent(0)
        print("\n   Stopped")
        
        print("\n📍 RIGHT motor - Direction pin LOW (should be REVERSE)")
        print("   If wheel goes FORWARD, right motor is reversed")
        input("   Press Enter to start (10 seconds)...")
        right_dir.value(0)  # LOW = should be reverse
        right_motor.pulse_width_percent(30)
        
        for i in range(10, 0, -1):
            print(f"   {i} seconds remaining...", end='\r')
            time.sleep(1)
        
        right_motor.pulse_width_percent(0)
        print("\n   Stopped")
        
        # Stop everything
        left_motor.pulse_width_percent(0)
        right_motor.pulse_width_percent(0)
        
        print("\n" + "=" * 60)
        print("DIAGNOSIS")
        print("=" * 60)
        
        print("\n🔍 What did you observe?")
        print("\n1. Both motors correct:")
        print("   - HIGH = Forward, LOW = Reverse for both")
        print("   → No changes needed!")
        
        print("\n2. LEFT motor reversed:")
        print("   - HIGH = Reverse, LOW = Forward")
        print("   → Need to swap left motor wires OR")
        print("   → Invert left direction in software")
        
        print("\n3. RIGHT motor reversed:")
        print("   - HIGH = Reverse, LOW = Forward")
        print("   → Need to swap right motor wires OR")
        print("   → Invert right direction in software")
        
        print("\n4. BOTH motors reversed:")
        print("   - Both go opposite of expected")
        print("   → Check if looking at robot from front/back")
        print("   → Or swap both motor connections")
        
        print("\n" + "=" * 60)
        print("SOFTWARE FIX")
        print("=" * 60)
        
        print("\nIf a motor is reversed, you can fix in software:")
        print("\n1. Edit raw_robot_hat_hal.py")
        print("2. In execute_motor_command(), change:")
        print("   self.left_dir_pin.value(1 if command.left_pwm_duty >= 0 else 0)")
        print("   To:")
        print("   self.left_dir_pin.value(0 if command.left_pwm_duty >= 0 else 1)")
        print("\nOr swap the motor wires physically.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Emergency stop
        try:
            left_motor.pulse_width_percent(0)
            right_motor.pulse_width_percent(0)
        except:
            pass

if __name__ == "__main__":
    test_motor_wiring()