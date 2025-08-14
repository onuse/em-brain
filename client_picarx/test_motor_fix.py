#!/usr/bin/env python3
"""
Motor Direction Fix Test

Test to identify and fix motor direction issues.
We'll systematically test each configuration.
"""

import sys
import time
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, 'robot_hat')

def test_current_setup():
    """Test current motor setup to understand the issue."""
    print("=" * 60)
    print("MOTOR DIRECTION FIX TEST")
    print("=" * 60)
    print("\n⚠️  WHEELS MUST BE OFF THE GROUND!")
    
    response = input("\nAre wheels OFF THE GROUND? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Test aborted")
        return
    
    from robot_hat import PWM, Pin
    
    # Setup
    left_pwm = PWM('P12')
    right_pwm = PWM('P13')
    left_pwm.freq(1000)
    right_pwm.freq(1000)
    
    left_dir = Pin("D4", mode=Pin.OUT)
    right_dir = Pin("D5", mode=Pin.OUT)
    
    print("\n" + "=" * 60)
    print("TEST 1: Verify Direction Pins Work")
    print("=" * 60)
    
    print("\n📍 LEFT motor - Dir=HIGH, PWM=30%")
    print("   Watch: Which direction does LEFT wheel spin?")
    input("   Press Enter to start (10 seconds)...")
    
    left_dir.value(1)
    left_pwm.pulse_width_percent(30)
    right_pwm.pulse_width_percent(0)
    
    for i in range(10, 0, -1):
        print(f"   {i} seconds...", end='\r')
        time.sleep(1)
    
    left_pwm.pulse_width_percent(0)
    direction_high = input("\n   Did LEFT wheel go FORWARD or BACKWARD? (f/b): ")
    
    print("\n📍 LEFT motor - Dir=LOW, PWM=30%")
    print("   Watch: Which direction does LEFT wheel spin NOW?")
    input("   Press Enter to start (10 seconds)...")
    
    left_dir.value(0)
    left_pwm.pulse_width_percent(30)
    
    for i in range(10, 0, -1):
        print(f"   {i} seconds...", end='\r')
        time.sleep(1)
    
    left_pwm.pulse_width_percent(0)
    direction_low = input("\n   Did LEFT wheel go FORWARD or BACKWARD? (f/b): ")
    
    # Check if direction pin actually changes anything
    if direction_high == direction_low:
        print("\n⚠️ PROBLEM: Direction pin doesn't affect LEFT motor!")
        print("   Motor always goes same direction regardless of pin.")
        left_dir_works = False
    else:
        print("\n✅ LEFT direction pin works!")
        left_dir_works = True
        if direction_high == 'f':
            print("   HIGH = Forward, LOW = Backward")
            left_inverted = False
        else:
            print("   HIGH = Backward, LOW = Forward (inverted)")
            left_inverted = True
    
    # Test RIGHT motor
    print("\n" + "=" * 60)
    print("TEST 2: RIGHT Motor Direction")
    print("=" * 60)
    
    print("\n📍 RIGHT motor - Dir=HIGH, PWM=30%")
    print("   Watch: Which direction does RIGHT wheel spin?")
    input("   Press Enter to start (10 seconds)...")
    
    right_dir.value(1)
    right_pwm.pulse_width_percent(30)
    left_pwm.pulse_width_percent(0)
    
    for i in range(10, 0, -1):
        print(f"   {i} seconds...", end='\r')
        time.sleep(1)
    
    right_pwm.pulse_width_percent(0)
    direction_high = input("\n   Did RIGHT wheel go FORWARD or BACKWARD? (f/b): ")
    
    print("\n📍 RIGHT motor - Dir=LOW, PWM=30%")
    print("   Watch: Which direction does RIGHT wheel spin NOW?")
    input("   Press Enter to start (10 seconds)...")
    
    right_dir.value(0)
    right_pwm.pulse_width_percent(30)
    
    for i in range(10, 0, -1):
        print(f"   {i} seconds...", end='\r')
        time.sleep(1)
    
    right_pwm.pulse_width_percent(0)
    direction_low = input("\n   Did RIGHT wheel go FORWARD or BACKWARD? (f/b): ")
    
    # Check if direction pin actually changes anything
    if direction_high == direction_low:
        print("\n⚠️ PROBLEM: Direction pin doesn't affect RIGHT motor!")
        print("   Motor always goes same direction regardless of pin.")
        right_dir_works = False
    else:
        print("\n✅ RIGHT direction pin works!")
        right_dir_works = True
        if direction_high == 'f':
            print("   HIGH = Forward, LOW = Backward")
            right_inverted = False
        else:
            print("   HIGH = Backward, LOW = Forward (inverted)")
            right_inverted = True
    
    # Stop everything
    left_pwm.pulse_width_percent(0)
    right_pwm.pulse_width_percent(0)
    
    # Generate fix
    print("\n" + "=" * 60)
    print("DIAGNOSIS & FIX")
    print("=" * 60)
    
    if not left_dir_works and not right_dir_works:
        print("\n❌ Direction pins not working at all!")
        print("\nPossible causes:")
        print("1. Motors might be single-direction only")
        print("2. Direction pins might not be connected")
        print("3. Motor driver might be in different mode")
        print("\nTry using Motor class instead:")
        print("""
# In raw_robot_hat_hal.py __init__:
from robot_hat import Motor
self.left_motor = Motor(PWM('P12'), Pin('D4'))
self.right_motor = Motor(PWM('P13'), Pin('D5'))

# In execute_motor_command:
self.left_motor.speed(command.left_pwm_duty * 100)
self.right_motor.speed(command.right_pwm_duty * 100)
""")
    else:
        print("\n✅ Direction pins are working!")
        
        if left_dir_works and left_inverted:
            print("\n⚠️ LEFT motor is wired reversed")
        if right_dir_works and right_inverted:
            print("\n⚠️ RIGHT motor is wired reversed")
        
        print("\nUpdate raw_robot_hat_hal.py execute_motor_command:")
        print("""
# Set motor directions (with inversions if needed)
left_forward = command.left_pwm_duty >= 0
right_forward = command.right_pwm_duty >= 0
""")
        
        if left_dir_works and left_inverted:
            print("# LEFT motor is inverted")
            print("self.left_dir_pin.value(0 if left_forward else 1)")
        else:
            print("self.left_dir_pin.value(1 if left_forward else 0)")
        
        if right_dir_works and right_inverted:
            print("# RIGHT motor is inverted")
            print("self.right_dir_pin.value(0 if right_forward else 1)")
        else:
            print("self.right_dir_pin.value(1 if right_forward else 0)")

def test_motor_class():
    """Test using Motor class as alternative."""
    print("\n" + "=" * 60)
    print("TEST: Motor Class")
    print("=" * 60)
    
    response = input("\nTest Motor class? (yes/no): ")
    if response.lower() != 'yes':
        return
    
    from robot_hat import Motor, PWM, Pin
    
    print("\nCreating Motor objects...")
    left_motor = Motor(PWM('P12'), Pin('D4'))
    right_motor = Motor(PWM('P13'), Pin('D5'))
    
    print("\n📍 LEFT motor forward (speed=30)")
    input("   Press Enter to start (5 seconds)...")
    left_motor.speed(30)
    right_motor.speed(0)
    time.sleep(5)
    
    print("\n📍 LEFT motor reverse (speed=-30)")
    input("   Press Enter to start (5 seconds)...")
    left_motor.speed(-30)
    time.sleep(5)
    
    left_motor.speed(0)
    
    works = input("\nDid LEFT motor change direction? (yes/no): ")
    
    if works.lower() == 'yes':
        print("✅ Motor class works! Use this approach.")
    else:
        print("❌ Motor class doesn't fix the issue.")
        print("   May need to check motor wiring or driver mode.")
    
    # Stop
    left_motor.speed(0)
    right_motor.speed(0)

def main():
    test_current_setup()
    test_motor_class()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nImplement the suggested fix in raw_robot_hat_hal.py")
    print("Then test with: python3 test_movement.py")

if __name__ == "__main__":
    main()