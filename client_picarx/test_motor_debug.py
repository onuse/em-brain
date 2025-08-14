#!/usr/bin/env python3
"""
Motor Debug Test - Comprehensive

Identifies and fixes motor control issues:
1. Direction pin not working
2. Motors reversed
3. PWM/Direction channel confusion
"""

import sys
import time
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, 'robot_hat')

def test_basic_pins():
    """Test if pins are working at all."""
    print("\n" + "=" * 60)
    print("BASIC PIN TEST")
    print("=" * 60)
    
    from robot_hat import Pin, PWM
    
    print("\n1. Testing direction pins can be toggled...")
    try:
        left_dir = Pin("D4", mode=Pin.OUT)
        right_dir = Pin("D5", mode=Pin.OUT)
        
        print("   Setting D4 (left dir) HIGH...")
        left_dir.value(1)
        time.sleep(1)
        print("   Setting D4 (left dir) LOW...")
        left_dir.value(0)
        time.sleep(1)
        
        print("   Setting D5 (right dir) HIGH...")
        right_dir.value(1)
        time.sleep(1)
        print("   Setting D5 (right dir) LOW...")
        right_dir.value(0)
        time.sleep(1)
        
        print("   ✅ Direction pins working")
    except Exception as e:
        print(f"   ❌ Direction pin error: {e}")
        return False
    
    print("\n2. Testing PWM channels...")
    try:
        left_pwm = PWM('P12')
        right_pwm = PWM('P13')
        left_pwm.freq(1000)
        right_pwm.freq(1000)
        
        print("   Setting P12 (left PWM) to 0%...")
        left_pwm.pulse_width_percent(0)
        time.sleep(1)
        print("   Setting P12 (left PWM) to 50%...")
        left_pwm.pulse_width_percent(50)
        time.sleep(1)
        print("   Setting P12 (left PWM) to 0%...")
        left_pwm.pulse_width_percent(0)
        
        print("   ✅ PWM channels working")
    except Exception as e:
        print(f"   ❌ PWM error: {e}")
        return False
    
    return True

def test_motor_control_combinations():
    """Test all combinations to find what actually controls the motors."""
    print("\n" + "=" * 60)
    print("MOTOR CONTROL COMBINATION TEST")
    print("=" * 60)
    print("\n⚠️  Wheels MUST be OFF THE GROUND!")
    
    response = input("\nAre wheels OFF THE GROUND? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Test aborted")
        return
    
    from robot_hat import Pin, PWM
    
    # Setup
    left_pwm = PWM('P12')
    right_pwm = PWM('P13')
    left_pwm.freq(1000)
    right_pwm.freq(1000)
    
    left_dir = Pin("D4", mode=Pin.OUT)
    right_dir = Pin("D5", mode=Pin.OUT)
    
    # Stop everything
    left_pwm.pulse_width_percent(0)
    right_pwm.pulse_width_percent(0)
    left_dir.value(0)
    right_dir.value(0)
    
    print("\n" + "-" * 60)
    print("TEST: Find what actually controls motor direction")
    print("-" * 60)
    
    tests = [
        # Test if direction pins actually do anything
        ("LEFT motor: PWM=30%, Dir=HIGH", 
         lambda: (left_pwm.pulse_width_percent(30), left_dir.value(1), right_pwm.pulse_width_percent(0))),
        
        ("LEFT motor: PWM=30%, Dir=LOW", 
         lambda: (left_pwm.pulse_width_percent(30), left_dir.value(0), right_pwm.pulse_width_percent(0))),
        
        ("LEFT motor: PWM=0%, Dir stays LOW", 
         lambda: (left_pwm.pulse_width_percent(0),)),
         
        # Maybe direction is controlled by PWM polarity?
        ("LEFT motor: Try negative PWM workaround (set dir=1, pwm=30)", 
         lambda: (left_dir.value(1), left_pwm.pulse_width_percent(30))),
         
        ("LEFT motor: Try opposite (set dir=0, pwm=30)", 
         lambda: (left_dir.value(0), left_pwm.pulse_width_percent(30))),
         
        ("Stop all",
         lambda: (left_pwm.pulse_width_percent(0), right_pwm.pulse_width_percent(0))),
         
        # Test right motor
        ("RIGHT motor: PWM=30%, Dir=HIGH", 
         lambda: (right_pwm.pulse_width_percent(30), right_dir.value(1), left_pwm.pulse_width_percent(0))),
        
        ("RIGHT motor: PWM=30%, Dir=LOW", 
         lambda: (right_pwm.pulse_width_percent(30), right_dir.value(0), left_pwm.pulse_width_percent(0))),
        
        ("Stop all",
         lambda: (left_pwm.pulse_width_percent(0), right_pwm.pulse_width_percent(0))),
    ]
    
    for desc, action in tests:
        print(f"\n📍 {desc}")
        print("   Watch: Does direction change? Which way does it spin?")
        input("   Press Enter to start (5 seconds)...")
        
        action()
        
        for i in range(5, 0, -1):
            print(f"   {i} seconds...", end='\r')
            time.sleep(1)
        print()
    
    # Stop everything
    left_pwm.pulse_width_percent(0)
    right_pwm.pulse_width_percent(0)
    left_dir.value(0)
    right_dir.value(0)

def test_alternative_control():
    """Test if we need different pins or different control method."""
    print("\n" + "=" * 60)
    print("ALTERNATIVE CONTROL TEST")
    print("=" * 60)
    
    response = input("\nTest alternative pins? (yes/no): ")
    if response.lower() != 'yes':
        return
    
    from robot_hat import Pin, PWM
    
    print("\nMaybe the motors use different pins...")
    print("Common alternatives:")
    print("  - Direction might be on P14/P15 instead of D4/D5")
    print("  - PWM might need to go negative for reverse")
    print("  - Might need Motor class instead of raw PWM")
    
    # Try using the Motor class if it exists
    try:
        from robot_hat import Motor
        print("\n✅ Motor class exists! Testing...")
        
        print("Creating Motor objects...")
        left_motor = Motor(12, 4)  # PWM channel 12, direction pin 4?
        right_motor = Motor(13, 5)  # PWM channel 13, direction pin 5?
        
        print("Testing left motor forward...")
        left_motor.speed(50)
        time.sleep(3)
        
        print("Testing left motor reverse...")
        left_motor.speed(-50)
        time.sleep(3)
        
        left_motor.speed(0)
        print("✅ Motor class might be the solution!")
        
    except ImportError:
        print("\n❌ No Motor class available")
    except Exception as e:
        print(f"\n❌ Motor class error: {e}")
    
    # Try different pin combinations
    print("\nTrying different pin combinations...")
    alt_pins = [
        ("P4", "P5"),    # PWM on P4/P5
        ("P14", "P15"),  # PWM on P14/P15
        ("D12", "D13"),  # Direction on D12/D13
    ]
    
    for p1, p2 in alt_pins:
        try:
            print(f"\nTrying {p1}/{p2} as direction pins...")
            pin1 = Pin(p1, mode=Pin.OUT)
            pin2 = Pin(p2, mode=Pin.OUT)
            pin1.value(1)
            pin2.value(1)
            time.sleep(1)
            pin1.value(0)
            pin2.value(0)
            print(f"  {p1}/{p2} accessible")
        except Exception as e:
            print(f"  {p1}/{p2} not available: {e}")

def suggest_fixes():
    """Suggest fixes based on observations."""
    print("\n" + "=" * 60)
    print("DIAGNOSIS & FIXES")
    print("=" * 60)
    
    print("\n🔍 Based on your observations:")
    
    print("\n1. If direction pins don't change motor direction:")
    print("   → The PiCar-X might use a different control method")
    print("   → Try: Using Motor class instead of raw PWM+Pin")
    print("   → Or: Direction might be built into the motor driver")
    
    print("\n2. If motors always spin the same direction:")
    print("   → The motor driver might be in 'PWM only' mode")
    print("   → Direction might be controlled by:")
    print("     - Two PWM channels per motor (forward/reverse)")
    print("     - Or Motor class with signed speed values")
    
    print("\n3. If motors spin opposite directions:")
    print("   → One motor is physically reversed")
    print("   → Fix: Swap motor wires or invert in software")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED FIX")
    print("=" * 60)
    
    print("\nEdit raw_robot_hat_hal.py and try this approach:")
    print("""
# Option 1: Try Motor class (if direction pins don't work)
from robot_hat import Motor
self.left_motor = Motor(12, 4)   # PWM channel, dir pin
self.right_motor = Motor(13, 5)

# In execute_motor_command:
self.left_motor.speed(command.left_pwm_duty * 100)  # -100 to 100
self.right_motor.speed(command.right_pwm_duty * 100)

# Option 2: Use two PWM channels per motor
# Some drivers use PWM1 for forward, PWM2 for reverse
self.left_forward_pwm = PWM('P12')
self.left_reverse_pwm = PWM('P14')  # Find the right channel

# Option 3: Invert one motor if they're opposite
# In execute_motor_command:
left_duty = command.left_pwm_duty
right_duty = -command.right_pwm_duty  # Invert right motor
""")

def main():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("COMPREHENSIVE MOTOR DEBUG")
    print("=" * 60)
    
    if not test_basic_pins():
        print("\n❌ Basic pin test failed. Check connections.")
        return
    
    test_motor_control_combinations()
    test_alternative_control()
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    print("\nBased on what you observed, implement the suggested fix")
    print("in raw_robot_hat_hal.py and test again.")

if __name__ == "__main__":
    main()