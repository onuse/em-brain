#!/usr/bin/env python3
"""
Trace through PiCar-X motor control to understand the logic.
"""

import sys
sys.path.insert(0, 'robot_hat')

def trace_picarx():
    """Trace what PiCar-X does for forward/backward."""
    print("=" * 60)
    print("PICAR-X MOTOR LOGIC TRACE")
    print("=" * 60)
    
    print("\nPiCar-X motor_pins: ['D4', 'D5', 'P13', 'P12']")
    print("  motor_pins[0] = D4 = left_rear_dir_pin")
    print("  motor_pins[1] = D5 = right_rear_dir_pin") 
    print("  motor_pins[2] = P13 = left_rear_pwm_pin")
    print("  motor_pins[3] = P12 = right_rear_pwm_pin")
    
    print("\nSo the mapping is:")
    print("  LEFT motor:  PWM=P13, DIR=D4")
    print("  RIGHT motor: PWM=P12, DIR=D5")
    
    print("\n" + "=" * 60)
    print("FORWARD(speed=50) calls:")
    print("=" * 60)
    
    print("\nforward(50) ->")
    print("  set_motor_speed(1, 50)    # Motor 1")
    print("  set_motor_speed(2, -50)   # Motor 2 NEGATIVE!")
    
    print("\nset_motor_speed(1, 50) ->")
    print("  motor = 0 (motor-1)")
    print("  speed = 50 >= 0, so direction = 1")
    print("  motor_direction_pins[0].low()   # D4 LOW")
    print("  motor_speed_pins[0].pwm(75)     # P13 = 75%")
    
    print("\nset_motor_speed(2, -50) ->")
    print("  motor = 1 (motor-1)")
    print("  speed = -50 < 0, so direction = -1")
    print("  motor_direction_pins[1].high()  # D5 HIGH") 
    print("  motor_speed_pins[1].pwm(75)     # P12 = 75%")
    
    print("\n" + "=" * 60)
    print("RESULT for FORWARD:")
    print("=" * 60)
    print("  LEFT motor (P13/D4):  DIR=LOW, PWM=75%")
    print("  RIGHT motor (P12/D5): DIR=HIGH, PWM=75%")
    
    print("\n" + "=" * 60)
    print("BACKWARD(speed=50) calls:")
    print("=" * 60)
    
    print("\nbackward(50) ->")
    print("  set_motor_speed(1, -50)   # Motor 1 NEGATIVE")
    print("  set_motor_speed(2, 50)    # Motor 2")
    
    print("\nset_motor_speed(1, -50) ->")
    print("  motor = 0")
    print("  speed = -50 < 0, so direction = -1")
    print("  motor_direction_pins[0].high()  # D4 HIGH")
    print("  motor_speed_pins[0].pwm(75)     # P13 = 75%")
    
    print("\nset_motor_speed(2, 50) ->")
    print("  motor = 1")
    print("  speed = 50 >= 0, so direction = 1")
    print("  motor_direction_pins[1].low()   # D5 LOW")
    print("  motor_speed_pins[1].pwm(75)     # P12 = 75%")
    
    print("\n" + "=" * 60)
    print("RESULT for BACKWARD:")
    print("=" * 60)
    print("  LEFT motor (P13/D4):  DIR=HIGH, PWM=75%")
    print("  RIGHT motor (P12/D5): DIR=LOW, PWM=75%")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("1. Motors are wired opposite each other")
    print("2. For robot FORWARD:")
    print("   - LEFT motor goes 'forward' (D4=LOW)")
    print("   - RIGHT motor goes 'backward' (D5=HIGH)")
    print("3. For robot BACKWARD:")
    print("   - LEFT motor goes 'backward' (D4=HIGH)")
    print("   - RIGHT motor goes 'forward' (D5=LOW)")
    print("\nThis is differential drive!")

if __name__ == "__main__":
    trace_picarx()