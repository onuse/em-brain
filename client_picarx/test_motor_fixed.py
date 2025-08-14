#!/usr/bin/env python3
"""
Test Motor Control with Fixed Direction Logic

Tests the corrected motor control based on PiCar-X implementation.
"""

import sys
import time
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def test_fixed_motors():
    """Test motors with corrected logic."""
    print("=" * 60)
    print("MOTOR TEST - FIXED LOGIC")
    print("=" * 60)
    print("\nUsing PiCar-X motor logic:")
    print("  - D5/P12 = LEFT motor")
    print("  - D4/P13 = RIGHT motor")
    print("  - Direction: LOW=forward, HIGH=reverse")
    print("  - Right motor inverted for differential drive")
    print("\n⚠️  WHEELS MUST BE OFF THE GROUND!")
    
    response = input("\nAre wheels OFF THE GROUND? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Test aborted")
        return
    
    from src.hardware.raw_robot_hat_hal import create_hal, RawMotorCommand
    
    print("\nCreating HAL with fixed motor logic...")
    hal = create_hal()
    
    if hal.mock_mode:
        print("⚠️ HAL in mock mode")
        return
    
    tests = [
        ("Stop", 0.0, 0.0, 2),
        ("Forward slow", 0.2, 0.2, 5),
        ("Forward medium", 0.4, 0.4, 5),
        ("Stop", 0.0, 0.0, 2),
        ("Backward slow", -0.2, -0.2, 5),
        ("Stop", 0.0, 0.0, 2),
        ("Turn left (left slower)", 0.1, 0.3, 5),
        ("Turn right (right slower)", 0.3, 0.1, 5),
        ("Stop", 0.0, 0.0, 2),
        ("Spin left", -0.2, 0.2, 5),
        ("Spin right", 0.2, -0.2, 5),
        ("Stop", 0.0, 0.0, 2),
    ]
    
    print("\nStarting motor tests...")
    print("-" * 40)
    
    for desc, left, right, duration in tests:
        print(f"\n📍 {desc}")
        print(f"   Left: {left:+.1f}, Right: {right:+.1f}")
        if desc != "Stop":
            print("   Watch wheel directions!")
        
        # Execute command
        hal.execute_motor_command(RawMotorCommand(left, right))
        
        # Wait with countdown
        for i in range(duration, 0, -1):
            print(f"   {i} seconds...", end='\r')
            time.sleep(1)
        print()
    
    # Emergency stop
    hal.emergency_stop()
    hal.cleanup()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    print("\n🔍 Did the robot:")
    print("  1. Go FORWARD when both motors positive?")
    print("  2. Go BACKWARD when both motors negative?")
    print("  3. Turn correctly?")
    print("\nIf YES to all: Motors are fixed! ✅")
    print("If NO: May need to adjust motor inversions.")

def test_with_picarx_class():
    """Test using actual PiCar-X class for comparison."""
    print("\n" + "=" * 60)
    print("COMPARISON TEST - PiCar-X Class")
    print("=" * 60)
    
    response = input("\nTest with PiCar-X class? (yes/no): ")
    if response.lower() != 'yes':
        return
    
    try:
        # Import from pasted file
        sys.path.insert(0, 'robot_hat')
        from picarx import Picarx
        
        print("\nUsing PiCar-X class...")
        px = Picarx()
        
        print("\nForward test (5 seconds)...")
        px.forward(50)
        time.sleep(5)
        
        print("Stop (2 seconds)...")
        px.stop()
        time.sleep(2)
        
        print("Backward test (5 seconds)...")
        px.backward(50)
        time.sleep(5)
        
        px.stop()
        print("\n✅ PiCar-X class test complete")
        
    except Exception as e:
        print(f"❌ PiCar-X test failed: {e}")

def main():
    test_fixed_motors()
    test_with_picarx_class()

if __name__ == "__main__":
    main()