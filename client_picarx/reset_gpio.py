#!/usr/bin/env python3
"""
Reset GPIO pins to clean state.
Run this if you get 'GPIO already in use' errors.
"""

import sys

try:
    # Try gpiozero cleanup
    try:
        from gpiozero import Device
        from gpiozero.pins.mock import MockFactory
        Device.pin_factory = MockFactory()
        print("Reset gpiozero to mock mode")
    except:
        pass
    
    # Try RPi.GPIO cleanup
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.cleanup()
        print("Cleaned up RPi.GPIO")
    except:
        pass
    
    # Try robot_hat cleanup
    try:
        sys.path.insert(0, 'robot_hat')
        from robot_hat import Pin
        # Reset common pins
        for pin_name in ['D2', 'D3', 'D4', 'D5']:
            try:
                pin = Pin(pin_name)
                pin.close()
            except:
                pass
        print("Reset robot_hat pins")
    except:
        pass
    
    print("\n✅ GPIO reset complete")
    print("You can now run your robot program.")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTry rebooting if GPIO issues persist:")
    print("  sudo reboot")