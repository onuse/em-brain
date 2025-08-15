#!/usr/bin/env python3
"""
Clean up GPIO pins before starting robot.
Run this if you get "pin already in use" errors.
"""

import sys
import os

print("Cleaning up GPIO pins...")

# Method 1: Try gpiozero cleanup
try:
    from gpiozero import Device
    from gpiozero.pins.rpigpio import RPiGPIOFactory
    
    # Reset the default pin factory
    Device.pin_factory = RPiGPIOFactory()
    
    # Clean up all pins
    if Device.pin_factory:
        Device.pin_factory.close()
        Device.pin_factory = None
    
    print("✅ Cleaned up gpiozero pins")
except Exception as e:
    print(f"⚠️ gpiozero cleanup: {e}")

# Method 2: Try RPi.GPIO cleanup
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.cleanup()
    print("✅ Cleaned up RPi.GPIO pins")
except Exception as e:
    print(f"⚠️ RPi.GPIO cleanup: {e}")

# Method 3: Force cleanup specific pins
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    
    # Clean up specific pins used by robot
    pins_to_clean = [
        23,  # Ultrasonic trigger
        24,  # Ultrasonic echo  
        4,   # Left motor direction
        5,   # Right motor direction
    ]
    
    for pin in pins_to_clean:
        try:
            GPIO.setup(pin, GPIO.IN)
            GPIO.cleanup(pin)
        except:
            pass
    
    print(f"✅ Cleaned up pins: {pins_to_clean}")
except Exception as e:
    print(f"⚠️ Specific pin cleanup: {e}")

# Method 4: Kill any running Python processes that might be holding pins
import subprocess
import os

try:
    # Get current process ID
    current_pid = os.getpid()
    
    # Find other Python processes
    result = subprocess.run(['pgrep', '-f', 'python.*picarx'], capture_output=True, text=True)
    if result.stdout:
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid and int(pid) != current_pid:
                try:
                    subprocess.run(['kill', pid])
                    print(f"✅ Killed process {pid}")
                except:
                    pass
except Exception as e:
    print(f"⚠️ Process cleanup: {e}")

print("\n✨ GPIO cleanup complete!")
print("You can now run the robot program.")