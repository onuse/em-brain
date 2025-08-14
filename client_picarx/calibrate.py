#!/usr/bin/env python3
"""
PiCar-X Servo Calibration Tool

Run this to calibrate servo center positions.
Adjust offsets in config/servo_calibration.json until:
- Steering is straight at 0°
- Camera points forward and level at 0°
"""

import json
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.hardware.bare_metal_hal import create_hal
from src.hardware.picarx_hardware_limits import (
    RawServoCommand, SERVO_PULSE_CENTER
)


def calibrate_servos():
    """Interactive servo calibration."""
    
    print("=" * 60)
    print("PICAR-X SERVO CALIBRATION")
    print("=" * 60)
    
    # Load current calibration
    config_path = Path("config/servo_calibration.json")
    with open(config_path, 'r') as f:
        calibration = json.load(f)
    
    print("\nCurrent offsets:")
    print(f"  Steering: {calibration['steering']['center_offset']}°")
    print(f"  Camera Pan: {calibration['camera_pan']['center_offset']}°")
    print(f"  Camera Tilt: {calibration['camera_tilt']['center_offset']}°")
    
    # Initialize hardware
    hal = create_hal()
    
    print("\n📐 Setting all servos to 0° position...")
    print("   (Wheels should be straight, camera forward and level)")
    
    # Set all to center
    cmd = RawServoCommand(
        steering_us=SERVO_PULSE_CENTER,
        camera_pan_us=SERVO_PULSE_CENTER,
        camera_tilt_us=SERVO_PULSE_CENTER
    )
    hal.execute_servo_command(cmd)
    
    print("\n⚙️ Check physical positions:")
    print("1. STEERING: Are the wheels pointing straight?")
    print("2. CAMERA PAN: Is the camera pointing forward?")
    print("3. CAMERA TILT: Is the camera level?")
    
    print("\n📝 To calibrate:")
    print("1. Measure how far off each servo is")
    print("2. Edit config/servo_calibration.json:")
    print("   - If steering points LEFT, set positive offset")
    print("   - If steering points RIGHT, set negative offset")
    print("   - Same for camera pan/tilt")
    print("3. Run this script again to verify")
    
    input("\nPress Enter to test calibration with movements...")
    
    # Test movements
    print("\n🔄 Testing movements...")
    
    movements = [
        ("Steering LEFT", RawServoCommand(steering_us=1333, camera_pan_us=1500, camera_tilt_us=1500)),
        ("Steering CENTER", RawServoCommand(steering_us=1500, camera_pan_us=1500, camera_tilt_us=1500)),
        ("Steering RIGHT", RawServoCommand(steering_us=1667, camera_pan_us=1500, camera_tilt_us=1500)),
        ("Camera PAN LEFT", RawServoCommand(steering_us=1500, camera_pan_us=1000, camera_tilt_us=1500)),
        ("Camera PAN CENTER", RawServoCommand(steering_us=1500, camera_pan_us=1500, camera_tilt_us=1500)),
        ("Camera PAN RIGHT", RawServoCommand(steering_us=1500, camera_pan_us=2000, camera_tilt_us=1500)),
        ("Camera TILT DOWN", RawServoCommand(steering_us=1500, camera_pan_us=1500, camera_tilt_us=1300)),
        ("Camera TILT CENTER", RawServoCommand(steering_us=1500, camera_pan_us=1500, camera_tilt_us=1500)),
        ("Camera TILT UP", RawServoCommand(steering_us=1500, camera_pan_us=1500, camera_tilt_us=1700)),
        ("All CENTER", RawServoCommand(steering_us=1500, camera_pan_us=1500, camera_tilt_us=1500)),
    ]
    
    for name, cmd in movements:
        print(f"  {name}...")
        hal.execute_servo_command(cmd)
        time.sleep(1)
    
    print("\n✅ Calibration test complete!")
    print("\nFinal calibration values in config/servo_calibration.json:")
    
    # Reload to show any manual edits
    with open(config_path, 'r') as f:
        calibration = json.load(f)
    
    print(f"  Steering offset: {calibration['steering']['center_offset']}°")
    print(f"  Camera pan offset: {calibration['camera_pan']['center_offset']}°")
    print(f"  Camera tilt offset: {calibration['camera_tilt']['center_offset']}°")
    
    # Clean shutdown
    hal.emergency_stop()


def quick_center():
    """Just center all servos for manual measurement."""
    hal = create_hal()
    
    print("Setting all servos to 1500µs (0°)...")
    cmd = RawServoCommand(
        steering_us=1500,
        camera_pan_us=1500,
        camera_tilt_us=1500
    )
    hal.execute_servo_command(cmd)
    
    print("Servos centered. Measure physical positions:")
    print("  - Steering should be straight")
    print("  - Camera should point forward")
    print("  - Camera should be level")
    
    input("\nPress Enter to exit...")
    hal.emergency_stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate PiCar-X servos')
    parser.add_argument('--quick', action='store_true',
                        help='Just center servos for measurement')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_center()
    else:
        calibrate_servos()