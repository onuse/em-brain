#!/usr/bin/env python3
"""
PiCar-X Hardware Limits

Based on SunFounder's PiCar-X implementation, these are the actual
hardware limits they use. We'll respect these for safety.

Reference: https://github.com/sunfounder/picar-x/blob/v2.0/picarx/picarx.py
"""

# Motor limits (from PiCar-X SDK)
MOTOR_SPEED_MIN = -100  # Full reverse
MOTOR_SPEED_MAX = 100   # Full forward

# Steering servo limits (degrees)
STEERING_MIN = -30  # Full left
STEERING_MAX = 30   # Full right
STEERING_CENTER = 0

# Camera servo limits (degrees)  
CAMERA_PAN_MIN = -90
CAMERA_PAN_MAX = 90
CAMERA_PAN_CENTER = 0

CAMERA_TILT_MIN = -35  # Look down
CAMERA_TILT_MAX = 65   # Look up
CAMERA_TILT_CENTER = 0

# Standard servo pulse ranges (microseconds)
# These are RC servo standards, not from PiCar SDK
SERVO_PULSE_MIN = 500   # Absolute minimum (rarely used)
SERVO_PULSE_MAX = 2500  # Absolute maximum (rarely used)
SERVO_PULSE_CENTER = 1500

# Safe servo ranges (what we'll actually use)
SERVO_SAFE_MIN = 1000   # -90 degrees typically
SERVO_SAFE_MAX = 2000   # +90 degrees typically

# Conversion factors
# Degrees to microseconds (approximate for standard servos)
# 180° range = 1000µs range → 5.56µs per degree
US_PER_DEGREE = 1000.0 / 180.0

def degrees_to_microseconds(degrees: float, center: float = SERVO_PULSE_CENTER) -> int:
    """Convert degrees to servo microseconds."""
    return int(center + (degrees * US_PER_DEGREE))

def steering_to_microseconds(steering_degrees: float) -> int:
    """Convert steering angle to servo pulse."""
    # Clamp to limits
    steering_degrees = max(STEERING_MIN, min(STEERING_MAX, steering_degrees))
    
    # For steering, we need to account for mechanical linkage
    # -30° to +30° steering maps to wider servo range
    # Assuming 1:1 for now (can be calibrated)
    return degrees_to_microseconds(steering_degrees)

def camera_pan_to_microseconds(pan_degrees: float) -> int:
    """Convert camera pan angle to servo pulse."""
    pan_degrees = max(CAMERA_PAN_MIN, min(CAMERA_PAN_MAX, pan_degrees))
    return degrees_to_microseconds(pan_degrees)

def camera_tilt_to_microseconds(tilt_degrees: float) -> int:
    """Convert camera tilt angle to servo pulse."""
    tilt_degrees = max(CAMERA_TILT_MIN, min(CAMERA_TILT_MAX, tilt_degrees))
    return degrees_to_microseconds(tilt_degrees)

def motor_speed_to_pwm_duty(speed: int) -> float:
    """
    Convert motor speed (-100 to 100) to PWM duty cycle (0.0 to 1.0).
    
    Note: The actual PWM to speed relationship is non-linear and
    depends on battery voltage, load, etc. The brain will learn this.
    """
    # Clamp to limits
    speed = max(MOTOR_SPEED_MIN, min(MOTOR_SPEED_MAX, speed))
    
    # Convert to duty cycle
    # This is a simple linear mapping, but reality is more complex
    # The brain will discover the actual response curve
    duty = abs(speed) / 100.0
    
    return duty


# Hardware configuration (verified from PiCar-X)
HARDWARE_CONFIG = {
    'motors': {
        'left_channel': 4,   # PCA9685 channel for left motor PWM
        'right_channel': 5,  # PCA9685 channel for right motor PWM
        'left_dir_pin': 23,  # GPIO23 for left motor direction
        'right_dir_pin': 24, # GPIO24 for right motor direction
    },
    'servos': {
        'steering_channel': 2,     # PCA9685 channel 2
        'camera_pan_channel': 0,   # PCA9685 channel 0
        'camera_tilt_channel': 1,  # PCA9685 channel 1
    },
    'sensors': {
        'ultrasonic_trig_pin': 27,  # GPIO27
        'ultrasonic_echo_pin': 22,  # GPIO22
        'grayscale_adc_channels': [0, 1, 2],  # ADC channels
        'battery_adc_channel': 6,   # Typical battery monitor channel
    }
}

if __name__ == "__main__":
    """Test conversions."""
    
    print("PiCar-X Hardware Limits")
    print("=" * 40)
    
    print("\nServo Limits:")
    print(f"  Steering: {STEERING_MIN}° to {STEERING_MAX}°")
    print(f"  Camera Pan: {CAMERA_PAN_MIN}° to {CAMERA_PAN_MAX}°")
    print(f"  Camera Tilt: {CAMERA_TILT_MIN}° to {CAMERA_TILT_MAX}°")
    
    print("\nServo Pulse Conversions:")
    print(f"  Steering 0°: {steering_to_microseconds(0)}µs")
    print(f"  Steering -30°: {steering_to_microseconds(-30)}µs")
    print(f"  Steering +30°: {steering_to_microseconds(30)}µs")
    
    print(f"\n  Camera Pan 0°: {camera_pan_to_microseconds(0)}µs")
    print(f"  Camera Pan -90°: {camera_pan_to_microseconds(-90)}µs")
    print(f"  Camera Pan +90°: {camera_pan_to_microseconds(90)}µs")
    
    print("\nMotor Speed to PWM:")
    for speed in [0, 25, 50, 75, 100]:
        duty = motor_speed_to_pwm_duty(speed)
        print(f"  Speed {speed:3d}: {duty:.2f} duty cycle")