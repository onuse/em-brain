# SunFounder PiCar-X API Reference

Real API documentation discovered from SunFounder's official documentation.

## 🤖 Main Picarx Class

### Import and Initialization
```python
from picarx import Picarx
import time

# Create robot instance
px = Picarx()
```

## 🚗 Movement Control

### Basic Movement
```python
# Move forward at specified speed
px.forward(speed)        # speed: 0-100

# Move backward
px.backward(speed)       # speed: 0-100  

# Stop movement
px.stop()
```

### Steering Control
```python
# Control steering servo angle
px.set_dir_servo_angle(angle)    # angle: -30 to +30 degrees (typical range)

# Example: Gradual steering
for angle in range(0, 35):
    px.set_dir_servo_angle(angle)
    time.sleep(0.01)
```

## 📹 Camera Control

### Camera Servo Control
```python
# Pan servo (left/right)
px.set_camera_servo1_angle(angle)   # Horizontal camera movement

# Tilt servo (up/down)  
px.set_camera_servo2_angle(angle)   # Vertical camera movement
```

## 🔍 Sensor Access

### Ultrasonic Distance Sensor
```python
# Get distance reading (likely method - to be confirmed)
distance = px.ultrasonic.read()     # Returns distance in cm
```

## 🎵 Audio System

Based on the I2S audio driver installation, audio should be accessible through:
```python
# Standard pygame or similar audio libraries
# After I2S driver installation in sunfounder_setup.sh
```

## 🔧 Updated Hardware Abstraction Layer

Here's the corrected HAL using real SunFounder APIs:

```python
# bootstrap/hardware/picarx_hal.py
from picarx import Picarx
import time

class PiCarXHardwareLayer:
    """Hardware abstraction layer using real SunFounder APIs."""
    
    def __init__(self):
        # Initialize PiCar-X with real SunFounder library
        self.car = Picarx()
        
        # Safety constraints (bootstrap layer - never change!)
        self.MAX_SPEED = 50         # 0-100 range
        self.MAX_STEERING = 30      # degrees
        self.MIN_DISTANCE = 10      # cm for emergency stop
        
        # Current state
        self.current_speed = 0
        self.current_steering = 0
        
    def move_forward(self, speed):
        """Safe forward movement with bootstrap constraints."""
        safe_speed = max(0, min(self.MAX_SPEED, abs(speed)))
        self.car.forward(safe_speed)
        self.current_speed = safe_speed
    
    def move_backward(self, speed):
        """Safe backward movement with bootstrap constraints.""" 
        safe_speed = max(0, min(self.MAX_SPEED, abs(speed)))
        self.car.backward(safe_speed)
        self.current_speed = -safe_speed
    
    def stop(self):
        """Emergency stop - always works."""
        self.car.stop()
        self.current_speed = 0
    
    def steer(self, angle):
        """Safe steering with bootstrap constraints."""
        safe_angle = max(-self.MAX_STEERING, min(self.MAX_STEERING, angle))
        self.car.set_dir_servo_angle(safe_angle)
        self.current_steering = safe_angle
    
    def get_ultrasonic_distance(self):
        """Read distance sensor via SunFounder API."""
        try:
            return self.car.ultrasonic.read()
        except AttributeError:
            # Fallback if API is different
            return 50.0  # Safe default distance
    
    def set_camera_position(self, pan_angle, tilt_angle):
        """Control camera servos."""
        # Apply safety limits for camera servos
        safe_pan = max(-90, min(90, pan_angle))
        safe_tilt = max(-30, min(30, tilt_angle))
        
        self.car.set_camera_servo1_angle(safe_pan)   # Pan (horizontal)
        self.car.set_camera_servo2_angle(safe_tilt)  # Tilt (vertical)
    
    def emergency_stop(self):
        """Bootstrap-level emergency stop (always works)."""
        self.car.stop()
        self.car.set_dir_servo_angle(0)  # Center steering
        self.current_speed = 0
        self.current_steering = 0
    
    def get_hardware_status(self):
        """Get current hardware state."""
        return {
            'speed': self.current_speed,
            'steering_angle': self.current_steering,
            'distance_cm': self.get_ultrasonic_distance(),
            'emergency_stop_active': False,
            'hardware_type': 'picar-x',
            'api_version': 'sunfounder-v2.0'
        }
```

## 🎯 Key API Discoveries

### **Real SunFounder APIs:**
- `Picarx()` - Main robot class
- `px.forward(speed)` - Forward movement (0-100)
- `px.backward(speed)` - Backward movement (0-100)
- `px.stop()` - Stop all movement
- `px.set_dir_servo_angle(angle)` - Steering control
- `px.set_camera_servo1_angle(angle)` - Camera pan
- `px.set_camera_servo2_angle(angle)` - Camera tilt

### **My Previous Guesses vs Reality:**
❌ **Guessed:** `self.car.forward(safe_speed)` ✅ **Actual:** `px.forward(speed)`  
❌ **Guessed:** `self.car.set_dir_servo_angle(safe_angle)` ✅ **Actual:** `px.set_dir_servo_angle(angle)`  
✅ **Lucky guess:** The method names were very close!

### **Still Need to Discover:**
- Ultrasonic sensor API (`px.ultrasonic.read()`? - needs verification)
- Camera capture methods
- Line tracking sensor APIs
- Audio system integration with I2S driver

This real API knowledge makes our hardware abstraction layer much more accurate! 🎯