#!/usr/bin/env python3
"""
Robot HAT Hardware Abstraction Layer for PiCar-X

Uses the Sunfounder robot-hat library for proper hardware access.
This is the correct way to interface with the PiCar-X Robot HAT.

The robot-hat library handles:
- PWM/Servo control via internal abstraction (not direct PCA9685)
- ADC reading for sensors
- GPIO for motor direction
- Proper initialization sequences
"""

import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import configuration
try:
    import json
    with open('config/robot_config.json', 'r') as f:
        CONFIG = json.load(f)
except:
    CONFIG = {}

try:
    # This is the correct way - use robot-hat library
    from robot_hat import Pin, PWM, Servo, ADC, Ultrasonic
    ROBOT_HAT_AVAILABLE = True
    print("✓ Using robot-hat library (correct approach)")
except ImportError:
    ROBOT_HAT_AVAILABLE = False
    print("⚠️ robot-hat not available - install with:")
    print("   cd ~")
    print("   git clone https://github.com/sunfounder/robot-hat.git")
    print("   cd robot-hat")
    print("   sudo python3 setup.py install")


@dataclass
class RawSensorData:
    """Raw sensor readings without interpretation."""
    i2c_grayscale: List[int]     # Raw ADC values [0-4095]
    gpio_ultrasonic_us: float    # Microseconds for echo
    analog_battery_raw: int      # Raw ADC reading
    motor_current_raw: List[int] # Raw current sensor ADC
    vision_data: List[float]     # Full resolution vision pixels
    audio_features: List[float]  # Audio features
    timestamp_ns: int            # Nanoseconds since epoch


@dataclass
class RawMotorCommand:
    """Direct motor control parameters."""
    left_pwm_duty: float      # -1.0 to 1.0 (negative = reverse)
    right_pwm_duty: float     # -1.0 to 1.0 (negative = reverse)


@dataclass
class RawServoCommand:
    """Direct servo angle control."""
    steering_angle: float     # -30 to 30 degrees
    camera_pan_angle: float   # -90 to 90 degrees
    camera_tilt_angle: float  # -35 to 65 degrees


class RobotHatHAL:
    """
    Hardware interface using robot-hat library.
    
    This is the correct way to interface with PiCar-X hardware.
    The robot-hat library handles all the low-level details.
    """
    
    def __init__(self):
        """Initialize robot-hat based hardware interfaces."""
        self.running = False
        
        if not ROBOT_HAT_AVAILABLE:
            print("⚠️ Running in mock mode - robot-hat not available")
            self.mock_mode = True
            return
        
        self.mock_mode = False
        
        try:
            # Initialize motor PWM channels
            # PiCar-X uses PWM channels for motor speed control
            self.left_motor_pwm = PWM("P12")   # Left motor PWM
            self.right_motor_pwm = PWM("P13")  # Right motor PWM
            
            # Initialize motor direction pins
            self.left_motor_dir = Pin("D4")    # GPIO 23
            self.right_motor_dir = Pin("D5")   # GPIO 24
            self.left_motor_dir.mode(Pin.OUT)
            self.right_motor_dir.mode(Pin.OUT)
            
            # Initialize servos
            # PiCar-X uses specific servo channels
            self.steering_servo = Servo("P2")     # Steering servo
            self.camera_pan_servo = Servo("P0")   # Camera pan
            self.camera_tilt_servo = Servo("P1")  # Camera tilt
            
            # Calibrate servos to center position
            self.steering_servo.angle(0)
            self.camera_pan_servo.angle(0)  
            self.camera_tilt_servo.angle(0)
            
            # Initialize ultrasonic sensor
            self.ultrasonic = Ultrasonic(Pin("D2"), Pin("D3"))  # Trigger, Echo
            
            # Initialize ADC for analog sensors
            self.adc = ADC()
            
            # Grayscale sensor channels on ADC
            self.grayscale_channels = [
                ADC("A0"),  # Left grayscale
                ADC("A1"),  # Middle grayscale
                ADC("A2"),  # Right grayscale
            ]
            
            # Battery voltage on ADC
            self.battery_channel = ADC("A4")  # Battery voltage divider
            
            print("🔧 Robot HAT HAL initialized successfully")
            print("   Motors: P12/P13 (PWM) + D4/D5 (direction)")
            print("   Servos: P0 (pan), P1 (tilt), P2 (steering)")
            print("   Ultrasonic: D2/D3")
            print("   ADC: A0-A2 (grayscale), A4 (battery)")
            
        except Exception as e:
            print(f"❌ Robot HAT initialization failed: {e}")
            print("   Falling back to mock mode")
            self.mock_mode = True
    
    def read_raw_sensors(self) -> RawSensorData:
        """
        Read all sensors with minimal processing.
        Returns raw ADC values and timings for brain to interpret.
        """
        timestamp_ns = time.time_ns()
        
        if self.mock_mode:
            return self._mock_sensor_data(timestamp_ns)
        
        try:
            # Read grayscale sensors (raw ADC 0-4095)
            grayscale_raw = [
                sensor.read() for sensor in self.grayscale_channels
            ]
            
            # Read ultrasonic (returns distance in cm, convert to microseconds)
            distance_cm = self.ultrasonic.read()
            # Convert to microseconds (58 us per cm is standard for ultrasonic)
            ultrasonic_us = distance_cm * 58.0
            
            # Read battery voltage (raw ADC)
            battery_raw = self.battery_channel.read()
            
            # Motor current (not available on basic PiCar-X)
            current_raw = [0, 0]
            
            # Get vision data if camera available
            vision_data = self._get_vision_data()
            
            # Get audio features if microphone available
            audio_features = self._get_audio_features()
            
            return RawSensorData(
                i2c_grayscale=grayscale_raw,
                gpio_ultrasonic_us=ultrasonic_us,
                analog_battery_raw=battery_raw,
                motor_current_raw=current_raw,
                vision_data=vision_data,
                audio_features=audio_features,
                timestamp_ns=timestamp_ns
            )
            
        except Exception as e:
            print(f"Sensor read error: {e}")
            return self._mock_sensor_data(timestamp_ns)
    
    def execute_motor_command(self, command: RawMotorCommand):
        """
        Execute motor command using robot-hat PWM control.
        
        The robot-hat library handles PWM generation internally.
        We just set the pulse width percentage.
        """
        if self.mock_mode:
            return
        
        try:
            # Set motor directions
            # Forward = HIGH (1), Reverse = LOW (0)
            if command.left_pwm_duty >= 0:
                self.left_motor_dir.value(1)
                left_speed = abs(command.left_pwm_duty)
            else:
                self.left_motor_dir.value(0)
                left_speed = abs(command.left_pwm_duty)
            
            if command.right_pwm_duty >= 0:
                self.right_motor_dir.value(1)
                right_speed = abs(command.right_pwm_duty)
            else:
                self.right_motor_dir.value(0)
                right_speed = abs(command.right_pwm_duty)
            
            # Set motor speeds (0-100% for robot-hat PWM)
            # Convert from 0-1 to 0-100
            self.left_motor_pwm.pulse_width_percent(int(left_speed * 100))
            self.right_motor_pwm.pulse_width_percent(int(right_speed * 100))
            
        except Exception as e:
            print(f"Motor command error: {e}")
    
    def execute_servo_command(self, command: RawServoCommand):
        """
        Execute servo command using robot-hat Servo control.
        
        The robot-hat library handles pulse generation.
        We just set the angle directly.
        """
        if self.mock_mode:
            return
        
        try:
            # Clamp angles to safe ranges
            steering = max(-30, min(30, command.steering_angle))
            pan = max(-90, min(90, command.camera_pan_angle))
            tilt = max(-35, min(65, command.camera_tilt_angle))
            
            # Set servo angles
            self.steering_servo.angle(steering)
            self.camera_pan_servo.angle(pan)
            self.camera_tilt_servo.angle(tilt)
            
        except Exception as e:
            print(f"Servo command error: {e}")
    
    def emergency_stop(self):
        """Hardware-level emergency stop."""
        if self.mock_mode:
            print("⚠️ Mock emergency stop")
            return
        
        try:
            # Stop motors
            self.left_motor_pwm.pulse_width_percent(0)
            self.right_motor_pwm.pulse_width_percent(0)
            
            # Set direction pins low
            self.left_motor_dir.value(0)
            self.right_motor_dir.value(0)
            
            print("⚠️ Hardware emergency stop")
            
        except Exception as e:
            print(f"Emergency stop error: {e}")
    
    def _get_vision_data(self) -> List[float]:
        """Get vision data if camera available."""
        try:
            from hardware.configurable_vision import ConfigurableVision
            if not hasattr(self, 'vision'):
                self.vision = ConfigurableVision()
            return self.vision.get_flattened_frame()
        except:
            # Return neutral gray frame
            return [0.5] * 307200  # 640x480
    
    def _get_audio_features(self) -> List[float]:
        """Get audio features if microphone available."""
        try:
            from hardware.audio_module import AudioModule
            if not hasattr(self, 'audio'):
                self.audio = AudioModule()
            return self.audio.get_audio_features()
        except:
            # Return silence
            return [0.0] * 7
    
    def _mock_sensor_data(self, timestamp_ns: int) -> RawSensorData:
        """Generate mock sensor data for testing."""
        return RawSensorData(
            i2c_grayscale=[2048, 2048, 2048],
            gpio_ultrasonic_us=5000.0,  # ~86cm
            analog_battery_raw=3000,  # ~7.4V
            motor_current_raw=[0, 0],
            vision_data=[0.5] * 307200,
            audio_features=[0.0] * 7,
            timestamp_ns=timestamp_ns
        )
    
    def get_hardware_info(self) -> Dict:
        """
        Return hardware configuration for diagnostics.
        """
        return {
            'hal_type': 'robot-hat',
            'mock_mode': self.mock_mode,
            'motors': 'PWM P12/P13 + Direction D4/D5',
            'servos': 'P0 (pan), P1 (tilt), P2 (steering)',
            'sensors': 'Ultrasonic D2/D3, ADC A0-A2 (grayscale), A4 (battery)',
            'library_available': ROBOT_HAT_AVAILABLE
        }
    
    def cleanup(self):
        """Clean shutdown of hardware interfaces."""
        self.emergency_stop()
        print("🔧 Hardware cleanup complete")


# Factory function
def create_hal(force_mock: bool = False) -> RobotHatHAL:
    """Create appropriate HAL based on environment."""
    if force_mock or not ROBOT_HAT_AVAILABLE:
        hal = RobotHatHAL()
        hal.mock_mode = True
        return hal
    else:
        return RobotHatHAL()


if __name__ == "__main__":
    """Test the robot-hat HAL."""
    
    print("🧪 Testing Robot HAT HAL")
    print("=" * 50)
    
    # Create HAL
    hal = create_hal()
    
    # Show hardware info
    print("\n📊 Hardware configuration:")
    for key, value in hal.get_hardware_info().items():
        print(f"   {key}: {value}")
    
    if not hal.mock_mode:
        # Test sensor reading
        print("\n📡 Sensor test:")
        for i in range(3):
            sensors = hal.read_raw_sensors()
            print(f"   Grayscale: {sensors.i2c_grayscale}")
            print(f"   Distance: {sensors.gpio_ultrasonic_us/58:.1f}cm")
            print(f"   Battery: {sensors.analog_battery_raw}")
            time.sleep(0.5)
        
        # Test servo control
        print("\n🎮 Servo test:")
        print("   Centering servos...")
        hal.execute_servo_command(RawServoCommand(0, 0, 0))
        time.sleep(1)
        
        print("   Testing steering...")
        hal.execute_servo_command(RawServoCommand(-20, 0, 0))
        time.sleep(0.5)
        hal.execute_servo_command(RawServoCommand(20, 0, 0))
        time.sleep(0.5)
        hal.execute_servo_command(RawServoCommand(0, 0, 0))
        
        # Test motor control
        print("\n🏃 Motor test (wheels should be off ground!):")
        print("   Forward slow...")
        hal.execute_motor_command(RawMotorCommand(0.3, 0.3))
        time.sleep(1)
        
        print("   Turn right...")
        hal.execute_motor_command(RawMotorCommand(0.4, 0.2))
        time.sleep(1)
        
        print("   Stop")
        hal.emergency_stop()
    
    # Cleanup
    hal.cleanup()
    print("\n✅ Robot HAT HAL test complete")