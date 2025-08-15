#!/usr/bin/env python3
"""
Raw Robot HAT Hardware Abstraction Layer for PiCar-X

Uses robot-hat library for I2C communication but exposes raw control.
Gives the brain direct access to hardware without artificial limits.

Philosophy: The brain should discover hardware limits through experience,
not have them imposed by software abstractions.
"""

import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os

# Add robot_hat to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../robot_hat'))

try:
    from robot_hat import PWM, ADC, Pin, I2C
    from robot_hat.modules import Ultrasonic
    ROBOT_HAT_AVAILABLE = True
    print("✓ Robot HAT library loaded")
except ImportError:
    ROBOT_HAT_AVAILABLE = False
    print("⚠️ Robot HAT not available - using mock")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


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
    """Direct servo PWM control - brain discovers limits."""
    steering_pw: int          # Raw pulse width (microseconds)
    camera_pan_pw: int        # Raw pulse width (microseconds)
    camera_tilt_pw: int       # Raw pulse width (microseconds)


class RawRobotHatHAL:
    """
    Raw hardware interface using robot-hat for I2C but exposing direct control.
    
    No artificial limits - the brain discovers what works through experience.
    """
    
    # Pin mappings (from PiCar-X documentation)
    # From PiCar-X: motor_pins = ['D4', 'D5', 'P13', 'P12']
    MOTOR_PINS = {
        'left_dir': 23,      # GPIO23 (D4 in robot-hat) - LEFT motor direction
        'right_dir': 24,     # GPIO24 (D5 in robot-hat) - RIGHT motor direction
    }
    
    ULTRASONIC_PINS = {
        'trigger': 27,       # GPIO27 (D2)
        'echo': 22,          # GPIO22 (D3)
    }
    
    # PWM channels (robot-hat channel names)
    # From PiCar-X: motor_pins = ['D4', 'D5', 'P13', 'P12']
    PWM_CHANNELS = {
        'steering': 'P2',
        'camera_pan': 'P0',
        'camera_tilt': 'P1',
        'left_motor': 'P13',   # LEFT motor PWM
        'right_motor': 'P12',  # RIGHT motor PWM
    }
    
    # ADC channels
    ADC_CHANNELS = {
        'grayscale_left': 'A0',
        'grayscale_middle': 'A1',
        'grayscale_right': 'A2',
        'battery': 'A4',
    }
    
    def __init__(self, debug: bool = False):
        """Initialize raw hardware interfaces."""
        self.running = False
        self.mock_mode = not ROBOT_HAT_AVAILABLE
        self.debug = debug
        
        if self.mock_mode:
            print("🤖 Running in mock mode")
            self._init_mock()
            return
        
        try:
            # Initialize motor PWM (using raw PWM class, not Motor abstraction)
            self.left_motor_pwm = PWM(self.PWM_CHANNELS['left_motor'])
            self.right_motor_pwm = PWM(self.PWM_CHANNELS['right_motor'])
            
            # Set motor PWM to high frequency (no artificial limits)
            # Brain can discover what frequencies work best
            self.left_motor_pwm.freq(1000)  # 1kHz default
            self.right_motor_pwm.freq(1000)
            
            # Initialize servo PWM (using raw PWM, not Servo class)
            # This avoids the -90 to +90 degree limits
            self.steering_pwm = PWM(self.PWM_CHANNELS['steering'])
            self.pan_pwm = PWM(self.PWM_CHANNELS['camera_pan'])
            self.tilt_pwm = PWM(self.PWM_CHANNELS['camera_tilt'])
            
            # Set servo frequency to 50Hz (standard)
            for pwm in [self.steering_pwm, self.pan_pwm, self.tilt_pwm]:
                pwm.freq(50)
            
            # Motor direction pins using robot-hat Pin class
            # From PiCar-X: D4=left_dir, D5=right_dir
            self.left_dir_pin = Pin("D4", mode=Pin.OUT)
            self.right_dir_pin = Pin("D5", mode=Pin.OUT)
            self.gpio_available = False  # Always use robot-hat
            
            # Initialize ADC channels for raw reading
            self.adc_channels = {
                name: ADC(channel) 
                for name, channel in self.ADC_CHANNELS.items()
            }
            
            # Ultrasonic sensor (Ultrasonic class will set modes)
            # Just pass Pin objects, Ultrasonic recreates them internally
            trig = Pin("D2")  # No mode needed, Ultrasonic sets it
            echo = Pin("D3")  # No mode needed, Ultrasonic sets it
            self.ultrasonic = Ultrasonic(trig, echo)
            
            # Vision and audio (if available)
            self._init_vision_audio()
            
            print("🔧 Raw Robot HAT HAL initialized")
            print(f"   Motor PWM: {self.PWM_CHANNELS['left_motor']}, {self.PWM_CHANNELS['right_motor']}")
            print(f"   Servo PWM: Raw control on P0, P1, P2")
            print(f"   ADC: Raw 12-bit values (0-4095)")
            print(f"   GPIO: Via robot-hat")
            
        except Exception as e:
            print(f"❌ Hardware init failed: {e}")
            print("   Falling back to mock mode")
            self.mock_mode = True
            self._init_mock()
    
    def read_raw_sensors(self) -> RawSensorData:
        """
        Read all sensors with NO processing.
        Returns raw ADC values and timings for brain to interpret.
        """
        timestamp_ns = time.time_ns()
        
        if self.mock_mode:
            return self._mock_sensor_data(timestamp_ns)
        
        try:
            # Read grayscale sensors - RAW ADC values (0-4095)
            grayscale_raw = [
                self.adc_channels['grayscale_left'].read(),
                self.adc_channels['grayscale_middle'].read(),
                self.adc_channels['grayscale_right'].read(),
            ]
            
            # Read ultrasonic - get raw echo time in microseconds
            # The Ultrasonic class returns cm, so we convert back to timing
            distance_cm = self.ultrasonic.read()
            if distance_cm > 0:
                # Standard conversion: 58 microseconds per cm
                ultrasonic_us = distance_cm * 58.0
            else:
                ultrasonic_us = 0.0  # Timeout or error
            
            # Read battery - RAW ADC value
            battery_raw = self.adc_channels['battery'].read()
            
            # Motor current (if available - PiCar-X doesn't have these)
            current_raw = [0, 0]
            
            # Vision and audio
            vision_data = self._get_vision_data()
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
        Execute raw motor command with direct PWM control.
        
        Brain specifies exact duty cycles (-1 to +1).
        Positive = forward, Negative = backward
        
        PiCar-X wiring (differential drive):
        - LEFT motor: P13 (PWM), D4 (dir) 
        - RIGHT motor: P12 (PWM), D5 (dir)
        - For FORWARD: left goes forward (D4=LOW), right goes backward (D5=HIGH)
        - For BACKWARD: left goes backward (D4=HIGH), right goes forward (D5=LOW)
        """
        if self.mock_mode:
            return
        
        try:
            # For differential drive, when robot goes forward:
            # - Left motor spins forward
            # - Right motor spins backward (physically reversed)
            # This is handled by the PiCar-X wiring
            
            # We want intuitive control:
            # command.left/right_pwm_duty: positive = forward, negative = backward
            
            # LEFT motor control (P13/D4)
            if command.left_pwm_duty >= 0:
                # Forward: D4 = LOW
                self.left_dir_pin.value(0)
            else:
                # Backward: D4 = HIGH
                self.left_dir_pin.value(1)
            
            # RIGHT motor control (P12/D5)
            # RIGHT motor is physically wired reversed!
            if command.right_pwm_duty >= 0:
                # Forward: D5 = HIGH (because motor is reversed)
                self.right_dir_pin.value(1)
            else:
                # Backward: D5 = LOW (because motor is reversed)
                self.right_dir_pin.value(0)
            
            # Convert to PWM percent (0-100)
            left_percent = abs(command.left_pwm_duty) * 100
            right_percent = abs(command.right_pwm_duty) * 100
            
            # Apply PiCar-X speed scaling: maps to 50-100% range
            # This ensures motors have enough power to move
            if left_percent > 0:
                left_percent = min(100, int(left_percent / 2) + 50)
            if right_percent > 0:
                right_percent = min(100, int(right_percent / 2) + 50)
            
            self.left_motor_pwm.pulse_width_percent(left_percent)
            self.right_motor_pwm.pulse_width_percent(right_percent)
            
        except Exception as e:
            print(f"Motor command error: {e}")
    
    def execute_servo_command(self, command: RawServoCommand):
        """
        Execute raw servo command with microsecond precision.
        
        Brain specifies exact pulse widths in microseconds.
        No angle conversion or limits - brain discovers ranges.
        Typical servo range: 500-2500μs, but brain can try anything.
        """
        if self.mock_mode:
            return
        
        try:
            # Convert microseconds to PWM counts
            # At 50Hz, period is 20ms = 20000μs
            # PWM uses 0-4095 counts per period
            
            def us_to_pwm(us: int) -> int:
                # Don't limit! Let brain discover limits
                counts = int((us / 20000.0) * 4095)
                return counts
            
            # Set raw pulse widths
            self.steering_pwm.pulse_width(us_to_pwm(command.steering_pw))
            self.pan_pwm.pulse_width(us_to_pwm(command.camera_pan_pw))
            self.tilt_pwm.pulse_width(us_to_pwm(command.camera_tilt_pw))
            
        except Exception as e:
            print(f"Servo command error: {e}")
    
    def set_motor_frequency(self, freq_hz: int):
        """
        Let brain experiment with motor PWM frequency.
        Different frequencies affect motor behavior.
        """
        if not self.mock_mode:
            self.left_motor_pwm.freq(freq_hz)
            self.right_motor_pwm.freq(freq_hz)
    
    def read_raw_adc(self, channel: str) -> int:
        """
        Read any ADC channel directly.
        Returns raw 12-bit value (0-4095).
        """
        if self.mock_mode:
            return 2048  # Mock middle value
        
        try:
            if channel in self.adc_channels:
                return self.adc_channels[channel].read()
            else:
                # Try to read arbitrary channel
                adc = ADC(channel)
                return adc.read()
        except:
            return 0
    
    def set_raw_pwm(self, channel: str, value: int):
        """
        Set any PWM channel to raw value (0-4095).
        Let brain discover what each channel does.
        """
        if self.mock_mode:
            return
        
        try:
            pwm = PWM(channel)
            pwm.pulse_width(value)
        except Exception as e:
            print(f"PWM set error on {channel}: {e}")
    
    def _get_vision_data(self) -> List[float]:
        """Get vision data from singleton (if available)."""
        try:
            from src.hardware.vision_singleton import VisionSingleton
            vision_instance = VisionSingleton.get_instance()
            if vision_instance and hasattr(vision_instance, 'get_flattened_frame'):
                # Get latest frame (already normalized 0-1)
                return vision_instance.get_flattened_frame()
        except Exception as e:
            if self.debug:
                print(f"Vision data error: {e}")
        
        # Return empty list if vision not available
        return []
    
    def _get_audio_features(self) -> List[float]:
        """Get audio features from singleton (if available)."""
        try:
            from src.hardware.vision_singleton import AudioSingleton
            audio_instance = AudioSingleton.get_instance()
            if audio_instance and hasattr(audio_instance, 'get_audio_features'):
                # Get latest audio features
                return audio_instance.get_audio_features()
        except Exception as e:
            if self.debug:
                print(f"Audio features error: {e}")
        
        # Return default audio features if not available
        return [0.0] * 7  # 7 audio channels expected
    
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
            if self.gpio_available:
                GPIO.output(self.MOTOR_PINS['left_dir'], GPIO.LOW)
                GPIO.output(self.MOTOR_PINS['right_dir'], GPIO.LOW)
            else:
                self.left_dir_pin.value(0)
                self.right_dir_pin.value(0)
            
            print("⚠️ Hardware emergency stop")
            
        except Exception as e:
            print(f"Emergency stop error: {e}")
    
    def get_hardware_info(self) -> Dict:
        """
        Return raw hardware topology for brain discovery.
        """
        return {
            'hal_type': 'raw_robot_hat',
            'mock_mode': self.mock_mode,
            'pwm_channels': list(self.PWM_CHANNELS.values()),
            'adc_channels': list(self.ADC_CHANNELS.values()),
            'gpio_pins': list(self.MOTOR_PINS.values()) + list(self.ULTRASONIC_PINS.values()),
            'pwm_resolution': 4095,
            'adc_resolution': 4095,
            'i2c_address': 0x14,  # Robot HAT MCU address
        }
    
    def _init_vision_audio(self):
        """Initialize vision and audio using singletons to prevent conflicts."""
        # Use singletons to prevent multiple initializations
        try:
            from src.hardware.vision_singleton import VisionSingleton, AudioSingleton
            
            # Get or create vision singleton
            self.vision = VisionSingleton.get_instance()
            if self.vision:
                print(f"   Vision: {self.vision.pixels:,} pixels")
            
            # Get or create audio singleton  
            self.audio = AudioSingleton.get_instance()
            if self.audio:
                print(f"   Audio: Ready")
                
        except Exception as e:
            print(f"   Vision/Audio init error: {e}")
            self.vision = None
            self.audio = None
    
    def _get_vision_data(self) -> List[float]:
        """Get vision data if available."""
        if self.vision:
            return self.vision.get_flattened_frame()
        return [0.5] * 307200  # Neutral gray
    
    def _get_audio_features(self) -> List[float]:
        """Get audio features if available."""
        if self.audio:
            return self.audio.get_audio_features()
        return [0.0] * 7  # Silence
    
    def _init_mock(self):
        """Initialize mock components."""
        self.mock_distance = 50.0
        self.mock_grayscale = [2048, 2048, 2048]
        self.mock_battery = 3000
        self.vision = None
        self.audio = None
    
    def _mock_sensor_data(self, timestamp_ns: int) -> RawSensorData:
        """Generate mock sensor data."""
        # Add some noise for realism
        noise = np.random.normal(0, 50, 3)
        grayscale = [int(g + n) for g, n in zip(self.mock_grayscale, noise)]
        
        distance_us = (self.mock_distance * 58.0) + np.random.normal(0, 100)
        
        return RawSensorData(
            i2c_grayscale=grayscale,
            gpio_ultrasonic_us=distance_us,
            analog_battery_raw=self.mock_battery + int(np.random.normal(0, 10)),
            motor_current_raw=[0, 0],
            vision_data=[0.5] * 307200,
            audio_features=[0.0] * 7,
            timestamp_ns=timestamp_ns
        )
    
    def cleanup(self):
        """Clean shutdown."""
        self.emergency_stop()
        if hasattr(self, 'gpio_available') and self.gpio_available and GPIO_AVAILABLE:
            GPIO.cleanup()
        
        # Clean up singletons
        try:
            from src.hardware.vision_singleton import VisionSingleton, AudioSingleton
            VisionSingleton.cleanup()
            AudioSingleton.cleanup()
        except:
            pass
            
        print("🔧 Hardware cleanup complete")


# Factory function
def create_hal(force_mock: bool = False) -> RawRobotHatHAL:
    """Create appropriate HAL based on environment."""
    if force_mock:
        hal = RawRobotHatHAL()
        hal.mock_mode = True
        return hal
    return RawRobotHatHAL()


if __name__ == "__main__":
    """Test the raw HAL."""
    
    print("🧪 Testing Raw Robot HAT HAL")
    print("=" * 50)
    
    # Create HAL
    hal = create_hal()
    
    # Show hardware info
    print("\n📊 Hardware topology:")
    for key, value in hal.get_hardware_info().items():
        print(f"   {key}: {value}")
    
    if not hal.mock_mode:
        # Test raw sensor reading
        print("\n📡 Raw sensor test:")
        for i in range(3):
            sensors = hal.read_raw_sensors()
            print(f"   Grayscale ADC: {sensors.i2c_grayscale}")
            print(f"   Ultrasonic μs: {sensors.gpio_ultrasonic_us:.1f}")
            print(f"   Battery ADC: {sensors.analog_battery_raw}")
            time.sleep(0.5)
        
        # Test raw servo control
        print("\n🎮 Raw servo test (discovering limits):")
        print("   Testing pulse widths from 500-2500μs...")
        
        # Center (typically 1500μs)
        hal.execute_servo_command(RawServoCommand(1500, 1500, 1500))
        time.sleep(1)
        
        # Try extremes (brain discovers actual limits)
        print("   Testing 1000μs...")
        hal.execute_servo_command(RawServoCommand(1000, 1500, 1500))
        time.sleep(0.5)
        
        print("   Testing 2000μs...")
        hal.execute_servo_command(RawServoCommand(2000, 1500, 1500))
        time.sleep(0.5)
        
        # Back to center
        hal.execute_servo_command(RawServoCommand(1500, 1500, 1500))
        
        # Test motor frequency experimentation
        print("\n🏃 Motor frequency test:")
        print("   100Hz...")
        hal.set_motor_frequency(100)
        hal.execute_motor_command(RawMotorCommand(0.3, 0.3))
        time.sleep(1)
        
        print("   1000Hz...")
        hal.set_motor_frequency(1000)
        time.sleep(1)
        
        print("   10000Hz...")
        hal.set_motor_frequency(10000)
        time.sleep(1)
        
        hal.emergency_stop()
    
    # Cleanup
    hal.cleanup()
    print("\n✅ Raw HAL test complete")