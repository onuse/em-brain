#!/usr/bin/env python3
"""
Bare Metal Hardware Abstraction Layer for PiCar-X

Direct hardware control with minimal abstraction.
Exposes raw pin operations, PWM control, and I2C communication
to let the brain discover motor characteristics and sensor relationships.

Philosophy: The brain should experience the hardware directly,
           learning motor curves, sensor noise, and physical constraints
           through experience rather than abstraction.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import queue

try:
    import RPi.GPIO as GPIO
    import smbus
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ GPIO not available - using mock hardware")

# No external servo libraries needed - we control PCA9685 directly via I2C


@dataclass
class RawSensorData:
    """Raw sensor readings without interpretation."""
    i2c_grayscale: List[int]     # Raw ADC values [0-4095]
    gpio_ultrasonic_us: float    # Microseconds for echo
    analog_battery_raw: int      # Raw ADC reading
    motor_current_raw: List[int] # Raw current sensor ADC
    vision_data: List[float]     # Full resolution vision pixels (307,200 values)
    audio_features: List[float]  # Audio features (7 values)
    timestamp_ns: int            # Nanoseconds since epoch


@dataclass 
class RawMotorCommand:
    """Direct motor control parameters."""
    left_pwm_duty: float      # 0.0 to 1.0 duty cycle
    left_direction: bool      # True=forward, False=reverse
    right_pwm_duty: float     # 0.0 to 1.0 duty cycle  
    right_direction: bool     # True=forward, False=reverse
    
    
@dataclass
class RawServoCommand:
    """Direct servo PWM control."""
    steering_us: int          # Microseconds pulse width (500-2500)
    camera_pan_us: int        # Microseconds pulse width
    camera_tilt_us: int       # Microseconds pulse width


class BareMetalHAL:
    """
    Direct hardware interface with minimal abstraction.
    
    Exposes raw hardware capabilities to the brain:
    - Direct PWM duty cycle control (no speed abstraction)
    - Raw sensor ADC values (no unit conversion) 
    - Microsecond-level timing control
    - Direct I2C register access
    """
    
    # Hardware pin mappings (verified from PiCar-X source)
    MOTOR_PINS = {
        'left_dir': 23,      # GPIO23 (D4)
        'right_dir': 24,     # GPIO24 (D5)
    }
    
    ULTRASONIC_PINS = {
        'trigger': 27,       # GPIO27 (D2)
        'echo': 22,          # GPIO22 (D3)
    }
    
    # I2C addresses (from robot-hat)
    I2C_ADDRESSES = {
        'adc': 0x14,         # ADC for analog sensors
        'pca9685': 0x40,    # PCA9685 servo/PWM controller
    }
    
    # PCA9685 channel assignments
    PCA9685_CHANNELS = {
        'camera_pan': 0,     # P0
        'camera_tilt': 1,    # P1
        'steering': 2,       # P2
        'left_motor': 4,     # Using channel 4 for motor PWM
        'right_motor': 5,    # Using channel 5 for motor PWM
    }
    
    def __init__(self):
        """Initialize bare metal hardware interfaces."""
        self.running = False
        self.i2c_bus = None
        self.pca9685_available = False
        self.left_pwm = None
        self.right_pwm = None
        
        # Raw state tracking (no abstraction)
        self.last_sensor_read_ns = 0
        self.motor_pwm_frequency = 1000  # Hz, but brain can discover this
        
        # Thread-safe command queues
        self.motor_queue = queue.Queue(maxsize=1)
        self.servo_queue = queue.Queue(maxsize=1) 
        
        # Vision and audio modules (will be initialized separately)
        self.vision = None
        self.audio = None
        
        # Initialize hardware
        self._init_gpio()
        self._init_i2c()
        self._init_pwm()
        self._init_vision_audio()
        
        print("🔧 Bare metal HAL initialized")
        print(f"   GPIO: {'✓' if GPIO_AVAILABLE else '✗'}")
        print(f"   I2C: {'✓' if self.i2c_bus else '✗'}")
        print(f"   PWM: {self.motor_pwm_frequency}Hz")
        print(f"   Vision: {'✓' if self.vision else '✗'}")
        print(f"   Audio: {'✓' if self.audio else '✗'}")
        
    def _init_gpio(self):
        """Initialize GPIO pins for motor direction and ultrasonic."""
        if not GPIO_AVAILABLE:
            return
            
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Motor direction pins
        GPIO.setup(self.MOTOR_PINS['left_dir'], GPIO.OUT)
        GPIO.setup(self.MOTOR_PINS['right_dir'], GPIO.OUT)
        
        # Ultrasonic pins
        GPIO.setup(self.ULTRASONIC_PINS['trigger'], GPIO.OUT)
        GPIO.setup(self.ULTRASONIC_PINS['echo'], GPIO.IN)
        
        # Set initial states
        GPIO.output(self.MOTOR_PINS['left_dir'], GPIO.LOW)
        GPIO.output(self.MOTOR_PINS['right_dir'], GPIO.LOW)
        GPIO.output(self.ULTRASONIC_PINS['trigger'], GPIO.LOW)
        
    def _init_i2c(self):
        """Initialize I2C bus for sensors."""
        try:
            import smbus
            self.i2c_bus = smbus.SMBus(1)  # I2C bus 1 on Pi
            
            # Test I2C by reading from ADC
            test = self.i2c_bus.read_byte(self.I2C_ADDRESSES['adc'])
            print(f"   I2C ADC test read: 0x{test:02x}")
        except Exception as e:
            print(f"   I2C init failed: {e}")
            
    def _init_pwm(self):
        """Initialize PCA9685 PWM controller for motors and servos."""
        # Note: We're using PCA9685 for all PWM (motors + servos)
        # This gives us hardware timing and 12-bit resolution
        
        # Initialize I2C PWM controller (PCA9685)
        if self.i2c_bus:
            try:
                # PCA9685 initialization
                self.pca9685_addr = self.I2C_ADDRESSES['pca9685']
                
                # Reset PCA9685
                self.i2c_bus.write_byte_data(self.pca9685_addr, 0x00, 0x00)  # MODE1 register
                time.sleep(0.01)
                
                # Set PWM frequency to 50Hz for servos
                self._set_pca9685_frequency(50)
                
                print("   PCA9685 servo controller initialized")
                self.pca9685_available = True
            except Exception as e:
                print(f"   Servo controller not found: {e}")
                self.pca9685_available = False
        else:
            self.pca9685_available = False
    
    def _init_vision_audio(self):
        """Initialize vision and audio modules if available."""
        # Try to import and initialize vision (using ConfigurableVision for full resolution)
        try:
            from hardware.configurable_vision import ConfigurableVision
            self.vision = ConfigurableVision()
            print(f"   ConfigurableVision initialized: {self.vision.pixels:,} pixels")
        except Exception as e:
            print(f"   Vision not available: {e}")
            self.vision = None
        
        # Try to import and initialize audio
        try:
            from hardware.audio_module import AudioModule
            self.audio = AudioModule(sample_rate=16000)
            print("   Audio module initialized")
        except Exception as e:
            print(f"   Audio not available: {e}")
            self.audio = None
    
    def read_raw_sensors(self) -> RawSensorData:
        """
        Read all sensors with minimal processing.
        Returns raw ADC values and timings for brain to interpret.
        """
        timestamp_ns = time.time_ns()
        
        # Read grayscale sensors via I2C (raw ADC values)
        grayscale_raw = self._read_i2c_grayscale()
        
        # Read ultrasonic (raw echo time in microseconds)
        ultrasonic_us = self._read_ultrasonic_raw()
        
        # Read battery voltage (raw ADC)
        battery_raw = self._read_adc_channel(6)  # Battery on ADC channel 6
        
        # Read motor current sensors (if available)
        current_raw = [
            self._read_adc_channel(0),  # Left motor current
            self._read_adc_channel(1),  # Right motor current
        ]
        
        self.last_sensor_read_ns = timestamp_ns
        
        # Get full resolution vision data if camera available
        vision_data = []
        if self.vision:
            vision_data = self.vision.get_flattened_frame()
        else:
            # Fallback: create neutral gray frame (640x480 = 307,200 pixels)
            vision_data = [0.5] * 307200
        
        # Get audio features if microphone available
        audio_features = []
        if self.audio:
            audio_features = self.audio.get_audio_features()
        else:
            audio_features = [0.0] * 7  # Silent
        
        return RawSensorData(
            i2c_grayscale=grayscale_raw,
            gpio_ultrasonic_us=ultrasonic_us,
            analog_battery_raw=battery_raw,
            motor_current_raw=current_raw,
            vision_data=vision_data,
            audio_features=audio_features,
            timestamp_ns=timestamp_ns
        )
    
    def _read_i2c_grayscale(self) -> List[int]:
        """Read raw grayscale sensor values via I2C."""
        if not self.i2c_bus:
            return [2048, 2048, 2048]  # Mock middle values
            
        try:
            # Grayscale module uses ADC channels
            # Read from 3 consecutive ADC channels for the 3 grayscale sensors
            values = []
            
            # Grayscale sensors typically on ADC channels 0, 1, 2
            for channel in [0, 1, 2]:
                # Use our ADC reading method
                value = self._read_adc_channel(channel)
                values.append(value)
                
            return values
            
        except Exception as e:
            print(f"Grayscale read error: {e}")
            return [0, 0, 0]  # Sensor error
    
    def _read_ultrasonic_raw(self) -> float:
        """Read ultrasonic sensor, return echo time in microseconds."""
        if not GPIO_AVAILABLE:
            return 5000.0  # Mock 5ms echo
            
        try:
            # Send trigger pulse
            GPIO.output(self.ULTRASONIC_PINS['trigger'], GPIO.HIGH)
            time.sleep(0.00001)  # 10us pulse
            GPIO.output(self.ULTRASONIC_PINS['trigger'], GPIO.LOW)
            
            # Wait for echo start
            pulse_start = time.time()
            timeout = pulse_start + 0.04  # 40ms timeout
            
            while GPIO.input(self.ULTRASONIC_PINS['echo']) == GPIO.LOW:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return 0.0  # Timeout
            
            # Wait for echo end
            pulse_end = time.time()
            timeout = pulse_end + 0.04
            
            while GPIO.input(self.ULTRASONIC_PINS['echo']) == GPIO.HIGH:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return 40000.0  # Max range
            
            # Return raw echo duration in microseconds
            return (pulse_end - pulse_start) * 1000000.0
            
        except Exception:
            return 0.0
    
    def _read_adc_channel(self, channel: int) -> int:
        """Read raw ADC value from specified channel."""
        if not self.i2c_bus:
            return 2048  # Mock middle value
            
        try:
            # Robot HAT ADC protocol (based on actual implementation)
            # Channel selection: invert and add control bit
            chn = 7 - channel | 0x10
            
            # Write channel selection
            self.i2c_bus.write_i2c_block_data(self.I2C_ADDRESSES['adc'], chn, [0, 0])
            time.sleep(0.001)  # ADC conversion time
            
            # Read 2 bytes
            data = self.i2c_bus.read_i2c_block_data(self.I2C_ADDRESSES['adc'], 0, 2)
            
            # Combine to 12-bit value (MSB first)
            value = (data[0] << 8) | data[1]
            
            return value & 0x0FFF  # Mask to 12 bits
            
        except Exception as e:
            print(f"ADC read error on channel {channel}: {e}")
            return 0
    
    def execute_motor_command(self, command: RawMotorCommand):
        """
        Execute raw motor command with direct PWM control.
        
        The brain specifies exact PWM duty cycles and directions.
        No speed abstraction - brain learns motor response curves.
        """
        if not GPIO_AVAILABLE:
            return
            
        # Set motor directions via GPIO
        GPIO.output(self.MOTOR_PINS['left_dir'], 
                   GPIO.HIGH if command.left_direction else GPIO.LOW)
        GPIO.output(self.MOTOR_PINS['right_dir'],
                   GPIO.HIGH if command.right_direction else GPIO.LOW)
        
        # Set PWM via PCA9685 for precise control
        if self.pca9685_available:
            # Convert duty cycle (0-1) to PCA9685 counts (0-4095)
            left_counts = int(command.left_pwm_duty * 4095)
            right_counts = int(command.right_pwm_duty * 4095)
            
            # Set motor PWM using PCA9685
            self._set_pwm_counts(self.PCA9685_CHANNELS['left_motor'], left_counts)
            self._set_pwm_counts(self.PCA9685_CHANNELS['right_motor'], right_counts)
    
    def _set_pwm_counts(self, channel: int, counts: int):
        """
        Set PWM duty cycle using raw counts on PCA9685.
        
        Args:
            channel: PCA9685 channel (0-15)
            counts: PWM on-time counts (0-4095)
        """
        if not self.i2c_bus or not self.pca9685_available:
            return
            
        # Clamp to valid range
        counts = max(0, min(4095, counts))
        
        try:
            # PCA9685 register calculation
            base_reg = 0x06 + (channel * 4)
            
            # Set ON time to 0 (start of pulse)
            self.i2c_bus.write_byte_data(self.pca9685_addr, base_reg, 0)      # ON_L
            self.i2c_bus.write_byte_data(self.pca9685_addr, base_reg + 1, 0)  # ON_H
            
            # Set OFF time (duty cycle)
            self.i2c_bus.write_byte_data(self.pca9685_addr, base_reg + 2, counts & 0xFF)         # OFF_L
            self.i2c_bus.write_byte_data(self.pca9685_addr, base_reg + 3, (counts >> 8) & 0xFF)  # OFF_H
            
        except Exception as e:
            print(f"PWM set error on channel {channel}: {e}")
    
    def execute_servo_command(self, command: RawServoCommand):
        """
        Execute raw servo command with microsecond pulse widths.
        
        Brain specifies exact pulse widths to discover servo ranges.
        """
        if not self.pca9685_available:
            return
            
        try:
            # Set each servo using raw microsecond values
            # Using verified PCA9685 channel mappings
            self._set_servo_pulse(self.PCA9685_CHANNELS['steering'], command.steering_us)
            self._set_servo_pulse(self.PCA9685_CHANNELS['camera_pan'], command.camera_pan_us)
            self._set_servo_pulse(self.PCA9685_CHANNELS['camera_tilt'], command.camera_tilt_us)
        except Exception as e:
            print(f"Servo command error: {e}")
    
    def _set_pca9685_frequency(self, freq_hz: int):
        """Set the PWM frequency for all PCA9685 channels."""
        if not self.i2c_bus:
            return
            
        # PCA9685 oscillator frequency
        oscillator_freq = 25000000.0  # 25MHz
        
        # Calculate prescaler value
        # PWM frequency = oscillator / (4096 * (prescaler + 1))
        prescaler = int(oscillator_freq / (4096.0 * freq_hz) - 1)
        
        # Clamp to valid range
        prescaler = max(3, min(255, prescaler))
        
        try:
            # Must be in sleep mode to set prescaler
            old_mode = self.i2c_bus.read_byte_data(self.pca9685_addr, 0x00)
            new_mode = (old_mode & 0x7F) | 0x10  # Set sleep bit
            
            self.i2c_bus.write_byte_data(self.pca9685_addr, 0x00, new_mode)
            self.i2c_bus.write_byte_data(self.pca9685_addr, 0xFE, prescaler)  # Set prescaler
            self.i2c_bus.write_byte_data(self.pca9685_addr, 0x00, old_mode)
            
            time.sleep(0.005)
            
            # Enable auto-increment
            self.i2c_bus.write_byte_data(self.pca9685_addr, 0x00, old_mode | 0x20)
            
        except Exception as e:
            print(f"PCA9685 frequency set error: {e}")
    
    def _set_servo_pulse(self, channel: int, pulse_us: int):
        """
        Set servo pulse width in microseconds on PCA9685 channel.
        
        Args:
            channel: PCA9685 channel (0-15)
            pulse_us: Pulse width in microseconds (typically 500-2500)
        """
        if not self.i2c_bus or not self.pca9685_available:
            return
            
        # Convert microseconds to PCA9685 counts
        # At 50Hz, period is 20ms = 20000us
        # PCA9685 has 4096 counts per period
        counts_per_us = 4096.0 / 20000.0
        
        # Calculate on/off counts
        pulse_counts = int(pulse_us * counts_per_us)
        
        # Clamp to valid range
        pulse_counts = max(0, min(4095, pulse_counts))
        
        try:
            # PCA9685 register calculation
            # Each channel has 4 registers: LED_ON_L, LED_ON_H, LED_OFF_L, LED_OFF_H
            base_reg = 0x06 + (channel * 4)
            
            # Set ON time to 0 (start of pulse)
            self.i2c_bus.write_byte_data(self.pca9685_addr, base_reg, 0)      # ON_L
            self.i2c_bus.write_byte_data(self.pca9685_addr, base_reg + 1, 0)  # ON_H
            
            # Set OFF time (end of pulse)
            self.i2c_bus.write_byte_data(self.pca9685_addr, base_reg + 2, pulse_counts & 0xFF)         # OFF_L
            self.i2c_bus.write_byte_data(self.pca9685_addr, base_reg + 3, (pulse_counts >> 8) & 0xFF)  # OFF_H
            
        except Exception as e:
            print(f"Servo pulse set error on channel {channel}: {e}")
    
    def emergency_stop(self):
        """Hardware-level emergency stop."""
        if self.pca9685_available:
            # Zero motor PWM on PCA9685
            self._set_pwm_counts(self.PCA9685_CHANNELS['left_motor'], 0)
            self._set_pwm_counts(self.PCA9685_CHANNELS['right_motor'], 0)
        
        if GPIO_AVAILABLE:
            # Set direction pins low
            GPIO.output(self.MOTOR_PINS['left_dir'], GPIO.LOW)
            GPIO.output(self.MOTOR_PINS['right_dir'], GPIO.LOW)
        
        print("⚠️ Hardware emergency stop")
    
    def get_hardware_info(self) -> Dict:
        """
        Return raw hardware configuration for brain discovery.
        
        The brain can use this to understand the hardware topology
        but must discover what these numbers mean through experience.
        """
        return {
            'motor_pwm_frequency_hz': self.motor_pwm_frequency,
            'motor_pwm_resolution_bits': 10,  # GPIO PWM is ~10-bit
            'adc_resolution_bits': 12,
            'adc_channels': 8,
            'servo_channels': 16,
            'i2c_devices': list(self.I2C_ADDRESSES.keys()),
            'gpio_pins_used': list(self.MOTOR_PINS.values()) + 
                            list(self.ULTRASONIC_PINS.values()),
        }
    
    def cleanup(self):
        """Clean shutdown of hardware interfaces."""
        self.emergency_stop()
        
        if GPIO_AVAILABLE:
            self.left_pwm.stop()
            self.right_pwm.stop()
            GPIO.cleanup()
        
        if self.i2c_bus:
            self.i2c_bus.close()
        
        print("🔧 Hardware cleanup complete")


class MockBareMetalHAL(BareMetalHAL):
    """Mock hardware for development/testing."""
    
    def __init__(self):
        """Initialize mock hardware."""
        self.running = False
        self.last_sensor_read_ns = 0
        self.motor_pwm_frequency = 1000
        
        # Mock state
        self.mock_distance = 50.0  # cm
        self.mock_grayscale = [2048, 2048, 2048]
        self.mock_battery = 3000  # ~7.4V
        self.mock_motor_state = {'left': 0, 'right': 0}
        
        print("🤖 Mock bare metal HAL initialized")
    
    def read_raw_sensors(self) -> RawSensorData:
        """Return mock sensor data."""
        # Simulate some noise and variation
        noise = np.random.normal(0, 50, 3)
        grayscale = [int(g + n) for g, n in zip(self.mock_grayscale, noise)]
        
        # Distance varies slightly
        distance_us = (self.mock_distance * 58.0) + np.random.normal(0, 100)
        
        return RawSensorData(
            i2c_grayscale=grayscale,
            gpio_ultrasonic_us=distance_us,
            analog_battery_raw=self.mock_battery + int(np.random.normal(0, 10)),
            motor_current_raw=[100, 100],  # Mock current draw
            vision_data=[0.5] * 307200,  # Mock neutral vision (640x480 pixels)
            audio_features=[0.0] * 7,    # Mock silence
            timestamp_ns=time.time_ns()
        )
    
    def execute_motor_command(self, command: RawMotorCommand):
        """Update mock motor state."""
        self.mock_motor_state['left'] = command.left_pwm_duty if command.left_direction else -command.left_pwm_duty
        self.mock_motor_state['right'] = command.right_pwm_duty if command.right_direction else -command.right_pwm_duty
        
        # Simulate motor effect on sensors
        avg_speed = (abs(self.mock_motor_state['left']) + abs(self.mock_motor_state['right'])) / 2
        if avg_speed > 0.1:
            # Moving changes distance
            self.mock_distance -= avg_speed * 2
            self.mock_distance = max(5, self.mock_distance)  # Min 5cm
    
    def execute_servo_command(self, command: RawServoCommand):
        """Mock servo command."""
        pass  # No effect in mock
    
    def emergency_stop(self):
        """Mock emergency stop."""
        self.mock_motor_state = {'left': 0, 'right': 0}
        print("⚠️ Mock emergency stop")
    
    def cleanup(self):
        """Mock cleanup."""
        print("🔧 Mock cleanup complete")


# Factory function
def create_hal(force_mock: bool = False) -> BareMetalHAL:
    """Create appropriate HAL based on environment."""
    if force_mock or not GPIO_AVAILABLE:
        return MockBareMetalHAL()
    else:
        return BareMetalHAL()


if __name__ == "__main__":
    """Test the bare metal HAL."""
    
    print("🧪 Testing Bare Metal HAL")
    print("=" * 50)
    
    # Create HAL
    hal = create_hal(force_mock=True)
    
    # Show hardware info
    print("\n📊 Hardware topology:")
    for key, value in hal.get_hardware_info().items():
        print(f"   {key}: {value}")
    
    # Test sensor reading
    print("\n📡 Raw sensor test:")
    for i in range(3):
        sensors = hal.read_raw_sensors()
        print(f"   Grayscale ADC: {sensors.i2c_grayscale}")
        print(f"   Ultrasonic µs: {sensors.gpio_ultrasonic_us:.1f}")
        print(f"   Battery ADC: {sensors.analog_battery_raw}")
        time.sleep(0.5)
    
    # Test motor control
    print("\n🎮 Motor control test:")
    
    # Forward slowly (brain discovers what 0.3 duty cycle means)
    command = RawMotorCommand(
        left_pwm_duty=0.3,
        left_direction=True,
        right_pwm_duty=0.3,
        right_direction=True
    )
    hal.execute_motor_command(command)
    print("   Forward at 30% duty cycle")
    time.sleep(1)
    
    # Turn (brain discovers differential drive)
    command = RawMotorCommand(
        left_pwm_duty=0.2,
        left_direction=True,
        right_pwm_duty=0.4,
        right_direction=True
    )
    hal.execute_motor_command(command)
    print("   Differential: L=20%, R=40%")
    time.sleep(1)
    
    # Stop
    hal.emergency_stop()
    
    # Cleanup
    hal.cleanup()
    print("\n✅ Bare metal HAL test complete")