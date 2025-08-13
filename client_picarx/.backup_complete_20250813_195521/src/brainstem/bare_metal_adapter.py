#!/usr/bin/env python3
"""
Bare Metal Brainstem Adapter

Bridges the brain's high-level motor commands with bare metal hardware control.
Implements safety reflexes at the lowest level while exposing raw sensor data
for the brain to interpret.

Philosophy: Safety lives in the brainstem (like biological reflexes),
           but everything else is learned by the brain through experience.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import threading

# Import our bare metal HAL
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from hardware.bare_metal_hal import (
    BareMetalHAL, create_hal, 
    RawMotorCommand, RawServoCommand, RawSensorData
)


@dataclass
class BrainstemReflexConfig:
    """Configuration for brainstem-level safety reflexes."""
    # These are the ONLY hardcoded safety values
    # Everything else is learned
    collision_distance_us: float = 1000.0   # ~17cm in microseconds
    cliff_adc_threshold: int = 3500         # High ADC = no ground
    current_spike_adc: int = 3000           # Motor stall detection
    battery_critical_adc: int = 2400        # ~6V critical
    
    # Reflex response parameters
    collision_backoff_duty: float = 0.3     # PWM duty for collision avoidance
    collision_backoff_ms: int = 500         # Duration of backoff
    

class BareMetalBrainstemAdapter:
    """
    Minimal brainstem layer for bare metal control.
    
    Responsibilities:
    1. Safety reflexes (collision, cliff, overcurrent)
    2. Sensor data packaging for brain
    3. Motor command translation from brain space to PWM
    4. NO interpretation or abstraction of sensor data
    """
    
    def __init__(self, force_mock: bool = False):
        """Initialize bare metal brainstem."""
        self.hal = create_hal(force_mock)
        self.reflex_config = BrainstemReflexConfig()
        
        # Reflex state
        self.reflex_active = False
        self.reflex_end_time = 0
        self.last_sensor_data = None
        
        # Brain has 24 input channels, 4 output channels
        self.brain_input_dim = 24
        self.brain_output_dim = 4
        
        # The brain will discover these relationships
        self.pwm_curve = None  # Brain learns motor response curve
        self.sensor_calibration = None  # Brain learns sensor meanings
        
        print("🧠 Bare metal brainstem initialized")
        print(f"   Reflexes: collision<{self.reflex_config.collision_distance_us}µs")
        print(f"   Brain I/O: {self.brain_input_dim} in, {self.brain_output_dim} out")
    
    def sensors_to_brain_input(self, raw_sensors: RawSensorData) -> List[float]:
        """
        Package raw sensor data for brain consumption.
        
        NO INTERPRETATION - just normalization to [0,1] range.
        Brain must discover what these signals mean.
        """
        brain_input = [0.5] * self.brain_input_dim  # Start neutral
        
        # Channels 0-2: Raw grayscale ADC (brain discovers line/edge)
        for i, adc_value in enumerate(raw_sensors.i2c_grayscale):
            brain_input[i] = adc_value / 4095.0  # Normalize 12-bit ADC
        
        # Channel 3: Ultrasonic echo time (brain discovers distance relationship)
        brain_input[3] = min(raw_sensors.gpio_ultrasonic_us / 40000.0, 1.0)  # Cap at 40ms
        
        # Channel 4-5: Motor current (brain discovers load/stall)
        brain_input[4] = raw_sensors.motor_current_raw[0] / 4095.0
        brain_input[5] = raw_sensors.motor_current_raw[1] / 4095.0
        
        # Channel 6: Battery voltage (brain discovers depletion effects)
        brain_input[6] = raw_sensors.analog_battery_raw / 4095.0
        
        # Channel 7-9: Time derivatives (brain discovers motion)
        if self.last_sensor_data:
            dt = (raw_sensors.timestamp_ns - self.last_sensor_data.timestamp_ns) / 1e9
            if dt > 0:
                # Grayscale change rate
                for i in range(3):
                    delta = raw_sensors.i2c_grayscale[i] - self.last_sensor_data.i2c_grayscale[i]
                    brain_input[7+i] = (delta / 4095.0 / dt + 1.0) / 2.0  # Normalize to [0,1]
        
        # Channel 10-11: Ultrasonic change (brain discovers approach/retreat)
        if self.last_sensor_data:
            echo_delta = raw_sensors.gpio_ultrasonic_us - self.last_sensor_data.gpio_ultrasonic_us
            brain_input[10] = (echo_delta / 40000.0 + 1.0) / 2.0  # Normalize
            brain_input[11] = abs(echo_delta) / 40000.0  # Magnitude of change
        
        # Channels 12-17: Frequency domain hints (brain discovers patterns)
        # Simple high/low pass of grayscale (brain learns edge detection)
        if len(raw_sensors.i2c_grayscale) == 3:
            # Spatial gradient
            left_right_diff = (raw_sensors.i2c_grayscale[0] - raw_sensors.i2c_grayscale[2]) / 4095.0
            brain_input[12] = (left_right_diff + 1.0) / 2.0
            
            # Center surround
            center = raw_sensors.i2c_grayscale[1]
            surround = (raw_sensors.i2c_grayscale[0] + raw_sensors.i2c_grayscale[2]) / 2
            brain_input[13] = (center - surround) / 4095.0 + 0.5
        
        # Channels 14-16: Raw timing signals (brain discovers periodicity)
        t = raw_sensors.timestamp_ns / 1e9
        brain_input[14] = (np.sin(t * 2 * np.pi) + 1.0) / 2.0      # 1Hz reference
        brain_input[15] = (np.sin(t * 10 * np.pi) + 1.0) / 2.0     # 5Hz reference  
        brain_input[16] = (np.sin(t * 40 * np.pi) + 1.0) / 2.0     # 20Hz reference
        
        # Channels 17-19: Current sensor derivatives (brain discovers motor dynamics)
        if self.last_sensor_data:
            dt = (raw_sensors.timestamp_ns - self.last_sensor_data.timestamp_ns) / 1e9
            if dt > 0:
                for i in range(2):
                    delta = raw_sensors.motor_current_raw[i] - self.last_sensor_data.motor_current_raw[i]
                    brain_input[17+i] = (delta / 4095.0 / dt + 1.0) / 2.0
        
        # Channel 20: Reflex state (brain learns to avoid triggering reflexes)
        brain_input[20] = 1.0 if self.reflex_active else 0.0
        
        # Channels 21-23: Reserved for expansion
        # Could add: temperature, IMU, microphone, etc.
        
        self.last_sensor_data = raw_sensors
        return brain_input
    
    def brain_output_to_motors(self, brain_output: List[float]) -> Tuple[RawMotorCommand, RawServoCommand]:
        """
        Convert brain outputs to raw motor commands.
        
        Brain outputs are in [-1, 1] space.
        We convert to PWM duty cycles but brain must learn the response curve.
        
        Fixed mapping:
        - Output 0: Forward/backward (affects both motors)
        - Output 1: Left/right differential
        - Output 2: Steering servo
        - Output 3: Camera pan (or tilt based on config)
        """
        # Ensure we have 4 outputs
        if len(brain_output) < self.brain_output_dim:
            brain_output.extend([0.0] * (self.brain_output_dim - len(brain_output)))
        
        # Channel 0: Primary thrust (forward/backward intent)
        # Channel 1: Differential (turning intent)
        thrust = brain_output[0]  # -1 to 1
        differential = brain_output[1]  # -1 to 1
        
        # Convert to differential drive (brain must learn this mapping)
        left_intent = thrust - differential * 0.5
        right_intent = thrust + differential * 0.5
        
        # Clamp to [-1, 1]
        left_intent = max(-1.0, min(1.0, left_intent))
        right_intent = max(-1.0, min(1.0, right_intent))
        
        # Convert to PWM duty cycle (brain learns actual motor response)
        # We use squared response for finer control at low speeds
        # But brain doesn't know this - it must discover it
        left_duty = abs(left_intent) ** 1.5  # Power curve for better control
        right_duty = abs(right_intent) ** 1.5
        
        # Motor command
        motor_cmd = RawMotorCommand(
            left_pwm_duty=left_duty,
            left_direction=left_intent >= 0,
            right_pwm_duty=right_duty,
            right_direction=right_intent >= 0
        )
        
        # Channel 2: Steering servo
        # Channel 3: Camera pan servo
        # Simple linear mapping with safety limits
        
        # Steering: -1 = full left, 0 = center, 1 = full right
        steering_us = int(1500 + (brain_output[2] * 500))  # ±500µs from center
        steering_us = max(1000, min(2000, steering_us))    # Safety limits
        
        # Camera: -1 = left/down, 0 = center, 1 = right/up
        camera_us = int(1500 + (brain_output[3] * 500))    # ±500µs from center
        camera_us = max(1000, min(2000, camera_us))        # Safety limits
        
        servo_cmd = RawServoCommand(
            steering_us=steering_us,
            camera_pan_us=camera_us,
            camera_tilt_us=1500  # Keep tilt centered for now
        )
        
        return motor_cmd, servo_cmd
    
    def check_safety_reflexes(self, raw_sensors: RawSensorData) -> bool:
        """
        Check for safety conditions that trigger reflexes.
        
        Returns True if reflex was triggered.
        These are the ONLY hardcoded behaviors - pure safety.
        """
        # Check if previous reflex is still active
        if self.reflex_active and time.time() * 1000 < self.reflex_end_time:
            return True
        
        self.reflex_active = False
        
        # Collision reflex
        if raw_sensors.gpio_ultrasonic_us < self.reflex_config.collision_distance_us:
            self._trigger_collision_reflex()
            return True
        
        # Cliff reflex (high grayscale = no ground)
        if any(adc > self.reflex_config.cliff_adc_threshold for adc in raw_sensors.i2c_grayscale):
            self._trigger_cliff_reflex()
            return True
        
        # Overcurrent reflex (motor stall)
        if any(adc > self.reflex_config.current_spike_adc for adc in raw_sensors.motor_current_raw):
            self._trigger_stall_reflex()
            return True
        
        # Critical battery (stop to prevent damage)
        if raw_sensors.analog_battery_raw < self.reflex_config.battery_critical_adc:
            self._trigger_battery_reflex()
            return True
        
        return False
    
    def _trigger_collision_reflex(self):
        """Collision avoidance reflex - back up."""
        print("⚠️ REFLEX: Collision avoidance triggered")
        
        # Back up
        cmd = RawMotorCommand(
            left_pwm_duty=self.reflex_config.collision_backoff_duty,
            left_direction=False,  # Reverse
            right_pwm_duty=self.reflex_config.collision_backoff_duty,
            right_direction=False  # Reverse
        )
        self.hal.execute_motor_command(cmd)
        
        self.reflex_active = True
        self.reflex_end_time = time.time() * 1000 + self.reflex_config.collision_backoff_ms
    
    def _trigger_cliff_reflex(self):
        """Cliff detection reflex - stop immediately."""
        print("⚠️ REFLEX: Cliff detected")
        self.hal.emergency_stop()
        self.reflex_active = True
        self.reflex_end_time = time.time() * 1000 + 1000  # 1 second freeze
    
    def _trigger_stall_reflex(self):
        """Motor stall reflex - stop motors."""
        print("⚠️ REFLEX: Motor stall detected")
        self.hal.emergency_stop()
        self.reflex_active = True
        self.reflex_end_time = time.time() * 1000 + 500  # 0.5 second pause
    
    def _trigger_battery_reflex(self):
        """Critical battery reflex - shutdown."""
        print("🔋 REFLEX: Critical battery - shutting down")
        self.hal.emergency_stop()
        self.reflex_active = True
        self.reflex_end_time = float('inf')  # Permanent until reboot
    
    def process_cycle(self, brain_output: List[float]) -> List[float]:
        """
        Main processing cycle: sensors → brain → motors.
        
        Returns brain input for next cycle.
        """
        # Read raw sensors
        raw_sensors = self.hal.read_raw_sensors()
        
        # Check safety reflexes
        if not self.check_safety_reflexes(raw_sensors):
            # No reflex active - execute brain commands
            motor_cmd, servo_cmd = self.brain_output_to_motors(brain_output)
            self.hal.execute_motor_command(motor_cmd)
            self.hal.execute_servo_command(servo_cmd)
        
        # Package sensors for brain (even during reflex)
        brain_input = self.sensors_to_brain_input(raw_sensors)
        
        return brain_input
    
    def get_hardware_info(self) -> Dict:
        """Expose hardware info for brain's understanding."""
        info = self.hal.get_hardware_info()
        info.update({
            'brain_input_channels': self.brain_input_dim,
            'brain_output_channels': self.brain_output_dim,
            'reflex_thresholds': {
                'collision_us': self.reflex_config.collision_distance_us,
                'cliff_adc': self.reflex_config.cliff_adc_threshold,
                'current_adc': self.reflex_config.current_spike_adc,
                'battery_adc': self.reflex_config.battery_critical_adc,
            }
        })
        return info
    
    def cleanup(self):
        """Clean shutdown."""
        self.hal.cleanup()
        print("🧠 Brainstem shutdown complete")


def test_bare_metal_brainstem():
    """Test the bare metal brainstem adapter."""
    print("🧪 Testing Bare Metal Brainstem")
    print("=" * 50)
    
    # Create brainstem
    brainstem = BareMetalBrainstemAdapter(force_mock=True)
    
    # Show hardware info
    print("\n📊 Hardware configuration:")
    info = brainstem.get_hardware_info()
    print(f"   Brain I/O: {info['brain_input_channels']}→{info['brain_output_channels']}")
    print(f"   PWM frequency: {info['motor_pwm_frequency_hz']}Hz")
    print(f"   ADC resolution: {info['adc_resolution_bits']} bits")
    
    # Simulate brain outputs
    print("\n🧠 Simulating brain control:")
    
    # Test 1: Forward motion
    brain_output = [0.3, 0.0, 0.0, 0.0]  # 30% forward, no turn
    print(f"   Brain output: {brain_output} (forward)")
    brain_input = brainstem.process_cycle(brain_output)
    print(f"   Brain input: [{brain_input[0]:.2f}, {brain_input[1]:.2f}, ...] (first 2 channels)")
    
    # Test 2: Turning
    brain_output = [0.2, 0.5, 0.0, 0.0]  # Forward with right turn
    print(f"\n   Brain output: {brain_output} (turn right)")
    brain_input = brainstem.process_cycle(brain_output)
    print(f"   Ultrasonic channel: {brain_input[3]:.3f}")
    
    # Test 3: Stop
    brain_output = [0.0, 0.0, 0.0, 0.0]  # Stop
    print(f"\n   Brain output: {brain_output} (stop)")
    brain_input = brainstem.process_cycle(brain_output)
    print(f"   Reflex state channel: {brain_input[20]}")
    
    # Cleanup
    brainstem.cleanup()
    print("\n✅ Bare metal brainstem test complete")


if __name__ == "__main__":
    test_bare_metal_brainstem()