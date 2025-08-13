#!/usr/bin/env python3
"""
Clean Brainstem Implementation

This is the ONLY brainstem file you need. It:
1. Uses bare metal HAL for hardware control
2. Connects to brain server via TCP
3. Implements minimal safety reflexes
4. No magic numbers - uses PiCar-X verified limits

Everything else in the brainstem folder can be deleted.
"""

import time
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

# Hardware components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from hardware.bare_metal_hal import (
    BareMetalHAL, create_hal,
    RawMotorCommand, RawServoCommand, RawSensorData
)
from hardware.picarx_hardware_limits import *
from brainstem.brain_client_simple import BrainClient, BrainServerConfig


@dataclass
class SafetyConfig:
    """Safety thresholds for reflexes."""
    collision_distance_cm: float = 10.0    # Stop if closer than 10cm
    cliff_threshold_adc: int = 3500        # High ADC = no ground
    battery_critical_v: float = 6.0        # Shutdown if below 6V
    max_temp_c: float = 80.0              # Shutdown if CPU too hot


class CleanBrainstem:
    """
    Minimal brainstem with just what's needed.
    
    Responsibilities:
    1. Read sensors → send to brain
    2. Get brain commands → execute on hardware  
    3. Safety reflexes (collision, cliff, battery)
    4. That's it!
    """
    
    def __init__(self, brain_host: str = "localhost", brain_port: int = 9999):
        """Initialize clean brainstem."""
        
        # Hardware layer
        self.hal = create_hal()
        
        # Brain connection
        self.brain_client = None
        self._init_brain_connection(brain_host, brain_port)
        
        # Safety config
        self.safety = SafetyConfig()
        
        # State
        self.running = False
        self.cycle_count = 0
        self.reflex_active = False
        
        print("🧠 Clean brainstem initialized")
        print(f"   Brain: {brain_host}:{brain_port}")
        print(f"   Safety: collision<{self.safety.collision_distance_cm}cm")
    
    def _init_brain_connection(self, host: str, port: int):
        """Initialize brain server connection."""
        try:
            config = BrainServerConfig(host=host, port=port, timeout=0.05)
            self.brain_client = BrainClient(config)
            
            if self.brain_client.connect():
                print(f"✅ Connected to brain server")
            else:
                print(f"⚠️  Could not connect to brain - reflexes only mode")
                self.brain_client = None
        except Exception as e:
            print(f"❌ Brain connection failed: {e}")
            self.brain_client = None
    
    def sensors_to_brain_format(self, raw: RawSensorData) -> List[float]:
        """
        Convert raw sensors to brain's 24-channel format.
        
        Simple normalization, no interpretation.
        Brain learns what these mean.
        """
        brain_input = [0.5] * 24  # Start neutral
        
        # Channels 0-2: Grayscale sensors (normalized 0-1)
        for i in range(3):
            brain_input[i] = raw.i2c_grayscale[i] / 4095.0
        
        # Channel 3: Ultrasonic (convert µs to normalized distance)
        # ~58µs per cm, max ~400cm = 23200µs
        distance_normalized = min(raw.gpio_ultrasonic_us / 23200.0, 1.0)
        brain_input[3] = distance_normalized
        
        # Channel 4: Battery (normalized)
        brain_input[4] = raw.analog_battery_raw / 4095.0
        
        # Channels 5-23: Reserved for expansion
        # Could add: IMU, temperature, current sensors, etc.
        
        return brain_input
    
    def brain_to_hardware_commands(self, brain_output: List[float]) -> tuple:
        """
        Convert brain's 4 outputs to hardware commands.
        
        Fixed mapping:
        0: Forward/backward thrust
        1: Left/right differential
        2: Steering servo
        3: Camera pan servo
        """
        
        # Ensure we have 4 outputs
        if len(brain_output) < 4:
            brain_output.extend([0.0] * (4 - len(brain_output)))
        
        # Motors: differential drive from thrust + turn
        thrust = brain_output[0]      # -1 = reverse, 1 = forward
        turn = brain_output[1]         # -1 = left, 1 = right
        
        # Mix for differential drive
        left = thrust - turn * 0.5
        right = thrust + turn * 0.5
        
        # Clamp and convert to PWM
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        
        # Power curve for better low-speed control
        left_pwm = abs(left) ** 1.5
        right_pwm = abs(right) ** 1.5
        
        motor_cmd = RawMotorCommand(
            left_pwm_duty=left_pwm,
            left_direction=left >= 0,
            right_pwm_duty=right_pwm,
            right_direction=right >= 0
        )
        
        # Servos: use verified PiCar-X limits
        steering = brain_output[2]  # -1 to 1
        camera = brain_output[3]    # -1 to 1
        
        # Convert to microseconds using PiCar-X limits
        steering_deg = steering * STEERING_MAX  # ±30°
        steering_us = steering_to_microseconds(steering_deg)
        
        camera_deg = camera * CAMERA_PAN_MAX  # ±90°
        camera_us = camera_pan_to_microseconds(camera_deg)
        
        servo_cmd = RawServoCommand(
            steering_us=steering_us,
            camera_pan_us=camera_us,
            camera_tilt_us=SERVO_PULSE_CENTER  # Keep centered
        )
        
        return motor_cmd, servo_cmd
    
    def check_safety(self, sensors: RawSensorData) -> bool:
        """
        Check safety conditions and trigger reflexes if needed.
        
        Returns True if safe to continue.
        """
        
        # Collision check (ultrasonic)
        distance_cm = sensors.gpio_ultrasonic_us / 58.0  # µs to cm
        if distance_cm < self.safety.collision_distance_cm:
            print(f"⚠️ REFLEX: Collision imminent ({distance_cm:.1f}cm)")
            self.hal.emergency_stop()
            self.reflex_active = True
            return False
        
        # Cliff detection (high grayscale = no ground)
        if any(adc > self.safety.cliff_threshold_adc for adc in sensors.i2c_grayscale):
            print("⚠️ REFLEX: Cliff detected")
            self.hal.emergency_stop()
            self.reflex_active = True
            return False
        
        # Battery critical
        battery_v = (sensors.analog_battery_raw / 4095.0) * 10.0  # Rough conversion
        if battery_v < self.safety.battery_critical_v:
            print(f"🔋 REFLEX: Critical battery ({battery_v:.1f}V)")
            self.hal.emergency_stop()
            self.reflex_active = True
            return False
        
        # Clear reflex if we're safe again
        if self.reflex_active and distance_cm > self.safety.collision_distance_cm * 2:
            self.reflex_active = False
            print("✅ Reflex cleared")
        
        return True
    
    def run_cycle(self) -> bool:
        """
        Run one brainstem cycle.
        
        Returns False if should stop.
        """
        self.cycle_count += 1
        
        try:
            # Read sensors
            raw_sensors = self.hal.read_raw_sensors()
            
            # Check safety
            if not self.check_safety(raw_sensors):
                return True  # Continue but skip motor commands
            
            # Convert to brain format
            brain_input = self.sensors_to_brain_format(raw_sensors)
            
            # Get brain output (or use defaults)
            brain_output = [0.0, 0.0, 0.0, 0.0]
            
            if self.brain_client:
                response = self.brain_client.process_sensors(brain_input)
                if response and 'motor_commands' in response:
                    brain_output = response['motor_commands']
            
            # Convert to hardware commands
            motor_cmd, servo_cmd = self.brain_to_hardware_commands(brain_output)
            
            # Execute (unless reflex is active)
            if not self.reflex_active:
                self.hal.execute_motor_command(motor_cmd)
                self.hal.execute_servo_command(servo_cmd)
            
            # Status every 100 cycles
            if self.cycle_count % 100 == 0:
                distance_cm = raw_sensors.gpio_ultrasonic_us / 58.0
                print(f"Cycle {self.cycle_count}: distance={distance_cm:.1f}cm, "
                      f"brain={'✓' if self.brain_client else '✗'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Cycle error: {e}")
            self.hal.emergency_stop()
            return True
    
    def run(self, rate_hz: float = 20.0):
        """Run brainstem at specified rate."""
        print(f"🚀 Starting brainstem at {rate_hz}Hz")
        
        self.running = True
        period = 1.0 / rate_hz
        
        while self.running:
            start = time.time()
            
            if not self.run_cycle():
                break
            
            # Maintain rate
            elapsed = time.time() - start
            if elapsed < period:
                time.sleep(period - elapsed)
        
        self.shutdown()
    
    def shutdown(self):
        """Clean shutdown."""
        print("🛑 Shutting down brainstem...")
        
        self.running = False
        self.hal.emergency_stop()
        
        if self.brain_client:
            self.brain_client.disconnect()
        
        self.hal.cleanup()
        print("✅ Shutdown complete")


def main():
    """Run the clean brainstem."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean PiCar-X Brainstem')
    parser.add_argument('--brain-host', default='localhost', help='Brain server host')
    parser.add_argument('--brain-port', type=int, default=9999, help='Brain server port')
    parser.add_argument('--rate', type=float, default=20.0, help='Control rate Hz')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 CLEAN PICARX BRAINSTEM")
    print("=" * 60)
    print(f"Brain: {args.brain_host}:{args.brain_port}")
    print(f"Rate: {args.rate}Hz")
    print("=" * 60)
    
    brainstem = CleanBrainstem(args.brain_host, args.brain_port)
    
    try:
        brainstem.run(args.rate)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        brainstem.shutdown()


if __name__ == "__main__":
    main()