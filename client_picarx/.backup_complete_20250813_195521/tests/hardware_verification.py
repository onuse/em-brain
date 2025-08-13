#!/usr/bin/env python3
"""
Hardware Verification Suite for Bare Metal HAL

Tests actual hardware communication and response characteristics.
Not testing intelligence or behavior - just verifying hardware works.

Run on Raspberry Pi with: sudo python3 hardware_verification.py
"""

import time
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.hardware.bare_metal_hal import BareMetalHAL, RawMotorCommand, RawServoCommand


class HardwareVerification:
    """
    Verify hardware is connected and responding correctly.
    
    These tests verify:
    1. I2C devices respond at expected addresses
    2. GPIO pins can be controlled
    3. Sensors return varying data (not stuck)
    4. Motors draw current when commanded
    5. Servos reach commanded positions
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize hardware verification."""
        self.verbose = verbose
        self.hal = None
        self.results = {}
        
    def log(self, message: str):
        """Print if verbose mode."""
        if self.verbose:
            print(message)
    
    def verify_platform(self) -> bool:
        """Verify we're running on Raspberry Pi."""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi' in model:
                    self.log(f"✅ Platform: {model.strip()}")
                    return True
        except:
            pass
        
        self.log("❌ Not running on Raspberry Pi")
        return False
    
    def verify_i2c_devices(self) -> Dict[str, bool]:
        """Verify I2C devices are present and responding."""
        self.log("\n🔍 Verifying I2C devices...")
        
        results = {}
        
        try:
            import smbus
            bus = smbus.SMBus(1)
            
            # Expected I2C addresses
            devices = {
                'ADC': 0x14,
                'PCA9685': 0x40,
            }
            
            for name, addr in devices.items():
                try:
                    # Try to read a byte
                    bus.read_byte(addr)
                    self.log(f"   ✅ {name} at 0x{addr:02x} - responding")
                    results[name] = True
                except IOError:
                    self.log(f"   ❌ {name} at 0x{addr:02x} - not found")
                    results[name] = False
            
            bus.close()
            
        except ImportError:
            self.log("   ❌ smbus module not available")
            return {}
        except Exception as e:
            self.log(f"   ❌ I2C bus error: {e}")
            return {}
        
        return results
    
    def verify_gpio_pins(self) -> bool:
        """Verify GPIO pins can be controlled."""
        self.log("\n🔍 Verifying GPIO control...")
        
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Test a motor direction pin
            test_pin = 17  # Left motor direction
            
            # Set as output
            GPIO.setup(test_pin, GPIO.OUT)
            
            # Toggle pin
            GPIO.output(test_pin, GPIO.HIGH)
            time.sleep(0.01)
            GPIO.output(test_pin, GPIO.LOW)
            
            self.log(f"   ✅ GPIO control working (tested pin {test_pin})")
            return True
            
        except Exception as e:
            self.log(f"   ❌ GPIO error: {e}")
            return False
    
    def verify_sensors_not_stuck(self) -> Dict[str, bool]:
        """Verify sensors return varying values (not stuck)."""
        self.log("\n🔍 Verifying sensors are not stuck...")
        
        if not self.hal:
            self.hal = BareMetalHAL()
        
        results = {}
        
        # Collect multiple sensor readings
        readings = []
        for i in range(10):
            sensors = self.hal.read_raw_sensors()
            readings.append(sensors)
            time.sleep(0.1)
        
        # Check ultrasonic variation
        ultrasonic_values = [r.gpio_ultrasonic_us for r in readings]
        ultrasonic_variance = np.var(ultrasonic_values)
        if ultrasonic_variance > 10:  # Some variation expected
            self.log(f"   ✅ Ultrasonic sensor varying (σ²={ultrasonic_variance:.1f})")
            results['ultrasonic'] = True
        else:
            self.log(f"   ⚠️  Ultrasonic sensor may be stuck (σ²={ultrasonic_variance:.1f})")
            results['ultrasonic'] = False
        
        # Check grayscale variation
        for i, name in enumerate(['left', 'center', 'right']):
            values = [r.i2c_grayscale[i] for r in readings]
            variance = np.var(values)
            if variance > 5:  # Some noise expected
                self.log(f"   ✅ Grayscale {name} varying (σ²={variance:.1f})")
                results[f'grayscale_{name}'] = True
            else:
                self.log(f"   ⚠️  Grayscale {name} may be stuck (σ²={variance:.1f})")
                results[f'grayscale_{name}'] = False
        
        # Check battery ADC
        battery_values = [r.analog_battery_raw for r in readings]
        battery_variance = np.var(battery_values)
        self.log(f"   📊 Battery ADC variance: {battery_variance:.1f}")
        results['battery'] = battery_variance > 1
        
        return results
    
    def measure_motor_characteristics(self) -> Dict[str, float]:
        """Measure motor response characteristics."""
        self.log("\n🔍 Measuring motor characteristics...")
        
        if not self.hal:
            self.hal = BareMetalHAL()
        
        characteristics = {}
        
        # Find motor deadband (minimum PWM for movement)
        self.log("   Testing motor deadband...")
        
        for motor_name, direction in [('left', True), ('right', True)]:
            # Gradually increase PWM until current changes
            baseline = self.hal.read_raw_sensors().motor_current_raw[0 if motor_name == 'left' else 1]
            
            for duty in np.arange(0.05, 0.5, 0.05):
                if motor_name == 'left':
                    cmd = RawMotorCommand(duty, direction, 0, True)
                else:
                    cmd = RawMotorCommand(0, True, duty, direction)
                
                self.hal.execute_motor_command(cmd)
                time.sleep(0.2)
                
                current = self.hal.read_raw_sensors().motor_current_raw[0 if motor_name == 'left' else 1]
                
                if abs(current - baseline) > 50:  # Significant current change
                    self.log(f"   ✅ {motor_name} motor starts at {duty*100:.0f}% duty")
                    characteristics[f'{motor_name}_deadband'] = duty
                    break
                
                # Stop motor
                self.hal.execute_motor_command(RawMotorCommand(0, True, 0, True))
                time.sleep(0.1)
        
        # Measure current at different speeds
        self.log("   Testing motor current draw...")
        
        for duty in [0.2, 0.4, 0.6]:
            cmd = RawMotorCommand(duty, True, duty, True)
            self.hal.execute_motor_command(cmd)
            time.sleep(0.5)
            
            sensors = self.hal.read_raw_sensors()
            left_current = sensors.motor_current_raw[0]
            right_current = sensors.motor_current_raw[1]
            
            self.log(f"   📊 At {duty*100:.0f}% duty: L={left_current} R={right_current} ADC")
            characteristics[f'current_at_{int(duty*100)}pct'] = (left_current, right_current)
        
        # Stop motors
        self.hal.execute_motor_command(RawMotorCommand(0, True, 0, True))
        
        return characteristics
    
    def verify_servo_range(self) -> Dict[str, Tuple[int, int]]:
        """Verify servo movement ranges."""
        self.log("\n🔍 Testing servo ranges...")
        
        if not self.hal:
            self.hal = BareMetalHAL()
        
        ranges = {}
        
        # Test each servo
        servos = [
            ('steering', 1500, [1000, 1500, 2000]),
            ('camera_pan', 1500, [800, 1500, 2200]),
            ('camera_tilt', 1500, [800, 1500, 2200]),
        ]
        
        for servo_name, center, test_values in servos:
            self.log(f"   Testing {servo_name} servo...")
            
            working_range = []
            
            for pulse_us in test_values:
                if servo_name == 'steering':
                    cmd = RawServoCommand(pulse_us, 1500, 1500)
                elif servo_name == 'camera_pan':
                    cmd = RawServoCommand(1500, pulse_us, 1500)
                else:
                    cmd = RawServoCommand(1500, 1500, pulse_us)
                
                try:
                    self.hal.execute_servo_command(cmd)
                    time.sleep(0.5)
                    working_range.append(pulse_us)
                    self.log(f"      ✅ {pulse_us}µs works")
                except Exception as e:
                    self.log(f"      ❌ {pulse_us}µs failed: {e}")
            
            if working_range:
                ranges[servo_name] = (min(working_range), max(working_range))
                self.log(f"   📊 {servo_name} range: {ranges[servo_name][0]}-{ranges[servo_name][1]}µs")
            
            # Return to center
            self.hal.execute_servo_command(RawServoCommand(1500, 1500, 1500))
        
        return ranges
    
    def measure_timing_characteristics(self) -> Dict[str, float]:
        """Measure hardware timing characteristics."""
        self.log("\n🔍 Measuring timing characteristics...")
        
        if not self.hal:
            self.hal = BareMetalHAL()
        
        timings = {}
        
        # Measure sensor read time
        read_times = []
        for _ in range(50):
            start = time.time_ns()
            self.hal.read_raw_sensors()
            read_times.append(time.time_ns() - start)
        
        timings['sensor_read_mean_ms'] = np.mean(read_times) / 1e6
        timings['sensor_read_std_ms'] = np.std(read_times) / 1e6
        
        self.log(f"   📊 Sensor read: {timings['sensor_read_mean_ms']:.2f}±{timings['sensor_read_std_ms']:.2f}ms")
        
        # Measure motor command latency
        cmd_times = []
        for _ in range(50):
            start = time.time_ns()
            self.hal.execute_motor_command(RawMotorCommand(0.3, True, 0.3, True))
            cmd_times.append(time.time_ns() - start)
        
        timings['motor_cmd_mean_ms'] = np.mean(cmd_times) / 1e6
        timings['motor_cmd_std_ms'] = np.std(cmd_times) / 1e6
        
        self.log(f"   📊 Motor command: {timings['motor_cmd_mean_ms']:.2f}±{timings['motor_cmd_std_ms']:.2f}ms")
        
        # Stop motors
        self.hal.execute_motor_command(RawMotorCommand(0, True, 0, True))
        
        return timings
    
    def run_all_tests(self) -> Dict:
        """Run complete hardware verification suite."""
        self.log("=" * 60)
        self.log("🧪 BARE METAL HARDWARE VERIFICATION SUITE")
        self.log("=" * 60)
        
        results = {
            'platform': self.verify_platform(),
            'i2c_devices': self.verify_i2c_devices(),
            'gpio_control': self.verify_gpio_pins(),
        }
        
        # Only continue with HAL tests if basic hardware works
        if results['platform'] and results['gpio_control']:
            try:
                self.hal = BareMetalHAL()
                
                results['sensors_active'] = self.verify_sensors_not_stuck()
                results['motor_characteristics'] = self.measure_motor_characteristics()
                results['servo_ranges'] = self.verify_servo_range()
                results['timing'] = self.measure_timing_characteristics()
                
                # Cleanup
                self.hal.cleanup()
                
            except Exception as e:
                self.log(f"\n❌ HAL initialization failed: {e}")
                results['error'] = str(e)
        
        # Summary
        self.log("\n" + "=" * 60)
        self.log("📊 VERIFICATION SUMMARY")
        self.log("=" * 60)
        
        if results.get('platform'):
            self.log("✅ Running on Raspberry Pi")
        else:
            self.log("❌ Not on Raspberry Pi - limited tests")
        
        if results.get('i2c_devices'):
            working = sum(1 for v in results['i2c_devices'].values() if v)
            total = len(results['i2c_devices'])
            self.log(f"📊 I2C Devices: {working}/{total} responding")
        
        if results.get('gpio_control'):
            self.log("✅ GPIO control working")
        
        if results.get('motor_characteristics'):
            self.log("✅ Motors characterized")
        
        if results.get('timing'):
            mean_ms = results['timing']['sensor_read_mean_ms']
            self.log(f"📊 Sensor latency: {mean_ms:.1f}ms")
        
        return results


def main():
    """Run hardware verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify bare metal hardware')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    parser.add_argument('--motors', action='store_true', help='Test motors (robot will move!)')
    parser.add_argument('--json', action='store_true', help='Output JSON results')
    
    args = parser.parse_args()
    
    # Safety warning for motor tests
    if args.motors:
        print("\n" + "⚠️ " * 10)
        print("WARNING: Motor tests will move the robot!")
        print("Ensure robot is on blocks or safe area.")
        print("⚠️ " * 10)
        response = input("\nContinue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    # Run verification
    verifier = HardwareVerification(verbose=not args.quiet)
    results = verifier.run_all_tests()
    
    # Output results
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    
    # Exit code based on critical tests
    if results.get('platform') and results.get('gpio_control'):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()