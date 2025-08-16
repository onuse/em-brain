#!/usr/bin/env python3
"""
Simple PiCar-X Robot Controller

This is the FINAL, CLEAN implementation using:
- Clean brainstem (minimal safety reflexes)
- Bare metal HAL (direct hardware control)
- Verified PiCar-X limits (from SDK analysis)

No magic, no discovery, just solid engineering.

Usage:
    sudo python3 picarx_robot_simple.py                      # Connect to localhost brain
    sudo python3 picarx_robot_simple.py --brain-host 192.168.1.100  # Remote brain
    sudo python3 picarx_robot_simple.py --no-brain          # Reflexes only
    sudo python3 picarx_robot_simple.py --calibrate         # Calibrate servos
"""

import sys
import os
import time
import signal
import argparse
import json
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from brainstem.brainstem import Brainstem
from hardware.bare_metal_hal import create_hal
from hardware.picarx_hardware_limits import *
from streams.battery_stream import BatteryStreamer


class PiCarXRobot:
    """
    Simple, clean robot controller.
    
    Just what's needed, nothing more.
    """
    
    def __init__(self, brain_host: str = None, brain_port: int = None, 
                 use_brain: bool = True, mock_hardware: bool = False, log_level: str = 'WARNING'):
        """Initialize robot."""
        
        print("🤖 Simple PiCar-X Robot Controller")
        print("=" * 50)
        
        # Check if on Raspberry Pi
        self.on_pi = self._check_platform()
        if not self.on_pi and not mock_hardware:
            print("⚠️  Not on Raspberry Pi - using mock hardware")
            mock_hardware = True
        
        # Initialize components
        if use_brain:
            # Let brainstem handle config loading if not specified
            if brain_host or brain_port:
                print(f"📡 Connecting to brain at {brain_host}:{brain_port}")
            else:
                print(f"📡 Using brain configuration from robot_config.json")
            self.brainstem = Brainstem(brain_host, brain_port, log_level=log_level)
            
            # Start battery stream (parallel to main connection)
            self.battery_stream = BatteryStreamer(
                brain_host or self.brainstem.brain_host,
                port=10004,
                use_tcp=False  # Try UDP first
            )
            self.battery_stream.start()
            print("🔋 Battery stream started on port 10004")
        else:
            print("🧠 Running in reflex-only mode (no brain)")
            self.brainstem = Brainstem(log_level=log_level)  # Will work without brain
            self.battery_stream = None
        
        self.running = False
        self.cycle_count = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("✅ Robot initialized and ready!")
        print("=" * 50)
    
    def _check_platform(self) -> bool:
        """Check if running on Raspberry Pi."""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                return 'Raspberry Pi' in f.read()
        except:
            return False
    
    def run(self, rate_hz: float = 20.0):
        """
        Run robot at specified rate.
        
        Simple control loop:
        1. Read sensors
        2. Check safety
        3. Get brain commands (or use reflexes)
        4. Execute on hardware
        """
        
        print(f"\n🚀 Starting robot control at {rate_hz}Hz")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        period = 1.0 / rate_hz
        
        # Print header for status
        print("Cycle | Distance | Battery | Brain | Status")
        print("-" * 50)
        
        while self.running:
            start = time.time()
            
            # Run one brainstem cycle
            self.brainstem.run_cycle()
            self.cycle_count += 1
            
            # Simple status every 20 cycles (1 second at 20Hz)
            if self.cycle_count % 20 == 0:
                self._print_status()
            
            # Maintain rate
            elapsed = time.time() - start
            if elapsed < period:
                time.sleep(period - elapsed)
        
        print("\n🛑 Robot stopped")
    
    def _print_status(self):
        """Print simple status line."""
        # Get latest sensor data
        try:
            sensors = self.brainstem.hal.read_raw_sensors()
            distance_cm = sensors.gpio_ultrasonic_us / 58.0
            battery_v = (sensors.analog_battery_raw / 4095.0) * 10.0
            brain_ok = "✓" if self.brainstem.brain_client else "✗"
            
            status = "OK"
            if self.brainstem.reflex_active:
                status = "REFLEX"
            elif distance_cm < 20:
                status = "CAUTION"
            
            print(f"{self.cycle_count:5d} | {distance_cm:7.1f}cm | {battery_v:5.1f}V | {brain_ok:5s} | {status}")
            
        except:
            pass  # Silent fail for status
    
    def calibrate_servos(self):
        """
        Interactive servo calibration.
        
        Tests each servo to find comfortable limits.
        """
        
        print("\n🔧 SERVO CALIBRATION MODE")
        print("=" * 50)
        print("This will move servos to find their limits.")
        print("Make sure the robot is secure!\n")
        
        hal = self.brainstem.hal
        calibration = {}
        
        # Calibrate steering
        print("STEERING SERVO (Channel 2)")
        print("-" * 30)
        hal._set_servo_pulse(2, 1500)
        input("Servo at center (1500µs). Press Enter to continue...")
        
        hal._set_servo_pulse(2, 1200)
        left_ok = input("Servo at 1200µs. Is this a safe LEFT limit? (y/n): ")
        if left_ok.lower() == 'y':
            calibration['steering_left'] = 1200
        else:
            new_val = int(input("Enter safe LEFT limit (µs): "))
            calibration['steering_left'] = new_val
        
        hal._set_servo_pulse(2, 1800)
        right_ok = input("Servo at 1800µs. Is this a safe RIGHT limit? (y/n): ")
        if right_ok.lower() == 'y':
            calibration['steering_right'] = 1800
        else:
            new_val = int(input("Enter safe RIGHT limit (µs): "))
            calibration['steering_right'] = new_val
        
        hal._set_servo_pulse(2, 1500)  # Return to center
        
        # Calibrate camera pan
        print("\nCAMERA PAN SERVO (Channel 0)")
        print("-" * 30)
        hal._set_servo_pulse(0, 1500)
        input("Servo at center (1500µs). Press Enter to continue...")
        
        hal._set_servo_pulse(0, 1000)
        left_ok = input("Servo at 1000µs. Is this a safe LEFT limit? (y/n): ")
        if left_ok.lower() == 'y':
            calibration['pan_left'] = 1000
        else:
            new_val = int(input("Enter safe LEFT limit (µs): "))
            calibration['pan_left'] = new_val
        
        hal._set_servo_pulse(0, 2000)
        right_ok = input("Servo at 2000µs. Is this a safe RIGHT limit? (y/n): ")
        if right_ok.lower() == 'y':
            calibration['pan_right'] = 2000
        else:
            new_val = int(input("Enter safe RIGHT limit (µs): "))
            calibration['pan_right'] = new_val
        
        hal._set_servo_pulse(0, 1500)  # Return to center
        
        # Save calibration
        with open('servo_calibration.json', 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print("\n✅ Calibration complete and saved!")
        print(json.dumps(calibration, indent=2))
    
    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        if self.battery_stream:
            self.battery_stream.stop()
            print("🔋 Battery stream stopped")
        self.brainstem.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n⚠️ Received signal {signum}")
        self.shutdown()
        sys.exit(0)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Simple PiCar-X Robot Controller')
    
    # Brain connection
    parser.add_argument('--brain-host', default=None,
                      help='Brain server hostname/IP')
    parser.add_argument('--brain-port', type=int, default=None,
                      help='Brain server port')
    parser.add_argument('--no-brain', action='store_true',
                      help='Run without brain (reflexes only)')
    
    # Operation modes
    parser.add_argument('--calibrate', action='store_true',
                      help='Run servo calibration')
    parser.add_argument('--test', action='store_true',
                      help='Run hardware test')
    
    # Settings
    parser.add_argument('--rate', type=float, default=20.0,
                      help='Control loop rate in Hz')
    parser.add_argument('--mock', action='store_true',
                      help='Use mock hardware')
    parser.add_argument('--log-level', type=str, default='WARNING',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level (default: WARNING for clean output)')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 60)
    print("🚗 SIMPLE PICAR-X ROBOT CONTROLLER")
    print("=" * 60)
    print(f"Version: 1.0 (Clean Architecture)")
    if args.brain_host or args.brain_port:
        print(f"Brain: {args.brain_host or 'config'}:{args.brain_port or 'config'}")
    else:
        print(f"Brain: Using robot_config.json")
    print(f"Rate: {args.rate}Hz")
    print("=" * 60)
    
    # Run requested mode
    if args.calibrate:
        # Calibration mode
        robot = PiCarXRobot(use_brain=False, mock_hardware=args.mock, log_level=args.log_level)
        robot.calibrate_servos()
        
    elif args.test:
        # Hardware test mode
        print("\n🧪 HARDWARE TEST MODE")
        print("-" * 40)
        
        robot = PiCarXRobot(use_brain=False, mock_hardware=args.mock, log_level=args.log_level)
        
        print("\nTesting sensors...")
        for i in range(5):
            sensors = robot.brainstem.hal.read_raw_sensors()
            distance_cm = sensors.gpio_ultrasonic_us / 58.0
            print(f"  Distance: {distance_cm:.1f}cm, "
                  f"Grayscale: {sensors.i2c_grayscale}")
            time.sleep(0.5)
        
        print("\nTesting motors (forward)...")
        from src.hardware.raw_robot_hat_hal import RawMotorCommand
        robot.brainstem.hal.execute_motor_command(
            RawMotorCommand(0.3, 0.3)  # left_pwm_duty, right_pwm_duty
        )
        time.sleep(1)
        robot.brainstem.hal.emergency_stop()
        
        print("\nTesting servos...")
        for pulse in [1200, 1500, 1800, 1500]:
            robot.brainstem.hal._set_servo_pulse(2, pulse)
            print(f"  Steering: {pulse}µs")
            time.sleep(0.5)
        
        print("\n✅ Hardware test complete!")
        
    else:
        # Normal operation
        robot = PiCarXRobot(
            brain_host=args.brain_host,
            brain_port=args.brain_port,
            use_brain=not args.no_brain,
            mock_hardware=args.mock,
            log_level=args.log_level
        )
        
        try:
            robot.run(args.rate)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            robot.shutdown()


if __name__ == "__main__":
    main()