#!/usr/bin/env python3
"""
PiCar-X Robot Controller with Bare Metal HAL

Direct hardware control implementation that lets the brain
experience the robot's body without abstractions.

Usage:
    python3 picarx_robot_bare_metal.py                     # Default settings
    python3 picarx_robot_bare_metal.py --brain-host 192.168.1.100  # Connect to brain
    python3 picarx_robot_bare_metal.py --verify            # Run hardware verification
"""

import sys
import os
import time
import signal
import argparse
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import bare metal components
from hardware.bare_metal_hal import (
    BareMetalHAL, create_hal,
    RawMotorCommand, RawServoCommand, RawSensorData
)
from brainstem.bare_metal_adapter import BareMetalBrainstemAdapter
from brainstem.brain_client import BrainClient, BrainServerConfig


@dataclass
class BareMetal RobotConfig:
    """Configuration for bare metal robot."""
    # Brain server settings
    brain_host: str = "localhost"
    brain_port: int = 9999
    use_brain: bool = True
    
    # Hardware settings  
    force_mock: bool = False
    control_rate_hz: float = 20.0
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Telemetry
    record_telemetry: bool = False
    telemetry_file: Optional[str] = None


class BareMetalPiCarXRobot:
    """
    Bare metal robot controller with direct hardware access.
    
    The brain experiences:
    - Raw PWM duty cycles instead of "speed"
    - Ultrasonic echo microseconds instead of "distance"
    - ADC values instead of "voltage"
    - Current draw patterns instead of "motor load"
    
    Only safety reflexes are hardcoded in the brainstem.
    Everything else emerges from experience.
    """
    
    def __init__(self, config: BareMetalRobotConfig):
        """Initialize bare metal robot controller."""
        self.config = config
        self.running = False
        
        # Setup logging
        self._setup_logging()
        
        # Initialize hardware
        self.logger.info("🔧 Initializing bare metal hardware...")
        self.hal = create_hal(force_mock=config.force_mock)
        
        # Initialize brainstem adapter
        self.logger.info("🧠 Initializing brainstem adapter...")
        self.brainstem_adapter = BareMetalBrainstemAdapter(
            force_mock=config.force_mock
        )
        
        # Initialize brain client if needed
        self.brain_client = None
        if config.use_brain:
            self._init_brain_client()
        
        # State tracking
        self.cycle_count = 0
        self.start_time = None
        self.telemetry_log = []
        
        # Performance metrics
        self.cycle_times = []
        self.max_cycle_time = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Show hardware info
        hw_info = self.brainstem_adapter.get_hardware_info()
        self.logger.info(f"📊 Hardware: {hw_info['motor_pwm_frequency_hz']}Hz PWM, "
                        f"{hw_info['adc_resolution_bits']}-bit ADC")
        self.logger.info(f"🧠 Brain I/O: {hw_info['brain_input_channels']} inputs, "
                        f"{hw_info['brain_output_channels']} outputs")
        
        self.logger.info("🚗 Bare metal robot controller ready!")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        
        handlers = [logging.StreamHandler()]
        if self.config.log_file:
            handlers.append(logging.FileHandler(self.config.log_file))
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _init_brain_client(self):
        """Initialize connection to brain server."""
        try:
            brain_config = BrainServerConfig(
                host=self.config.brain_host,
                port=self.config.brain_port,
                timeout=0.05  # 50ms for 20Hz operation
            )
            
            self.brain_client = BrainClient(brain_config)
            
            if self.brain_client.connect():
                self.logger.info(f"✅ Connected to brain at "
                               f"{self.config.brain_host}:{self.config.brain_port}")
            else:
                self.logger.warning("⚠️  Could not connect to brain - "
                                  "running with reflexes only")
                self.brain_client = None
                
        except Exception as e:
            self.logger.error(f"Brain client initialization failed: {e}")
            self.brain_client = None
    
    def control_loop(self):
        """
        Main control loop - sensors → brain → motors.
        
        The brain experiences raw hardware signals and learns
        what they mean through trial and error.
        """
        self.logger.info(f"Starting control loop at {self.config.control_rate_hz}Hz")
        
        cycle_period = 1.0 / self.config.control_rate_hz
        brain_output = [0.0] * 4  # Start with neutral outputs
        
        while self.running:
            cycle_start = time.time()
            self.cycle_count += 1
            
            try:
                # Get brain output (from brain server or local reflexes)
                if self.brain_client:
                    # Send previous sensors, get motor commands
                    brain_response = self.brain_client.process_sensors(brain_input)
                    if brain_response:
                        brain_output = brain_response.get('motor_commands', brain_output)
                
                # Process through brainstem (applies reflexes, executes motors)
                brain_input = self.brainstem_adapter.process_cycle(brain_output)
                
                # Record telemetry if enabled
                if self.config.record_telemetry:
                    self._record_telemetry(brain_input, brain_output)
                
                # Track performance
                cycle_time = time.time() - cycle_start
                self.cycle_times.append(cycle_time)
                if cycle_time > self.max_cycle_time:
                    self.max_cycle_time = cycle_time
                
                # Warn on slow cycles
                if cycle_time > cycle_period * 1.5:
                    self.logger.warning(f"Slow cycle {self.cycle_count}: "
                                      f"{cycle_time*1000:.1f}ms")
                
                # Periodic status
                if self.cycle_count % 100 == 0:
                    self._log_status()
                
                # Maintain cycle rate
                sleep_time = cycle_period - cycle_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                # Safety: stop on error
                self.hal.emergency_stop()
                time.sleep(0.1)
        
        self.logger.info("Control loop stopped")
    
    def _record_telemetry(self, brain_input: List[float], brain_output: List[float]):
        """Record telemetry data for analysis."""
        entry = {
            'timestamp': time.time(),
            'cycle': self.cycle_count,
            'brain_input': brain_input,
            'brain_output': brain_output,
            'reflex_active': brain_input[20] > 0.5,  # Reflex state channel
        }
        
        self.telemetry_log.append(entry)
        
        # Keep log manageable
        if len(self.telemetry_log) > 10000:
            self.telemetry_log = self.telemetry_log[-5000:]
    
    def _log_status(self):
        """Log periodic status information."""
        # Calculate metrics
        recent_cycles = self.cycle_times[-100:] if self.cycle_times else [0]
        avg_cycle = np.mean(recent_cycles) if recent_cycles else 0
        
        # Get current sensor state
        sensors = self.hal.read_raw_sensors()
        
        status = f"Cycle {self.cycle_count} | "
        status += f"Avg: {avg_cycle*1000:.1f}ms | "
        status += f"Max: {self.max_cycle_time*1000:.1f}ms | "
        status += f"Echo: {sensors.gpio_ultrasonic_us:.0f}µs | "
        status += f"Battery: {sensors.analog_battery_raw}"
        
        if self.brain_client:
            status += " | Brain: ✓"
        else:
            status += " | Brain: ✗"
        
        self.logger.info(status)
    
    def start(self):
        """Start the robot controller."""
        self.logger.info("🚀 Starting bare metal PiCar-X robot...")
        self.running = True
        self.start_time = time.time()
        
        # Initialize brain input with current sensors
        sensors = self.hal.read_raw_sensors()
        brain_input = self.brainstem_adapter.sensors_to_brain_input(sensors)
        
        if self.brain_client:
            # Send initial sensors to brain
            self.brain_client.process_sensors(brain_input)
        
        # Run control loop
        try:
            self.control_loop()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown the robot."""
        if not self.running:
            return
        
        self.logger.info("🛑 Shutting down robot...")
        self.running = False
        
        # Emergency stop
        self.hal.emergency_stop()
        
        # Disconnect brain
        if self.brain_client:
            self.brain_client.disconnect()
        
        # Save telemetry if recorded
        if self.config.record_telemetry and self.telemetry_log:
            self._save_telemetry()
        
        # Log final statistics
        if self.start_time:
            runtime = time.time() - self.start_time
            self.logger.info(f"Total runtime: {runtime:.1f}s")
            self.logger.info(f"Total cycles: {self.cycle_count}")
            self.logger.info(f"Average rate: {self.cycle_count/runtime:.1f}Hz")
        
        # Cleanup hardware
        self.brainstem_adapter.cleanup()
        
        self.logger.info("✅ Shutdown complete")
    
    def _save_telemetry(self):
        """Save telemetry data to file."""
        import json
        
        filename = self.config.telemetry_file or f"telemetry_{int(time.time())}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.telemetry_log, f)
            self.logger.info(f"📊 Telemetry saved to {filename} "
                           f"({len(self.telemetry_log)} entries)")
        except Exception as e:
            self.logger.error(f"Failed to save telemetry: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}")
        self.shutdown()
        sys.exit(0)


def run_hardware_verification():
    """Run hardware verification suite."""
    print("🧪 Running hardware verification...")
    
    from tests.hardware_verification import HardwareVerification
    
    verifier = HardwareVerification(verbose=True)
    results = verifier.run_all_tests()
    
    # Check critical components
    if results.get('platform') and results.get('gpio_control'):
        print("\n✅ Hardware verification passed!")
        return True
    else:
        print("\n❌ Hardware verification failed!")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Bare Metal PiCar-X Robot Controller')
    
    # Operation mode
    parser.add_argument('--verify', action='store_true',
                      help='Run hardware verification tests')
    
    # Brain settings
    parser.add_argument('--brain-host', default='localhost',
                      help='Brain server hostname/IP')
    parser.add_argument('--brain-port', type=int, default=9999,
                      help='Brain server port')
    parser.add_argument('--no-brain', action='store_true',
                      help='Run with reflexes only (no brain)')
    
    # Hardware settings
    parser.add_argument('--mock', action='store_true',
                      help='Use mock hardware for testing')
    parser.add_argument('--rate', type=float, default=20.0,
                      help='Control loop rate in Hz')
    
    # Telemetry
    parser.add_argument('--record', action='store_true',
                      help='Record telemetry data')
    parser.add_argument('--telemetry-file',
                      help='Telemetry output file')
    
    # Logging
    parser.add_argument('--log-level', default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level')
    parser.add_argument('--log-file',
                      help='Log to file')
    
    args = parser.parse_args()
    
    # Run verification if requested
    if args.verify:
        success = run_hardware_verification()
        sys.exit(0 if success else 1)
    
    # Check platform
    try:
        with open('/proc/device-tree/model', 'r') as f:
            if 'Raspberry Pi' not in f.read() and not args.mock:
                print("⚠️  Not on Raspberry Pi - enabling mock mode")
                args.mock = True
    except:
        if not args.mock:
            print("⚠️  Platform detection failed - enabling mock mode")
            args.mock = True
    
    # Create configuration
    config = BareMetalRobotConfig(
        brain_host=args.brain_host,
        brain_port=args.brain_port,
        use_brain=not args.no_brain,
        force_mock=args.mock,
        control_rate_hz=args.rate,
        record_telemetry=args.record,
        telemetry_file=args.telemetry_file,
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    # Print banner
    print("=" * 60)
    print("🤖 BARE METAL PiCar-X ROBOT CONTROLLER")
    print("=" * 60)
    print(f"Brain: {config.brain_host}:{config.brain_port}")
    print(f"Hardware: {'Mock' if config.force_mock else 'Real'}")
    print(f"Control: {config.control_rate_hz}Hz")
    print(f"Telemetry: {'Recording' if config.record_telemetry else 'Off'}")
    print("=" * 60)
    print("The brain will experience raw hardware signals:")
    print("  - PWM duty cycles (not 'speed')")
    print("  - Echo microseconds (not 'distance')")
    print("  - ADC values (not 'voltage')")
    print("  - Current patterns (not 'load')")
    print("=" * 60)
    
    # Create and start robot
    robot = BareMetalPiCarXRobot(config)
    robot.start()


if __name__ == "__main__":
    # Fix import for numpy
    import numpy as np
    main()