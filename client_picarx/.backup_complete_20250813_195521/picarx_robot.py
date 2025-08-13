#!/usr/bin/env python3
"""
PiCar-X Robot Controller

Main entry point for running the PiCar-X robot with brain integration.
Handles hardware initialization, sensor reading, motor control, and safety features.

Usage:
    python3 picarx_robot.py                          # Default settings
    python3 picarx_robot.py --brain-host 192.168.1.100  # Custom brain server
    python3 picarx_robot.py --mock                      # Run with mock hardware
    python3 picarx_robot.py --no-brain                  # Autonomous mode only
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

# Check if we're on Raspberry Pi
def is_raspberry_pi():
    """Check if running on Raspberry Pi hardware."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            return 'Raspberry Pi' in model
    except:
        return False

# Try to import PiCar-X library
PICARX_AVAILABLE = False
try:
    from picarx import Picarx
    from robot_hat import Music, TTS
    PICARX_AVAILABLE = True
    print("✅ PiCar-X hardware library loaded")
except ImportError:
    print("⚠️  PiCar-X library not available - using mock hardware")

# Import our brainstem components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from brainstem.integrated_brainstem import IntegratedBrainstem, BrainstemConfig
from brainstem.brain_client import BrainServerConfig


@dataclass
class RobotConfig:
    """Configuration for the PiCar-X robot."""
    # Brain server settings
    brain_host: str = "localhost"
    brain_port: int = 9999  # Standard brain server port
    use_brain: bool = True
    
    # Hardware settings
    use_mock_hardware: bool = False
    control_rate_hz: float = 20.0
    
    # Safety settings
    enable_safety: bool = True
    emergency_stop_distance: float = 0.05  # 5cm
    low_battery_threshold: float = 6.2     # Volts
    high_temp_threshold: float = 70        # Celsius
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Features
    enable_voice: bool = False
    enable_line_following: bool = True
    enable_obstacle_avoidance: bool = True


class MockPicarx:
    """Mock PiCar-X implementation for testing without hardware."""
    
    def __init__(self):
        self.servo_positions = [0, 0, 0]  # pan, tilt, steering
        self.motor_speeds = [0, 0]  # left, right
        self.battery_voltage = 7.4
        self.cpu_temp = 45.0
        self.ultrasonic_distance = 50  # cm
        self.grayscale_values = [500, 500, 500]
        print("🤖 Mock PiCar-X initialized")
    
    def get_distance(self):
        """Get ultrasonic distance in cm."""
        return self.ultrasonic_distance
    
    def get_grayscale_data(self):
        """Get grayscale sensor data."""
        return self.grayscale_values
    
    def get_battery_voltage(self):
        """Get battery voltage."""
        return self.battery_voltage
    
    def get_cpu_temp(self):
        """Get CPU temperature."""
        return self.cpu_temp
    
    def set_motor_speed(self, motor, speed):
        """Set motor speed for a specific motor."""
        # motor: 1 = left, 2 = right
        # speed: -100 to 100
        if motor == 1:
            self.motor_speeds[0] = speed
        elif motor == 2:
            self.motor_speeds[1] = speed
    
    def set_servo_angle(self, servo_id, angle):
        """Set servo angle."""
        if 0 <= servo_id <= 2:
            self.servo_positions[servo_id] = angle
    
    def forward(self, speed):
        """Move forward."""
        self.set_motor_speed(1, speed)
        self.set_motor_speed(2, speed)
    
    def backward(self, speed):
        """Move backward."""
        self.set_motor_speed(1, -speed)
        self.set_motor_speed(2, -speed)
    
    def stop(self):
        """Stop all motors."""
        self.set_motor_speed(1, 0)
        self.set_motor_speed(2, 0)
    
    def set_dir_servo_angle(self, angle):
        """Set steering servo angle."""
        self.servo_positions[2] = angle
    
    def set_cam_pan_angle(self, angle):
        """Set camera pan servo angle."""
        self.servo_positions[0] = angle
    
    def set_cam_tilt_angle(self, angle):
        """Set camera tilt servo angle."""
        self.servo_positions[1] = angle


class PiCarXRobot:
    """
    Main PiCar-X robot controller with brain integration.
    
    Handles:
    - Hardware initialization and management
    - Sensor reading and normalization
    - Brain communication via brainstem
    - Motor command execution
    - Safety monitoring and emergency stops
    - Graceful shutdown
    """
    
    def __init__(self, config: RobotConfig):
        """Initialize the robot controller."""
        self.config = config
        self.running = False
        self.emergency_stop_active = False
        
        # Setup logging
        self._setup_logging()
        
        # Initialize hardware
        self._init_hardware()
        
        # Initialize brainstem if using brain
        self.brainstem = None
        if config.use_brain:
            self._init_brainstem()
        
        # State tracking
        self.cycle_count = 0
        self.last_cycle_time = time.time()
        self.sensor_history = []
        self.motor_history = []
        
        # Performance tracking
        self.cycle_times = []
        self.max_cycle_time = 0
        self.total_runtime = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🚗 PiCar-X Robot Controller initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        
        if self.config.log_file:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format=log_format,
                handlers=[
                    logging.FileHandler(self.config.log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format=log_format
            )
        
        self.logger = logging.getLogger(__name__)
    
    def _init_hardware(self):
        """Initialize PiCar-X hardware or mock."""
        if self.config.use_mock_hardware or not PICARX_AVAILABLE:
            self.logger.info("Using mock hardware")
            self.px = MockPicarx()
        else:
            try:
                self.logger.info("Initializing real PiCar-X hardware...")
                self.px = Picarx()
                
                # Calibrate servos to center position
                self.px.set_dir_servo_angle(0)
                
                # Initialize camera servos using correct API
                self.px.set_cam_pan_angle(0)
                self.px.set_cam_tilt_angle(0)
                
                # Initialize voice if available and enabled
                if self.config.enable_voice:
                    try:
                        self.tts = TTS()
                        self.music = Music()
                        self.tts.say("PiCar-X ready")
                    except:
                        self.logger.warning("Voice features not available")
                        self.tts = None
                        self.music = None
                
                self.logger.info("✅ Hardware initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize hardware: {e}")
                self.logger.info("Falling back to mock hardware")
                self.px = MockPicarx()
    
    def _init_brainstem(self):
        """Initialize the brainstem for brain communication."""
        from brainstem.integrated_brainstem import IntegratedBrainstemConfig
        
        brain_config = BrainServerConfig(
            host=self.config.brain_host,
            port=self.config.brain_port,
            timeout=0.05  # 50ms timeout for 20Hz operation
        )
        
        # Get default brainstem config
        from config.brainstem_config import get_config
        brainstem_config = get_config()
        
        # Create integrated config
        integrated_config = IntegratedBrainstemConfig(
            brain_server_config=brain_config,
            brainstem_config=brainstem_config,
            use_mock_brain=False
        )
        
        self.brainstem = IntegratedBrainstem(integrated_config)
        
        # Try to connect to brain
        if self.brainstem.connect():
            self.logger.info("Connected to brain server")
        else:
            self.logger.warning("Could not connect to brain - running in autonomous mode")
    
    def _read_sensors(self) -> List[float]:
        """
        Read all sensors from PiCar-X hardware.
        
        Returns:
            List of 16 sensor values matching expected format
        """
        try:
            # Read primary sensors
            distance = self.px.get_distance() / 100.0  # Convert cm to meters
            grayscale = self.px.get_grayscale_data()
            
            # Normalize grayscale (0-1000 → 0-1)
            grayscale_norm = [g / 1000.0 for g in grayscale]
            
            # Get motor speeds (if available)
            try:
                left_motor = self.px.motor_speeds[0] / 100.0
                right_motor = self.px.motor_speeds[1] / 100.0
            except:
                left_motor = 0.0
                right_motor = 0.0
            
            # Get servo positions
            try:
                camera_pan = self.px.servo_positions[0]
                camera_tilt = self.px.servo_positions[1]
                steering = self.px.servo_positions[2]
            except:
                camera_pan = 0
                camera_tilt = 0
                steering = 0
            
            # Get system info
            battery = self.px.get_battery_voltage()
            cpu_temp = self.px.get_cpu_temp()
            
            # Line and cliff detection (basic thresholding)
            line_detected = 1 if grayscale_norm[1] > 0.6 else 0
            cliff_detected = 1 if distance > 1.0 else 0  # Large distance = cliff
            
            # Memory usage estimate
            import psutil
            memory_usage = psutil.virtual_memory().percent / 100.0
            
            # Compile sensor array
            sensors = [
                distance,                    # 0: ultrasonic distance (meters)
                grayscale_norm[2],          # 1: right grayscale
                grayscale_norm[1],          # 2: center grayscale
                grayscale_norm[0],          # 3: left grayscale
                left_motor,                 # 4: left motor speed
                right_motor,                # 5: right motor speed
                camera_pan,                 # 6: camera pan angle
                camera_tilt,                # 7: camera tilt angle
                steering,                   # 8: steering angle
                battery,                    # 9: battery voltage
                line_detected,              # 10: line detected flag
                cliff_detected,             # 11: cliff detected flag
                cpu_temp,                   # 12: CPU temperature
                memory_usage,               # 13: memory usage
                time.time() * 1000,         # 14: timestamp (ms)
                0                           # 15: reserved
            ]
            
            return sensors
            
        except Exception as e:
            self.logger.error(f"Error reading sensors: {e}")
            # Return safe default values
            return [0.5] * 16
    
    def _execute_motors(self, motor_commands: Dict[str, float]):
        """
        Execute motor commands on hardware.
        
        Args:
            motor_commands: Dictionary with motor command values
        """
        try:
            # Check for emergency stop
            if self.emergency_stop_active:
                self.px.stop()
                return
            
            # Apply motor commands
            left = motor_commands.get('left_motor', 0)
            right = motor_commands.get('right_motor', 0)
            
            # Set motor speeds
            if abs(left) < 5 and abs(right) < 5:
                self.px.stop()
            else:
                # PiCar-X API requires calling set_motor_speed twice - once per motor
                self.px.set_motor_speed(1, int(left))   # Motor 1 = left
                self.px.set_motor_speed(2, int(right))  # Motor 2 = right
            
            # Set servo positions
            steering = motor_commands.get('steering_servo', 0)
            self.px.set_dir_servo_angle(int(steering))
            
            camera_pan = motor_commands.get('camera_pan_servo', 0)
            camera_tilt = motor_commands.get('camera_tilt_servo', 0)
            
            # Set camera angles using correct API
            self.px.set_cam_pan_angle(int(camera_pan))
            self.px.set_cam_tilt_angle(int(camera_tilt))
            
        except Exception as e:
            self.logger.error(f"Error executing motors: {e}")
            self.px.stop()
    
    def _check_safety(self, sensors: List[float]) -> bool:
        """
        Check safety conditions and trigger emergency stop if needed.
        
        Returns:
            True if safe to continue, False if emergency stop triggered
        """
        if not self.config.enable_safety:
            return True
        
        # Check distance
        if sensors[0] < self.config.emergency_stop_distance:
            self.logger.warning(f"⚠️  Emergency stop - obstacle at {sensors[0]:.2f}m")
            self.emergency_stop_active = True
            self.px.stop()
            return False
        
        # Check battery
        if sensors[9] < self.config.low_battery_threshold:
            self.logger.warning(f"⚠️  Low battery: {sensors[9]:.1f}V")
            if sensors[9] < 6.0:  # Critical level
                self.logger.error("🔋 Critical battery - shutting down")
                self.shutdown()
                return False
        
        # Check temperature
        if sensors[12] > self.config.high_temp_threshold:
            self.logger.warning(f"🌡️  High CPU temperature: {sensors[12]:.1f}°C")
            if sensors[12] > 80:  # Critical level
                self.logger.error("🔥 Critical temperature - shutting down")
                self.shutdown()
                return False
        
        # Check cliff
        if sensors[11] > 0:
            self.logger.warning("⚠️  Cliff detected!")
            self.emergency_stop_active = True
            self.px.stop()
            return False
        
        # Clear emergency stop if conditions are safe
        if self.emergency_stop_active and sensors[0] > 0.2:
            self.emergency_stop_active = False
            self.logger.info("✅ Emergency stop cleared")
        
        return True
    
    def _autonomous_behavior(self, sensors: List[float]) -> Dict[str, float]:
        """
        Generate autonomous behavior when brain is not available.
        
        Simple obstacle avoidance and line following.
        """
        distance = sensors[0]
        grayscale_left = sensors[3]
        grayscale_center = sensors[2]
        grayscale_right = sensors[1]
        
        # Obstacle avoidance takes priority
        if distance < 0.2:
            # Too close - back up and turn
            return {
                'left_motor': -20,
                'right_motor': -20,
                'steering_servo': 20 if sensors[8] >= 0 else -20,
                'camera_pan_servo': 0,
                'camera_tilt_servo': 0
            }
        elif distance < 0.4:
            # Getting close - slow down and prepare to turn
            base_speed = 15
        else:
            # Safe distance
            base_speed = 25
        
        # Line following
        if self.config.enable_line_following and grayscale_center > 0.6:
            # On the line
            steering = 0
        elif grayscale_left > grayscale_right:
            # Line is to the left
            steering = -15
        elif grayscale_right > grayscale_left:
            # Line is to the right
            steering = 15
        else:
            # No line - just go straight
            steering = 0
        
        return {
            'left_motor': base_speed,
            'right_motor': base_speed,
            'steering_servo': steering,
            'camera_pan_servo': 0,
            'camera_tilt_servo': 0
        }
    
    def control_loop(self):
        """Main control loop - runs at configured rate."""
        self.logger.info(f"Starting control loop at {self.config.control_rate_hz}Hz")
        
        cycle_period = 1.0 / self.config.control_rate_hz
        
        while self.running:
            cycle_start = time.time()
            self.cycle_count += 1
            
            try:
                # Read sensors
                sensors = self._read_sensors()
                self.sensor_history.append(sensors)
                if len(self.sensor_history) > 100:
                    self.sensor_history.pop(0)
                
                # Check safety
                if not self._check_safety(sensors):
                    continue
                
                # Get motor commands
                if self.brainstem and self.config.use_brain:
                    # Use brain via brainstem
                    motor_commands = self.brainstem.process_cycle(sensors)
                    
                    # Check for vocal commands
                    vocal = self.brainstem.get_vocal_commands()
                    if vocal and self.tts:
                        self.tts.say(vocal.get('text', ''))
                else:
                    # Use autonomous behavior
                    motor_commands = self._autonomous_behavior(sensors)
                
                # Execute motor commands
                self._execute_motors(motor_commands)
                
                # Store motor history
                self.motor_history.append(motor_commands)
                if len(self.motor_history) > 100:
                    self.motor_history.pop(0)
                
                # Performance tracking
                cycle_time = time.time() - cycle_start
                self.cycle_times.append(cycle_time)
                if cycle_time > self.max_cycle_time:
                    self.max_cycle_time = cycle_time
                
                # Log slow cycles
                if cycle_time > cycle_period * 1.5:
                    self.logger.warning(f"Slow cycle {self.cycle_count}: {cycle_time*1000:.1f}ms")
                
                # Periodic status update
                if self.cycle_count % 100 == 0:
                    self._log_status()
                
                # Sleep to maintain rate
                sleep_time = cycle_period - cycle_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Error in control loop: {e}")
                self.px.stop()
                time.sleep(0.1)
        
        self.logger.info("Control loop stopped")
    
    def _log_status(self):
        """Log periodic status information."""
        avg_cycle_time = sum(self.cycle_times[-100:]) / min(100, len(self.cycle_times))
        
        status_msg = f"Cycle {self.cycle_count} | "
        status_msg += f"Avg: {avg_cycle_time*1000:.1f}ms | "
        status_msg += f"Max: {self.max_cycle_time*1000:.1f}ms | "
        
        if self.brainstem:
            brain_status = self.brainstem.get_status()
            status_msg += f"Brain: {'✓' if brain_status['connected'] else '✗'} | "
            status_msg += f"Reflexes: {brain_status['reflex_activations']}"
        
        self.logger.info(status_msg)
    
    def start(self):
        """Start the robot controller."""
        self.logger.info("🚀 Starting PiCar-X robot...")
        self.running = True
        self.total_runtime = time.time()
        
        # Start control loop in main thread
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
        
        # Stop motors
        self.px.stop()
        
        # Center servos
        try:
            self.px.set_dir_servo_angle(0)
            # Reset camera servos using correct API
            self.px.set_cam_pan_angle(0)
            self.px.set_cam_tilt_angle(0)
        except:
            pass
        
        # Disconnect brainstem
        if self.brainstem:
            self.brainstem.shutdown()
        
        # Log final statistics
        runtime = time.time() - self.total_runtime
        self.logger.info(f"Total runtime: {runtime:.1f}s")
        self.logger.info(f"Total cycles: {self.cycle_count}")
        self.logger.info(f"Average rate: {self.cycle_count/runtime:.1f}Hz")
        
        if self.tts:
            self.tts.say("Goodbye")
        
        self.logger.info("✅ Shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}")
        self.shutdown()
        sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='PiCar-X Robot Controller')
    
    # Brain settings
    parser.add_argument('--brain-host', default='localhost',
                      help='Brain server hostname/IP (default: localhost)')
    parser.add_argument('--brain-port', type=int, default=9999,
                      help='Brain server port (default: 9999)')
    parser.add_argument('--no-brain', action='store_true',
                      help='Run in autonomous mode without brain')
    
    # Hardware settings
    parser.add_argument('--mock', action='store_true',
                      help='Use mock hardware for testing')
    parser.add_argument('--rate', type=float, default=20.0,
                      help='Control loop rate in Hz (default: 20)')
    
    # Safety settings
    parser.add_argument('--no-safety', action='store_true',
                      help='Disable safety features (not recommended)')
    
    # Features
    parser.add_argument('--voice', action='store_true',
                      help='Enable voice output')
    parser.add_argument('--no-line-following', action='store_true',
                      help='Disable line following behavior')
    
    # Logging
    parser.add_argument('--log-level', default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level (default: INFO)')
    parser.add_argument('--log-file', help='Log to file')
    
    args = parser.parse_args()
    
    # Check platform
    if not args.mock and not is_raspberry_pi():
        print("⚠️  Not running on Raspberry Pi - enabling mock mode")
        args.mock = True
    
    # Create configuration
    config = RobotConfig(
        brain_host=args.brain_host,
        brain_port=args.brain_port,
        use_brain=not args.no_brain,
        use_mock_hardware=args.mock,
        control_rate_hz=args.rate,
        enable_safety=not args.no_safety,
        enable_voice=args.voice,
        enable_line_following=not args.no_line_following,
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    # Print startup banner
    print("=" * 60)
    print("🚗 PiCar-X Robot Controller")
    print("=" * 60)
    print(f"Brain Server: {config.brain_host}:{config.brain_port}")
    print(f"Hardware: {'Mock' if config.use_mock_hardware else 'Real'}")
    print(f"Control Rate: {config.control_rate_hz}Hz")
    print(f"Safety: {'Enabled' if config.enable_safety else 'DISABLED'}")
    print("=" * 60)
    
    # Create and start robot
    robot = PiCarXRobot(config)
    robot.start()


if __name__ == "__main__":
    main()