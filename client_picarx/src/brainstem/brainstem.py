#!/usr/bin/env python3
"""
Clean Brainstem Implementation

This is the ONLY brainstem file you need. It:
1. Uses bare metal HAL for hardware control
2. Connects to brain server via TCP
3. Implements minimal safety reflexes
4. Loads all settings from robot_config.json
5. Provides telemetry monitoring on port 9997

Everything else in the brainstem folder can be deleted.
"""

import time
import json
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Hardware components
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Try new raw HAL first, fallback to bare metal HAL
try:
    from hardware.raw_robot_hat_hal import (
        RawRobotHatHAL, create_hal,
        RawMotorCommand, RawServoCommand, RawSensorData
    )
    print("Using raw_robot_hat_hal (with local robot-hat library)")
except ImportError:
    from hardware.bare_metal_hal import (
        BareMetalHAL, create_hal,
        RawMotorCommand, RawServoCommand, RawSensorData
    )
    print("Fallback to bare_metal_hal")
    
from hardware.picarx_hardware_limits import *

# Import brain client with fallback handling
try:
    from brainstem.brain_client import BrainClient, BrainServerConfig
except ImportError:
    try:
        from brain_client import BrainClient, BrainServerConfig  
    except ImportError:
        from .brain_client import BrainClient, BrainServerConfig

# Import monitoring
try:
    from brainstem.brainstem_monitor import BrainstemMonitor
except ImportError:
    try:
        from brainstem_monitor import BrainstemMonitor
    except ImportError:
        from .brainstem_monitor import BrainstemMonitor

# Import vision streamer for UDP (optional)
try:
    from streams.vision_stream import VisionStreamAdapter
    VISION_STREAM_AVAILABLE = True
except ImportError:
    VISION_STREAM_AVAILABLE = False


def load_robot_config() -> Dict[str, Any]:
    """Load robot configuration from JSON file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "robot_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print(f"⚠️ Config file not found at {config_path}, using defaults")
        return {
            "safety": {
                "collision_distance_cm": 3.0,
                "cliff_threshold_adc": 3500,
                "max_speed": 0.8
            },
            "sensors": {
                "battery": {"critical_voltage": 6.0}
            },
            "brain": {
                "timeout": 0.05
            }
        }


@dataclass
class SafetyConfig:
    """Safety thresholds for reflexes - loaded from config."""
    collision_distance_cm: float = 3.0
    cliff_threshold_adc: int = 3500
    battery_critical_v: float = 6.0
    max_temp_c: float = 80.0
    max_speed: float = 0.8
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SafetyConfig':
        """Create SafetyConfig from robot configuration."""
        safety = config.get("safety", {})
        sensors = config.get("sensors", {})
        
        return cls(
            collision_distance_cm=safety.get("collision_distance_cm", 3.0),
            cliff_threshold_adc=safety.get("cliff_threshold_adc", 3500),
            battery_critical_v=sensors.get("battery", {}).get("critical_voltage", 6.0),
            max_temp_c=80.0,  # Not in config yet, using default
            max_speed=safety.get("max_speed", 0.8)
        )


class Brainstem:
    """
    Minimal brainstem with just what's needed.
    
    Responsibilities:
    1. Read sensors → send to brain
    2. Get brain commands → execute on hardware  
    3. Safety reflexes (collision, cliff, battery)
    4. That's it!
    """
    
    def __init__(self, brain_host: str = None, brain_port: int = None, config_path: str = None, 
                 enable_monitor: bool = True, enable_brain: bool = True):
        """Initialize clean brainstem with configuration."""
        
        # Load configuration
        self.config = load_robot_config() if not config_path else json.load(open(config_path))
        
        # Hardware layer
        self.hal = create_hal()
        
        # Brain connection (use config values if not explicitly overridden)
        brain_config = self.config.get("brain", {})
        self.brain_host = brain_host if brain_host is not None else brain_config.get("host", "localhost") 
        self.brain_port = brain_port if brain_port is not None else brain_config.get("port", 9999)
        self.brain_client = None
        self.last_connect_attempt = 0
        self.reconnect_interval = 5.0  # Try reconnecting every 5 seconds
        
        # Vision stream adapter (for UDP streaming) - MUST BE BEFORE BRAIN CONNECTION
        self.vision_stream = None
        if VISION_STREAM_AVAILABLE and enable_brain:
            try:
                self.vision_stream = VisionStreamAdapter(self.brain_host, enabled=True)
                print(f"   ✅ Vision UDP streaming enabled to {self.brain_host}:10002")
            except Exception as e:
                print(f"   ⚠️  Vision streaming setup failed: {e}")
        
        # Now init brain connection (after vision_stream exists)
        if enable_brain:
            self._init_brain_connection(self.brain_host, self.brain_port)
        
        # Safety config from JSON
        self.safety = SafetyConfig.from_config(self.config)
        
        # Performance settings
        perf_config = self.config.get("performance", {})
        self.control_loop_hz = perf_config.get("control_loop_hz", 20)
        
        # State
        self.running = False
        self.cycle_count = 0
        self.reflex_active = False
        
        # Monitoring
        self.monitor = None
        if enable_monitor:
            monitor_port = self.config.get("debug", {}).get("monitor_port", 9997)
            self.monitor = BrainstemMonitor(port=monitor_port)
            self.monitor.start()
        
        # Setup logging for brainstem debugging
        self.logger = logging.getLogger('Brainstem')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        print("🧠 Clean brainstem initialized")
        print(f"   Config: Loaded from robot_config.json")
        print(f"   Brain: {self.brain_host}:{self.brain_port}")
        print(f"   Monitor: Port {monitor_port if enable_monitor else 'disabled'}")
        print(f"   Safety: collision<{self.safety.collision_distance_cm}cm, battery>{self.safety.battery_critical_v}V")
        print(f"   Performance: {self.control_loop_hz}Hz control loop")
    
    def _init_brain_connection(self, host: str, port: int):
        """Initialize brain server connection."""
        self.logger.info(f"BRAIN_INIT: Initializing connection to {host}:{port}")
        
        try:
            # Dynamically determine our sensor dimensions based on vision resolution
            basic_sensors = 5      # Grayscale (3) + ultrasonic + battery
            audio_features = 7     # Audio feature channels
            
            # Check if vision is going via UDP (parallel) or TCP
            if self.vision_stream and self.vision_stream.enabled:
                # Vision via UDP - don't count in TCP dimensions!
                vision_pixels = 0
                print(f"🔍 Vision via UDP stream (parallel processing)")
            else:
                # Vision via TCP - count in dimensions
                vision_config = self.config.get("vision", {})
                resolution = vision_config.get("resolution", [640, 480])
                vision_pixels = resolution[0] * resolution[1]
                
                # Override with actual HAL resolution if available
                if self.hal and hasattr(self.hal, 'vision') and self.hal.vision:
                    vision_pixels = self.hal.vision.output_dim
                    print(f"🔍 Vision resolution from HAL: {vision_pixels:,} pixels (TCP)")
                else:
                    print(f"🔍 Vision resolution from config: {vision_pixels:,} pixels ({resolution[0]}x{resolution[1]}) (TCP)")
            
            sensory_dims = basic_sensors + vision_pixels + audio_features
            action_dims = 6    # We use 6 outputs (motors + servos + audio)
            
            print(f"📊 Sensor dimensions: {sensory_dims:,} total")
            print(f"   - Basic sensors: {basic_sensors}")
            print(f"   - Vision pixels: {vision_pixels:,}")
            print(f"   - Audio features: {audio_features}")
            
            config = BrainServerConfig(
                host=host, 
                port=port, 
                timeout=2.0,  # Increased for large vision data (was 0.05)
                sensory_dimensions=sensory_dims,
                action_dimensions=action_dims
            )
            self.brain_client = BrainClient(config)
            
            self.logger.info("CONNECTION_ATTEMPT: Attempting to connect to brain server")
            if self.brain_client.connect():
                self.logger.info("CONNECTION_SUCCESS: Successfully connected to brain server")
                print(f"✅ Connected to brain server")
            else:
                self.logger.warning("CONNECTION_FAILED: Could not connect to brain server, entering reflexes-only mode")
                print(f"⚠️  Could not connect to brain - reflexes only mode")
                self.brain_client = None
        except Exception as e:
            self.logger.error(f"BRAIN_INIT_ERROR: Brain connection initialization failed: {e}")
            print(f"❌ Brain connection failed: {e}")
            import traceback
            self.logger.debug(f"BRAIN_INIT_TRACEBACK: {traceback.format_exc()}")
            self.brain_client = None
        
        self.last_connect_attempt = time.time()
    
    def _try_reconnect(self):
        """Try to reconnect to brain server."""
        current_time = time.time()
        if current_time - self.last_connect_attempt < self.reconnect_interval:
            return  # Too soon to retry
        
        self.logger.info(f"RECONNECT_ATTEMPT: Trying to reconnect (last attempt {current_time - self.last_connect_attempt:.1f}s ago)")
        print(f"🔄 Attempting to reconnect to brain server...")
        self.last_connect_attempt = current_time
        
        # Clean up old client
        if self.brain_client:
            self.logger.debug("RECONNECT_CLEANUP: Disconnecting old client")
            self.brain_client.disconnect()
            self.brain_client = None
        
        # Try connecting again
        self._init_brain_connection(self.brain_host, self.brain_port)
    
    def sensors_to_brain_format(self, raw: RawSensorData) -> List[float]:
        """
        Convert raw sensors to brain's expanded input format.
        
        Dynamic sizing based on actual vision data:
        - Basic sensors: 5 channels
        - Vision: len(raw.vision_data) pixels
        - Audio: 7 channels
        """
        brain_input = []
        
        # Channels 0-2: Grayscale sensors (normalized 0-1)
        for i in range(3):
            brain_input.append(raw.i2c_grayscale[i] / 4095.0)
        
        # Channel 3: Ultrasonic (convert µs to normalized distance)
        # ~58µs per cm, max ~400cm = 23200µs
        distance_normalized = min(raw.gpio_ultrasonic_us / 23200.0, 1.0)
        brain_input.append(distance_normalized)
        
        # Channel 4: Battery (normalized)
        brain_input.append(raw.analog_battery_raw / 4095.0)
        
        # Vision data - send via UDP if available, otherwise TCP
        if self.vision_stream and self.vision_stream.enabled:
            # Send vision via UDP (parallel processing!)
            self.vision_stream.process_vision(raw.vision_data)
            # Don't send vision via TCP - dimensions already exclude it
        else:
            # Send vision via TCP (may block brain at high res)
            brain_input.extend(raw.vision_data)
        
        # Audio features (7 channels)
        # Volume + 4 frequency bands + pitch + onset
        brain_input.extend(raw.audio_features)
        
        return brain_input
    
    def brain_to_hardware_commands(self, brain_output: List[float]) -> tuple:
        """
        Convert brain's outputs to hardware commands.
        
        Using 6 outputs now:
        0: Forward/backward thrust
        1: Left/right differential
        2: Steering servo
        3: Camera pan servo
        4: Audio frequency (for vocalizations)
        5: Audio volume
        """
        
        # Ensure we have at least 6 outputs
        while len(brain_output) < 6:
            brain_output.append(0.0)
        
        # Motors: differential drive from thrust + turn
        thrust = brain_output[0]      # -1 = reverse, 1 = forward
        turn = brain_output[1]         # -1 = left, 1 = right
        
        # Mix for differential drive
        left = thrust - turn * 0.5
        right = thrust + turn * 0.5
        
        # Debug output every 50 cycles
        if self.cycle_count % 50 == 0:
            print(f"🧠 Brain output: thrust={thrust:.2f}, turn={turn:.2f} → L={left:.2f}, R={right:.2f}")
        
        # Clamp and convert to PWM
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        
        # Apply power curve for better low-speed control
        # Then apply safety speed limit
        left_motor = (abs(left) ** 1.5) * (1 if left >= 0 else -1)
        right_motor = (abs(right) ** 1.5) * (1 if right >= 0 else -1)
        
        # Apply safety limits
        left_motor = left_motor * self.safety.max_speed
        right_motor = right_motor * self.safety.max_speed
        
        # New raw HAL uses signed duty cycles (-1 to +1)
        motor_cmd = RawMotorCommand(
            left_pwm_duty=left_motor,   # Signed: negative = reverse
            right_pwm_duty=right_motor  # Signed: negative = reverse
        )
        
        # Servos: use verified PiCar-X limits
        steering = brain_output[2]  # -1 to 1
        camera = brain_output[3]    # -1 to 1
        
        # Convert to microseconds using PiCar-X limits
        steering_deg = steering * STEERING_MAX  # ±30°
        steering_us = steering_to_microseconds(steering_deg)
        
        camera_deg = camera * CAMERA_PAN_MAX  # ±90°
        camera_us = camera_pan_to_microseconds(camera_deg)
        
        # Use correct field names for raw HAL (pw = pulse width)
        servo_cmd = RawServoCommand(
            steering_pw=steering_us,      # Raw pulse width in microseconds
            camera_pan_pw=camera_us,       # Raw pulse width in microseconds
            camera_tilt_pw=SERVO_PULSE_CENTER  # Keep centered
        )
        
        # Audio output (channels 4-5)
        audio_freq = brain_output[4]  # 0-1 frequency control
        audio_vol = brain_output[5]   # 0-1 volume control
        
        # Generate audio if HAL has audio capability
        if self.hal and hasattr(self.hal, 'audio') and self.hal.audio:
            self.hal.audio.generate_sound_from_brain([audio_freq, audio_vol])
        
        return motor_cmd, servo_cmd
    
    def check_safety(self, sensors: RawSensorData) -> bool:
        """
        Check safety conditions and trigger reflexes if needed.
        
        Returns True if safe to continue.
        """
        
        # Collision check (ultrasonic)
        us_per_cm = self.config.get("sensors", {}).get("ultrasonic", {}).get("us_per_cm", 58.0)
        
        # Check for invalid readings (sensor error)
        min_valid_us = self.config.get("safety", {}).get("ignore_ultrasonic_below_us")
        if sensors.gpio_ultrasonic_us < min_valid_us:  
            # Sensor not reading properly - skip collision check
            if self.cycle_count % 50 == 0:  # Don't spam
                print(f"⚠️  Ultrasonic sensor error (reading: {sensors.gpio_ultrasonic_us}µs < {min_valid_us}µs)")
            distance_cm = 999  # Assume far away when sensor fails
        else:
            distance_cm = sensors.gpio_ultrasonic_us / us_per_cm
        
        if distance_cm < self.safety.collision_distance_cm:
            print(f"⚠️ REFLEX: Collision imminent ({distance_cm:.1f}cm)")
            self.hal.emergency_stop()
            self.reflex_active = True
            if self.monitor:
                self.monitor.trigger_reflex('collision')
            return False
        
        # Cliff detection (high grayscale = no ground)
        if any(adc > self.safety.cliff_threshold_adc for adc in sensors.i2c_grayscale):
            print("⚠️ REFLEX: Cliff detected")
            self.hal.emergency_stop()
            self.reflex_active = True
            if self.monitor:
                self.monitor.trigger_reflex('cliff')
            return False
        
        # Battery critical
        battery_config = self.config.get("sensors", {}).get("battery", {})
        max_adc = battery_config.get("max_adc_value", 4095)
        voltage_mult = battery_config.get("adc_to_voltage_multiplier", 10.0)
        battery_v = (sensors.analog_battery_raw / max_adc) * voltage_mult
        if battery_v < self.safety.battery_critical_v:
            print(f"🔋 REFLEX: Critical battery ({battery_v:.1f}V)")
            self.hal.emergency_stop()
            self.reflex_active = True
            if self.monitor:
                self.monitor.trigger_reflex('battery')
            return False
        
        # Clear reflex if we're safe again
        if self.reflex_active and distance_cm > self.safety.collision_distance_cm * 2:
            self.reflex_active = False
            if self.monitor:
                self.monitor.clear_reflex()
            print("✅ Reflex cleared")
        
        # Update sensor metrics
        if self.monitor:
            self.monitor.update_sensors(
                distance_cm=distance_cm,
                battery_v=battery_v,
                grayscale=list(sensors.i2c_grayscale)
            )
        
        return True
    
    def run_cycle(self) -> bool:
        """
        Run one brainstem cycle with telemetry.
        
        Returns False if should stop.
        """
        cycle_start = time.perf_counter()
        self.cycle_count += 1
        
        try:
            # Read sensors (with timing)
            sensor_start = time.perf_counter()
            raw_sensors = self.hal.read_raw_sensors()
            sensor_time = (time.perf_counter() - sensor_start) * 1000
            
            # Check safety
            if not self.check_safety(raw_sensors):
                return True  # Continue but skip motor commands
            
            # Convert to brain format
            brain_input = self.sensors_to_brain_format(raw_sensors)
            
            # Debug logging for vision issue
            if self.cycle_count % 100 == 0:  # Every 100 cycles (~5 seconds at 20Hz)
                print(f"📊 Sensor data: basic={len(raw_sensors.i2c_grayscale)+2}, " +
                      f"vision={len(raw_sensors.vision_data)}, audio={len(raw_sensors.audio_features)}, " +
                      f"total={len(brain_input)}")
            
            # Get brain output (or use defaults)
            brain_output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 outputs
            comm_time = 0
            
            if self.brain_client:
                comm_start = time.perf_counter()
                response = self.brain_client.process_sensors(brain_input)
                comm_time = (time.perf_counter() - comm_start) * 1000
                
                if response and 'motor_commands' in response:
                    brain_output = response['motor_commands']
                    if self.monitor:
                        self.monitor.update_connection(connected=True, timeout=False)
                elif response is None and not self.brain_client.connected:
                    # Lost connection, try to reconnect
                    self._try_reconnect()
                    if self.monitor:
                        self.monitor.update_connection(connected=False, timeout=True)
            elif time.time() - self.last_connect_attempt > self.reconnect_interval:
                # No client and enough time has passed, try reconnecting
                self._try_reconnect()
                if self.monitor:
                    self.monitor.update_connection(connected=False)
            
            # Convert to hardware commands (with timing)
            action_start = time.perf_counter()
            motor_cmd, servo_cmd = self.brain_to_hardware_commands(brain_output)
            
            # Execute (unless reflex is active)
            if not self.reflex_active:
                self.hal.execute_motor_command(motor_cmd)
                self.hal.execute_servo_command(servo_cmd)
            action_time = (time.perf_counter() - action_start) * 1000
            
            # Update motor state metrics
            if self.monitor:
                self.monitor.update_motors(
                    left=motor_cmd.left_pwm_duty,
                    right=motor_cmd.right_pwm_duty,
                    steering=servo_cmd.steering_pw,  # Using raw pulse width
                    camera=servo_cmd.camera_pan_pw   # Using raw pulse width
                )
                self.monitor.update_timing(
                    sensor_ms=sensor_time,
                    comm_ms=comm_time,
                    action_ms=action_time
                )
            
            # Status every 100 cycles
            if self.cycle_count % 100 == 0:
                distance_cm = raw_sensors.gpio_ultrasonic_us / 58.0
                print(f"Cycle {self.cycle_count}: distance={distance_cm:.1f}cm, "
                      f"brain={'✓' if self.brain_client else '✗'}")
                
                # Print telemetry summary occasionally
                if self.monitor and self.cycle_count % 500 == 0:
                    print(self.monitor.get_summary())
            
            # Update cycle time metric
            cycle_time = (time.perf_counter() - cycle_start) * 1000
            if self.monitor:
                self.monitor.update_cycle_time(cycle_time)
            
            return True
            
        except Exception as e:
            print(f"❌ Cycle error: {e}")
            self.hal.emergency_stop()
            if self.monitor:
                self.monitor.trigger_reflex('emergency')
            return True
    
    def run(self, rate_hz: float = None):
        """Run brainstem at configured rate."""
        # Use config rate if not specified
        rate_hz = rate_hz or self.control_loop_hz
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
        """Clean shutdown with telemetry summary."""
        print("🛑 Shutting down brainstem...")
        
        self.running = False
        self.hal.emergency_stop()
        
        if self.brain_client:
            self.brain_client.disconnect()
        
        if self.vision_stream:
            self.vision_stream.stop()
        
        # Print final telemetry summary
        if self.monitor:
            print("\n📊 Final Telemetry Report:")
            print(self.monitor.get_summary())
            self.monitor.stop()
        
        self.hal.cleanup()
        print("✅ Shutdown complete")


def main():
    """Run the clean brainstem."""
    import argparse
    
    # Load config first to get defaults
    config = load_robot_config()
    brain_config = config.get("brain", {})
    
    parser = argparse.ArgumentParser(description='Clean PiCar-X Brainstem')
    parser.add_argument('--brain-host', 
                       default=None,  # None means use config
                       help=f'Brain server host (default: {brain_config.get("host", "localhost")} from config)')
    parser.add_argument('--brain-port', 
                       type=int, 
                       default=None,  # None means use config
                       help=f'Brain server port (default: {brain_config.get("port", 9999)} from config)')
    parser.add_argument('--rate', 
                       type=float, 
                       default=None,  # None means use config
                       help=f'Control rate Hz (default: {config.get("performance", {}).get("control_loop_hz", 20.0)} from config)')
    
    args = parser.parse_args()
    
    # Use config values if args are None
    brain_host = args.brain_host or brain_config.get('host', 'localhost')
    brain_port = args.brain_port or brain_config.get('port', 9999)
    rate = args.rate or config.get('performance', {}).get('control_loop_hz', 20.0)
    
    print("=" * 60)
    print("🧠 CLEAN PICARX BRAINSTEM")
    print("=" * 60)
    print(f"Brain: {brain_host}:{brain_port}")
    print(f"Rate: {rate}Hz")
    print(f"Config: robot_config.json")
    print("=" * 60)
    
    # Pass None to Brainstem to let it use config values
    brainstem = Brainstem(args.brain_host, args.brain_port)
    
    try:
        brainstem.run(rate)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        brainstem.shutdown()


if __name__ == "__main__":
    main()