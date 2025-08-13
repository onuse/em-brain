#!/usr/bin/env python3
"""
Brainstem Configuration

Centralized configuration for all brainstem components.
Eliminates magic numbers and provides clear documentation of all parameters.

This configuration can be:
1. Loaded from environment variables
2. Overridden by command-line arguments
3. Persisted to/from JSON for different robot profiles
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
import os
from pathlib import Path


@dataclass
class SensorConfig:
    """Sensor processing configuration."""
    
    # Dimensions
    picarx_sensor_count: int = 16
    brain_input_dimensions: int = 16  # Match actual sensor count - no padding!
    
    # Normalization ranges
    ultrasonic_max_distance: float = 2.0  # meters
    grayscale_threshold: float = 0.3      # Line detection threshold
    battery_nominal: float = 7.4          # Nominal battery voltage
    battery_min: float = 6.0              # Minimum safe voltage
    battery_max: float = 8.4              # Maximum voltage (fully charged)
    cpu_temp_normal: float = 40.0         # Normal CPU temperature (C)
    cpu_temp_warning: float = 60.0        # Warning threshold (C)
    cpu_temp_critical: float = 80.0       # Critical threshold (C)
    
    # Derived sensor parameters
    distance_gradient_window: float = 0.5  # seconds for gradient calculation
    line_quality_threshold: float = 0.3    # Minimum for line following
    exploration_variance_scale: float = 10.0  # Scaling for exploration metric
    
    # Normalization constants
    spatial_distance_scale: float = 2.0    # Max distance for spatial normalization
    motor_range_half: float = 1.0          # Motor range (-1 to 1)
    steering_range: float = 60.0           # Total steering range for normalization
    camera_pan_range: float = 180.0        # Total camera pan range
    camera_tilt_range: float = 100.0       # Total camera tilt range  
    steering_offset: float = 30.0          # Steering angle offset
    camera_pan_offset: float = 90.0        # Camera pan offset
    camera_tilt_offset: float = 35.0       # Camera tilt offset
    
    # Thresholds for sensor processing
    neutral_value: float = 0.5             # Neutral sensor value
    line_detection_strength: float = 0.6   # Strong line signal threshold
    forward_speed_threshold: float = 0.1   # Minimum forward speed threshold
    straight_steering_threshold: float = 5.0  # Threshold for "going straight"
    distance_change_scale: float = 1.0     # Scale for distance change normalization
    distance_change_offset: float = 0.5    # Offset for distance change
    steering_effort_scale: float = 30.0    # Scale for steering effort calculation
    
    # System health thresholds
    low_battery_threshold: float = 6.5     # Low battery voltage threshold
    high_temp_threshold: float = 60.0      # High temperature threshold
    health_battery_penalty: float = 0.3    # Health penalty for low battery
    health_temp_penalty: float = 0.2       # Health penalty for high temperature
    health_cliff_penalty: float = 0.5      # Health penalty for cliff detection


@dataclass
class MotorConfig:
    """Motor control configuration."""
    
    # Dimensions
    picarx_motor_count: int = 5
    brain_output_dimensions: int = 4
    
    # Safety limits
    max_motor_speed: float = 50.0         # Maximum motor speed (%)
    max_steering_angle: float = 25.0      # Maximum steering angle (degrees)
    camera_pan_min: float = -90.0         # Minimum pan angle (degrees)
    camera_pan_max: float = 90.0          # Maximum pan angle (degrees)
    camera_tilt_min: float = -35.0        # Minimum tilt angle (degrees)
    camera_tilt_max: float = 65.0         # Maximum tilt angle (degrees)
    
    # Smoothing parameters
    motor_smoothing_alpha: float = 0.3    # Exponential smoothing factor
    steering_smoothing_alpha: float = 0.5 # Steering smoothing factor
    servo_smoothing_alpha: float = 0.7    # Camera servo smoothing
    
    # Acceleration limits
    max_acceleration: float = 100.0       # Max speed change per second (%)
    max_deceleration: float = 200.0       # Max braking per second (%)
    emergency_deceleration: float = 500.0 # Emergency stop rate (%)
    
    # Motor control parameters
    turn_speed_reduction_base: float = 0.5  # Base speed when turning
    turn_speed_reduction_factor: float = 0.5  # Additional reduction based on turn
    camera_pan_range: float = 45.0        # Camera pan control range (±degrees)
    camera_tilt_range: float = 30.0       # Camera tilt control range (degrees)


@dataclass
class RewardConfig:
    """Reward signal calculation configuration."""
    
    # Baseline reward
    neutral_baseline: float = 0.5         # Neutral reward value
    
    # Distance-based rewards/penalties
    collision_penalty: float = 0.4        # Penalty for being too close
    near_obstacle_penalty: float = 0.2    # Penalty for getting close to obstacles
    collision_cooldown_penalty: float = 0.1  # Ongoing penalty during cooldown
    forward_movement_reward: float = 0.2  # Reward for safe forward movement
    
    # Line following rewards
    line_detection_reward: float = 0.15   # Reward for detecting line
    line_score_accumulation: float = 0.1  # Line score increment
    line_score_decay: float = 0.9         # Line score decay factor
    line_score_weight: float = 0.1        # Weight for line following score
    
    # Exploration rewards
    exploration_reward: float = 0.05      # Reward for exploration behavior
    exploration_cooldown: int = 10        # Cooldown cycles for exploration bonus
    exploration_steering_threshold: float = 10.0  # Min steering change for exploration
    
    # System health penalties
    low_battery_penalty: float = 0.1      # Penalty for low battery
    high_temperature_penalty: float = 0.05  # Penalty for high CPU temperature
    cliff_detection_penalty: float = 0.3  # Penalty for cliff detection
    
    # Timers and counters
    collision_cooldown_cycles: int = 20   # Cycles to maintain collision penalty


@dataclass
class SafetyConfig:
    """Safety system configuration."""
    
    # Collision avoidance
    min_safe_distance: float = 0.2        # Minimum safe distance (meters)
    emergency_stop_distance: float = 0.1  # Emergency stop threshold (meters)
    
    # Speed reduction factors
    obstacle_speed_factor: float = 0.5    # Speed reduction near obstacles
    cliff_speed_factor: float = 0.0       # Speed when cliff detected (stop)
    low_battery_speed_factor: float = 0.7 # Speed reduction on low battery
    high_temp_speed_factor: float = 0.5   # Speed reduction on high temperature
    
    # Thresholds
    cliff_detection_threshold: float = 0.5  # Cliff sensor threshold
    line_lost_timeout: float = 2.0          # Seconds before line lost panic
    
    # Watchdog timers
    command_timeout: float = 0.5          # Max time between commands (seconds)
    sensor_timeout: float = 0.1           # Max time between sensor updates
    heartbeat_timeout: float = 1.0        # Brain connection heartbeat


@dataclass
class NetworkConfig:
    """Network communication configuration."""
    
    # Connection
    brain_host: str = "localhost"
    brain_port: int = 9999
    monitoring_port: int = 8888
    
    # Timeouts
    connection_timeout: float = 5.0       # Initial connection timeout
    receive_timeout: float = 0.5          # Message receive timeout
    send_timeout: float = 0.1             # Message send timeout
    
    # Retry logic
    max_reconnect_attempts: int = 3
    reconnect_delay: float = 1.0          # Base delay between attempts
    reconnect_backoff: float = 2.0        # Exponential backoff multiplier
    
    # Protocol
    max_message_size: int = 4096          # Maximum message size (bytes)
    max_vector_size: int = 1024           # Maximum vector elements
    
    # Performance
    tcp_nodelay: bool = True              # Disable Nagle's algorithm
    socket_keepalive: bool = True         # Enable TCP keepalive
    keepalive_interval: int = 1           # Keepalive interval (seconds)


@dataclass
class ThreadingConfig:
    """Threading and concurrency configuration."""
    
    # Thread pools
    sensor_thread_priority: int = 10      # Real-time priority for sensors
    motor_thread_priority: int = 10       # Real-time priority for motors
    network_thread_priority: int = 5      # Lower priority for network
    
    # Queue sizes
    sensor_queue_size: int = 100          # Sensor message queue
    motor_queue_size: int = 100           # Motor command queue
    telemetry_queue_size: int = 1000      # Telemetry data queue
    
    # Timing
    sensor_poll_rate: float = 0.01        # 100Hz sensor polling
    motor_update_rate: float = 0.02       # 50Hz motor updates
    telemetry_rate: float = 0.1           # 10Hz telemetry
    
    # Synchronization
    use_locks: bool = True                # Enable thread synchronization
    lock_timeout: float = 0.1              # Maximum lock wait time


@dataclass
class BrainstemConfig:
    """Complete brainstem configuration."""
    
    # Component configs
    sensors: SensorConfig = field(default_factory=SensorConfig)
    motors: MotorConfig = field(default_factory=MotorConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    threading: ThreadingConfig = field(default_factory=ThreadingConfig)
    
    # Robot identification
    robot_id: str = "picarx_001"
    robot_type: str = "picarx"
    firmware_version: str = "1.0.0"
    
    # Operating modes
    debug_mode: bool = False
    simulation_mode: bool = False
    safe_mode: bool = True               # Start in safe mode
    
    @classmethod
    def from_file(cls, path: str) -> "BrainstemConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrainstemConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Update nested configs
        if 'sensors' in data:
            config.sensors = SensorConfig(**data['sensors'])
        if 'motors' in data:
            config.motors = MotorConfig(**data['motors'])
        if 'safety' in data:
            config.safety = SafetyConfig(**data['safety'])
        if 'rewards' in data:
            config.rewards = RewardConfig(**data['rewards'])
        if 'network' in data:
            config.network = NetworkConfig(**data['network'])
        if 'threading' in data:
            config.threading = ThreadingConfig(**data['threading'])
        
        # Update top-level fields
        for key in ['robot_id', 'robot_type', 'firmware_version', 
                   'debug_mode', 'simulation_mode', 'safe_mode']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    @classmethod
    def from_env(cls) -> "BrainstemConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override from environment
        if host := os.getenv('BRAIN_HOST'):
            config.network.brain_host = host
        if port := os.getenv('BRAIN_PORT'):
            config.network.brain_port = int(port)
        if robot_id := os.getenv('ROBOT_ID'):
            config.robot_id = robot_id
        if debug := os.getenv('DEBUG_MODE'):
            config.debug_mode = debug.lower() in ('true', '1', 'yes')
        if safe := os.getenv('SAFE_MODE'):
            config.safe_mode = safe.lower() in ('true', '1', 'yes')
        
        # Safety overrides (always respect env safety settings)
        if max_speed := os.getenv('MAX_MOTOR_SPEED'):
            config.motors.max_motor_speed = float(max_speed)
        if min_distance := os.getenv('MIN_SAFE_DISTANCE'):
            config.safety.min_safe_distance = float(min_distance)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'sensors': asdict(self.sensors),
            'motors': asdict(self.motors),
            'safety': asdict(self.safety),
            'rewards': asdict(self.rewards),
            'network': asdict(self.network),
            'threading': asdict(self.threading),
            'robot_id': self.robot_id,
            'robot_type': self.robot_type,
            'firmware_version': self.firmware_version,
            'debug_mode': self.debug_mode,
            'simulation_mode': self.simulation_mode,
            'safe_mode': self.safe_mode,
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration for consistency and safety."""
        errors = []
        
        # Validate dimensions
        if self.sensors.brain_input_dimensions < self.sensors.picarx_sensor_count:
            errors.append("Brain input dimensions less than sensor count")
        
        # Validate safety limits
        if self.motors.max_motor_speed > 100:
            errors.append(f"Motor speed {self.motors.max_motor_speed} exceeds 100%")
        
        if self.safety.min_safe_distance < 0.05:
            errors.append(f"Minimum safe distance {self.safety.min_safe_distance} too small")
        
        # Validate network settings
        if self.network.receive_timeout > self.safety.command_timeout:
            errors.append("Network timeout exceeds command timeout")
        
        # Validate threading
        if self.threading.sensor_poll_rate > 0.1:
            errors.append("Sensor poll rate too slow for real-time control")
        
        if errors:
            for error in errors:
                print(f"❌ Config validation error: {error}")
            return False
        
        return True


# Default configurations for different scenarios
PROFILES = {
    "default": BrainstemConfig(),
    
    "aggressive": BrainstemConfig(
        motors=MotorConfig(
            max_motor_speed=80.0,
            max_acceleration=200.0,
            motor_smoothing_alpha=0.1
        ),
        safety=SafetyConfig(
            min_safe_distance=0.1,
            obstacle_speed_factor=0.8
        )
    ),
    
    "cautious": BrainstemConfig(
        motors=MotorConfig(
            max_motor_speed=30.0,
            max_acceleration=50.0,
            motor_smoothing_alpha=0.5
        ),
        safety=SafetyConfig(
            min_safe_distance=0.5,
            obstacle_speed_factor=0.2
        )
    ),
    
    "simulation": BrainstemConfig(
        simulation_mode=True,
        network=NetworkConfig(
            brain_host="localhost",
            connection_timeout=1.0
        ),
        threading=ThreadingConfig(
            sensor_poll_rate=0.1,  # Slower for simulation
            motor_update_rate=0.1
        )
    ),
    
    "testing": BrainstemConfig(
        debug_mode=True,
        safe_mode=True,
        motors=MotorConfig(max_motor_speed=10.0),
        safety=SafetyConfig(min_safe_distance=1.0)
    )
}


def get_config(profile: str = "default", config_file: Optional[str] = None) -> BrainstemConfig:
    """
    Get configuration with priority:
    1. Config file (if specified)
    2. Environment variables
    3. Profile defaults
    """
    
    # Start with profile
    if profile in PROFILES:
        config = PROFILES[profile]
    else:
        print(f"⚠️  Unknown profile '{profile}', using default")
        config = PROFILES["default"]
    
    # Override with file if specified
    if config_file and Path(config_file).exists():
        try:
            config = BrainstemConfig.from_file(config_file)
            print(f"✓ Loaded config from {config_file}")
        except Exception as e:
            print(f"❌ Failed to load config file: {e}")
    
    # Override with environment
    env_config = BrainstemConfig.from_env()
    
    # Merge environment overrides
    if os.getenv('BRAIN_HOST'):
        config.network.brain_host = env_config.network.brain_host
    if os.getenv('BRAIN_PORT'):
        config.network.brain_port = env_config.network.brain_port
    if os.getenv('DEBUG_MODE'):
        config.debug_mode = env_config.debug_mode
    if os.getenv('SAFE_MODE'):
        config.safe_mode = env_config.safe_mode
    
    # Validate final configuration
    if not config.validate():
        print("⚠️  Configuration validation failed, using safe defaults")
        config = PROFILES["testing"]
    
    return config


if __name__ == "__main__":
    # Test configuration system
    print("Testing Brainstem Configuration System")
    print("=" * 50)
    
    # Test default config
    default = get_config()
    print(f"\nDefault config: {default.motors.max_motor_speed}% max speed")
    
    # Test profiles
    for profile_name in PROFILES:
        config = get_config(profile_name)
        print(f"{profile_name}: max_speed={config.motors.max_motor_speed}, "
              f"min_distance={config.safety.min_safe_distance}")
    
    # Test saving/loading
    test_file = "/tmp/test_config.json"
    default.save(test_file)
    loaded = BrainstemConfig.from_file(test_file)
    assert loaded.motors.max_motor_speed == default.motors.max_motor_speed
    print("\n✓ Configuration save/load test passed")
    
    # Test validation
    bad_config = BrainstemConfig(
        motors=MotorConfig(max_motor_speed=150.0)  # Invalid
    )
    assert not bad_config.validate()
    print("✓ Validation test passed")
    
    print("\n✅ All configuration tests passed!")