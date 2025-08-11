#!/usr/bin/env python3
"""
Integrated Brainstem for PiCar-X

Combines:
1. Sensor-motor adaptation layer
2. Brain server communication
3. Local fallback behaviors
4. Safety monitoring
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from .brain_client import BrainServerClient, BrainServerConfig, MockBrainServerClient
    from .sensor_motor_adapter_fixed import PiCarXBrainAdapter  # Use fixed adapter
    from ..config.brainstem_config import BrainstemConfig, get_config
except ImportError:
    # For standalone testing
    from brain_client import BrainServerClient, BrainServerConfig, MockBrainServerClient
    from sensor_motor_adapter import PiCarXBrainAdapter
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.brainstem_config import BrainstemConfig, get_config


@dataclass
class IntegratedBrainstemConfig:
    """Configuration wrapper for integrated brainstem."""
    brain_server_config: BrainServerConfig
    use_mock_brain: bool = False
    brainstem_config: BrainstemConfig = None
    
    def __post_init__(self):
        if self.brainstem_config is None:
            self.brainstem_config = get_config()


class IntegratedBrainstem:
    """
    Complete brainstem implementation for PiCar-X.
    
    Handles:
    - Sensor normalization
    - Brain communication
    - Motor command execution
    - Safety reflexes
    - Fallback behaviors
    """
    
    def __init__(self, config: IntegratedBrainstemConfig):
        """Initialize integrated brainstem."""
        self.config = config
        self.bsc = config.brainstem_config  # Shorthand for brainstem config
        
        # Initialize components
        self.adapter = PiCarXBrainAdapter()
        
        if config.use_mock_brain:
            self.brain_client = MockBrainServerClient(config.brain_server_config)
        else:
            self.brain_client = BrainServerClient(config.brain_server_config)
        
        # State tracking
        self.last_sensor_data = None
        self.last_motor_commands = None
        self.last_brain_response = None
        self.reflex_active = False
        
        # Statistics
        self.cycle_count = 0
        self.brain_timeouts = 0
        self.reflex_activations = 0
        
        # Threading
        self.running = False
        self.update_thread = None
        
        print("🧠 Integrated Brainstem initialized")
        print(f"   Update rate: {self.bsc.threading.motor_update_rate}Hz")
        print(f"   Mock brain: {config.use_mock_brain}")
        print(f"   Safe mode: {self.bsc.safe_mode}")
        print(f"   Profile: {self.bsc.robot_id}")
    
    def connect(self) -> bool:
        """Connect to brain server."""
        success = self.brain_client.connect()
        if success:
            print("✅ Brainstem connected to brain server")
        else:
            print("⚠️  Brainstem running in autonomous mode")
        return success
    
    def process_cycle(self, raw_sensor_data: List[float]) -> Dict[str, float]:
        """
        Process one sensory-motor cycle.
        
        Args:
            raw_sensor_data: Raw 16-channel sensor data from PiCar-X
            
        Returns:
            Motor commands dictionary
        """
        self.cycle_count += 1
        cycle_start = time.time()
        
        # Store raw sensors
        self.last_sensor_data = raw_sensor_data
        
        # Convert to brain format
        brain_input = self.adapter.sensors_to_brain_input(raw_sensor_data)
        
        # Check for immediate safety reflexes
        if self.bsc.safe_mode:
            reflex_commands = self._check_reflexes(raw_sensor_data)
            if reflex_commands:
                self.reflex_active = True
                self.reflex_activations += 1
                self.last_motor_commands = reflex_commands
                return reflex_commands
        else:
            self.reflex_active = False
        
        # Send to brain if connected
        motor_commands = None
        if self.brain_client.is_connected():
            # Prepare sensor package for brain (NO REWARD!)
            sensor_package = {
                'raw_sensors': raw_sensor_data,
                'normalized_sensors': brain_input,  # All 16 channels
                'cycle': self.cycle_count,
                'reflex_active': self.reflex_active
            }
            
            # Send and get response
            success = self.brain_client.send_sensor_data(sensor_package)
            
            if success:
                # Get motor commands from brain
                brain_response = self.brain_client.get_latest_motor_commands()
                if brain_response:
                    self.last_brain_response = brain_response
                    
                    # Extract brain's motor outputs (4 channels)
                    brain_motors = [
                        brain_response.get('motor_x', 0.0),
                        brain_response.get('motor_y', 0.0),
                        brain_response.get('motor_z', 0.0),
                        brain_response.get('motor_w', 0.0)
                    ]
                    
                    # Convert to PiCar-X motors
                    motor_commands = self.adapter.brain_output_to_motors(brain_motors)
            else:
                self.brain_timeouts += 1
        
        # Fallback to local behavior if no brain response
        if motor_commands is None:
            motor_commands = self._generate_fallback_behavior(raw_sensor_data)
        
        # Apply safety overrides
        if self.bsc.safe_mode:
            motor_commands = self._apply_safety_limits(motor_commands, raw_sensor_data)
        
        # Store and return
        self.last_motor_commands = motor_commands
        
        # Track timing
        cycle_time = time.time() - cycle_start
        expected_cycle_time = self.bsc.threading.motor_update_rate
        if cycle_time > expected_cycle_time * 1.5:  # 50% slower than expected
            print(f"⚠️  Slow cycle: {cycle_time*1000:.1f}ms (expected <{expected_cycle_time*1000:.1f}ms)")
        
        return motor_commands
    
    def _check_reflexes(self, sensor_data: List[float]) -> Optional[Dict[str, float]]:
        """
        Check for immediate reflex responses.
        
        Returns motor commands if reflex triggered, None otherwise.
        """
        # Emergency stop for imminent collision
        distance = sensor_data[0]
        if distance < self.bsc.safety.emergency_stop_distance:
            return {
                'left_motor': 0.0,
                'right_motor': 0.0,
                'steering_servo': 0.0,
                'camera_pan_servo': 0.0,
                'camera_tilt_servo': 0.0
            }
        
        # Cliff detection - stop and reverse
        if sensor_data[11] > self.bsc.safety.cliff_detection_threshold:
            return {
                'left_motor': self.bsc.motors.max_motor_speed * -0.4,  # 40% reverse
                'right_motor': self.bsc.motors.max_motor_speed * -0.4,
                'steering_servo': 0.0,
                'camera_pan_servo': 0.0,
                'camera_tilt_servo': self.bsc.motors.camera_tilt_min / 2  # Look down
            }
        
        # No reflex needed
        return None
    
    def _generate_fallback_behavior(self, sensor_data: List[float]) -> Dict[str, float]:
        """
        Generate fallback behavior when brain is unavailable.
        
        Simple obstacle avoidance + line following.
        """
        distance = sensor_data[0]
        grayscale_left = sensor_data[3]
        grayscale_center = sensor_data[2]
        grayscale_right = sensor_data[1]
        
        # Base speed depends on distance
        if distance < self.bsc.safety.min_safe_distance:
            base_speed = self.bsc.motors.max_motor_speed * -0.2  # 20% reverse
            steering = self.bsc.motors.max_steering_angle * 0.6 if sensor_data[8] >= 0 else -self.bsc.motors.max_steering_angle * 0.6
        elif distance < self.bsc.safety.min_safe_distance * 2.5:
            base_speed = self.bsc.motors.max_motor_speed * 0.4  # 40% forward
            # Simple line following
            line_threshold = self.bsc.sensors.grayscale_threshold * 2  # Higher threshold for line
            if grayscale_center > line_threshold:
                steering = 0.0  # On line
            elif grayscale_left > grayscale_right:
                steering = -self.bsc.motors.max_steering_angle * 0.4  # Gentle left
            else:
                steering = self.bsc.motors.max_steering_angle * 0.4  # Gentle right
        else:
            base_speed = self.bsc.motors.max_motor_speed * 0.6  # 60% normal speed
            steering = 0.0
        
        return {
            'left_motor': base_speed,
            'right_motor': base_speed,
            'steering_servo': steering,
            'camera_pan_servo': 0.0,
            'camera_tilt_servo': 0.0
        }
    
    def _apply_safety_limits(self, commands: Dict[str, float], 
                           sensor_data: List[float]) -> Dict[str, float]:
        """Apply safety limits to motor commands."""
        # Reduce speed if battery is low
        if sensor_data[9] < self.bsc.sensors.battery_min:
            speed_factor = self.bsc.safety.low_battery_speed_factor
            commands['left_motor'] *= speed_factor
            commands['right_motor'] *= speed_factor
        
        # Limit speed based on CPU temperature
        if sensor_data[12] > self.bsc.sensors.cpu_temp_critical:
            speed_factor = self.bsc.safety.high_temp_speed_factor
            commands['left_motor'] *= speed_factor
            commands['right_motor'] *= speed_factor
        
        return commands
    
    def get_status(self) -> Dict[str, Any]:
        """Get brainstem status information."""
        return {
            'connected': self.brain_client.is_connected(),
            'cycles': self.cycle_count,
            'brain_timeouts': self.brain_timeouts,
            'reflex_activations': self.reflex_activations,
            'reflex_active': self.reflex_active,
            'last_brain_response': self.last_brain_response,
            'adapter_debug': self.adapter.get_debug_info()
        }
    
    def get_vocal_commands(self) -> Optional[Dict[str, Any]]:
        """Get any vocal commands from brain."""
        if self.brain_client.is_connected():
            return self.brain_client.get_latest_vocal_commands()
        return None
    
    def shutdown(self):
        """Shutdown brainstem."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        print("🛑 Brainstem shutdown complete")


def test_integrated_brainstem():
    """Test the integrated brainstem."""
    print("🧪 Testing Integrated Brainstem")
    print("=" * 50)
    
    # Configure for mock brain
    config = IntegratedBrainstemConfig(
        brain_server_config=BrainServerConfig(),
        use_mock_brain=True,
        brainstem_config=get_config(profile="testing")  # Use testing profile
    )
    
    brainstem = IntegratedBrainstem(config)
    brainstem.connect()
    
    # Simulate some cycles
    print("\nSimulating robot cycles...")
    
    test_scenarios = [
        ("Normal driving", [0.8, 0.3, 0.3, 0.3, 0.2, 0.2, 0, 0, 0, 7.4, 0, 0, 45, 0.3, 1000, 0]),
        ("Obstacle ahead", [0.15, 0.3, 0.3, 0.3, 0.1, 0.1, 0, 0, 0, 7.4, 0, 0, 45, 0.3, 1000, 0]),
        ("On line", [0.5, 0.2, 0.8, 0.2, 0.2, 0.2, 0, 0, 0, 7.4, 1, 0, 45, 0.3, 1000, 0]),
        ("Cliff detected", [0.5, 0.3, 0.3, 0.3, 0.2, 0.2, 0, 0, 0, 7.4, 0, 1, 45, 0.3, 1000, 0]),
    ]
    
    for scenario, sensor_data in test_scenarios:
        print(f"\n{scenario}:")
        motor_commands = brainstem.process_cycle(sensor_data)
        
        print(f"  Distance: {sensor_data[0]:.2f}m")
        print(f"  Motors: L={motor_commands['left_motor']:.1f}, R={motor_commands['right_motor']:.1f}")
        print(f"  Steering: {motor_commands['steering_servo']:.1f}°")
        
        if brainstem.reflex_active:
            print("  ⚡ Reflex active!")
    
    # Show status
    print(f"\nBrainstem status:")
    status = brainstem.get_status()
    print(f"  Cycles: {status['cycles']}")
    print(f"  Reflex activations: {status['reflex_activations']}")
    print(f"  Brain timeouts: {status['brain_timeouts']}")
    
    brainstem.shutdown()
    print("\n✅ Integrated brainstem test complete!")


if __name__ == "__main__":
    test_integrated_brainstem()