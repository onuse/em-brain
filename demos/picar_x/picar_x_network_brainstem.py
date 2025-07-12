#!/usr/bin/env python3
"""
PiCar-X Network Brainstem

A realistic PiCar-X brainstem that communicates with the minimal brain server over TCP.
This demonstrates the proper client-server separation for real robot deployment.

This is how a real robot would work:
- Brainstem runs on Pi Zero (robot hardware)
- Brain server runs on powerful machine (desktop/cloud)
- Communication over TCP sockets
"""

import sys
import os

# Add the brain/ directory to import minimal as a package
current_dir = os.path.dirname(__file__)  # picar_x/
demos_dir = os.path.dirname(current_dir)  # demos/
minimal_dir = os.path.dirname(demos_dir)  # minimal/
brain_dir = os.path.dirname(minimal_dir)   # brain/
sys.path.insert(0, brain_dir)

import numpy as np
import time
import math
import random
from typing import List, Dict, Any, Optional

from src.communication.client import MinimalBrainClient


class PiCarXNetworkBrainstem:
    """
    Network-enabled PiCar-X brainstem that communicates with remote brain server.
    
    This runs on the robot hardware and handles:
    - Hardware sensor interface
    - Motor control
    - Network communication with brain server
    - Local safety/emergency logic
    """
    
    def __init__(self, brain_server_host: str = 'localhost', brain_server_port: int = 9999):
        """Initialize the network brainstem."""
        
        # Network brain client
        self.brain_client = MinimalBrainClient(brain_server_host, brain_server_port)
        
        # Robot state (simulated hardware)
        self.position = [5.0, 5.0, 0.0]  # x, y, heading
        self.velocity = [0.0, 0.0]       # forward_speed, angular_speed
        self.last_update = time.time()
        
        # Sensor state
        self.ultrasonic_distance = 100.0
        self.camera_pan_angle = 0.0
        self.camera_tilt_angle = 0.0
        self.line_sensors = [0.0, 0.0]
        self.camera_rgb = [0.5, 0.5, 0.5]
        
        # Motor state
        self.motor_speed = 0.0
        self.steering_angle = 0.0
        
        # Safety and performance
        self.emergency_stop = False
        self.total_control_cycles = 0
        self.network_errors = 0
        self.last_brain_response_time = 0.0
        
        # Environment simulation (obstacles)
        self.obstacles = [
            (10, 10, 3),   # x, y, radius
            (20, 15, 2.5),
            (15, 25, 2),
            (25, 8, 3),
            (8, 20, 2.5)
        ]
        
        print("🤖 PiCar-X Network Brainstem initialized")
        print(f"   Brain server: {brain_server_host}:{brain_server_port}")
        print("   Ready to connect and operate autonomously")
    
    def connect_to_brain(self) -> bool:
        """Connect to the brain server."""
        
        print("🔌 Connecting to brain server...")
        
        # Define robot capabilities for handshake
        robot_capabilities = [
            1.0,  # Robot type: 1.0 = PiCar-X
            16.0, # Sensory vector dimensions
            4.0,  # Action vector dimensions  
            1.0   # Hardware version
        ]
        
        success = self.brain_client.connect(robot_capabilities)
        
        if success:
            server_info = self.brain_client.get_server_info()
            print(f"✅ Connected to brain server!")
            print(f"   Brain version: {server_info['brain_version']}")
            print(f"   GPU available: {server_info['gpu_available']}")
            print(f"   Expected sensor dims: {server_info['expected_sensory_size']}")
        else:
            print("❌ Failed to connect to brain server")
        
        return success
    
    def autonomous_operation(self, duration_seconds: int = 60, update_rate_hz: float = 10.0):
        """
        Run autonomous operation loop.
        
        Args:
            duration_seconds: How long to operate
            update_rate_hz: Control loop frequency
        """
        
        if not self.brain_client.is_connected():
            print("⚠️  Not connected to brain server - cannot operate")
            return False
        
        print(f"🚀 Starting autonomous operation ({duration_seconds}s at {update_rate_hz}Hz)")
        
        start_time = time.time()
        target_cycle_time = 1.0 / update_rate_hz
        
        while time.time() - start_time < duration_seconds:
            cycle_start = time.time()
            
            # Check for emergency conditions
            if self._check_emergency_conditions():
                print("🚨 Emergency stop activated!")
                self.emergency_stop = True
                break
            
            # Execute control cycle
            try:
                cycle_result = self._network_control_cycle()
                
                if cycle_result is None:
                    self.network_errors += 1
                    print(f"⚠️  Network error #{self.network_errors}")
                    
                    # If too many network errors, stop operation
                    if self.network_errors > 10:
                        print("💔 Too many network errors - stopping operation")
                        break
                else:
                    # Reset error count on successful cycle
                    if self.network_errors > 0:
                        self.network_errors = max(0, self.network_errors - 1)
                
                self.total_control_cycles += 1
                
                # Status updates
                if self.total_control_cycles % int(update_rate_hz * 5) == 0:  # Every 5 seconds
                    self._print_status_update(cycle_result)
                
            except KeyboardInterrupt:
                print("\n⏹️  Operation interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error in control cycle: {e}")
                break
            
            # Maintain control rate
            cycle_time = time.time() - cycle_start
            if cycle_time < target_cycle_time:
                time.sleep(target_cycle_time - cycle_time)
        
        # Operation complete
        operation_time = time.time() - start_time
        print(f"\n✅ Autonomous operation complete ({operation_time:.1f}s)")
        self._print_operation_summary()
        
        return True
    
    def _network_control_cycle(self) -> Optional[Dict[str, Any]]:
        """
        Execute one complete network-based control cycle.
        
        Returns:
            Control cycle information, or None if network error
        """
        cycle_start = time.time()
        
        # 1. Read sensors
        sensory_vector = self._read_sensors()
        
        # 2. Send to brain server and get action
        action_vector = self.brain_client.get_action(sensory_vector, timeout=0.5)
        
        if action_vector is None:
            # Network error - use emergency/default action
            action_vector = self._emergency_action()
            brain_response_time = 0.0
            network_success = False
        else:
            brain_response_time = time.time() - cycle_start
            self.last_brain_response_time = brain_response_time
            network_success = True
        
        # 3. Execute motor commands
        motor_result = self._execute_motors(action_vector)
        
        # 4. Update robot physics
        self._update_robot_physics()
        
        cycle_time = time.time() - cycle_start
        
        return {
            'cycle_time': cycle_time,
            'brain_response_time': brain_response_time,
            'network_success': network_success,
            'sensory_vector': sensory_vector,
            'action_vector': action_vector,
            'motor_result': motor_result,
            'robot_position': self.position.copy()
        }
    
    def _read_sensors(self) -> List[float]:
        """Read all robot sensors and create unified sensory vector."""
        
        # Update sensor readings based on current position
        self._update_sensors()
        
        # Build sensory vector (16 dimensions)
        sensors = []
        
        # Ultrasonic sensors (5 directions)
        for angle_offset in [-45, -22.5, 0, 22.5, 45]:
            distance = self._simulate_ultrasonic_reading(angle_offset)
            normalized = max(0.0, min(1.0, (200.0 - distance) / 200.0))
            sensors.append(normalized)
        
        # Camera RGB (3 channels)
        sensors.extend(self.camera_rgb)
        
        # Line sensors (2 channels)
        sensors.extend(self.line_sensors)
        
        # Camera pose (2 dimensions)
        sensors.append(self.camera_pan_angle / 180.0)   # -1 to 1
        sensors.append(self.camera_tilt_angle / 90.0)   # -1 to 1
        
        # Robot motion state (4 dimensions)
        sensors.append(self.motor_speed / 100.0)        # -1 to 1
        sensors.append(self.steering_angle / 30.0)      # -1 to 1
        sensors.append(math.sin(math.radians(self.position[2])))  # heading sin
        sensors.append(math.cos(math.radians(self.position[2])))  # heading cos
        
        return sensors
    
    def _execute_motors(self, action_vector: List[float]) -> Dict[str, float]:
        """Execute motor commands from action vector."""
        
        if len(action_vector) >= 4:
            # Extract and scale actions
            target_motor_speed = np.clip(action_vector[0] * 50.0, -50.0, 50.0)  # Reduced for safety
            target_steering = np.clip(action_vector[1] * 25.0, -25.0, 25.0)
            target_camera_pan = np.clip(action_vector[2] * 90.0, -90.0, 90.0)
            target_camera_tilt = np.clip(action_vector[3] * 45.0, -45.0, 45.0)
        else:
            # Safety defaults
            target_motor_speed = 0.0
            target_steering = 0.0
            target_camera_pan = 0.0
            target_camera_tilt = 0.0
        
        # Apply motor limits for smooth control
        motor_change_limit = 10.0
        steering_change_limit = 5.0
        camera_change_limit = 10.0
        
        self.motor_speed += np.clip(target_motor_speed - self.motor_speed,
                                   -motor_change_limit, motor_change_limit)
        self.steering_angle += np.clip(target_steering - self.steering_angle,
                                      -steering_change_limit, steering_change_limit)
        self.camera_pan_angle += np.clip(target_camera_pan - self.camera_pan_angle,
                                        -camera_change_limit, camera_change_limit)
        self.camera_tilt_angle += np.clip(target_camera_tilt - self.camera_tilt_angle,
                                         -camera_change_limit, camera_change_limit)
        
        return {
            'motor_speed': self.motor_speed,
            'steering_angle': self.steering_angle,
            'camera_pan': self.camera_pan_angle,
            'camera_tilt': self.camera_tilt_angle
        }
    
    def _emergency_action(self) -> List[float]:
        """Generate emergency action when brain server is unavailable."""
        # Emergency: stop and turn away from obstacles
        
        if self.ultrasonic_distance < 30:
            # Obstacle detected - back up and turn
            return [-0.3, 0.5, 0.0, 0.0]  # Reverse and turn right
        else:
            # No immediate danger - slow forward
            return [0.1, 0.0, 0.0, 0.0]  # Slow forward
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency stop conditions."""
        
        # Emergency if obstacle too close
        if self.ultrasonic_distance < 10:
            return True
        
        # Emergency if no brain response for too long
        if time.time() - self.last_brain_response_time > 5.0:
            return True
        
        # Emergency if too many network errors
        if self.network_errors > 20:
            return True
        
        return False
    
    def _update_robot_physics(self):
        """Update robot position based on motor commands."""
        dt = time.time() - self.last_update
        self.last_update = time.time()
        
        if dt > 0.2:  # Skip large time jumps
            dt = 0.1
        
        # Convert motor commands to motion
        forward_speed = self.motor_speed * 0.02  # m/s
        
        # Ackermann steering
        if abs(self.steering_angle) > 1.0:
            turning_radius = 2.0 / math.tan(math.radians(abs(self.steering_angle)))
            angular_speed = forward_speed / turning_radius
            if self.steering_angle < 0:
                angular_speed = -angular_speed
        else:
            angular_speed = 0.0
        
        # Update position
        self.position[2] += math.degrees(angular_speed * dt)
        self.position[2] = self.position[2] % 360.0
        
        heading_rad = math.radians(self.position[2])
        self.position[0] += forward_speed * math.cos(heading_rad) * dt
        self.position[1] += forward_speed * math.sin(heading_rad) * dt
        
        # Keep in bounds
        self.position[0] = max(2, min(28, self.position[0]))
        self.position[1] = max(2, min(28, self.position[1]))
    
    def _update_sensors(self):
        """Update sensor readings based on current position."""
        
        # Ultrasonic distance
        self.ultrasonic_distance = self._simulate_ultrasonic_reading(0)
        
        # Camera RGB (varies with environment)
        self.camera_rgb = [
            0.4 + 0.3 * math.sin(self.position[0] * 0.2),
            0.4 + 0.3 * math.cos(self.position[1] * 0.2),
            0.4 + 0.3 * math.sin((self.position[0] + self.position[1]) * 0.1)
        ]
        
        # Line sensors (detect virtual lines)
        line_positions = [(10, 15), (20, 10), (15, 20)]
        min_distance_to_line = min(
            math.sqrt((self.position[0] - lx)**2 + (self.position[1] - ly)**2)
            for lx, ly in line_positions
        )
        
        line_strength = max(0.0, 1.0 - min_distance_to_line / 3.0)
        self.line_sensors = [
            max(0.0, min(1.0, line_strength + random.uniform(-0.1, 0.1))),
            max(0.0, min(1.0, line_strength + random.uniform(-0.1, 0.1)))
        ]
    
    def _simulate_ultrasonic_reading(self, pan_angle_offset: float) -> float:
        """Simulate ultrasonic sensor reading."""
        
        sensor_angle = math.radians(self.position[2] + pan_angle_offset)
        sensor_dir = [math.cos(sensor_angle), math.sin(sensor_angle)]
        
        min_distance = 200.0
        
        for obs_x, obs_y, obs_radius in self.obstacles:
            to_obstacle = [obs_x - self.position[0], obs_y - self.position[1]]
            projection = (to_obstacle[0] * sensor_dir[0] + to_obstacle[1] * sensor_dir[1])
            
            if projection > 0:
                perp_distance = math.sqrt(
                    to_obstacle[0]**2 + to_obstacle[1]**2 - projection**2
                )
                
                if perp_distance <= obs_radius:
                    hit_distance = projection - math.sqrt(obs_radius**2 - perp_distance**2)
                    if hit_distance > 0:
                        min_distance = min(min_distance, hit_distance * 100)  # Convert to cm
        
        return max(10.0, min(200.0, min_distance + random.uniform(-3.0, 3.0)))
    
    def _print_status_update(self, cycle_result: Optional[Dict]):
        """Print status update."""
        client_stats = self.brain_client.get_client_stats()
        
        print(f"\n📍 Cycle {self.total_control_cycles}: "
              f"Position ({self.position[0]:5.1f}, {self.position[1]:5.1f}) "
              f"Heading {self.position[2]:5.1f}°")
        
        if cycle_result:
            print(f"   🤖 Motors: Speed {cycle_result['motor_result']['motor_speed']:5.1f}, "
                  f"Steering {cycle_result['motor_result']['steering_angle']:5.1f}°")
            print(f"   📡 Network: Success {cycle_result['network_success']}, "
                  f"Brain response: {cycle_result['brain_response_time']*1000:.1f}ms")
        
        print(f"   🌐 Client: {client_stats['performance']['success_rate']:.2%} success, "
              f"{client_stats['performance']['avg_request_time']*1000:.1f}ms avg")
        print(f"   📊 Sensors: Ultrasonic {self.ultrasonic_distance:.1f}cm, "
              f"Lines [{self.line_sensors[0]:.2f}, {self.line_sensors[1]:.2f}]")
    
    def _print_operation_summary(self):
        """Print comprehensive operation summary."""
        client_stats = self.brain_client.get_client_stats()
        
        print("\n" + "="*60)
        print("🎯 AUTONOMOUS OPERATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Robot Performance:")
        print(f"   • Control cycles: {self.total_control_cycles}")
        print(f"   • Network errors: {self.network_errors}")
        print(f"   • Emergency stops: {'Yes' if self.emergency_stop else 'No'}")
        print(f"   • Final position: ({self.position[0]:.1f}, {self.position[1]:.1f})")
        
        print(f"\n🌐 Network Performance:")
        print(f"   • Success rate: {client_stats['performance']['success_rate']:.2%}")
        print(f"   • Average brain response: {client_stats['performance']['avg_request_time']*1000:.1f}ms")
        print(f"   • Requests per second: {client_stats['performance']['requests_per_second']:.1f}")
        print(f"   • Total requests: {client_stats['performance']['requests_sent']}")
        
        print(f"\n🧠 Brain Server Info:")
        server_info = self.brain_client.get_server_info()
        if server_info:
            print(f"   • Brain version: {server_info['brain_version']}")
            print(f"   • GPU acceleration: {server_info['gpu_available']}")
            print(f"   • Action dimensions: {server_info['action_size']}")
        
        print("\n" + "="*60)
    
    def disconnect(self):
        """Disconnect from brain server."""
        print("👋 Disconnecting from brain server...")
        self.brain_client.disconnect()


def main():
    """Main demonstration of network-based robot control."""
    
    print("🤖 PiCar-X Network Brainstem Demo")
    print("   Demonstrating proper client-server separation for robot deployment")
    print("   Brain server must be running separately!")
    
    # Initialize robot brainstem
    robot = PiCarXNetworkBrainstem(brain_server_host='localhost', brain_server_port=9999)
    
    try:
        # Connect to brain server
        if not robot.connect_to_brain():
            print("❌ Cannot proceed without brain server connection")
            return
        
        # Run autonomous operation
        robot.autonomous_operation(duration_seconds=30, update_rate_hz=5.0)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
    
    print("\n🎉 Network brainstem demo complete!")
    print("   Key achievement: Robot controlled by remote brain over TCP")


if __name__ == "__main__":
    main()