#!/usr/bin/env python3
"""
Embodied PiCar-X Brainstem with Free Energy Action Selection

This is the updated brainstem that integrates the embodied Free Energy system
with the PiCar-X robot simulation. Unlike the original brainstem that goes
directly from brain to motors, this version:

1. Sensors → Brain (pattern recognition)
2. Brain + Current State → Embodied Free Energy System (action selection)
3. Selected Action → Motors

This demonstrates how embodied Free Energy creates goal-directed behavior
that emerges from the robot's physical constraints.
"""

import sys
import os

# Add the brain/ directory to sys.path
current_dir = os.path.dirname(__file__)  # picar_x_simulation/
demos_dir = os.path.dirname(current_dir)  # demos/
brain_dir = os.path.dirname(demos_dir)   # brain/ (project root)
sys.path.insert(0, brain_dir)

# Import original brainstem as base
from .picar_x_brainstem import PiCarXBrainstem

# Import embodied Free Energy system
from server.embodied_free_energy import (
    EmbodiedFreeEnergySystem,
    EmbodiedBrainAdapter,
    HardwareInterface,
    HardwareTelemetry
)

import numpy as np
import time
import math
from typing import List, Tuple, Dict, Any


class PiCarXHardwareInterface(HardwareInterface):
    """Hardware interface for PiCar-X robot simulation."""
    
    def __init__(self, brainstem):
        self.brainstem = brainstem
        self.battery_level = 1.0  # Start at full battery
        self.motor_temperature = 25.0  # Start at room temperature
        self.last_telemetry_time = time.time()
    
    def get_telemetry(self) -> HardwareTelemetry:
        """Get current robot hardware state."""
        
        # Simulate realistic hardware degradation
        current_time = time.time()
        dt = current_time - self.last_telemetry_time
        self.last_telemetry_time = current_time
        
        # Battery drain based on motor activity
        motor_load = abs(self.brainstem.motor_speed) / 100.0
        battery_drain = (0.001 + motor_load * 0.01) * dt  # Base drain + activity drain
        self.battery_level = max(0.0, self.battery_level - battery_drain)
        
        # Motor heating based on activity
        target_temp = 25.0 + motor_load * 30.0  # Up to 55°C under full load
        temp_rate = 5.0  # Temperature change rate
        temp_diff = target_temp - self.motor_temperature
        self.motor_temperature += temp_diff * min(1.0, dt * temp_rate)
        
        # Memory usage (simulate based on brain complexity)
        memory_usage = min(90.0, 30.0 + self.brainstem.total_control_cycles * 0.01)
        
        # Sensor noise (simulate based on distance and obstacles)
        base_noise = 0.02
        if self.brainstem.ultrasonic_distance < 20.0:
            sensor_noise = base_noise + 0.05  # Higher noise near obstacles
        else:
            sensor_noise = base_noise
        
        return HardwareTelemetry(
            battery_voltage=self.battery_level * 12.6,  # 12.6V = full battery
            battery_percentage=self.battery_level,
            motor_temperatures={'drive': self.motor_temperature, 'steering': self.motor_temperature - 5},
            cpu_temperature=40.0 + motor_load * 10.0,
            memory_usage_percentage=memory_usage,
            disk_usage_percentage=25.0,
            wifi_signal_strength=-40.0,
            sensor_noise_levels={'ultrasonic': sensor_noise, 'camera': base_noise},
            timestamp=current_time
        )
    
    def predict_hardware_effects(self, action: Any, current_state: HardwareTelemetry) -> HardwareTelemetry:
        """Predict how action will affect robot hardware."""
        
        if not isinstance(action, dict):
            return current_state
        
        # Predict battery drain
        predicted_battery = current_state.battery_percentage
        predicted_temp = max(current_state.motor_temperatures.values())
        
        action_type = action.get('type', 'unknown')
        
        if action_type == 'move':
            speed = action.get('speed', 0.5)
            # Energy cost scales with speed
            energy_cost = speed * 0.02  # 2% per speed unit
            predicted_battery -= energy_cost
            
            # Heating scales with speed
            heat_increase = speed * 5.0  # Up to 5°C per speed unit
            predicted_temp += heat_increase
            
        elif action_type == 'rotate':
            angle = abs(action.get('angle', 0))
            # Rotation energy cost
            energy_cost = (angle / 90) * 0.01  # 1% per 90 degree turn
            predicted_battery -= energy_cost
            
            # Heating from rotation
            heat_increase = (angle / 90) * 2.0
            predicted_temp += heat_increase
            
        elif action_type == 'stop':
            # Stopping allows cooling and minimal energy use
            predicted_battery -= 0.001  # Minimal idle drain
            predicted_temp = max(25.0, predicted_temp - 2.0)  # Cooling
            
        elif action_type == 'seek_charger':
            # Energy seeking predicts successful charging
            urgency = action.get('urgency', 'moderate')
            if urgency == 'high':
                predicted_battery += 0.08  # Significant energy gain prediction
                predicted_temp += 3.0  # Some heating from movement to charger
            else:
                predicted_battery += 0.04  # Moderate energy gain
                predicted_temp += 1.5  # Minimal heating
        
        # Keep values in realistic bounds
        predicted_battery = max(0.0, min(1.0, predicted_battery))
        predicted_temp = max(25.0, min(80.0, predicted_temp))
        
        # Create predicted telemetry
        predicted_telemetry = HardwareTelemetry(
            battery_voltage=predicted_battery * 12.6,
            battery_percentage=predicted_battery,
            motor_temperatures={'drive': predicted_temp, 'steering': predicted_temp - 5},
            cpu_temperature=current_state.cpu_temperature,
            memory_usage_percentage=current_state.memory_usage_percentage,
            disk_usage_percentage=current_state.disk_usage_percentage,
            wifi_signal_strength=current_state.wifi_signal_strength,
            sensor_noise_levels=current_state.sensor_noise_levels,
            timestamp=time.time()
        )
        
        return predicted_telemetry


class RobotState:
    """Structured robot state for embodied Free Energy system."""
    
    def __init__(self, brainstem):
        # Physical state
        self.battery = brainstem.battery_level if hasattr(brainstem, 'battery_level') else 0.8
        self.obstacle_distance = brainstem.ultrasonic_distance
        self.location = tuple(brainstem.position[:2])  # (x, y)
        self.heading = brainstem.position[2]
        
        # Sensor state
        self.camera_pan = brainstem.camera_pan_angle
        self.camera_tilt = brainstem.camera_tilt_angle
        self.line_sensors = tuple(brainstem.line_sensors)
        self.camera_rgb = tuple(brainstem.camera_rgb)
        
        # Motor state
        self.motor_speed = brainstem.motor_speed
        self.steering_angle = brainstem.steering_angle
        
        # Performance metrics
        self.total_cycles = brainstem.total_control_cycles
        self.obstacle_encounters = brainstem.obstacle_encounters


class EmbodiedPiCarXBrainstem(PiCarXBrainstem):
    """
    Enhanced PiCar-X brainstem with embodied Free Energy action selection.
    
    This version uses the embodied Free Energy system for action selection,
    creating goal-directed behavior that emerges from physical constraints.
    """
    
    def __init__(self, enable_camera=True, enable_ultrasonics=True, enable_line_tracking=True):
        """Initialize embodied brainstem."""
        
        # Initialize base brainstem
        super().__init__(enable_camera, enable_ultrasonics, enable_line_tracking)
        
        # Create hardware interface
        self.hardware_interface = PiCarXHardwareInterface(self)
        
        # Create brain adapter
        self.brain_adapter = EmbodiedBrainAdapter(self.brain)
        
        # Create embodied Free Energy system
        self.embodied_system = EmbodiedFreeEnergySystem(
            self.brain_adapter, 
            self.hardware_interface
        )
        
        # Performance tracking for embodied system
        self.embodied_decisions = 0
        self.energy_seeking_actions = 0
        self.thermal_management_actions = 0
        
        print("🧬 Embodied PiCar-X Brainstem initialized")
        print("   Free Energy action selection enabled")
        print("   Physics-grounded behavior active")
    
    def control_cycle(self) -> Dict[str, Any]:
        """
        Enhanced control cycle with embodied Free Energy action selection.
        
        Flow: Sensors → Brain → Embodied Free Energy → Action Selection → Motors
        """
        cycle_start = time.time()
        
        # 1. Get sensory input from robot hardware  
        sensory_vector = self.get_sensory_vector()
        
        # 2. Create structured robot state for embodied system
        robot_state = RobotState(self)
        
        # 3. Embodied Free Energy system selects action
        selected_action = self.embodied_system.select_action(robot_state)
        
        # 4. Convert embodied action to motor commands
        action_vector = self._embodied_action_to_motor_vector(selected_action)
        
        # 5. Execute action on robot hardware
        motor_state = self.execute_action_vector(action_vector)
        
        # 6. Get outcome sensors for brain learning
        outcome_vector = self.get_sensory_vector()
        
        # 7. Store experience in brain (using converted action vector)
        experience_id = self.brain.store_experience(
            sensory_vector, action_vector, outcome_vector, action_vector
        )
        
        # 8. Performance tracking
        self.total_control_cycles += 1
        self.embodied_decisions += 1
        
        # Track embodied action types
        if isinstance(selected_action, dict):
            action_type = selected_action.get('type', 'unknown')
            if action_type == 'seek_charger':
                self.energy_seeking_actions += 1
            elif action_type == 'stop' and self.hardware_interface.motor_temperature > 50.0:
                self.thermal_management_actions += 1
        
        cycle_time = time.time() - cycle_start
        
        # Detect obstacles and successful navigation (inherited behavior)
        if self.ultrasonic_distance < 20.0:
            self.obstacle_encounters += 1
        elif self.motor_speed > 20.0 and self.ultrasonic_distance > 50.0:
            self.successful_navigations += 1
        
        return {
            'success': True,
            'cycle_time': cycle_time,
            'sensory_vector': sensory_vector,
            'selected_action': selected_action,
            'action_vector': action_vector,
            'motor_state': motor_state,
            'brain_state': {'embodied_free_energy': True},
            'experience_id': experience_id,
            'robot_stats': {
                'position': self.position,
                'ultrasonic_distance': self.ultrasonic_distance,
                'motor_speed': self.motor_speed,
                'steering_angle': self.steering_angle,
                'total_cycles': self.total_control_cycles,
                'embodied_decisions': self.embodied_decisions,
                'energy_seeking_actions': self.energy_seeking_actions,
                'thermal_management_actions': self.thermal_management_actions,
                'battery_level': self.hardware_interface.battery_level,
                'motor_temperature': self.hardware_interface.motor_temperature
            }
        }
    
    def _embodied_action_to_motor_vector(self, embodied_action: Any) -> List[float]:
        """Convert embodied Free Energy action to PiCar-X motor vector."""
        
        if not isinstance(embodied_action, dict):
            # Fallback to neutral action
            return [0.0, 0.0, 0.0, 0.0]  # [motor_speed, steering, camera_pan, camera_tilt]
        
        action_type = embodied_action.get('type', 'stop')
        
        if action_type == 'move':
            direction = embodied_action.get('direction', 'forward')
            speed = embodied_action.get('speed', 0.5)
            
            # Convert to motor commands
            if direction == 'forward':
                motor_speed = speed * 100.0  # Scale to -100 to 100
                steering = 0.0
            elif direction == 'backward':
                motor_speed = -speed * 100.0
                steering = 0.0
            elif direction == 'left':
                motor_speed = speed * 80.0  # Slightly slower for turns
                steering = -30.0  # Full left turn
            elif direction == 'right':
                motor_speed = speed * 80.0
                steering = 30.0  # Full right turn
            else:
                motor_speed = 0.0
                steering = 0.0
                
        elif action_type == 'rotate':
            angle = embodied_action.get('angle', 0)
            # Stationary rotation
            motor_speed = 0.0
            steering = max(-30.0, min(30.0, angle / 2.0))  # Scale angle to steering range
            
        elif action_type == 'stop':
            motor_speed = 0.0
            steering = 0.0
            
        elif action_type == 'seek_charger':
            # Energy-seeking: move forward slowly while scanning
            urgency = embodied_action.get('urgency', 'moderate')
            if urgency == 'high':
                motor_speed = 40.0  # Moderate speed
                steering = 0.0
            else:
                motor_speed = 20.0  # Slow speed
                steering = 0.0
                
        else:
            # Unknown action - stop
            motor_speed = 0.0
            steering = 0.0
        
        # Camera remains neutral for now (could be enhanced)
        camera_pan = 0.0
        camera_tilt = 0.0
        
        return [motor_speed, steering, camera_pan, camera_tilt]
    
    def get_embodied_statistics(self) -> Dict[str, Any]:
        """Get statistics specific to embodied Free Energy system."""
        
        base_stats = self.get_robot_stats()
        
        # Add embodied-specific statistics
        embodied_stats = {
            'total_embodied_decisions': self.embodied_decisions,
            'energy_seeking_rate': self.energy_seeking_actions / max(1, self.embodied_decisions),
            'thermal_management_rate': self.thermal_management_actions / max(1, self.embodied_decisions),
            'current_battery_level': self.hardware_interface.battery_level,
            'current_motor_temperature': self.hardware_interface.motor_temperature,
            'embodied_system_stats': self.embodied_system.get_system_statistics(),
            'hardware_telemetry': self.hardware_interface.get_telemetry().__dict__
        }
        
        # Merge with base stats
        return {**base_stats, **embodied_stats}
    
    def set_verbose_embodied(self, verbose: bool):
        """Enable/disable verbose embodied Free Energy logging."""
        self.embodied_system.set_verbose(verbose)
    
    def print_embodied_report(self):
        """Print comprehensive embodied system report."""
        
        print(f"\\n🧬 EMBODIED PICAR-X REPORT")
        print(f"=" * 50)
        
        # Hardware state
        telemetry = self.hardware_interface.get_telemetry()
        print(f"🔋 Hardware State:")
        print(f"   Battery: {telemetry.battery_percentage:.1%} ({telemetry.battery_voltage:.1f}V)")
        print(f"   Motor Temp: {max(telemetry.motor_temperatures.values()):.1f}°C")
        print(f"   Memory Usage: {telemetry.memory_usage_percentage:.1f}%")
        
        # Embodied behavior statistics
        print(f"\\n🎯 Embodied Behavior:")
        print(f"   Total embodied decisions: {self.embodied_decisions}")
        print(f"   Energy-seeking actions: {self.energy_seeking_actions} ({self.energy_seeking_actions/max(1,self.embodied_decisions)*100:.1f}%)")
        print(f"   Thermal management: {self.thermal_management_actions} ({self.thermal_management_actions/max(1,self.embodied_decisions)*100:.1f}%)")
        
        # Let embodied system print its own report
        self.embodied_system.print_system_report()


def main():
    """Test the embodied PiCar-X brainstem."""
    
    print("🧬 Testing Embodied PiCar-X Brainstem")
    print("=" * 60)
    
    # Create embodied brainstem
    brainstem = EmbodiedPiCarXBrainstem()
    brainstem.set_verbose_embodied(True)
    
    # Run several control cycles to see embodied behavior
    print(f"\\n🚀 Running embodied control cycles...")
    
    for cycle in range(10):
        print(f"\\n--- Cycle {cycle+1} ---")
        result = brainstem.control_cycle()
        
        if result['success']:
            selected_action = result['selected_action']
            robot_stats = result['robot_stats']
            
            print(f"Selected action: {selected_action}")
            print(f"Battery: {robot_stats['battery_level']:.1%}, "
                  f"Motor temp: {robot_stats['motor_temperature']:.1f}°C")
        
        # Simulate time passage
        time.sleep(0.1)
    
    # Print final report
    brainstem.print_embodied_report()
    
    print(f"\\n✅ Embodied brainstem test completed!")


if __name__ == "__main__":
    main()