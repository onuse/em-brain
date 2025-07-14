"""
Embodied Free Energy Framework - Core Components

Implements the Free Energy Principle for robotic systems where behavior emerges
from minimizing prediction error across embodied priors that arise from physical constraints.

Based on Karl Friston's Free Energy Principle but grounded in actual robot hardware.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import time
import numpy as np


@dataclass
class EmbodiedPrior:
    """
    A prior expectation grounded in physical reality.
    
    Unlike abstract motivations, these emerge from the robot's embodied constraints:
    - Battery voltage expectations
    - Motor temperature expectations  
    - Memory availability expectations
    - Sensor reliability expectations
    """
    name: str
    expected_value: float
    current_precision: float
    base_precision: float
    description: str
    
    def update_precision(self, context_factor: float):
        """Precision adapts based on context - core Free Energy mechanism."""
        # Higher context_factor = more important in current situation
        self.current_precision = self.base_precision * context_factor
    
    def calculate_prediction_error(self, predicted_value: float) -> float:
        """Calculate prediction error for this prior."""
        return abs(predicted_value - self.expected_value)
    
    def calculate_free_energy(self, predicted_value: float) -> float:
        """Free Energy = precision-weighted prediction error."""
        prediction_error = self.calculate_prediction_error(predicted_value)
        return prediction_error * self.current_precision


@dataclass
class HardwareTelemetry:
    """Current physical state of the robot - the embodied foundation."""
    battery_voltage: float
    battery_percentage: float
    motor_temperatures: Dict[str, float]
    cpu_temperature: float
    memory_usage_percentage: float
    disk_usage_percentage: float
    wifi_signal_strength: float
    sensor_noise_levels: Dict[str, float]
    timestamp: float
    
    @classmethod
    def mock_telemetry(cls, battery=0.7, motor_temp=40.0) -> 'HardwareTelemetry':
        """Create mock telemetry for testing."""
        return cls(
            battery_voltage=battery * 12.6,  # 12.6V = full battery
            battery_percentage=battery,
            motor_temperatures={'left': motor_temp, 'right': motor_temp},
            cpu_temperature=45.0,
            memory_usage_percentage=30.0,
            disk_usage_percentage=20.0,
            wifi_signal_strength=-45.0,
            sensor_noise_levels={'camera': 0.1, 'ultrasonic': 0.05},
            timestamp=time.time()
        )


class HardwareInterface(ABC):
    """Abstract interface for reading robot hardware state."""
    
    @abstractmethod
    def get_telemetry(self) -> HardwareTelemetry:
        """Get current hardware telemetry."""
        pass
    
    @abstractmethod
    def predict_hardware_effects(self, action: Any, current_state: HardwareTelemetry) -> HardwareTelemetry:
        """Predict how an action will affect hardware state."""
        pass


class MockHardwareInterface(HardwareInterface):
    """Mock hardware interface for testing and simulation."""
    
    def __init__(self, initial_battery=0.8, initial_motor_temp=35.0):
        self.battery = initial_battery
        self.motor_temp = initial_motor_temp
        self.time_step = 0
    
    def get_telemetry(self) -> HardwareTelemetry:
        """Simulate realistic hardware degradation."""
        # Simulate battery drain over time
        self.battery = max(0.0, self.battery - 0.001)  # Slow drain
        
        # Simulate motor heating with activity
        self.motor_temp = min(80.0, self.motor_temp + 0.1)  # Slow heating
        
        self.time_step += 1
        
        return HardwareTelemetry.mock_telemetry(self.battery, self.motor_temp)
    
    def predict_hardware_effects(self, action: Any, current_state: HardwareTelemetry) -> HardwareTelemetry:
        """Predict hardware effects of actions."""
        if not isinstance(action, dict):
            return current_state
        
        # Predict battery drain based on action
        predicted_battery = current_state.battery_percentage
        predicted_motor_temp = current_state.motor_temperatures.get('left', 40.0)
        
        action_type = action.get('type', 'stop')
        
        if action_type == 'move':
            speed = action.get('speed', 0.5)
            # Higher speed = more battery drain and heat
            predicted_battery -= speed * 0.02  # 2% drain per speed unit
            predicted_motor_temp += speed * 5.0  # 5°C heating per speed unit
            
        elif action_type == 'rotate':
            angle = abs(action.get('angle', 0))
            # Rotation uses moderate energy
            predicted_battery -= (angle / 180) * 0.01  # 1% drain per 180° rotation
            predicted_motor_temp += (angle / 180) * 2.0  # 2°C heating per 180° rotation
            
        elif action_type == 'stop':
            # Stopping allows cooling but still drains idle power
            predicted_battery -= 0.001  # Minimal idle drain
            predicted_motor_temp = max(25.0, predicted_motor_temp - 1.0)  # Cooling
        
        elif action_type == 'seek_charger':
            # Energy-seeking: predict eventual energy restoration despite short-term cost
            urgency = action.get('urgency', 'moderate')
            if urgency == 'high':
                # High urgency seeking predicts successful charging
                predicted_battery += 0.05  # Net positive energy gain prediction
                predicted_motor_temp += 2.0  # Some heat from movement to charger
            else:
                # Moderate urgency predicts smaller gain
                predicted_battery += 0.02  # Small net positive energy gain
                predicted_motor_temp += 1.0  # Minimal heat
        
        # Keep values in realistic bounds
        predicted_battery = max(0.0, min(1.0, predicted_battery))
        predicted_motor_temp = max(25.0, min(100.0, predicted_motor_temp))
        
        return HardwareTelemetry.mock_telemetry(predicted_battery, predicted_motor_temp)


@dataclass
class ActionProposal:
    """A proposed action with its predicted Free Energy cost."""
    action: Any
    predicted_outcome: Any
    predicted_hardware_state: HardwareTelemetry
    total_free_energy: float
    prior_contributions: Dict[str, float]
    reasoning: str
    confidence: float = 0.8
    
    def __str__(self):
        return f"Action({self.action}) FE={self.total_free_energy:.3f} - {self.reasoning}"


class EmbodiedPriorSystem:
    """Manages the set of embodied priors for a robot."""
    
    def __init__(self):
        self.priors: Dict[str, EmbodiedPrior] = {}
        self._initialize_standard_priors()
    
    def _initialize_standard_priors(self):
        """Initialize biologically-inspired embodied priors."""
        
        # Energy homeostasis - expect adequate power
        self.priors['energy_homeostasis'] = EmbodiedPrior(
            name='energy_homeostasis',
            expected_value=0.6,  # Expect 60% battery (more conservative)
            current_precision=5.0,  # Higher base precision for energy
            base_precision=5.0,
            description='Maintain adequate energy for continued operation'
        )
        
        # Thermal regulation - expect normal operating temperature
        self.priors['thermal_regulation'] = EmbodiedPrior(
            name='thermal_regulation', 
            expected_value=40.0,  # Expect 40°C motor temperature
            current_precision=2.0,
            base_precision=2.0,
            description='Maintain normal operating temperature'
        )
        
        # Cognitive capacity - expect available processing resources
        self.priors['cognitive_capacity'] = EmbodiedPrior(
            name='cognitive_capacity',
            expected_value=0.3,  # Expect 30% memory usage (plenty free)
            current_precision=1.5,
            base_precision=1.5,
            description='Maintain cognitive processing capacity'
        )
        
        # System integrity - expect reliable sensor function
        self.priors['system_integrity'] = EmbodiedPrior(
            name='system_integrity',
            expected_value=0.05,  # Expect low sensor noise
            current_precision=2.5,
            base_precision=2.5,
            description='Maintain system reliability and sensor accuracy'
        )
    
    def update_precision_weights(self, hardware_state: HardwareTelemetry):
        """Update precision weights based on current hardware context."""
        
        # Energy precision increases as battery gets low (survival pressure)
        battery_factor = 1.0 + (1.0 - hardware_state.battery_percentage) * 3.0
        self.priors['energy_homeostasis'].update_precision(battery_factor)
        
        # Thermal precision increases as motors get hot
        max_motor_temp = max(hardware_state.motor_temperatures.values())
        thermal_factor = 1.0 + max(0, (max_motor_temp - 50.0) / 30.0) * 2.0
        self.priors['thermal_regulation'].update_precision(thermal_factor)
        
        # Cognitive precision increases with memory pressure
        memory_factor = 1.0 + (hardware_state.memory_usage_percentage / 100.0) * 1.5
        self.priors['cognitive_capacity'].update_precision(memory_factor)
        
        # System integrity precision increases with sensor noise
        avg_noise = np.mean(list(hardware_state.sensor_noise_levels.values()))
        integrity_factor = 1.0 + avg_noise * 10.0
        self.priors['system_integrity'].update_precision(integrity_factor)
    
    def calculate_total_free_energy(self, predicted_hardware: HardwareTelemetry) -> Tuple[float, Dict[str, float]]:
        """Calculate total Free Energy across all embodied priors."""
        
        total_free_energy = 0.0
        prior_contributions = {}
        
        # Energy homeostasis contribution
        energy_fe = self.priors['energy_homeostasis'].calculate_free_energy(
            predicted_hardware.battery_percentage
        )
        prior_contributions['energy_homeostasis'] = energy_fe
        total_free_energy += energy_fe
        
        # Thermal regulation contribution  
        max_motor_temp = max(predicted_hardware.motor_temperatures.values())
        thermal_fe = self.priors['thermal_regulation'].calculate_free_energy(max_motor_temp)
        prior_contributions['thermal_regulation'] = thermal_fe
        total_free_energy += thermal_fe
        
        # Cognitive capacity contribution
        cognitive_fe = self.priors['cognitive_capacity'].calculate_free_energy(
            predicted_hardware.memory_usage_percentage / 100.0
        )
        prior_contributions['cognitive_capacity'] = cognitive_fe
        total_free_energy += cognitive_fe
        
        # System integrity contribution
        avg_noise = np.mean(list(predicted_hardware.sensor_noise_levels.values()))
        integrity_fe = self.priors['system_integrity'].calculate_free_energy(avg_noise)
        prior_contributions['system_integrity'] = integrity_fe
        total_free_energy += integrity_fe
        
        return total_free_energy, prior_contributions
    
    def get_precision_report(self) -> Dict[str, float]:
        """Get current precision weights for all priors."""
        return {name: prior.current_precision for name, prior in self.priors.items()}