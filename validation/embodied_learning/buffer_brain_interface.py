#!/usr/bin/env python3
"""
Buffer-Based Brain Interface for Validation

Simulates the sensor/motor buffer interface that real robots use,
while providing direct monitoring access for validation experiments.

Architecture:
- SIMULATE: sensor_buffer ← sensory_data ← validation_experiment
- SIMULATE: motor_buffer → motor_actions → validation_experiment  
- MONITOR: brain_state → metrics → validation_experiment
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent.parent / 'server'
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.communication.sensor_buffer import get_sensor_buffer


class BufferBrainInterface:
    """
    Buffer-based brain interface for validation experiments.
    
    Simulates the exact sensor→buffer→brain→buffer→motor pipeline
    that real robots use, with additional monitoring capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None, quiet_mode: bool = True):
        """Initialize buffer-based brain interface."""
        
        # Create brain factory (same as TCP server would)
        self.brain_factory = BrainFactory(config=config, quiet_mode=quiet_mode)
        
        # Get sensor buffer (same as TCP server uses)
        self.sensor_buffer = get_sensor_buffer()
        
        # Validation experiment client ID
        self.client_id = "validation_experiment"
        
        # Motor buffer simulation
        self.latest_motor_actions = [0.0, 0.0, 0.0, 0.0]
        self.latest_brain_state = {}
        
        print(f"🔬 BufferBrainInterface initialized")
        print(f"   Brain: {self.brain_factory}")
        print(f"   Architecture: Validation ←buffer→ Brain")
    
    def feed_sensors(self, sensory_input: List[float]) -> None:
        """
        Feed sensory data to brain via sensor buffer (simulates robot sensors).
        
        This is the same path real robots use.
        """
        # Add sensor input to buffer (exactly like TCP server does)
        self.sensor_buffer.add_sensor_input(self.client_id, sensory_input)
    
    def get_motor_actions(self) -> List[float]:
        """
        Get motor actions from brain via motor buffer (simulates robot actuators).
        
        This processes the brain and returns motor actions.
        """
        try:
            # Get latest sensor data from buffer (like brain_loop would)
            latest_data = self.sensor_buffer.get_latest_data(self.client_id)
            
            if latest_data:
                # Process through brain (exactly like TCP server does)
                action_vector, brain_state = self.brain_factory.process_sensory_input(
                    latest_data, action_dimensions=4
                )
                
                # Store results in motor buffer simulation
                self.latest_motor_actions = action_vector
                self.latest_brain_state = brain_state
                
                # Clear processed sensor data (like TCP server does)
                self.sensor_buffer.clear_client_data(self.client_id)
                
                return action_vector
            else:
                # No new sensor data - return previous actions
                return self.latest_motor_actions
                
        except Exception as e:
            print(f"⚠️  Brain processing error: {e}")
            return [0.0, 0.0, 0.0, 0.0]  # Safe fallback
    
    def get_brain_state(self) -> Dict[str, Any]:
        """
        Get current brain state for monitoring (validation privilege).
        
        This is NOT available to real robots - validation experiments only.
        """
        return self.latest_brain_state.copy()
    
    def get_brain_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive brain statistics (validation privilege).
        
        Includes all internal metrics not available to robots.
        """
        try:
            brain_stats = self.brain_factory.get_brain_stats()
            return brain_stats
        except Exception as e:
            print(f"⚠️  Error getting brain statistics: {e}")
            return {}
    
    def process_cycle(self, sensory_input: List[float]) -> List[float]:
        """
        Complete sensor→motor cycle (convenience method for validation).
        
        Equivalent to:
        1. feed_sensors(sensory_input)
        2. get_motor_actions()
        """
        self.feed_sensors(sensory_input)
        return self.get_motor_actions()
    
    def get_prediction_metrics(self) -> Dict[str, float]:
        """
        Get prediction-driven learning metrics (validation privilege).
        
        Returns the new prediction-based efficiency and learning metrics.
        """
        brain_state = self.get_brain_state()
        
        return {
            'prediction_efficiency': brain_state.get('prediction_efficiency', 0.0),
            'learning_detected': brain_state.get('learning_detected', False),
            'prediction_accuracy': brain_state.get('prediction_accuracy', 0.0),
            'field_evolution_cycles': brain_state.get('field_evolution_cycles', 0),
            'prediction_confidence': brain_state.get('prediction_confidence', 0.0)
        }
    
    def finalize(self):
        """Cleanup brain interface."""
        try:
            self.brain_factory.finalize_session()
            self.sensor_buffer.clear_client_data(self.client_id)
            print("🔬 BufferBrainInterface finalized")
        except Exception as e:
            print(f"⚠️  Error finalizing brain interface: {e}")


def create_buffer_brain_interface(config: Dict[str, Any] = None, quiet_mode: bool = True) -> BufferBrainInterface:
    """
    Create buffer-based brain interface for validation experiments.
    
    This simulates the exact sensor/motor buffer pipeline that real robots use.
    """
    return BufferBrainInterface(config=config, quiet_mode=quiet_mode)


if __name__ == "__main__":
    # Test the buffer brain interface
    brain_interface = create_buffer_brain_interface()
    
    # Simulate sensor→motor cycle
    test_sensors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    motor_actions = brain_interface.process_cycle(test_sensors)
    
    print(f"Test cycle:")
    print(f"  Sensors: {test_sensors}")
    print(f"  Motors: {motor_actions}")
    
    # Check prediction metrics
    prediction_metrics = brain_interface.get_prediction_metrics()
    print(f"  Prediction metrics: {prediction_metrics}")
    
    brain_interface.finalize()