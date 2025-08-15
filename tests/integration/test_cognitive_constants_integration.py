#!/usr/bin/env python3
"""
Test Cognitive Constants Integration

Verifies that cognitive constants are properly integrated throughout the system.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'server'))

import unittest
import torch
from src.parameters.cognitive_config import get_cognitive_config, reset_cognitive_config
from src.parameters.cognitive_constants import (
    PredictionErrorConstants, 
    StabilityConstants,
    TemporalConstants
)
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.brain_loop import DecoupledBrainLoop


class TestCognitiveConstantsIntegration(unittest.TestCase):
    """Test that cognitive constants are used throughout the system."""
    
    def setUp(self):
        """Reset configuration before each test."""
        reset_cognitive_config()
    
    def test_cognitive_config_loading(self):
        """Test that cognitive configuration loads properly."""
        config = get_cognitive_config()
        
        # Check brain config values match constants
        self.assertEqual(
            config.brain_config.optimal_prediction_error,
            PredictionErrorConstants.OPTIMAL_PREDICTION_ERROR_TARGET
        )
        self.assertEqual(
            config.brain_config.activation_threshold,
            StabilityConstants.MIN_ACTIVATION_VALUE
        )
        
        # Check sensor config
        self.assertEqual(config.sensor_config.autopilot_sensor_probability, 0.2)
        self.assertEqual(config.sensor_config.focused_sensor_probability, 0.5)
        self.assertEqual(config.sensor_config.deep_think_sensor_probability, 0.9)
    
    def test_brain_factory_uses_config(self):
        """Test that brain factory uses cognitive configuration."""
        factory = DynamicBrainFactory({'quiet_mode': True})
        
        # Check factory loaded config
        self.assertIsNotNone(factory.cognitive_config)
        self.assertIsNotNone(factory.brain_config)
        
        # Create a brain and check it gets config values
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=17,
            motor_dim=4
        )
        
        brain = brain_wrapper.brain
        
        # Check brain parameters match config
        self.assertEqual(
            brain.field_evolution_rate,
            factory.brain_config.field_evolution_rate
        )
        self.assertEqual(
            brain.constraint_discovery_rate,
            factory.brain_config.constraint_discovery_rate
        )
    
    def test_brain_uses_cognitive_constants(self):
        """Test that brain internals use cognitive constants."""
        factory = DynamicBrainFactory({'quiet_mode': True})
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=17,
            motor_dim=4
        )
        
        brain = brain_wrapper.brain
        config = get_cognitive_config().brain_config
        
        # Check field parameters
        self.assertEqual(brain.field_decay_rate, config.field_decay_rate)
        self.assertEqual(brain.field_diffusion_rate, config.field_diffusion_rate)
        self.assertEqual(brain.topology_stability_threshold, config.topology_stability_threshold)
        
        # Check activation threshold in field
        min_activation = torch.min(brain.unified_field).item()
        self.assertAlmostEqual(min_activation, config.activation_threshold, places=5)
        
        # Check prediction confidence
        self.assertEqual(brain._current_prediction_confidence, config.default_prediction_confidence)
        
        # Check spontaneous dynamics parameters
        self.assertEqual(brain.spontaneous.resting_potential, config.resting_potential)
        self.assertEqual(brain.spontaneous.spontaneous_rate, config.spontaneous_rate)
        
        # Check cognitive autopilot thresholds
        # Note: CognitiveAutopilot stores these as private attributes
        # The brain passes them correctly during initialization
    
    def test_brain_loop_uses_config(self):
        """Test that brain loop uses cognitive configuration."""
        # Create a simple brain for testing
        factory = DynamicBrainFactory({'quiet_mode': True})
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=17,
            motor_dim=4
        )
        
        # Create brain loop without specifying cycle time
        loop = DecoupledBrainLoop(brain_wrapper)
        
        # Check it uses temporal constants
        temporal_config = get_cognitive_config().get_temporal_config()
        expected_cycle_time = temporal_config['control_cycle_target']
        self.assertAlmostEqual(loop.base_cycle_time_s, expected_cycle_time, places=3)
        
        # Check sensor config is loaded
        self.assertIsNotNone(loop.sensor_config)
        self.assertEqual(
            loop.last_prediction_confidence,
            get_cognitive_config().brain_config.default_prediction_confidence
        )
    
    def test_environment_override(self):
        """Test that environment variables can override config."""
        # Set environment variable
        os.environ['BRAIN_BRAIN_FIELD_EVOLUTION_RATE'] = '0.2'
        
        # Reset and reload config
        reset_cognitive_config()
        config = get_cognitive_config()
        
        # Check override was applied
        self.assertEqual(config.brain_config.field_evolution_rate, 0.2)
        
        # Clean up
        del os.environ['BRAIN_BRAIN_FIELD_EVOLUTION_RATE']
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = get_cognitive_config()
        
        # Should validate successfully with defaults
        self.assertTrue(config.validate_config())
        
        # Test invalid config
        config.brain_config.optimal_prediction_error = 1.5  # Out of bounds
        self.assertFalse(config.validate_config())


if __name__ == '__main__':
    unittest.main()