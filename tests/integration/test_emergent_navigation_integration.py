#!/usr/bin/env python3
"""
Test Emergent Navigation Integration

Tests that emergent navigation (coordinate-free spatial understanding)
is properly integrated into the main brain.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'server'))

import unittest
import torch
import numpy as np
from src.core.dynamic_brain_factory import DynamicBrainFactory


class TestEmergentNavigationIntegration(unittest.TestCase):
    """Test emergent navigation features in the main brain."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a brain with emergent navigation enabled
        self.factory = DynamicBrainFactory({
            'use_dynamic_brain': True,
            'use_full_features': True,
            'emergent_navigation': True,  # Enable emergent navigation
            'quiet_mode': True
        })
        brain_wrapper = self.factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=24,
            motor_dim=4
        )
        self.brain = brain_wrapper.brain
    
    def test_emergent_navigation_initialized(self):
        """Test that emergent navigation is properly initialized."""
        # Check that emergent navigation is enabled
        self.assertTrue(hasattr(self.brain, 'emergent_navigation_enabled'))
        self.assertTrue(self.brain.emergent_navigation_enabled)
        
        # Check that emergent spatial dynamics exists
        self.assertTrue(hasattr(self.brain, 'emergent_spatial'))
        self.assertIsNotNone(self.brain.emergent_spatial)
        
        # Check that emergent interface exists
        self.assertTrue(hasattr(self.brain, 'emergent_interface'))
        self.assertIsNotNone(self.brain.emergent_interface)
    
    def test_place_discovery_in_brain(self):
        """Test that places are discovered during normal brain operation."""
        # Process several cycles with distinctive sensory patterns
        patterns = [
            [1.0] * 12 + [0.0] * 12 + [0.5],  # Pattern 1 with neutral reward
            [0.0] * 12 + [1.0] * 12 + [0.8],  # Pattern 2 with positive reward
            [0.5] * 24 + [0.2],                # Pattern 3 with low reward
        ]
        
        navigation_states = []
        places_discovered = 0
        
        # Process each pattern multiple times to build up field energy
        for pattern in patterns:
            # Process the same pattern 5 times to build field stability
            for _ in range(5):
                motor_output, brain_state = self.brain.process_robot_cycle(pattern)
                if 'navigation' in brain_state:
                    navigation_states.append(brain_state['navigation'])
                    places_discovered = brain_state['navigation'].get('places_discovered', places_discovered)
        
        # Check that places were discovered
        self.assertGreater(len(navigation_states), 0)
        self.assertGreater(places_discovered, 0, "Should have discovered at least one place")
    
    def test_emergent_motor_generation(self):
        """Test that motor commands emerge from field dynamics."""
        # Create a stable sensory pattern
        sensory_input = [0.7] * 10 + [0.3] * 10 + [0.5] * 4 + [0.0]
        
        # Process multiple cycles to establish field evolution
        motor_outputs = []
        for i in range(5):
            motor_output, brain_state = self.brain.process_robot_cycle(sensory_input)
            motor_outputs.append(motor_output)
            
            # Check that we have 4 motor outputs
            self.assertEqual(len(motor_output), 4)
        
        # Check that motor outputs vary (not all zeros)
        all_zeros = all(all(m == 0.0 for m in output) for output in motor_outputs)
        self.assertFalse(all_zeros, "Motor outputs should not all be zero")
    
    def test_navigation_state_in_brain_state(self):
        """Test that navigation information is included in brain state."""
        sensory_input = [0.5] * 24 + [0.0]
        
        motor_output, brain_state = self.brain.process_robot_cycle(sensory_input)
        
        # Check that navigation state exists
        self.assertIn('navigation', brain_state)
        nav_state = brain_state['navigation']
        
        # Check navigation state fields
        self.assertIn('current_place', nav_state)
        self.assertIn('known_places', nav_state)
        self.assertIn('navigation_active', nav_state)
        self.assertIn('field_stability', nav_state)
    
    def test_place_recognition(self):
        """Test that similar patterns are recognized as the same place."""
        # Create a distinctive pattern
        base_pattern = [0.9, 0.1, 0.9, 0.1] * 6 + [0.7]
        
        # Process it multiple times with slight variations
        places = []
        for i in range(5):
            # Add small noise
            pattern = [val + np.random.randn() * 0.05 for val in base_pattern[:-1]] + [base_pattern[-1]]
            motor_output, brain_state = self.brain.process_robot_cycle(pattern)
            
            if 'navigation' in brain_state:
                current_place = brain_state['navigation'].get('current_place')
                if current_place:
                    places.append(current_place)
        
        # Should recognize as same place (if any were discovered)
        if len(places) > 2:
            # Most common place should appear multiple times
            place_counts = {place: places.count(place) for place in set(places)}
            max_count = max(place_counts.values())
            self.assertGreater(max_count, 1, "Should recognize same place multiple times")
    
    def test_pattern_diversity_tracking(self):
        """Test that pattern diversity is tracked."""
        # Feed various patterns
        patterns = [
            [1.0] * 24 + [0.0],         # All high
            [0.0] * 24 + [0.0],         # All low
            [0.5] * 24 + [0.0],         # All medium
            [1.0, 0.0] * 12 + [0.0],   # Alternating
            np.random.rand(25).tolist() # Random
        ]
        
        for pattern in patterns:
            motor_output, brain_state = self.brain.process_robot_cycle(pattern)
        
        # Get final statistics
        if hasattr(self.brain, 'emergent_interface'):
            stats = self.brain.emergent_interface.get_statistics()
            self.assertGreater(stats['unique_patterns'], 0)
            self.assertIn('pattern_diversity', stats)
    
    def test_navigation_with_standard_features(self):
        """Test that emergent navigation works alongside other features."""
        # Process cycles to activate various features
        for i in range(10):
            sensory_input = np.random.rand(24).tolist() + [0.0]
            motor_output, brain_state = self.brain.process_robot_cycle(sensory_input)
            
            # Check that all systems are reporting
            self.assertIn('cognitive_mode', brain_state)  # Autopilot system
            self.assertIn('field_energy', brain_state)    # Field dynamics
            self.assertIn('prediction_confidence', brain_state)  # Prediction system
            
            # If we have enhanced dynamics, check for phase transitions
            if hasattr(self.brain, 'enhanced_dynamics'):
                phase = self.brain.enhanced_dynamics.current_phase
                self.assertIn(phase, ["stable", "high_energy", "chaotic", "low_energy"])


if __name__ == '__main__':
    unittest.main()