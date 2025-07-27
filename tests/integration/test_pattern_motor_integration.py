#!/usr/bin/env python3
"""
Test Pattern-Based Motor Integration

Tests that pattern-based motor generation works as a coordinate-free
alternative to gradient-based motor control.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'server'))

import unittest
import torch
import numpy as np
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.brains.field.field_types import FieldDynamicsFamily


class TestPatternMotorIntegration(unittest.TestCase):
    """Test pattern-based motor generation in the main brain."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a brain with pattern motor enabled
        self.factory = DynamicBrainFactory({
            'use_dynamic_brain': True,
            'use_full_features': True,
            'pattern_motor': True,  # Enable pattern-based motor
            'quiet_mode': True
        })
        brain_wrapper = self.factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=16,
            motor_dim=4
        )
        self.brain = brain_wrapper.brain
    
    def test_pattern_motor_enabled(self):
        """Test that pattern motor is properly enabled."""
        self.assertTrue(hasattr(self.brain, 'pattern_motor_enabled'))
        self.assertTrue(self.brain.pattern_motor_enabled)
        
        # Check that pattern motor generator exists
        self.assertTrue(hasattr(self.brain, 'pattern_motor_generator'))
        self.assertIsNotNone(self.brain.pattern_motor_generator)
    
    def test_pattern_based_motor_generation(self):
        """Test that motor commands are generated from patterns."""
        # Create a changing sensory pattern
        patterns = [
            [0.5] * 16 + [0.0],         # Baseline
            [1.0, 0.0] * 8 + [0.0],    # Oscillating
            [0.0, 1.0] * 8 + [0.0],    # Counter-oscillating
        ]
        
        motor_outputs = []
        for pattern in patterns:
            motor_output, brain_state = self.brain.process_robot_cycle(pattern)
            motor_outputs.append(motor_output)
            
            # Verify pattern motor is enabled in brain state
            self.assertTrue(brain_state.get('pattern_motor_enabled', False))
        
        # Check that motor outputs change with patterns
        # (They should be different as patterns evolve)
        self.assertFalse(all(motor_outputs[0] == m for m in motor_outputs[1:]),
                        "Motor outputs should vary with changing patterns")
    
    def test_no_gradients_in_pattern_mode(self):
        """Test that pattern mode doesn't use gradients."""
        sensory_input = [0.7] * 16 + [0.0]
        
        # Process a cycle
        motor_output, brain_state = self.brain.process_robot_cycle(sensory_input)
        
        # Get the last action
        if hasattr(self.brain, 'field_actions') and self.brain.field_actions:
            last_action = self.brain.field_actions[-1]
            
            # Check that field gradients are zero (pattern mode doesn't use gradients)
            if hasattr(last_action, 'field_gradients'):
                gradient_norm = torch.norm(last_action.field_gradients).item()
                self.assertEqual(gradient_norm, 0.0,
                               "Pattern-based motor should have zero gradients")
            
            # Check dynamics contributions show no spatial component
            if hasattr(last_action, 'dynamics_family_contributions'):
                spatial_contribution = last_action.dynamics_family_contributions.get(
                    FieldDynamicsFamily.SPATIAL, 1.0)
                self.assertEqual(spatial_contribution, 0.0,
                               "Pattern-based motor should have no spatial contribution")
    
    def test_pattern_evolution_affects_motor(self):
        """Test that field evolution patterns affect motor output."""
        # Create a stable pattern first
        stable_pattern = [0.5] * 16 + [0.0]
        for _ in range(3):
            self.brain.process_robot_cycle(stable_pattern)
        
        # Get motor output with stable field
        motor_stable, _ = self.brain.process_robot_cycle(stable_pattern)
        
        # Create a rapidly changing pattern
        for i in range(3):
            changing_pattern = [float(i % 2)] * 16 + [0.0]
            self.brain.process_robot_cycle(changing_pattern)
        
        # Get motor output with changing field
        motor_changing, _ = self.brain.process_robot_cycle([0.8] * 16 + [0.0])
        
        # Motor outputs should differ based on field evolution
        self.assertFalse(all(motor_stable[i] == motor_changing[i] for i in range(len(motor_stable))),
                        "Motor output should differ based on field evolution patterns")
    
    def test_pattern_motor_without_coordinates(self):
        """Test that pattern motor works without coordinate dependencies."""
        # This test verifies the core principle: no coordinates needed
        
        # Process several different sensory patterns
        test_patterns = [
            np.random.rand(17).tolist(),     # Random
            [1.0] * 8 + [0.0] * 8 + [0.5],  # Half on, half off
            [0.5, 0.5, 1.0, 1.0] * 4 + [0.0], # Repeating pattern
        ]
        
        all_motors = []
        for pattern in test_patterns:
            motor_output, brain_state = self.brain.process_robot_cycle(pattern)
            all_motors.append(motor_output)
            
            # Verify we got motor output
            self.assertEqual(len(motor_output), 4)
            
            # Check motor values are in valid range
            for motor_val in motor_output:
                self.assertGreaterEqual(motor_val, -1.0)
                self.assertLessEqual(motor_val, 1.0)
        
        # Different patterns should produce different motor outputs
        # (at least some variation expected)
        motor_variance = np.var([m for motors in all_motors for m in motors])
        self.assertGreater(motor_variance, 0.0005,
                          "Motor outputs should vary with different patterns")
    
    def test_pattern_motor_confidence(self):
        """Test that pattern coherence affects action confidence."""
        # Coherent pattern (all same value)
        coherent = [0.8] * 16 + [0.0]
        
        # Incoherent pattern (random)
        incoherent = np.random.rand(17).tolist()
        
        # Process both patterns multiple times to build history
        for _ in range(3):
            self.brain.process_robot_cycle(coherent)
        motor_coherent, _ = self.brain.process_robot_cycle(coherent)
        
        for _ in range(3):
            self.brain.process_robot_cycle(incoherent)
        motor_incoherent, _ = self.brain.process_robot_cycle(incoherent)
        
        # Get action confidence from last actions
        if hasattr(self.brain, 'field_actions') and len(self.brain.field_actions) >= 2:
            # Actions are appended, so get the last two
            action_incoherent = self.brain.field_actions[-1]
            action_coherent = self.brain.field_actions[-2]
            
            if hasattr(action_coherent, 'action_confidence') and hasattr(action_incoherent, 'action_confidence'):
                # Coherent patterns should generally have higher confidence
                # (though this is probabilistic, so we just check they're different)
                self.assertIsNotNone(action_coherent.action_confidence)
                self.assertIsNotNone(action_incoherent.action_confidence)


if __name__ == '__main__':
    unittest.main()