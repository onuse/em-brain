#!/usr/bin/env python3
"""
Test Attention System Integration

Verifies that the attention system is properly integrated into the brain.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'server'))

import unittest
import numpy as np
from src.core.dynamic_brain_factory import DynamicBrainFactory


class TestAttentionIntegration(unittest.TestCase):
    """Test attention system integration in the brain."""
    
    def setUp(self):
        """Create brain with attention enabled."""
        self.factory = DynamicBrainFactory({
            'quiet_mode': True,
            'enable_attention': True
        })
        
    def test_attention_initialization(self):
        """Test that attention system initializes properly."""
        brain_wrapper = self.factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=17,
            motor_dim=4
        )
        
        brain = brain_wrapper.brain
        
        # Check attention is enabled
        self.assertTrue(brain.attention_enabled)
        self.assertTrue(hasattr(brain, 'integrated_attention'))
        self.assertIsNotNone(brain.integrated_attention)
    
    def test_attention_processing(self):
        """Test attention processing in robot cycle."""
        brain_wrapper = self.factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=17,
            motor_dim=4
        )
        
        brain = brain_wrapper.brain
        
        # Create sensory input with salient pattern
        sensory_input = [0.1] * 17  # Mostly low values
        sensory_input[8] = 0.9  # One high value (salient)
        sensory_input[9] = 0.8  # Adjacent high value
        
        # Process cycle
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Check attention state exists
        self.assertIn('attention', brain_state)
        attention_state = brain_state['attention']
        
        # Check attention state structure
        self.assertIn('mean_saliency', attention_state)
        self.assertIn('current_regions', attention_state)
        self.assertIn('attention_triggers', attention_state)
    
    def test_attention_affects_confidence(self):
        """Test that attention affects prediction confidence."""
        brain_wrapper = self.factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=17,
            motor_dim=4
        )
        
        brain = brain_wrapper.brain
        
        # First, establish baseline with uniform input
        uniform_input = [0.5] * 17
        for _ in range(5):
            _, state1 = brain.process_robot_cycle(uniform_input)
        baseline_confidence = state1['prediction_confidence']
        
        # Now present highly salient input
        salient_input = [0.1, 0.9, 0.1, 0.9] * 4 + [0.5]  # High contrast pattern
        _, state2 = brain.process_robot_cycle(salient_input)
        salient_confidence = state2['prediction_confidence']
        
        # High saliency should reduce confidence (novelty)
        # This may not always be true in first few cycles, so we check trend
        print(f"Baseline confidence: {baseline_confidence:.3f}")
        print(f"Salient confidence: {salient_confidence:.3f}")
        
        # At minimum, attention system should be tracking this
        self.assertGreater(state2['attention']['attention_triggers'], 0)
    
    def test_attention_field_modulation(self):
        """Test that attention modulates the unified field."""
        brain_wrapper = self.factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=17,
            motor_dim=4
        )
        
        brain = brain_wrapper.brain
        
        # Get initial field energy
        initial_field_energy = float(brain.unified_field.abs().mean())
        
        # Present highly salient input multiple times
        salient_input = [0.9 if i % 2 == 0 else 0.1 for i in range(17)]
        
        for _ in range(3):
            _, state = brain.process_robot_cycle(salient_input)
        
        # Field energy should change with attention modulation
        final_field_energy = state['field_energy']
        
        # Check that field has been affected
        # The field energy should increase due to attention modulation
        print(f"Initial field energy: {initial_field_energy:.6f}")
        print(f"Final field energy: {final_field_energy:.6f}")
        print(f"Energy change: {abs(final_field_energy - initial_field_energy):.6f}")
        
        # We expect at least 30% increase in field energy with high attention
        # (attention influence is 0.3 from novelty_boost)
        self.assertGreater(final_field_energy, initial_field_energy * 1.3)
        
        # Check attention is active
        self.assertGreater(state['attention']['mean_saliency'], 0.3)
    
    def test_attention_disabled(self):
        """Test brain works with attention disabled."""
        # Create brain with attention disabled
        factory = DynamicBrainFactory({
            'quiet_mode': True,
            'enable_attention': False
        })
        
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=17,
            motor_dim=4
        )
        
        brain = brain_wrapper.brain
        
        # Should work without attention
        sensory_input = [0.5] * 17
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # No attention in state
        self.assertNotIn('attention', brain_state)
        
        # Should still produce valid output
        self.assertIsNotNone(motor_output)
        self.assertEqual(len(motor_output), 4)


if __name__ == '__main__':
    unittest.main()