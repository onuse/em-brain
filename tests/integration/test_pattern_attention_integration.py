#!/usr/bin/env python3
"""
Test Pattern-Based Attention Integration

Tests that pattern-based attention works as a coordinate-free
alternative to gradient-based attention systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'server'))

import unittest
import torch
import numpy as np
from src.core.dynamic_brain_factory import DynamicBrainFactory


class TestPatternAttentionIntegration(unittest.TestCase):
    """Test pattern-based attention in the main brain."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a brain with pattern attention enabled
        self.factory = DynamicBrainFactory({
            'use_dynamic_brain': True,
            'use_full_features': True,
            'pattern_attention': True,  # Enable pattern-based attention
            'enable_attention': False,  # Disable gradient-based attention
            'quiet_mode': True
        })
        brain_wrapper = self.factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=16,
            motor_dim=4
        )
        self.brain = brain_wrapper.brain
    
    def test_pattern_attention_enabled(self):
        """Test that pattern attention is properly enabled."""
        self.assertTrue(hasattr(self.brain, 'pattern_attention_enabled'))
        self.assertTrue(self.brain.pattern_attention_enabled)
        
        # Check that pattern attention system exists
        self.assertTrue(hasattr(self.brain, 'pattern_attention'))
        self.assertIsNotNone(self.brain.pattern_attention)
        
        # Check that gradient attention is disabled
        self.assertFalse(self.brain.attention_enabled)
    
    def test_pattern_attention_processing(self):
        """Test that attention processes field patterns."""
        # Create distinctive sensory patterns
        patterns = [
            [1.0] * 8 + [0.0] * 8 + [0.5],   # Half on/off pattern
            [0.0, 1.0] * 8 + [0.8],          # Alternating pattern
            np.random.rand(17).tolist(),     # Random pattern
        ]
        
        attention_states = []
        for pattern in patterns:
            motor_output, brain_state = self.brain.process_robot_cycle(pattern)
            
            # Check that pattern attention is enabled
            self.assertTrue(brain_state.get('pattern_attention_enabled', False))
            
            # Check for attention state
            if 'attention' in brain_state:
                attention_states.append(brain_state['attention'])
                self.assertEqual(brain_state['attention']['type'], 'pattern-based')
        
        # Should have attention states
        self.assertGreater(len(attention_states), 0)
        
        # Check attention metrics
        if attention_states:
            last_attention = attention_states[-1]
            self.assertIn('known_patterns', last_attention)
            self.assertIn('current_focus', last_attention)
            self.assertIn('focus_strength', last_attention)
    
    def test_pattern_salience_detection(self):
        """Test that salient patterns capture attention."""
        # Start with baseline pattern (low variance)
        baseline = [0.5] * 16 + [0.0]
        for _ in range(5):
            self.brain.process_robot_cycle(baseline)
        
        # Get baseline attention
        _, baseline_state = self.brain.process_robot_cycle(baseline)
        baseline_focus = baseline_state.get('attention', {}).get('focus_strength', 0)
        
        # Introduce highly salient pattern (high contrast, novel structure)
        # Mix of extremes with rhythmic pattern - very different from baseline
        salient = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0] * 2 + [1.0]  
        _, salient_state = self.brain.process_robot_cycle(salient)
        salient_focus = salient_state.get('attention', {}).get('focus_strength', 0)
        
        # Debug output
        print(f"\nBaseline focus: {baseline_focus:.6f}")
        print(f"Salient focus: {salient_focus:.6f}")
        print(f"Difference: {salient_focus - baseline_focus:.6f}")
        
        # Salient pattern should have higher focus strength
        # Allow for small numerical differences
        self.assertGreater(salient_focus, baseline_focus + 0.001,
                          f"Salient patterns should have higher focus strength. "
                          f"Baseline: {baseline_focus:.6f}, Salient: {salient_focus:.6f}")
    
    def test_pattern_memory_formation(self):
        """Test that patterns are remembered."""
        # Feed a unique pattern multiple times
        unique_pattern = [0.9, 0.1, 0.8, 0.2] * 4 + [0.7]
        
        initial_patterns = 0
        final_patterns = 0
        
        for i in range(5):
            motor_output, brain_state = self.brain.process_robot_cycle(unique_pattern)
            
            if 'attention' in brain_state:
                if i == 0:
                    initial_patterns = brain_state['attention'].get('known_patterns', 0)
                elif i == 4:
                    final_patterns = brain_state['attention'].get('known_patterns', 0)
        
        # Should discover new patterns
        self.assertGreater(final_patterns, 0, "Should have discovered patterns")
    
    def test_attention_focus_shifting(self):
        """Test that attention shifts between patterns."""
        # Create very different patterns
        pattern1 = [1.0] * 16 + [0.0]       # All high
        pattern2 = [0.0] * 16 + [0.0]       # All low  
        pattern3 = [0.5, 1.0] * 8 + [0.0]  # Mixed
        
        focus_patterns = []
        all_attention_states = []
        
        # Process each pattern multiple times
        for i, pattern in enumerate([pattern1, pattern1, pattern2, pattern2, pattern3, pattern3]):
            motor_output, brain_state = self.brain.process_robot_cycle(pattern)
            
            if 'attention' in brain_state:
                all_attention_states.append((i, brain_state['attention']))
                current_focus = brain_state['attention'].get('current_focus')
                if current_focus:
                    focus_patterns.append(current_focus)
        
        # Debug: print what we got
        print(f"\nTotal attention states: {len(all_attention_states)}")
        for idx, att_state in all_attention_states:
            print(f"  Step {idx}: focus={att_state.get('current_focus')}, "
                  f"type={att_state.get('type')}, patterns={att_state.get('known_patterns')}")
        print(f"Unique focus patterns: {set(focus_patterns)}")
        
        # Should have different focus patterns (attention shifting)
        unique_focuses = set(focus_patterns)
        self.assertGreater(len(unique_focuses), 1,
                          "Attention should shift between different patterns")
    
    def test_field_modulation_by_attention(self):
        """Test that attention modulates field activity."""
        # Create a high-salience pattern
        high_salience = [1.0, 0.0, 1.0, 0.0] * 4 + [0.0]
        
        # Process multiple times to build attention
        field_energies = []
        for _ in range(5):
            motor_output, brain_state = self.brain.process_robot_cycle(high_salience)
            field_energy = brain_state.get('field_energy', 0)
            field_energies.append(field_energy)
        
        # Field energy should vary with attention modulation
        energy_variance = np.var(field_energies)
        self.assertGreater(energy_variance, 0,
                          "Field energy should vary with attention modulation")
    
    def test_pattern_attention_without_coordinates(self):
        """Test that pattern attention works without spatial coordinates."""
        # This is the key test - no coordinates involved
        
        # Create patterns with no spatial structure, just temporal
        temporal_patterns = [
            [0.5 + 0.5 * np.sin(i * 0.5) for i in range(17)],  # Sine wave
            [0.5 + 0.5 * np.cos(i * 0.3) for i in range(17)],  # Cosine wave
            [float(i % 2) for i in range(17)],                  # Square wave
        ]
        
        attention_metrics = []
        for pattern in temporal_patterns:
            motor_output, brain_state = self.brain.process_robot_cycle(pattern)
            
            if 'attention' in brain_state:
                metrics = brain_state['attention']
                attention_metrics.append(metrics)
                
                # Verify no spatial/coordinate information
                self.assertNotIn('attention_regions', metrics)
                self.assertNotIn('spatial_focus', metrics)
                self.assertNotIn('gradient_peaks', metrics)
        
        # All patterns should be processable
        self.assertEqual(len(attention_metrics), len(temporal_patterns))
        
        # Different patterns should produce different attention states
        if len(attention_metrics) >= 2:
            # At least some metrics should differ
            self.assertNotEqual(attention_metrics[0], attention_metrics[-1],
                              "Different patterns should produce different attention states")
    
    def test_cross_modal_binding(self):
        """Test pattern binding through synchrony."""
        # Simulate multi-modal input (first 8 = visual, next 8 = audio)
        # Synchronized pattern (both modalities have same rhythm)
        sync_pattern = ([1.0, 0.0] * 4) + ([1.0, 0.0] * 4) + [0.0]
        
        # Process synchronized pattern
        motor_output, brain_state = self.brain.process_robot_cycle(sync_pattern)
        
        # Pattern attention should detect synchrony
        if 'attention' in brain_state:
            # Check that patterns were processed
            self.assertGreater(brain_state['attention'].get('known_patterns', 0), 0)
            
            # In full implementation, would check for bindings
            # For now, just verify attention processed the multi-modal pattern
            self.assertIsNotNone(brain_state['attention'].get('current_focus'))


if __name__ == '__main__':
    unittest.main()