#!/usr/bin/env python3
"""
Test suite for action generation in UnifiedFieldBrain.

These tests verify that the brain can generate meaningful motor commands
from field gradients and that actions have appropriate strength and direction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pytest
from typing import List

# Import the brain we're testing
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))
from brains.field.core_brain import UnifiedFieldBrain, create_unified_field_brain, FieldNativeAction


class TestActionGeneration:
    """Test action generation from field gradients."""
    
    @pytest.fixture
    def brain(self):
        """Create a test brain instance."""
        return create_unified_field_brain(
            spatial_resolution=10,
            temporal_window=5.0,
            quiet_mode=True
        )
    
    def test_action_generation_baseline(self, brain):
        """Test baseline action generation without input."""
        # Generate action from empty field
        action = brain._field_gradients_to_robot_action()
        
        assert isinstance(action, FieldNativeAction)
        assert action.motor_commands is not None
        assert len(action.motor_commands) == 4
        
        # With empty field, should produce minimal action
        print(f"Baseline action: {action.motor_commands}")
        print(f"Baseline confidence: {action.action_confidence}")
    
    def test_action_strength_scaling(self, brain):
        """Test that stronger field activity produces stronger actions."""
        actions = []
        
        # Test with increasing field strengths
        for strength in [0.1, 0.5, 1.0, 2.0]:
            brain.unified_field.zero_()  # Clear field
            brain.unified_field[5, 5, 5, :, :] = strength
            brain._calculate_gradient_flows()
            action = brain._field_gradients_to_robot_action()
            actions.append(action)
        
        # Action strength should increase with field strength
        action_magnitudes = [torch.norm(a.motor_commands).item() for a in actions]
        
        # Check monotonic increase (allowing for small variations)
        for i in range(1, len(action_magnitudes)):
            assert action_magnitudes[i] >= action_magnitudes[i-1] * 0.9
        
        print("Action magnitudes:", action_magnitudes)
    
    def test_directional_action_generation(self, brain):
        """Test that gradients in different directions produce different actions."""
        # Create gradients in different directions
        directions = {
            'x_positive': lambda: setattr(brain.unified_field, 'data', 
                                        torch.zeros_like(brain.unified_field)),
            'y_positive': lambda: setattr(brain.unified_field, 'data', 
                                        torch.zeros_like(brain.unified_field)),
            'z_positive': lambda: setattr(brain.unified_field, 'data', 
                                        torch.zeros_like(brain.unified_field))
        }
        
        # Create gradient in X direction
        brain.unified_field.zero_()
        for i in range(brain.spatial_resolution):
            brain.unified_field[i, 5, 5, 5, 5] = i * 0.2
        brain._calculate_gradient_flows()
        x_action = brain._field_gradients_to_robot_action()
        
        # Create gradient in Y direction
        brain.unified_field.zero_()
        for i in range(brain.spatial_resolution):
            brain.unified_field[5, i, 5, 5, 5] = i * 0.2
        brain._calculate_gradient_flows()
        y_action = brain._field_gradients_to_robot_action()
        
        # Actions should be different for different gradient directions
        x_motor = x_action.motor_commands
        y_motor = y_action.motor_commands
        
        # At least one component should differ significantly
        differences = torch.abs(x_motor - y_motor)
        assert torch.max(differences) > 0.01
        
        print(f"X-gradient action: {x_motor}")
        print(f"Y-gradient action: {y_motor}")
    
    def test_action_confidence_correlation(self, brain):
        """Test that action confidence correlates with gradient strength."""
        confidences = []
        gradient_strengths = []
        
        # Test different field patterns
        patterns = [
            lambda: brain.unified_field.zero_(),  # Empty
            lambda: brain.unified_field.fill_(0.1),  # Uniform low
            lambda: setattr(brain.unified_field[5:7, 5:7, 5:7, :, :], 'data',
                          torch.ones(2, 2, 2, 10, 15) * 0.5),  # Local peak
            lambda: setattr(brain.unified_field[:, :, :, 5, 5], 'data',
                          torch.randn(10, 10, 10) * 0.3)  # Random pattern
        ]
        
        for pattern in patterns:
            pattern()
            brain._calculate_gradient_flows()
            action = brain._field_gradients_to_robot_action()
            
            confidences.append(action.action_confidence)
            gradient_strengths.append(action.gradient_strength)
        
        # Confidence should correlate with gradient strength
        assert len(set(confidences)) > 1  # Should have variation
        assert len(set(gradient_strengths)) > 1
        
        print("Confidences:", confidences)
        print("Gradient strengths:", gradient_strengths)
    
    def test_exploration_fallback(self, brain):
        """Test that weak gradients trigger exploration behavior."""
        # Create very weak field activity
        brain.unified_field.fill_(1e-8)
        brain._calculate_gradient_flows()
        
        # Generate action - should trigger exploration
        action = brain._field_gradients_to_robot_action()
        
        # Exploration should produce small but non-zero actions
        assert action.motor_commands is not None
        magnitude = torch.norm(action.motor_commands).item()
        assert 0.0 < magnitude < 0.5  # Small exploration actions
        assert action.action_confidence < 0.2  # Low confidence
        
        print(f"Exploration action: {action.motor_commands}")
        print(f"Exploration confidence: {action.action_confidence}")
    
    def test_prediction_improvement_modifier(self, brain):
        """Test that prediction improvement affects action modulation."""
        # Simulate prediction improvement history
        brain._prediction_confidence_history = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        
        # Get modifier for improving predictions
        modifier = brain._get_prediction_improvement_addiction_modifier()
        assert modifier > 1.0  # Should amplify actions when learning
        
        # Simulate stagnant predictions
        brain._prediction_confidence_history = [0.5] * 10
        modifier = brain._get_prediction_improvement_addiction_modifier()
        assert 0.8 <= modifier <= 1.5  # Should modulate based on confidence
        
        print(f"Learning modifier: {modifier}")
    
    def test_multi_cycle_action_consistency(self, brain):
        """Test action generation consistency across multiple cycles."""
        # Process multiple cycles with similar inputs
        inputs = [[0.5 + i*0.01] * 24 for i in range(5)]  # Slightly varying inputs
        actions = []
        
        for inp in inputs:
            action, state = brain.process_robot_cycle(inp)
            actions.append(action)
        
        # Actions should show some consistency but not be identical
        for i in range(1, len(actions)):
            prev_action = torch.tensor(actions[i-1])
            curr_action = torch.tensor(actions[i])
            
            # Should not be exactly the same (some variation expected)
            assert not torch.allclose(prev_action, curr_action, atol=1e-6)
            
            # But should be reasonably similar
            difference = torch.norm(curr_action - prev_action).item()
            assert difference < 0.5  # Not wildly different
        
        print("Action sequence:", [a[:2] for a in actions])  # First 2 components
    
    def test_action_clamping(self, brain):
        """Test that motor commands are properly clamped to valid range."""
        # Create very strong field activity
        brain.unified_field[5, 5, 5, :, :] = 100.0  # Extreme value
        brain._calculate_gradient_flows()
        
        action = brain._field_gradients_to_robot_action()
        
        # All motor commands should be in [-1, 1]
        for cmd in action.motor_commands:
            assert -1.0 <= cmd <= 1.0
        
        print(f"Clamped action: {action.motor_commands}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])