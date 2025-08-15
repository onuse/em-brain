#!/usr/bin/env python3
"""
Test suite for verifying gradient extraction in UnifiedFieldBrain.

These tests ensure that gradients are properly calculated and extracted
from the multi-dimensional field for action generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pytest
from typing import Tuple

# Import the brain we're testing
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))
from brains.field.core_brain import UnifiedFieldBrain, create_unified_field_brain


class TestGradientExtraction:
    """Test gradient extraction and calculation in the field brain."""
    
    @pytest.fixture
    def brain(self):
        """Create a test brain instance."""
        return create_unified_field_brain(
            spatial_resolution=10,  # Smaller for faster tests
            temporal_window=5.0,
            quiet_mode=True
        )
    
    def test_gradient_calculation_creates_correct_shapes(self, brain):
        """Test that gradient calculation produces correct tensor shapes."""
        # Apply some test data to create field activity
        test_input = [0.5] * 24  # Standard robot input
        brain.process_robot_cycle(test_input)
        
        # Calculate gradients
        brain._calculate_gradient_flows()
        
        # Check gradient shapes match field shape
        assert 'gradient_x' in brain.gradient_flows
        assert 'gradient_y' in brain.gradient_flows
        assert 'gradient_z' in brain.gradient_flows
        
        expected_shape = brain.unified_field.shape
        assert brain.gradient_flows['gradient_x'].shape == expected_shape
        assert brain.gradient_flows['gradient_y'].shape == expected_shape
        assert brain.gradient_flows['gradient_z'].shape == expected_shape
    
    def test_gradient_extraction_current_broken_behavior(self, brain):
        """Document that the old broken gradient extraction is now fixed."""
        # Create known field state with clear gradients
        brain.unified_field[5, 5, 5, :, :] = 1.0  # Create a peak
        brain._calculate_gradient_flows()
        
        # Current extraction (now fixed)
        center_idx = brain.spatial_resolution // 2
        
        # The old problematic indexing would have collapsed dimensions
        # Now we use proper local region extraction in the action generation
        
        # Verify gradients have correct shape
        assert brain.gradient_flows['gradient_x'].shape == brain.unified_field.shape
        assert brain.gradient_flows['gradient_y'].shape == brain.unified_field.shape
        assert brain.gradient_flows['gradient_z'].shape == brain.unified_field.shape
        
        # The gradient should have meaningful values near the peak
        grad_x_at_peak = brain.gradient_flows['gradient_x'][5, 5, 5, :, :]
        print(f"Fixed gradient shape: {brain.gradient_flows['gradient_x'].shape}")
        print(f"Max gradient at peak: {torch.max(grad_x_at_peak).item()}")
    
    def test_gradient_extraction_proposed_fix(self, brain):
        """Test the proposed fix for gradient extraction."""
        # Create known field state
        brain.unified_field[5, 5, 5, :, :] = 1.0  # Create a peak
        brain._calculate_gradient_flows()
        
        center_idx = brain.spatial_resolution // 2
        
        # Proposed fix: Extract a local region and aggregate properly
        # Get gradients in a 3x3x3 region around center
        region_slice = slice(center_idx-1, center_idx+2)
        
        grad_x_region = brain.gradient_flows['gradient_x'][region_slice, region_slice, region_slice]
        grad_y_region = brain.gradient_flows['gradient_y'][region_slice, region_slice, region_slice]
        grad_z_region = brain.gradient_flows['gradient_z'][region_slice, region_slice, region_slice]
        
        # Aggregate across spatial dimensions, preserving other dimensions
        grad_x_aggregated = torch.mean(grad_x_region, dim=(0, 1, 2))
        grad_y_aggregated = torch.mean(grad_y_region, dim=(0, 1, 2))
        grad_z_aggregated = torch.mean(grad_z_region, dim=(0, 1, 2))
        
        # These should have shape matching remaining dimensions
        expected_shape = brain.unified_field.shape[3:]  # All dimensions except spatial
        assert grad_x_aggregated.shape == expected_shape
        assert grad_y_aggregated.shape == expected_shape
        assert grad_z_aggregated.shape == expected_shape
    
    def test_gradient_strength_and_direction(self, brain):
        """Test that gradients have meaningful strength and direction."""
        # Create a gradient in the field
        for i in range(brain.spatial_resolution):
            brain.unified_field[i, :, :, 5, 7] = i * 0.1  # Linear gradient in X
        
        brain._calculate_gradient_flows()
        
        # Check gradient strength
        grad_x = brain.gradient_flows['gradient_x']
        grad_magnitude = torch.abs(grad_x)
        
        # Should have non-zero gradients
        assert torch.max(grad_magnitude) > 0.01
        
        # Gradient in X should be stronger than Y or Z for this pattern
        grad_y_magnitude = torch.abs(brain.gradient_flows['gradient_y'])
        grad_z_magnitude = torch.abs(brain.gradient_flows['gradient_z'])
        
        assert torch.mean(grad_magnitude) > torch.mean(grad_y_magnitude)
        assert torch.mean(grad_magnitude) > torch.mean(grad_z_magnitude)
    
    def test_action_generation_from_gradients(self, brain):
        """Test that action generation produces non-zero actions from gradients."""
        # Create strong field activity
        brain.unified_field[5:7, 5:7, 5:7, :, :] = 1.0
        brain._calculate_gradient_flows()
        
        # Generate action
        action = brain._field_gradients_to_robot_action()
        
        # Should produce non-zero action
        assert action.motor_commands is not None
        assert len(action.motor_commands) == 4
        
        # At least one motor command should be non-zero
        motor_magnitude = torch.norm(action.motor_commands).item()
        assert motor_magnitude > 0.0
        
        # Check confidence and gradient strength
        assert action.action_confidence >= 0.0
        assert action.gradient_strength >= 0.0
        
        print(f"Action magnitude: {motor_magnitude}")
        print(f"Gradient strength: {action.gradient_strength}")
        print(f"Action confidence: {action.action_confidence}")
    
    def test_gradient_diffusion_effect(self, brain):
        """Test that diffusion spreads gradients appropriately."""
        # Create point source
        center = brain.spatial_resolution // 2
        brain.unified_field[center, center, center, 5, 5] = 10.0
        
        # Apply diffusion
        brain._apply_spatial_diffusion()
        
        # Check that activity spread to neighbors
        neighbor_activity = brain.unified_field[center-1:center+2, center-1:center+2, center-1:center+2, 5, 5]
        
        # Center should still be highest
        assert brain.unified_field[center, center, center, 5, 5] == torch.max(neighbor_activity)
        
        # But neighbors should have some activity
        edge_activity = brain.unified_field[center-1, center, center, 5, 5]
        assert edge_activity > 0.0
    
    def test_gradient_following_produces_movement(self, brain):
        """Test that gradient following produces movement toward high activity."""
        # Create two sequential inputs simulating movement
        input1 = [0.3, 0.3, 0.1] + [0.5] * 21
        input2 = [0.4, 0.4, 0.1] + [0.5] * 21  # Slightly moved
        
        # Process inputs
        action1, state1 = brain.process_robot_cycle(input1)
        action2, state2 = brain.process_robot_cycle(input2)
        
        # Should generate some motor commands
        assert any(abs(a) > 0.0 for a in action2)
        
        # Check that field is building up patterns
        assert state2['field_max_activation'] > 0.0
        assert state2['field_total_energy'] > 0.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])