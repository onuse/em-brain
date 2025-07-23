#!/usr/bin/env python3
"""
Test suite for field evolution and dynamics in UnifiedFieldBrain.

These tests verify that the field evolves correctly, including decay,
diffusion, and constraint-guided evolution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pytest
import time

# Import the brain we're testing
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../server/src'))
from brains.field.core_brain import UnifiedFieldBrain, create_unified_field_brain


class TestFieldEvolution:
    """Test field evolution and dynamics."""
    
    @pytest.fixture
    def brain(self):
        """Create a test brain instance."""
        return create_unified_field_brain(
            spatial_resolution=10,
            temporal_window=5.0,
            field_evolution_rate=0.1,
            quiet_mode=True
        )
    
    def test_field_decay(self, brain):
        """Test that field values decay over time."""
        # Set initial field values
        initial_value = 1.0
        brain.unified_field[5, 5, 5, 5, 5] = initial_value
        
        # Get initial energy
        initial_energy = torch.sum(torch.abs(brain.unified_field)).item()
        
        # Evolve field multiple times
        for _ in range(10):
            brain._evolve_unified_field()
        
        # Check decay
        final_value = brain.unified_field[5, 5, 5, 5, 5].item()
        final_energy = torch.sum(torch.abs(brain.unified_field)).item()
        
        # Value should decay
        assert final_value < initial_value
        assert final_energy < initial_energy
        
        # Check decay rate
        expected_value = initial_value * (brain.field_decay_rate ** 10)
        assert abs(final_value - expected_value) < 0.1
        
        print(f"Decay: {initial_value:.3f} -> {final_value:.3f} (expected {expected_value:.3f})")
    
    def test_spatial_diffusion(self, brain):
        """Test spatial diffusion spreads activity."""
        # Create point source
        center = brain.spatial_resolution // 2
        brain.unified_field[center, center, center, 5, 5] = 10.0
        
        # Store initial state
        initial_center = brain.unified_field[center, center, center, 5, 5].item()
        initial_neighbor = brain.unified_field[center+1, center, center, 5, 5].item()
        
        # Force diffusion (normally runs every 10 cycles)
        brain.field_evolution_cycles = 9  # Next evolution will trigger diffusion
        brain._evolve_unified_field()
        
        # Check diffusion effect
        final_center = brain.unified_field[center, center, center, 5, 5].item()
        final_neighbor = brain.unified_field[center+1, center, center, 5, 5].item()
        
        # Center should decrease, neighbors should increase
        assert final_center < initial_center
        assert final_neighbor > initial_neighbor
        
        print(f"Diffusion - Center: {initial_center:.3f} -> {final_center:.3f}")
        print(f"Diffusion - Neighbor: {initial_neighbor:.3f} -> {final_neighbor:.3f}")
    
    def test_topology_region_discovery(self, brain):
        """Test that stable topology regions are discovered."""
        # Create stable high-activation region
        brain.unified_field[4:7, 4:7, 4:7, :, :] = 0.5
        
        # Process an experience to trigger topology discovery
        test_input = [0.5] * 24
        experience = brain._robot_sensors_to_field_experience(test_input)
        brain._apply_field_experience(experience)
        
        # Should discover topology regions
        assert len(brain.topology_regions) > 0
        
        # Check region properties
        for region_key, region_info in brain.topology_regions.items():
            assert 'activation' in region_info
            assert 'stability' in region_info
            assert region_info['activation'] > 0
            
        print(f"Discovered {len(brain.topology_regions)} topology regions")
    
    def test_gradient_flow_calculation(self, brain):
        """Test gradient flow calculation."""
        # Create gradient pattern
        for i in range(brain.spatial_resolution):
            brain.unified_field[i, :, :, 5, 5] = i * 0.1
        
        # Calculate gradients
        brain._calculate_gradient_flows()
        
        # Should have gradient flows
        assert 'gradient_x' in brain.gradient_flows
        assert 'gradient_y' in brain.gradient_flows
        assert 'gradient_z' in brain.gradient_flows
        
        # X gradient should be strongest
        grad_x_max = torch.max(torch.abs(brain.gradient_flows['gradient_x'])).item()
        grad_y_max = torch.max(torch.abs(brain.gradient_flows['gradient_y'])).item()
        grad_z_max = torch.max(torch.abs(brain.gradient_flows['gradient_z'])).item()
        
        assert grad_x_max > grad_y_max
        assert grad_x_max > grad_z_max
        
        print(f"Gradient magnitudes - X: {grad_x_max:.4f}, Y: {grad_y_max:.4f}, Z: {grad_z_max:.4f}")
    
    def test_constraint_guided_evolution(self, brain):
        """Test constraint-guided field evolution."""
        # The brain uses ConstraintField4D internally
        # Set up field with constraints
        brain.unified_field[5, 5, 5, :, :] = 1.0
        
        # Multiple evolution cycles
        initial_energy = torch.sum(torch.abs(brain.unified_field)).item()
        
        for _ in range(20):
            brain._evolve_unified_field()
        
        final_energy = torch.sum(torch.abs(brain.unified_field)).item()
        
        # Energy should decrease but be constrained
        assert final_energy < initial_energy
        assert final_energy > 0  # Should not decay to zero
        
        print(f"Constraint evolution: {initial_energy:.3f} -> {final_energy:.3f}")
    
    def test_field_memory_influence(self, brain):
        """Test how field topology influences future evolution."""
        # Create strong pattern
        pattern_region = brain.unified_field[3:6, 3:6, 3:6, 5, 5]
        pattern_region.fill_(2.0)
        
        # Evolve field
        for _ in range(5):
            brain._evolve_unified_field()
        
        # Pattern should persist (though decayed)
        pattern_strength = torch.mean(brain.unified_field[3:6, 3:6, 3:6, 5, 5]).item()
        background_strength = torch.mean(brain.unified_field[7:9, 7:9, 7:9, 5, 5]).item()
        
        # Pattern region should still be stronger than background
        assert pattern_strength > background_strength * 2
        
        print(f"Pattern persistence - Pattern: {pattern_strength:.4f}, Background: {background_strength:.4f}")
    
    def test_field_evolution_performance(self, brain):
        """Test field evolution performance."""
        # Time field evolution
        start_time = time.perf_counter()
        
        for _ in range(100):
            brain._evolve_unified_field()
        
        elapsed = time.perf_counter() - start_time
        avg_time_ms = (elapsed / 100) * 1000
        
        # Should be reasonably fast
        assert avg_time_ms < 10  # Less than 10ms per evolution
        
        print(f"Field evolution performance: {avg_time_ms:.2f}ms per cycle")
    
    def test_experience_novelty_calculation(self, brain):
        """Test experience novelty calculation."""
        # First experience should be maximally novel
        coords1 = torch.randn(brain.total_dimensions)
        novelty1 = brain._calculate_experience_novelty(coords1)
        assert novelty1 == 1.0
        
        # Add experience
        brain.field_experiences.append(
            brain._robot_sensors_to_field_experience([0.5] * 24)
        )
        
        # Similar experience should have low novelty
        coords2 = brain.field_experiences[-1].field_coordinates
        novelty2 = brain._calculate_experience_novelty(coords2)
        assert novelty2 < 0.5
        
        # Very different experience should have high novelty
        coords3 = torch.randn(brain.total_dimensions) * 2
        novelty3 = brain._calculate_experience_novelty(coords3)
        assert novelty3 > 0.5
        
        print(f"Novelty scores - First: {novelty1:.3f}, Similar: {novelty2:.3f}, Different: {novelty3:.3f}")
    
    def test_field_state_statistics(self, brain):
        """Test field state statistics calculation."""
        # Create some field activity
        brain.unified_field[4:6, 4:6, 4:6, :, :] = 0.5
        brain._evolve_unified_field()
        
        # Get statistics
        stats = brain._get_field_brain_state(0.0)
        
        # Check required fields
        assert 'field_dimensions' in stats
        assert 'field_max_activation' in stats
        assert 'field_mean_activation' in stats
        assert 'field_total_energy' in stats
        
        # Values should be reasonable
        assert stats['field_dimensions'] == brain.total_dimensions
        assert stats['field_max_activation'] > 0
        assert stats['field_mean_activation'] > 0
        assert stats['field_total_energy'] > 0
        
        print(f"Field stats - Max: {stats['field_max_activation']:.4f}, "
              f"Mean: {stats['field_mean_activation']:.6f}, "
              f"Energy: {stats['field_total_energy']:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])