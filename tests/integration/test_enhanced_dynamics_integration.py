#!/usr/bin/env python3
"""
Test Enhanced Dynamics Integration

Tests that enhanced dynamics (phase transitions, attractors, energy redistribution)
are properly integrated into the main brain.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'server'))

import unittest
import torch
import numpy as np
from src.core.dynamic_brain_factory import DynamicBrainFactory


class TestEnhancedDynamicsIntegration(unittest.TestCase):
    """Test enhanced dynamics features in the main brain."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a brain with enhanced dynamics enabled
        self.factory = DynamicBrainFactory({
            'use_dynamic_brain': True,
            'use_full_features': True,
            'enhanced_dynamics': True,  # Explicitly enable
            'quiet_mode': True
        })
        brain_wrapper = self.factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=24,
            motor_dim=4
        )
        self.brain = brain_wrapper.brain
    
    def test_enhanced_dynamics_initialized(self):
        """Test that enhanced dynamics is properly initialized."""
        # Check that enhanced dynamics is enabled
        self.assertTrue(hasattr(self.brain, 'enhanced_dynamics_enabled'))
        self.assertTrue(self.brain.enhanced_dynamics_enabled)
        
        # Check that enhanced dynamics object exists
        self.assertTrue(hasattr(self.brain, 'enhanced_dynamics'))
        self.assertIsNotNone(self.brain.enhanced_dynamics)
        
        # Check that field adapter exists
        self.assertTrue(hasattr(self.brain, 'field_adapter'))
        self.assertIsNotNone(self.brain.field_adapter)
    
    def test_phase_transitions(self):
        """Test that phase transitions can occur."""
        # Get initial phase
        initial_phase = self.brain.enhanced_dynamics.current_phase
        self.assertEqual(initial_phase, "stable")
        
        # Create high energy condition by feeding strong sensory input
        for i in range(10):
            sensory_input = np.random.rand(24) * 0.9 + 0.1  # High values
            action, brain_state = self.brain.process_robot_cycle(sensory_input.tolist())
        
        # Check phase energy history is being tracked
        self.assertGreater(len(self.brain.enhanced_dynamics.phase_energy_history), 5)
        
        # Phase might have changed based on field energy
        current_phase = self.brain.enhanced_dynamics.current_phase
        print(f"Phase after high energy input: {current_phase}")
    
    def test_attractor_creation(self):
        """Test that attractors can be manually created."""
        initial_attractors = len(self.brain.enhanced_dynamics.active_attractors)
        
        # Create a manual attractor
        coordinates = torch.randn(36, device=self.brain.device) * 0.3
        self.brain.enhanced_dynamics.add_manual_attractor(
            coordinates=coordinates,
            intensity=0.5,
            persistence=5.0
        )
        
        # Check attractor was added
        self.assertEqual(len(self.brain.enhanced_dynamics.active_attractors), 
                        initial_attractors + 1)
        
        # Check attractor properties
        last_attractor = self.brain.enhanced_dynamics.active_attractors[-1]
        self.assertEqual(last_attractor['type'], 'manual')
        self.assertEqual(last_attractor['intensity'], 0.5)
        self.assertEqual(last_attractor['persistence'], 5.0)
    
    def test_energy_metrics_tracking(self):
        """Test that energy metrics are being tracked."""
        # Process several cycles to generate energy metrics
        for i in range(5):
            sensory_input = np.random.rand(24) * 0.5
            action, brain_state = self.brain.process_robot_cycle(sensory_input.tolist())
        
        # Check energy metrics
        self.assertGreater(self.brain.enhanced_dynamics.global_energy_level, 0)
        self.assertIsInstance(self.brain.enhanced_dynamics.energy_flow_directions, dict)
        
        # Check field statistics through adapter
        stats = self.brain.field_adapter.get_field_statistics()
        self.assertIn('total_activation', stats)
        self.assertIn('mean_activation', stats)
        self.assertIn('field_energy', stats)
        self.assertGreater(stats['total_activation'], 0)
    
    def test_field_evolution_with_enhanced_dynamics(self):
        """Test that field evolution includes enhanced dynamics."""
        # Get initial field state
        initial_field = self.brain.unified_field.clone()
        
        # Process multiple cycles
        for i in range(10):
            sensory_input = np.random.rand(24) * 0.3
            action, brain_state = self.brain.process_robot_cycle(sensory_input.tolist())
        
        # Field should have evolved
        field_difference = torch.sum(torch.abs(self.brain.unified_field - initial_field)).item()
        self.assertGreater(field_difference, 0.01)
        
        # Enhanced dynamics should have contributed
        self.assertGreater(self.brain.enhanced_dynamics.global_energy_level, 0)
    
    def test_enhanced_dynamics_configuration(self):
        """Test that enhanced dynamics uses proper configuration."""
        # Check phase config
        phase_config = self.brain.enhanced_dynamics.phase_config
        self.assertAlmostEqual(phase_config.energy_threshold, 0.7)
        self.assertAlmostEqual(phase_config.stability_threshold, 0.3)
        
        # Check attractor config
        attractor_config = self.brain.enhanced_dynamics.attractor_config
        self.assertTrue(attractor_config.auto_discovery)
        self.assertGreater(attractor_config.temporal_persistence, 0)
    
    def test_maintenance_with_enhanced_dynamics(self):
        """Test that maintenance includes enhanced dynamics operations."""
        # Process cycles to generate some activity
        for i in range(20):
            sensory_input = np.random.rand(24) * 0.4
            action, brain_state = self.brain.process_robot_cycle(sensory_input.tolist())
        
        # Trigger maintenance
        if hasattr(self.brain, '_perform_maintenance'):
            maintenance_results = self.brain._perform_maintenance()
            print(f"Maintenance results: {maintenance_results}")
        
        # Check if energy redistribution was considered
        if hasattr(self.brain.enhanced_dynamics, '_redistribution_pending'):
            print(f"Redistributions pending: {len(self.brain.enhanced_dynamics._redistribution_pending)}")


if __name__ == '__main__':
    unittest.main()