#!/usr/bin/env python3
"""
Test Emergent Spatial Navigation

Demonstrates navigation without coordinates - places emerge from field dynamics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server'))

import unittest
import torch
import numpy as np
from src.brains.field.emergent_spatial_dynamics import EmergentSpatialDynamics
from src.brains.field.emergent_robot_interface import EmergentRobotInterface
from src.brains.field.field_types import FieldDimension, FieldDynamicsFamily, FieldNativeAction


class TestEmergentSpatialNavigation(unittest.TestCase):
    """Test emergent spatial navigation without coordinates."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cpu')
        self.field_shape = (4, 4, 4, 3, 3, 2, 2)  # Simple field shape
        
        # Create field dimensions with proper initialization
        self.field_dimensions = [
            FieldDimension("space_pattern_1", FieldDynamicsFamily.SPATIAL, 0),
            FieldDimension("space_pattern_2", FieldDynamicsFamily.SPATIAL, 1),
            FieldDimension("space_pattern_3", FieldDynamicsFamily.SPATIAL, 2),
            FieldDimension("oscillation", FieldDynamicsFamily.OSCILLATORY, 3),
            FieldDimension("flow", FieldDynamicsFamily.FLOW, 4),
            FieldDimension("topology", FieldDynamicsFamily.TOPOLOGY, 5),
            FieldDimension("energy", FieldDynamicsFamily.ENERGY, 6),
        ]
        
        # Initialize systems
        self.spatial_dynamics = EmergentSpatialDynamics(
            field_shape=self.field_shape,
            device=self.device,
            quiet_mode=False  # Enable debug output
        )
        
        self.robot_interface = EmergentRobotInterface(
            sensory_dim=10,
            motor_dim=4,
            field_dimensions=self.field_dimensions,
            device=self.device,
            quiet_mode=True
        )
    
    def test_place_discovery(self):
        """Test that places emerge from stable field configurations."""
        # Create a stable field configuration
        stable_field = torch.randn(self.field_shape, device=self.device) * 0.3
        stable_field[1:3, 1:3, 1:3] = 0.8  # High activation region
        
        # Create associated sensory pattern as list (robot interface expects list)
        sensory_input = (torch.randn(10, device=self.device) * 0.5).tolist()
        sensory_input[3:6] = [0.9, 0.9, 0.9]  # Distinctive pattern
        
        # Process the experience
        spatial_state = self.spatial_dynamics.process_spatial_experience(
            current_field=stable_field,
            sensory_input=sensory_input,
            reward=0.5  # Positive experience
        )
        
        # Should discover a new place
        self.assertEqual(spatial_state['known_places'], 1)
        self.assertIsNotNone(spatial_state['current_place'])
        self.assertEqual(self.spatial_dynamics.places_discovered, 1)
    
    def test_place_recognition(self):
        """Test that similar field states are recognized as the same place."""
        # Create and learn a place
        field1 = torch.randn(self.field_shape, device=self.device) * 0.2
        field1[2, 2, 2] = 1.0  # Distinctive feature
        sensory1 = (torch.ones(10, device=self.device) * 0.5).tolist()
        
        state1 = self.spatial_dynamics.process_spatial_experience(
            current_field=field1,
            sensory_input=sensory1,
            reward=0.0
        )
        
        first_place = state1['current_place']
        
        # Create similar field (with noise)
        field2 = field1 + torch.randn_like(field1) * 0.1
        sensory2_tensor = torch.tensor(sensory1) + torch.randn(10) * 0.1
        sensory2 = sensory2_tensor.tolist()
        
        state2 = self.spatial_dynamics.process_spatial_experience(
            current_field=field2,
            sensory_input=sensory2,
            reward=0.0
        )
        
        # Should recognize as same place
        self.assertEqual(state2['current_place'], first_place)
        self.assertEqual(state2['known_places'], 1)  # No new place
    
    def test_transition_learning(self):
        """Test that transitions between places are learned."""
        # Create place A
        field_a = torch.randn(self.field_shape, device=self.device) * 0.2
        field_a[0, 0, 0] = 1.0
        sensory_a = [0.0] * 10
        sensory_a[0] = 1.0
        
        self.spatial_dynamics.process_spatial_experience(
            current_field=field_a,
            sensory_input=sensory_a,
            reward=0.0
        )
        
        # Create place B  
        field_b = torch.randn(self.field_shape, device=self.device) * 0.2
        field_b[3, 3, 3] = 1.0
        sensory_b = [0.0] * 10
        sensory_b[9] = 1.0
        
        self.spatial_dynamics.process_spatial_experience(
            current_field=field_b,
            sensory_input=sensory_b,
            reward=0.0
        )
        
        # Check that transition was learned
        nav_graph = self.spatial_dynamics.get_navigation_graph()
        self.assertIn('place_1', nav_graph)
        self.assertGreater(len(nav_graph['place_1']), 0)
        
        # The connection from place_1 to place_2 should exist
        connections = dict(nav_graph['place_1'])
        self.assertIn('place_2', connections)
    
    def test_motor_emergence(self):
        """Test that motor commands emerge from field evolution."""
        # Create current field
        current_field = torch.randn(self.field_shape, device=self.device) * 0.3
        
        # Simulate field evolution with specific patterns
        field_evolution = torch.zeros_like(current_field)
        
        # Add some random evolution to test motor emergence
        field_evolution += torch.randn_like(field_evolution) * 0.2
        
        # Generate motor commands
        action = self.spatial_dynamics.compute_motor_emergence(
            current_field=current_field,
            field_evolution=field_evolution
        )
        
        # Check that motor commands were generated
        self.assertEqual(len(action.output_stream), 4)
        self.assertIsInstance(action.output_stream, torch.Tensor)
        
        # Should have non-zero motor values
        self.assertGreater(torch.sum(torch.abs(action.output_stream)).item(), 0.0)
    
    def test_navigation_to_place(self):
        """Test navigation to a known place."""
        # Create and learn two places
        # Place A
        field_a = torch.randn(self.field_shape, device=self.device) * 0.2
        field_a[1, 1, 1] = 1.0
        sensory_a = [0.0] * 10
        sensory_a[1] = 1.0
        
        self.spatial_dynamics.process_spatial_experience(
            current_field=field_a,
            sensory_input=sensory_a,
            reward=1.0  # Important place
        )
        
        # Place B
        field_b = torch.randn(self.field_shape, device=self.device) * 0.2
        field_b[2, 2, 2] = 1.0
        sensory_b = [0.0] * 10
        sensory_b[5] = 1.0
        
        self.spatial_dynamics.process_spatial_experience(
            current_field=field_b,
            sensory_input=sensory_b,
            reward=0.0
        )
        
        # Start navigation from B to A
        success = self.spatial_dynamics.navigate_to_place('place_1')
        self.assertTrue(success)
        
        # Generate motor commands with navigation active
        field_evolution = field_a - field_b  # Evolution toward target
        action = self.spatial_dynamics.compute_motor_emergence(
            current_field=field_b,
            field_evolution=field_evolution * 0.1
        )
        
        # Should have confidence due to navigation
        self.assertGreater(action.confidence, 0.0)
    
    def test_sensory_pattern_to_field_impression(self):
        """Test that sensory patterns create field impressions without coordinates."""
        # Create distinctive sensory pattern
        sensory_input = [0.1, 0.9, 0.1, 0.9, 0.1, 0.5, 0.5, 0.5, 0.8, 0.0]
        
        # Convert to field experience
        experience = self.robot_interface.sensory_pattern_to_field_experience(sensory_input)
        
        # Check that field coordinates are not spatial coordinates
        self.assertIsInstance(experience.field_coordinates, torch.Tensor)
        self.assertEqual(len(experience.field_coordinates), len(self.field_dimensions))
        
        # Field impression should reflect pattern features
        self.assertGreater(experience.field_intensity, 0.4)  # Pattern has variation
        
        # Test that similar patterns create similar impressions
        similar_input = [0.1, 0.8, 0.2, 0.9, 0.1, 0.5, 0.6, 0.4, 0.8, 0.0]
        similar_experience = self.robot_interface.sensory_pattern_to_field_experience(similar_input)
        
        # Calculate similarity of field impressions
        similarity = torch.nn.functional.cosine_similarity(
            experience.field_coordinates.unsqueeze(0),
            similar_experience.field_coordinates.unsqueeze(0)
        ).item()
        
        self.assertGreater(similarity, 0.7)  # Similar patterns → similar impressions
    
    def test_pattern_diversity_tracking(self):
        """Test that the system tracks pattern diversity."""
        # Feed various patterns
        patterns = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Unique
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Unique
            [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Similar to first
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0],  # Unique
        ]
        
        for pattern in patterns:
            self.robot_interface.sensory_pattern_to_field_experience(pattern)
        
        stats = self.robot_interface.get_statistics()
        self.assertEqual(stats['unique_patterns'], 3)
        self.assertEqual(stats['pattern_matches'], 1)
        self.assertGreater(stats['pattern_diversity'], 0.7)


if __name__ == '__main__':
    unittest.main()