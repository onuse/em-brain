#!/usr/bin/env python3
"""
Emergent Navigation Demo

Demonstrates spatial navigation without coordinates - places and navigation
emerge from field dynamics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'server'))

import torch
import numpy as np
import time
from src.brains.field.emergent_spatial_dynamics import EmergentSpatialDynamics
from src.brains.field.emergent_robot_interface import EmergentRobotInterface
from src.brains.field.field_types import FieldDimension, FieldDynamicsFamily


def create_field_dimensions():
    """Create field dimensions for the demo."""
    return [
        FieldDimension("space_pattern_1", FieldDynamicsFamily.SPATIAL, 0),
        FieldDimension("space_pattern_2", FieldDynamicsFamily.SPATIAL, 1),
        FieldDimension("space_pattern_3", FieldDynamicsFamily.SPATIAL, 2),
        FieldDimension("oscillation", FieldDynamicsFamily.OSCILLATORY, 3),
        FieldDimension("flow", FieldDynamicsFamily.FLOW, 4),
        FieldDimension("topology", FieldDynamicsFamily.TOPOLOGY, 5),
        FieldDimension("energy", FieldDynamicsFamily.ENERGY, 6),
    ]


def simulate_robot_exploration():
    """Simulate a robot exploring and learning places."""
    print("\n" + "="*60)
    print("EMERGENT SPATIAL NAVIGATION DEMO")
    print("="*60)
    print("\nThis demo shows how spatial understanding emerges from")
    print("field dynamics without any coordinate system.")
    print("\n" + "="*60)
    
    # Initialize systems
    device = torch.device('cpu')
    field_shape = (4, 4, 4, 3, 3, 2, 2)
    field_dimensions = create_field_dimensions()
    
    spatial_dynamics = EmergentSpatialDynamics(
        field_shape=field_shape,
        device=device,
        quiet_mode=False
    )
    
    robot_interface = EmergentRobotInterface(
        sensory_dim=10,
        motor_dim=4,
        field_dimensions=field_dimensions,
        device=device,
        quiet_mode=False
    )
    
    print("\n📍 Phase 1: Discovering Places Through Experience")
    print("-" * 50)
    
    # Define some distinctive sensory patterns (simulating different locations)
    places_sensory_patterns = [
        {
            'name': 'Home Base',
            'pattern': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # High reward
            'field_signature': lambda: torch.randn(field_shape) * 0.2 + torch.tensor([[[[[[[1.0]]]]]]])
        },
        {
            'name': 'Food Source',
            'pattern': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.8],  # Good reward
            'field_signature': lambda: torch.randn(field_shape) * 0.2 + torch.tensor([[[[[[[0.0]]]]]]])
        },
        {
            'name': 'Water Source',
            'pattern': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.7],  # Moderate reward
            'field_signature': lambda: torch.randn(field_shape) * 0.3
        },
        {
            'name': 'Danger Zone',
            'pattern': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5],  # Negative reward
            'field_signature': lambda: torch.randn(field_shape) * 0.4
        }
    ]
    
    # Robot discovers places
    discovered_places = {}
    for i, place_info in enumerate(places_sensory_patterns):
        print(f"\n🤖 Robot encounters: {place_info['name']}")
        
        # Create unique field state for this place
        field_state = place_info['field_signature']()
        
        # Convert sensory pattern to field experience
        experience = robot_interface.sensory_pattern_to_field_experience(place_info['pattern'])
        
        # Process spatial experience
        spatial_state = spatial_dynamics.process_spatial_experience(
            current_field=field_state,
            sensory_input=place_info['pattern'],
            reward=place_info['pattern'][-1]  # Last element is reward
        )
        
        if spatial_state['current_place']:
            discovered_places[place_info['name']] = spatial_state['current_place']
            print(f"   ✓ Learned as: {spatial_state['current_place']}")
        
        # Simulate some movement between places
        if i < len(places_sensory_patterns) - 1:
            time.sleep(0.1)  # Brief pause
    
    print(f"\n📊 Total places discovered: {spatial_state['known_places']}")
    
    print("\n\n🔄 Phase 2: Recognition and Navigation")
    print("-" * 50)
    
    # Test place recognition
    print("\n🧪 Testing place recognition...")
    
    # Return to home base with slight variations
    home_pattern_varied = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9]
    home_field_varied = places_sensory_patterns[0]['field_signature']() + torch.randn(field_shape) * 0.1
    
    spatial_state = spatial_dynamics.process_spatial_experience(
        current_field=home_field_varied,
        sensory_input=home_pattern_varied,
        reward=0.9
    )
    
    print(f"   Robot thinks it's at: {spatial_state['current_place']}")
    print(f"   Should be: {discovered_places['Home Base']}")
    print(f"   ✓ Recognition: {'SUCCESS' if spatial_state['current_place'] == discovered_places['Home Base'] else 'FAILED'}")
    
    # Test navigation
    print("\n\n🧭 Phase 3: Emergent Navigation")
    print("-" * 50)
    
    # Navigate from current location to food source
    target_place = discovered_places.get('Food Source')
    if target_place:
        print(f"\n🎯 Initiating navigation to: {target_place} (Food Source)")
        success = spatial_dynamics.navigate_to_place(target_place)
        
        if success:
            # Simulate field evolution toward target
            current_field = home_field_varied
            target_field = places_sensory_patterns[1]['field_signature']()
            
            # Generate motor commands from field dynamics
            for step in range(3):
                # Simulate field evolution
                field_evolution = (target_field - current_field) * 0.1
                
                # Get motor commands
                action = spatial_dynamics.compute_motor_emergence(
                    current_field=current_field,
                    field_evolution=field_evolution
                )
                
                print(f"\n   Step {step + 1}:")
                print(f"   Motor output: [{action.output_stream[0]:.2f}, {action.output_stream[1]:.2f}, "
                      f"{action.output_stream[2]:.2f}, {action.output_stream[3]:.2f}]")
                print(f"   Confidence: {action.confidence:.2f}")
                
                # Update field toward target
                current_field += field_evolution
    
    # Show navigation graph
    print("\n\n📈 Phase 4: Learned Spatial Relationships")
    print("-" * 50)
    
    nav_graph = spatial_dynamics.get_navigation_graph()
    print("\nLearned connections between places:")
    for place, connections in nav_graph.items():
        if connections:
            print(f"\n{place}:")
            for target, strength in connections:
                print(f"  → {target} (strength: {strength:.2f})")
    
    # Show statistics
    print("\n\n📊 Final Statistics")
    print("-" * 50)
    
    stats = spatial_dynamics.get_statistics()
    print(f"Places discovered: {stats['places_discovered']}")
    print(f"Current location: {stats['current_place']}")
    print(f"Navigation success rate: {stats['navigation_success_rate']:.1%}")
    
    interface_stats = robot_interface.get_statistics()
    print(f"Unique sensory patterns: {interface_stats['unique_patterns']}")
    print(f"Pattern diversity: {interface_stats['pattern_diversity']:.1%}")
    
    print("\n\n✨ Key Insights")
    print("-" * 50)
    print("• Places emerged from stable field configurations, not coordinates")
    print("• Navigation happened through field tension, not gradient following")
    print("• Sensory patterns created field impressions without spatial mapping")
    print("• Motor commands emerged from field dynamics patterns")
    print("• The robot learned spatial relationships through experience")
    
    print("\n" + "="*60)
    print("Demo complete! The robot has learned to navigate without")
    print("any coordinate system - just field dynamics and experience.")
    print("="*60 + "\n")


if __name__ == "__main__":
    simulate_robot_exploration()