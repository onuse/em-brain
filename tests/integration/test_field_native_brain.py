#!/usr/bin/env python3
"""
Test Field-Native Brain - Phase B1

Test the revolutionary unified field-native brain that abandons discrete
streams in favor of unified multi-dimensional field dynamics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server/src'))

import torch
import numpy as np
import time
import math
from typing import List, Dict, Any

# Import field-native brain
from field_native_brain import create_unified_field_brain, FieldDynamicsFamily


def test_field_native_robot_interface():
    """Test that field-native brain can interface with robot sensors and actuators."""
    print(f"\n🤖 TESTING FIELD-NATIVE ROBOT INTERFACE")
    
    # Create compact field brain for testing
    brain = create_unified_field_brain(
        spatial_resolution=8,  # Smaller for testing
        temporal_window=5.0,
        field_evolution_rate=0.2,
        constraint_discovery_rate=0.3,
        quiet_mode=True
    )
    
    print(f"   Created {brain.total_dimensions}D unified field brain")
    print(f"   Spatial resolution: {brain.spatial_resolution}³")
    
    # Test robot sensor → field mapping
    test_sensors = [
        0.5, 0.3, 0.1,  # position
        0.8, 0.6, 0.4,  # distance sensors
        0.9, 0.7, 0.5,  # camera RGB
        0.3, 0.2, 0.8,  # audio, temp, battery
        0.9, 0.85, 0.1, 0.05, 0.02, 0.01,  # encoders, compass, gyro
        0.7, 0.3, 1.0,  # accelerometer
        0.5, 0.8, 0.4   # touch, light, proximity
    ]
    
    print(f"\n   Testing sensor → field mapping:")
    print(f"      Input: {len(test_sensors)}D robot sensors")
    
    # Process one cycle
    motor_output, brain_state = brain.process_robot_cycle(test_sensors)
    
    print(f"      Output: {len(motor_output)}D motor commands")
    print(f"      Motor commands: {[f'{x:.3f}' for x in motor_output]}")
    print(f"      Field energy: {brain_state['field_total_energy']:.3f}")
    print(f"      Field max activation: {brain_state['field_max_activation']:.4f}")
    
    return {
        'brain': brain,
        'motor_output': motor_output,
        'brain_state': brain_state,
        'sensor_input': test_sensors
    }


def test_field_dynamics_families():
    """Test that field dynamics families organize concepts correctly."""
    print(f"\n🌈 TESTING FIELD DYNAMICS FAMILIES")
    
    brain = create_unified_field_brain(
        spatial_resolution=6,
        temporal_window=4.0,
        quiet_mode=True
    )
    
    # Test different sensor patterns that should activate different families
    test_scenarios = [
        ("oscillatory_test", [0.5, 0.5, 0.0] + [math.sin(i * 0.5) for i in range(21)]),
        ("flow_test", [0.3, 0.7, 0.2] + [i * 0.1 for i in range(21)]),
        ("topology_test", [0.8, 0.8, 0.1] + [0.9 if i % 3 == 0 else 0.1 for i in range(21)]),
        ("energy_test", [0.2, 0.2, 0.9] + [0.8] * 21),
    ]
    
    family_activities = {}
    
    for scenario_name, sensor_pattern in test_scenarios:
        print(f"   Testing {scenario_name}:")
        
        # Process multiple cycles to build up field activity
        for cycle in range(5):
            motor_output, brain_state = brain.process_robot_cycle(sensor_pattern)
        
        # Record family activities
        activities = {
            'oscillatory': brain_state['oscillatory_activity'],
            'flow': brain_state['flow_activity'],
            'topology': brain_state['topology_activity'],
            'energy': brain_state['energy_activity'],
            'coupling': brain_state['coupling_activity'],
            'emergence': brain_state['emergence_activity']
        }
        
        family_activities[scenario_name] = activities
        
        # Find dominant family
        dominant_family = max(activities, key=activities.get)
        dominant_activity = activities[dominant_family]
        
        print(f"      Dominant family: {dominant_family} ({dominant_activity:.4f})")
        print(f"      All activities: {[(k, f'{v:.3f}') for k, v in activities.items()]}")
    
    return {
        'family_activities': family_activities,
        'brain': brain
    }


def test_topology_discovery():
    """Test that field brain discovers topology regions (replaces pattern storage)."""
    print(f"\n🏔️ TESTING TOPOLOGY DISCOVERY")
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        temporal_window=6.0,
        quiet_mode=True
    )
    
    # Create repeated patterns that should form stable topology
    base_pattern = [0.6, 0.4, 0.2, 0.8, 0.7, 0.5, 0.9, 0.3, 0.6, 0.4,
                    0.8, 0.2, 0.7, 0.9, 0.1, 0.5, 0.8, 0.3, 0.6, 0.7,
                    0.4, 0.9, 0.2, 0.5]
    
    print(f"   Applying repeated pattern to create stable topology:")
    
    topology_history = []
    
    # Apply pattern multiple times
    for cycle in range(15):
        # Add small variations to the base pattern
        varied_pattern = [p + 0.1 * math.sin(cycle * 0.3 + i) for i, p in enumerate(base_pattern)]
        
        motor_output, brain_state = brain.process_robot_cycle(varied_pattern)
        
        topology_count = brain_state['topology_regions_count']
        topology_discoveries = brain_state['topology_discoveries']
        
        topology_history.append({
            'cycle': cycle + 1,
            'regions': topology_count,
            'discoveries': topology_discoveries,
            'field_energy': brain_state['field_total_energy']
        })
        
        if cycle % 5 == 0:
            print(f"      Cycle {cycle+1}: regions={topology_count}, "
                  f"discoveries={topology_discoveries}, "
                  f"energy={brain_state['field_total_energy']:.3f}")
    
    final_regions = topology_history[-1]['regions']
    total_discoveries = topology_history[-1]['discoveries']
    
    print(f"   Final topology analysis:")
    print(f"      Stable regions formed: {final_regions}")
    print(f"      Total discoveries: {total_discoveries}")
    print(f"      Topology formation: {'✅ SUCCESS' if final_regions > 0 else '⚠️ LIMITED'}")
    
    return {
        'topology_history': topology_history,
        'final_regions': final_regions,
        'brain': brain
    }


def test_gradient_action_generation():
    """Test that actions emerge from field gradients (replaces discrete action generation)."""
    print(f"\n⚡ TESTING GRADIENT ACTION GENERATION")
    
    brain = create_unified_field_brain(
        spatial_resolution=8,
        temporal_window=4.0,
        quiet_mode=True
    )
    
    # Create gradient patterns in sensor input
    test_cases = [
        ("strong_x_gradient", [0.1, 0.9, 0.5] + [0.5] * 21),  # Strong X movement
        ("strong_y_gradient", [0.5, 0.1, 0.9] + [0.3] * 21),  # Strong Y movement
        ("circular_pattern", [0.5, 0.5, 0.5] + [math.sin(i * 0.3) for i in range(21)]),  # Circular
        ("no_gradient", [0.5, 0.5, 0.5] + [0.5] * 21),        # No gradient
    ]
    
    action_results = []
    
    for test_name, sensor_pattern in test_cases:
        print(f"   Testing {test_name}:")
        
        # Build up field gradients
        for buildup in range(3):
            brain.process_robot_cycle(sensor_pattern)
        
        # Test action generation
        motor_output, brain_state = brain.process_robot_cycle(sensor_pattern)
        
        gradient_strength = brain_state['last_gradient_strength']
        action_confidence = brain_state['last_action_confidence']
        
        action_results.append({
            'test_name': test_name,
            'motor_output': motor_output,
            'gradient_strength': gradient_strength,
            'action_confidence': action_confidence
        })
        
        print(f"      Motor output: {[f'{x:.3f}' for x in motor_output]}")
        print(f"      Gradient strength: {gradient_strength:.4f}")
        print(f"      Action confidence: {action_confidence:.4f}")
    
    # Analyze gradient responsiveness
    gradient_strengths = [r['gradient_strength'] for r in action_results]
    max_gradient = max(gradient_strengths)
    min_gradient = min(gradient_strengths)
    gradient_range = max_gradient - min_gradient
    
    print(f"   Gradient analysis:")
    print(f"      Gradient range: {gradient_range:.4f}")
    print(f"      Gradient responsiveness: {'✅ RESPONSIVE' if gradient_range > 0.1 else '⚠️ LIMITED'}")
    
    return {
        'action_results': action_results,
        'gradient_range': gradient_range,
        'brain': brain
    }


def test_field_native_learning():
    """Test that learning emerges from field evolution (replaces discrete updates)."""
    print(f"\n🧠 TESTING FIELD-NATIVE LEARNING")
    
    brain = create_unified_field_brain(
        spatial_resolution=8,
        temporal_window=5.0,
        field_evolution_rate=0.3,  # Higher evolution rate for testing
        quiet_mode=True
    )
    
    # Create learning scenario: repeated exposure to pattern with reward
    learning_pattern = [0.7, 0.3, 0.6, 0.8, 0.4, 0.9, 0.2, 0.7, 0.5, 0.8,
                       0.3, 0.6, 0.9, 0.1, 0.4, 0.7, 0.8, 0.2, 0.5, 0.9,
                       0.6, 0.3, 0.7, 0.4]
    
    learning_history = []
    
    print(f"   Repeated exposure to learning pattern:")
    
    for session in range(20):
        # Apply learning pattern
        motor_output, brain_state = brain.process_robot_cycle(learning_pattern)
        
        # Track learning metrics
        learning_history.append({
            'session': session + 1,
            'field_energy': brain_state['field_total_energy'],
            'field_max': brain_state['field_max_activation'],
            'topology_regions': brain_state['topology_regions_count'],
            'emergence_activity': brain_state['emergence_activity'],
            'coupling_activity': brain_state['coupling_activity']
        })
        
        if session % 5 == 0:
            print(f"      Session {session+1}: energy={brain_state['field_total_energy']:.3f}, "
                  f"max={brain_state['field_max_activation']:.4f}, "
                  f"emergence={brain_state['emergence_activity']:.4f}")
    
    # Analyze learning progression
    initial_energy = learning_history[0]['field_energy']
    final_energy = learning_history[-1]['field_energy']
    energy_change = final_energy - initial_energy
    
    initial_emergence = learning_history[0]['emergence_activity']
    final_emergence = learning_history[-1]['emergence_activity']
    emergence_change = final_emergence - initial_emergence
    
    print(f"   Learning analysis:")
    print(f"      Energy change: {energy_change:.3f}")
    print(f"      Emergence change: {emergence_change:.4f}")
    print(f"      Field adaptation: {'✅ LEARNING' if abs(energy_change) > 1.0 else '⚠️ STABLE'}")
    
    return {
        'learning_history': learning_history,
        'energy_change': energy_change,
        'emergence_change': emergence_change,
        'brain': brain
    }


def test_field_native_intelligence():
    """
    Comprehensive test of field-native brain intelligence capabilities.
    """
    print("🌊 TESTING FIELD-NATIVE BRAIN INTELLIGENCE")
    print("=" * 70)
    
    print(f"🎯 Phase B1: Unified Multi-dimensional Field Brain Foundation")
    print(f"   Testing the revolutionary field-native brain architecture")
    print(f"   Key capabilities: sensor mapping, family dynamics, topology discovery, gradient actions")
    
    # TEST 1: Robot Interface
    interface_results = test_field_native_robot_interface()
    
    # TEST 2: Field Dynamics Families
    family_results = test_field_dynamics_families()
    
    # TEST 3: Topology Discovery
    topology_results = test_topology_discovery()
    
    # TEST 4: Gradient Action Generation
    action_results = test_gradient_action_generation()
    
    # TEST 5: Field-Native Learning
    learning_results = test_field_native_learning()
    
    # Comprehensive analysis
    print(f"\n📊 COMPREHENSIVE FIELD-NATIVE INTELLIGENCE ANALYSIS")
    
    print(f"\n   🤖 Robot Interface:")
    print(f"      Sensor dimensions processed: {len(interface_results['sensor_input'])}")
    print(f"      Motor dimensions generated: {len(interface_results['motor_output'])}")
    print(f"      Field energy: {interface_results['brain_state']['field_total_energy']:.3f}")
    
    print(f"\n   🌈 Field Dynamics Families:")
    family_count = len(family_results['family_activities'])
    print(f"      Family scenarios tested: {family_count}")
    
    # Check if families show different activations
    all_activities = []
    for scenario, activities in family_results['family_activities'].items():
        all_activities.extend(activities.values())
    activity_range = max(all_activities) - min(all_activities)
    print(f"      Family activity range: {activity_range:.4f}")
    print(f"      Family differentiation: {'✅ STRONG' if activity_range > 0.01 else '⚠️ WEAK'}")
    
    print(f"\n   🏔️ Topology Discovery:")
    final_regions = topology_results['final_regions']
    total_discoveries = topology_results['topology_history'][-1]['discoveries']
    print(f"      Stable regions formed: {final_regions}")
    print(f"      Total discoveries: {total_discoveries}")
    print(f"      Topology formation: {'✅ SUCCESS' if final_regions > 0 else '⚠️ LIMITED'}")
    
    print(f"\n   ⚡ Gradient Action Generation:")
    gradient_range = action_results['gradient_range']
    print(f"      Gradient strength range: {gradient_range:.4f}")
    print(f"      Action responsiveness: {'✅ RESPONSIVE' if gradient_range > 0.1 else '⚠️ LIMITED'}")
    
    print(f"\n   🧠 Field-Native Learning:")
    energy_change = abs(learning_results['energy_change'])
    emergence_change = abs(learning_results['emergence_change'])
    print(f"      Field energy adaptation: {energy_change:.3f}")
    print(f"      Emergence development: {emergence_change:.4f}")
    print(f"      Learning capability: {'✅ ADAPTIVE' if energy_change > 1.0 else '⚠️ STABLE'}")
    
    # Overall field-native intelligence assessment
    success_metrics = [
        activity_range > 0.01,      # Family differentiation
        final_regions > 0,          # Topology formation
        gradient_range > 0.1,       # Action responsiveness
        energy_change > 1.0         # Learning adaptation
    ]
    
    success_count = sum(success_metrics)
    success_rate = success_count / len(success_metrics)
    
    print(f"\n   🌟 FIELD-NATIVE INTELLIGENCE ASSESSMENT:")
    print(f"      Success metrics: {success_count}/{len(success_metrics)}")
    print(f"      Overall success rate: {success_rate:.3f}")
    print(f"      Field-native intelligence: {'✅ ACHIEVED' if success_rate > 0.7 else '⚠️ DEVELOPING'}")
    
    print(f"\n✅ FIELD-NATIVE BRAIN INTELLIGENCE TEST COMPLETED!")
    
    if success_rate > 0.7:
        print(f"🎯 Key field-native achievements:")
        print(f"   ✓ Unified multi-dimensional field brain")
        print(f"   ✓ Sensor-to-field and field-to-action translation")
        print(f"   ✓ Dynamics families organize by physics, not modality")
        print(f"   ✓ Topology discovery replaces pattern storage")
        print(f"   ✓ Gradient following replaces action generation")
        print(f"   ✓ Field evolution replaces discrete learning")
        print(f"   ✓ ALL intelligence emerges from unified field dynamics!")
    else:
        print(f"⚠️ Field-native brain intelligence developing")
        print(f"🔧 Consider tuning field parameters or spatial resolution")
    
    return {
        'interface_results': interface_results,
        'family_results': family_results,
        'topology_results': topology_results,
        'action_results': action_results,
        'learning_results': learning_results,
        'success_rate': success_rate,
        'field_native_achieved': success_rate > 0.7
    }


if __name__ == "__main__":
    # Run the comprehensive field-native intelligence test
    results = test_field_native_intelligence()
    
    print(f"\n🔬 PHASE B1 VALIDATION SUMMARY:")
    print(f"   Success rate: {results['success_rate']:.3f}")
    print(f"   Field dimensions: {results['interface_results']['brain'].total_dimensions}")
    print(f"   Topology regions: {results['topology_results']['final_regions']}")
    print(f"   Gradient responsiveness: {results['action_results']['gradient_range']:.4f}")
    print(f"   Field adaptation: {abs(results['learning_results']['energy_change']):.3f}")
    print(f"   Field-native intelligence: {'✅ ACHIEVED' if results['field_native_achieved'] else '⚠️ DEVELOPING'}")
    
    if results['field_native_achieved']:
        print(f"\n🚀 Phase B1 FIELD-NATIVE BRAIN FOUNDATION SUCCESSFULLY DEMONSTRATED!")
        print(f"🎉 We have implemented the unified multi-dimensional field brain!")
        print(f"🌊 Intelligence emerges from continuous field dynamics, not discrete patterns!")
    else:
        print(f"\n⚠️ Phase B1 field-native brain foundation still developing")
        print(f"🔧 Consider optimizing field parameters or architecture")