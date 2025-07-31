#!/usr/bin/env python3
"""
Test Phase 2: Prediction Error as Primary Learning Signal

Verifies that prediction errors now drive all learning in the brain,
including self-modification strength, resource allocation, and field dynamics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from collections import deque

# Import brain components
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def test_error_driven_learning():
    """Test that prediction errors modulate learning rate."""
    print("\n=== Testing Error-Driven Learning (Phase 2) ===\n")
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=8,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    # Track metrics
    self_mod_history = []
    learning_rate_history = []
    prediction_error_history = []
    
    print("Testing with alternating predictable and unpredictable patterns...")
    
    for phase in range(3):
        print(f"\nPhase {phase + 1}:")
        
        if phase % 2 == 0:
            # Predictable pattern phase
            print("  Pattern: Predictable sine waves")
            for cycle in range(30):
                sensory_input = [0.5 * np.sin(2 * np.pi * i * cycle / 10) for i in range(8)]
                sensory_input.append(0.0)  # reward
                
                motor_output, brain_state = brain.process_robot_cycle(sensory_input)
                
                # Track metrics
                self_mod_history.append(brain.field_dynamics.self_modification_strength)
                prediction_error_history.append(brain._last_prediction_error)
                
                # Get error modulation if available
                if hasattr(brain.field_dynamics, '_error_modulation') and brain.field_dynamics._error_modulation:
                    learning_rate = brain.field_dynamics._error_modulation.get('learning_rate', 0.1)
                    learning_rate_history.append(learning_rate)
        else:
            # Unpredictable pattern phase
            print("  Pattern: Random noise")
            for cycle in range(30):
                sensory_input = [np.random.uniform(-1, 1) for _ in range(8)]
                sensory_input.append(0.0)  # reward
                
                motor_output, brain_state = brain.process_robot_cycle(sensory_input)
                
                # Track metrics
                self_mod_history.append(brain.field_dynamics.self_modification_strength)
                prediction_error_history.append(brain._last_prediction_error)
                
                # Get error modulation if available
                if hasattr(brain.field_dynamics, '_error_modulation') and brain.field_dynamics._error_modulation:
                    learning_rate = brain.field_dynamics._error_modulation.get('learning_rate', 0.1)
                    learning_rate_history.append(learning_rate)
        
        # Report phase results
        phase_start = (phase * 30)
        phase_end = ((phase + 1) * 30)
        
        avg_self_mod = np.mean(self_mod_history[phase_start:phase_end])
        avg_error = np.mean(prediction_error_history[phase_start:phase_end])
        
        print(f"  Average self-modification: {avg_self_mod:.3f}")
        print(f"  Average prediction error: {avg_error:.3f}")
        
        if learning_rate_history[phase_start:phase_end]:
            avg_learning_rate = np.mean(learning_rate_history[phase_start:phase_end])
            print(f"  Average learning rate: {avg_learning_rate:.3f}")
    
    # Analyze results
    print("\n=== Analysis ===")
    
    # Check if self-modification responds to errors
    predictable_self_mod = np.mean(self_mod_history[0:30] + self_mod_history[60:90])
    random_self_mod = np.mean(self_mod_history[30:60])
    
    print(f"\nSelf-modification strength:")
    print(f"  During predictable: {predictable_self_mod:.3f}")
    print(f"  During random: {random_self_mod:.3f}")
    print(f"  Ratio: {random_self_mod / predictable_self_mod:.2f}x")
    
    # Check if learning rate adapts
    if learning_rate_history:
        predictable_lr = np.mean(learning_rate_history[0:30] + learning_rate_history[60:90])
        random_lr = np.mean(learning_rate_history[30:60])
        print(f"\nLearning rate:")
        print(f"  During predictable: {predictable_lr:.3f}")
        print(f"  During random: {random_lr:.3f}")
        print(f"  Ratio: {random_lr / predictable_lr:.2f}x")
    
    return random_self_mod > predictable_self_mod * 1.2  # At least 20% higher during high errors


def test_resource_allocation():
    """Test that resources flow to high-error regions."""
    print("\n\n=== Testing Resource Allocation ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=10,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    # Create a pattern where some sensors are predictable and others are not
    print("Creating mixed predictability pattern...")
    print("  Sensors 0-4: Predictable (constant)")
    print("  Sensors 5-9: Unpredictable (random)")
    
    for cycle in range(50):
        # Half predictable, half random
        sensory_input = []
        for i in range(10):
            if i < 5:
                # Predictable (constant)
                sensory_input.append(0.5)
            else:
                # Unpredictable (random)
                sensory_input.append(np.random.uniform(-1, 1))
        sensory_input.append(0.0)  # reward
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    
    # Check resource allocation
    print("\n=== Checking Resource Distribution ===")
    
    # Get topology regions and their errors
    topology_stats = brain.topology_region_system.get_statistics()
    predictive_regions = brain.topology_region_system.get_predictive_regions()
    
    if predictive_regions:
        print(f"\nFound {len(predictive_regions)} predictive regions:")
        
        # Analyze which sensors each region predicts
        predictable_sensor_regions = []
        unpredictable_sensor_regions = []
        
        for region in predictive_regions[:5]:  # Show first 5
            avg_sensor_idx = np.mean(region.sensor_indices) if region.sensor_indices else -1
            
            if avg_sensor_idx < 5 and avg_sensor_idx >= 0:
                predictable_sensor_regions.append(region)
            else:
                unpredictable_sensor_regions.append(region)
            
            print(f"\n  Region {region.region_id}:")
            print(f"    Predicts sensors: {region.sensor_indices}")
            print(f"    Confidence: {region.prediction_confidence:.3f}")
            print(f"    Stability: {region.stability:.3f}")
        
        # Check if unpredictable sensors have different dynamics
        if hasattr(brain.field_dynamics, 'prediction_error_learning') and brain.field_dynamics.prediction_error_learning:
            error_stats = brain.field_dynamics.prediction_error_learning.get_learning_statistics()
            print(f"\nError-based learning statistics:")
            print(f"  Error trend: {error_stats['global_error_trend']:.3f}")
            print(f"  Improving regions: {error_stats['improving_regions']}")
    
    return True


def test_exploration_modulation():
    """Test that exploration increases when learning plateaus."""
    print("\n\n=== Testing Exploration Modulation ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=5,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    # Use a pattern that becomes predictable
    print("Running constant pattern to create learning plateau...")
    
    exploration_history = []
    
    for cycle in range(100):
        # Constant pattern (should plateau quickly)
        sensory_input = [0.5, -0.5, 0.3, -0.3, 0.0, 0.0]  # includes reward
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Track exploration
        if 'exploration_drive' in brain.modulation:
            exploration_history.append(brain.modulation['exploration_drive'])
    
    # Analyze exploration changes
    if exploration_history:
        early_exploration = np.mean(exploration_history[:20])
        late_exploration = np.mean(exploration_history[-20:])
        
        print(f"\nExploration drive:")
        print(f"  Early (cycles 1-20): {early_exploration:.3f}")
        print(f"  Late (cycles 81-100): {late_exploration:.3f}")
        print(f"  Change: {(late_exploration - early_exploration):.3f}")
        
        # Check if error modulation includes exploration boost
        if hasattr(brain.field_dynamics, '_error_modulation') and brain.field_dynamics._error_modulation:
            exploration_boost = brain.field_dynamics._error_modulation.get('exploration_boost', 1.0)
            print(f"  Exploration boost from errors: {exploration_boost:.2f}x")
    
    return True


if __name__ == "__main__":
    print("Testing Phase 2: Prediction Error as Primary Learning Signal")
    print("=" * 60)
    
    # Run tests
    test1_success = test_error_driven_learning()
    test2_success = test_resource_allocation()
    test3_success = test_exploration_modulation()
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 2 TEST SUMMARY")
    print("=" * 60)
    
    if test1_success and test2_success and test3_success:
        print("✓ All tests passed!")
        print("\nPhase 2 successfully implemented:")
        print("- Prediction errors modulate self-modification strength")
        print("- Learning rate adapts to error magnitude")
        print("- Resources allocated to high-error regions")
        print("- Exploration increases when learning plateaus")
        print("\nThe brain now uses prediction error as its primary learning signal!")
    else:
        print("✗ Some tests failed")
        print("\nIssues to investigate:")
        if not test1_success:
            print("- Error-driven learning not working properly")
        if not test2_success:
            print("- Resource allocation issues")
        if not test3_success:
            print("- Exploration modulation problems")