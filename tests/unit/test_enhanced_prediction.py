#!/usr/bin/env python3
"""
Test Enhanced Prediction System - Phase 1 Implementation

Tests the closed prediction loop where topology regions learn to predict sensors.
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


def test_prediction_improvement():
    """Test that prediction confidence improves over time with predictable input."""
    print("\n=== Testing Enhanced Prediction System ===\n")
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=10,
        motor_dim=3,
        spatial_resolution=32,
        quiet_mode=False
    )
    
    print("Initial state:")
    print(f"  Prediction confidence: {brain._current_prediction_confidence:.3f}")
    print(f"  Prediction error: {brain._last_prediction_error:.3f}")
    
    # Generate predictable sensory pattern (sine waves)
    def generate_sensory_pattern(cycle):
        pattern = []
        for i in range(10):
            # Each sensor has different frequency
            value = 0.5 * np.sin(2 * np.pi * (i + 1) * cycle / 20.0)
            pattern.append(value)
        # Add reward
        pattern.append(0.0)
        return pattern
    
    # Track metrics
    confidence_history = []
    error_history = []
    predictive_regions_history = []
    
    print("\nRunning 100 cycles with predictable sensory input...")
    
    for cycle in range(100):
        # Generate predictable input
        sensory_input = generate_sensory_pattern(cycle)
        
        # Process cycle
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Track metrics
        confidence_history.append(brain._current_prediction_confidence)
        error_history.append(brain._last_prediction_error)
        
        # Check topology statistics
        topology_stats = brain.topology_region_system.get_statistics()
        predictive_regions_history.append(topology_stats.get('predictive_regions', 0))
        
        # Progress report every 20 cycles
        if (cycle + 1) % 20 == 0:
            avg_confidence = np.mean(confidence_history[-20:])
            avg_error = np.mean(error_history[-20:])
            num_predictive = topology_stats.get('predictive_regions', 0)
            avg_region_confidence = topology_stats.get('avg_prediction_confidence', 0.0)
            
            print(f"\nCycle {cycle + 1}:")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Average error: {avg_error:.3f}")
            print(f"  Predictive regions: {num_predictive}")
            print(f"  Region confidence: {avg_region_confidence:.3f}")
            print(f"  Total regions: {topology_stats['total_regions']}")
    
    # Analyze results
    print("\n=== Results Analysis ===")
    
    # Check if prediction improved
    early_confidence = np.mean(confidence_history[:20])
    late_confidence = np.mean(confidence_history[-20:])
    confidence_improvement = late_confidence - early_confidence
    
    early_error = np.mean(error_history[:20])
    late_error = np.mean(error_history[-20:])
    error_reduction = early_error - late_error
    
    print(f"\nConfidence improvement: {confidence_improvement:.3f} ({early_confidence:.3f} → {late_confidence:.3f})")
    print(f"Error reduction: {error_reduction:.3f} ({early_error:.3f} → {late_error:.3f})")
    print(f"Final predictive regions: {predictive_regions_history[-1]}")
    
    # Test specific predictions
    print("\n=== Testing Specific Predictions ===")
    
    # Get current predictions
    if brain._predicted_sensory is not None:
        print("\nPredicted vs Actual (first 5 sensors):")
        next_input = generate_sensory_pattern(100)
        for i in range(min(5, len(brain._predicted_sensory))):
            predicted = brain._predicted_sensory[i].item()
            actual = next_input[i]
            error = abs(predicted - actual)
            sensor_confidence = brain._prediction_confidence_per_sensor[i].item() if brain._prediction_confidence_per_sensor is not None else 0.0
            print(f"  Sensor {i}: predicted={predicted:.3f}, actual={actual:.3f}, error={error:.3f}, confidence={sensor_confidence:.3f}")
    
    # Check which regions are predicting
    predictive_regions = brain.topology_region_system.get_predictive_regions()
    if predictive_regions:
        print(f"\n{len(predictive_regions)} regions are making predictions:")
        for i, region in enumerate(predictive_regions[:3]):  # Show first 3
            print(f"  Region {region.region_id}:")
            print(f"    Sensors: {region.sensor_indices}")
            print(f"    Confidence: {region.prediction_confidence:.3f}")
            print(f"    Stability: {region.stability:.3f}")
    
    # Success criteria
    print("\n=== Success Criteria ===")
    success = True
    
    if late_confidence > early_confidence:
        print("✓ Confidence improved over time")
    else:
        print("✗ Confidence did not improve")
        success = False
    
    if late_error < early_error:
        print("✓ Prediction error decreased")
    else:
        print("✗ Prediction error did not decrease")
        success = False
    
    if predictive_regions_history[-1] > 0:
        print("✓ Regions learned to predict sensors")
    else:
        print("✗ No predictive regions formed")
        success = False
    
    if late_confidence > 0.0:  # Was 0% before
        print("✓ Non-zero confidence achieved")
    else:
        print("✗ Still at 0% confidence")
        success = False
    
    return success


def test_unpredictable_input():
    """Test behavior with random unpredictable input."""
    print("\n\n=== Testing with Unpredictable Input ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=5,
        motor_dim=2,
        spatial_resolution=32,
        quiet_mode=True
    )
    
    confidence_history = []
    
    print("Running 50 cycles with random input...")
    
    for cycle in range(50):
        # Generate random input
        sensory_input = [np.random.uniform(-1, 1) for _ in range(5)]
        sensory_input.append(0.0)  # reward
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        confidence_history.append(brain._current_prediction_confidence)
    
    avg_confidence = np.mean(confidence_history)
    print(f"\nAverage confidence with random input: {avg_confidence:.3f}")
    print("Expected: Low confidence due to unpredictability")
    
    return avg_confidence < 0.3  # Should maintain low confidence


if __name__ == "__main__":
    print("Testing Enhanced Prediction System - Phase 1")
    print("=" * 50)
    
    # Run tests
    test1_success = test_prediction_improvement()
    test2_success = test_unpredictable_input()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if test1_success and test2_success:
        print("✓ All tests passed!")
        print("\nThe enhanced prediction system is working:")
        print("- Topology regions learn to predict sensors")
        print("- Confidence improves with predictable patterns")
        print("- Low confidence maintained for unpredictable input")
        print("- Prediction errors drive learning")
    else:
        print("✗ Some tests failed")
        print("\nIssues to investigate:")
        if not test1_success:
            print("- Prediction not improving with predictable input")
        if not test2_success:
            print("- Unexpected behavior with random input")