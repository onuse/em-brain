#!/usr/bin/env python3
"""
Test the tension-based strategic planning system.

This tests that the field's intrinsic drives (tensions) properly guide
pattern discovery and evaluation without external reward signals.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from server.src.brains.field.field_strategic_planner import FieldStrategicPlanner


def test_tension_measurement():
    """Test that field tensions are correctly measured."""
    print("Testing tension measurement...")
    
    # Create planner
    field_shape = (16, 16, 16, 64)
    planner = FieldStrategicPlanner(field_shape, sensory_dim=10, motor_dim=4)
    
    # Test 1: Low energy field should have high information tension
    low_energy_field = torch.zeros(field_shape, device=planner.device)
    low_energy_field[:, :, :, :32] = 0.1  # Very low content energy
    
    tensions = planner._measure_field_tensions(low_energy_field)
    print(f"Low energy field tensions: {tensions}")
    assert tensions['information'] > 0.7, f"Expected high information tension, got {tensions['information']}"
    
    # Test 2: High energy field should have low information tension
    high_energy_field = torch.ones(field_shape, device=planner.device) * 0.8
    tensions = planner._measure_field_tensions(high_energy_field)
    print(f"High energy field tensions: {tensions}")
    assert tensions['information'] < 0.3, f"Expected low information tension, got {tensions['information']}"
    
    # Test 3: High prediction errors should create prediction tension
    error_field = torch.zeros(field_shape, device=planner.device)
    error_field[:, :, :, 48:52] = 0.8  # High error signals
    tensions = planner._measure_field_tensions(error_field)
    print(f"High error field tensions: {tensions}")
    assert tensions['prediction'] > 0.5, f"Expected high prediction tension, got {tensions['prediction']}"
    
    print("✓ Tension measurement tests passed\n")


def test_tension_targeted_pattern_generation():
    """Test that patterns are generated to address specific tensions."""
    print("Testing tension-targeted pattern generation...")
    
    # Create planner
    field_shape = (16, 16, 16, 64)
    planner = FieldStrategicPlanner(field_shape, sensory_dim=10, motor_dim=4)
    
    # Test different tension scenarios
    test_tensions = [
        {'information': 0.9, 'learning': 0.2, 'confidence': 0.3, 'prediction': 0.2, 'novelty': 0.1, 'total': 0.34},
        {'information': 0.2, 'learning': 0.9, 'confidence': 0.3, 'prediction': 0.2, 'novelty': 0.1, 'total': 0.34},
        {'information': 0.2, 'learning': 0.2, 'confidence': 0.9, 'prediction': 0.2, 'novelty': 0.1, 'total': 0.34},
    ]
    
    expected_patterns = ['information', 'learning', 'confidence']
    
    for tensions, expected in zip(test_tensions, expected_patterns):
        pattern = planner._generate_tension_targeted_pattern(tensions)
        
        # Check pattern has correct shape
        assert pattern.shape == (16, 16, 16, 16), f"Wrong pattern shape: {pattern.shape}"
        
        # Check pattern has meaningful content
        assert pattern.abs().mean() > 0.01, "Pattern is too weak"
        assert pattern.abs().max() < 2.0, "Pattern is too strong"
        
        print(f"✓ Generated {expected}-targeted pattern with mean activation {pattern.abs().mean():.3f}")
    
    print("✓ Pattern generation tests passed\n")


def test_tension_based_evaluation():
    """Test that pattern evaluation is based on tension reduction."""
    print("Testing tension-based pattern evaluation...")
    
    # Create planner
    field_shape = (32, 32, 32, 64)
    planner = FieldStrategicPlanner(field_shape, sensory_dim=10, motor_dim=4)
    
    # Create a field with high information tension
    tense_field = torch.zeros(field_shape, device=planner.device)
    tense_field[:, :, :, :32] = 0.1  # Low energy creates tension
    
    initial_tensions = planner._measure_field_tensions(tense_field)
    print(f"Initial field tensions: {initial_tensions}")
    
    # Generate a pattern to address the tension
    good_pattern = planner._generate_tension_targeted_pattern(initial_tensions)
    
    # Also create a bad pattern that doesn't help
    bad_pattern = torch.zeros_like(good_pattern)
    
    # Evaluate both patterns
    good_score, _, _ = planner._evaluate_pattern_tension_based(tense_field, good_pattern)
    bad_score, _, _ = planner._evaluate_pattern_tension_based(tense_field, bad_pattern)
    
    print(f"Good pattern score: {good_score:.2f}")
    print(f"Bad pattern score: {bad_score:.2f}")
    
    # Good pattern should score higher
    assert good_score > bad_score, f"Good pattern ({good_score}) should score higher than bad pattern ({bad_score})"
    
    print("✓ Tension-based evaluation tests passed\n")


def test_full_discovery_cycle():
    """Test complete pattern discovery with tension-based system."""
    print("Testing full discovery cycle...")
    
    # Create planner
    field_shape = (16, 16, 16, 64)
    planner = FieldStrategicPlanner(field_shape, sensory_dim=10, motor_dim=4)
    
    # Create field with multiple tensions
    field = torch.zeros(field_shape, device=planner.device)
    field[:, :, :, :32] = 0.2  # Low energy
    field[:, :, :, 48:52] = 0.3  # Some errors
    field[:, :, :, 58] = 0.3  # Low confidence
    
    initial_tensions = planner._measure_field_tensions(field)
    print(f"Initial tensions: {initial_tensions}")
    
    # Discover pattern (reward_signal is ignored internally)
    pattern = planner.discover_strategic_pattern(
        field, 
        reward_signal=0.0,  # Not used anymore!
        exploration_level=0.5,
        n_candidates=8
    )
    
    assert pattern is not None, "Failed to discover pattern"
    print(f"Discovered pattern with score: {pattern.score:.2f}")
    print(f"Pattern persistence: {pattern.persistence:.1f} cycles")
    print(f"Behavioral signature: {pattern.behavioral_signature}")
    
    # Check that pattern library was updated
    assert len(planner.pattern_library) > 0, "Pattern not added to library"
    
    print("✓ Full discovery cycle tests passed\n")


def test_intrinsic_motivation_emergence():
    """Test that behavior emerges from tension resolution, not rewards."""
    print("Testing intrinsic motivation emergence...")
    
    # Create planner
    field_shape = (16, 16, 16, 64)
    planner = FieldStrategicPlanner(field_shape, sensory_dim=10, motor_dim=4)
    
    # Scenario 1: Low information field (should generate exploration)
    low_info_field = torch.zeros(field_shape, device=planner.device)
    low_info_field[:, :, :, :32] = 0.05
    
    pattern1 = planner.discover_strategic_pattern(low_info_field, reward_signal=0.0)
    behavior1 = pattern1.behavioral_signature
    
    # Scenario 2: High confidence field (should generate exploitation)
    high_conf_field = torch.ones(field_shape, device=planner.device) * 0.5
    high_conf_field[:, :, :, 58] = 0.9  # High confidence
    
    pattern2 = planner.discover_strategic_pattern(high_conf_field, reward_signal=0.0)
    behavior2 = pattern2.behavioral_signature
    
    print(f"Low info behavior: {behavior1}")
    print(f"High conf behavior: {behavior2}")
    
    # Behaviors should be different
    behavior_diff = torch.norm(behavior1 - behavior2).item()
    assert behavior_diff > 0.1, f"Behaviors too similar (diff={behavior_diff})"
    
    print(f"✓ Different tensions create different behaviors (diff={behavior_diff:.3f})")
    print("✓ Intrinsic motivation tests passed\n")


if __name__ == "__main__":
    print("=== Testing Tension-Based Strategic Planning ===\n")
    
    # Run all tests
    test_tension_measurement()
    test_tension_targeted_pattern_generation()
    test_tension_based_evaluation()
    test_full_discovery_cycle()
    test_intrinsic_motivation_emergence()
    
    print("=== All tests passed! ===")
    print("\nThe field brain now operates on intrinsic drives alone.")
    print("No external rewards needed - behavior emerges from tension resolution.")